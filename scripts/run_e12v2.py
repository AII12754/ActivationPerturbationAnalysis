#!/usr/bin/env python3
"""E12v2: Standalone trigram pipeline experiment with natural lengths + decode phase.

Self-contained script — no dependency on BaseExperiment framework.
Uses datasets at their natural lengths (no concatenation/truncation).
Separate warmup and test phases. Includes 128-token decode phase.

Output: results_e12v2/{dataset_name}/*.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from activation_science.core.extraction import (
    ActivationBatch,
    decode_step,
    prefill,
    select_next_token,
)
from activation_science.metrics.delta_coding import (
    DeltaPacket,
    compute_affine_params,
    apply_affine,
    compute_delta,
    compute_reconstruction_quality,
    compute_transfer_size,
    compute_transfer_size_int8_outlier,
    groupwise_int4_dequantize_topk,
    groupwise_int4_quantize_topk,
    groupwise_int8_dequantize_topk,
    groupwise_int8_quantize_topk,
    reconstruct_activation,
)
from activation_science.metrics.ngram_table import NgramTable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("e12v2")

# ===================================================================
# Config defaults
# ===================================================================
DEFAULTS = {
    "warmup_requests": 100,
    "test_requests": 100,
    "decode_tokens": 128,
    "layer_boundary": 6,
    "group_size": 128,
    "top_k": 1,
    "int8_group_size": 128,
    "int8_outlier_top_k": 1,
    "max_seq_len": 4096,
    "model_path": "Qwen/Qwen2.5-32B-Instruct",
    "gpu": 0,
    "seed": 42,
    "output_dir": "results_e12v2",
}

# ===================================================================
# Dataset loading
# ===================================================================
DATASET_ROOT = Path("/root/share/dataset")


def load_dataset_texts(name: str) -> List[str]:
    """Load raw texts from each dataset using pyarrow/json directly."""

    if name == "cnn_dm":
        files = sorted(DATASET_ROOT.glob("cnndn/3.0.0/train-*.parquet"))
        texts = []
        for f in files:
            tbl = pq.read_table(f, columns=["article"])
            texts.extend(tbl.column("article").to_pylist())
        return texts

    if name == "sharegpt":
        path = DATASET_ROOT / "ShareGPT_Vicuna_unfiltered" / "ShareGPT_V3_unfiltered_cleaned_split.json"
        with open(path, "r") as f:
            data = json.load(f)
        texts = []
        for item in data:
            convs = item.get("conversations", [])
            joined = "\n".join(turn.get("value", "") for turn in convs)
            if joined.strip():
                texts.append(joined)
        return texts

    if name == "wikitext2":
        path = DATASET_ROOT / "wikitext-2-raw-v1" / "test-00000-of-00001.parquet"
        tbl = pq.read_table(path, columns=["text"])
        texts = [t for t in tbl.column("text").to_pylist() if t and t.strip()]
        return texts

    if name == "gsm8k":
        path = DATASET_ROOT / "gsm8k" / "main" / "train-00000-of-00001.parquet"
        tbl = pq.read_table(path, columns=["question", "answer"])
        questions = tbl.column("question").to_pylist()
        answers = tbl.column("answer").to_pylist()
        return [f"{q}\n{a}" for q, a in zip(questions, answers)]

    if name == "triviaqa":
        files = sorted(DATASET_ROOT.glob("trivia_qa-rc/test-*.parquet"))
        texts = []
        for f in files:
            tbl = pq.read_table(f, columns=["question", "entity_pages"])
            questions = tbl.column("question").to_pylist()
            entity_pages = tbl.column("entity_pages").to_pylist()
            for q, ep in zip(questions, entity_pages):
                parts = [q]
                if ep and isinstance(ep, dict):
                    contexts = ep.get("wiki_context", [])
                    if contexts and isinstance(contexts, list):
                        # Join first context paragraph(s)
                        for ctx in contexts[:2]:
                            if ctx:
                                parts.append(str(ctx)[:3000])
                joined = "\n".join(parts)
                if joined.strip():
                    texts.append(joined)
        return texts

    if name == "alpaca":
        files = sorted(DATASET_ROOT.glob("alpaca/data/train-*.parquet"))
        texts = []
        for f in files:
            tbl = pq.read_table(f)
            cols = tbl.column_names
            for i in range(len(tbl)):
                parts = []
                for col in ["instruction", "input", "output"]:
                    if col in cols:
                        val = tbl.column(col)[i].as_py()
                        if val:
                            parts.append(val)
                if not parts:
                    # fallback: use 'text' column if available
                    if "text" in cols:
                        val = tbl.column("text")[i].as_py()
                        if val:
                            parts.append(val)
                if parts:
                    texts.append("\n".join(parts))
        return texts

    raise ValueError(f"Unknown dataset: {name}")


ALL_DATASETS = ["cnn_dm", "sharegpt", "wikitext2", "gsm8k", "triviaqa", "alpaca"]


# ===================================================================
# Table schemas
# ===================================================================
PIPELINE_QUALITY_COLUMNS = [
    "dataset_name", "request_index", "phase",
    "seq_len", "hidden_dim", "layer_boundary",
    "group_size", "top_k", "int8_group_size", "int8_outlier_top_k",
    "num_trigram", "num_bigram", "num_self_ref", "num_unigram",
    "pct_trigram", "pct_bigram", "pct_self_ref", "pct_unigram",
    "raw_cosine_mean", "raw_cosine_min",
    "cosine_similarity_mean", "cosine_similarity_min",
    "mse_mean", "mse_max",
    "total_transfer_bytes", "raw_fp16_bytes", "compression_ratio",
    "transfer_bytes_trigram", "transfer_bytes_bigram",
    "transfer_bytes_self_ref", "transfer_bytes_unigram",
]

TIER_DETAIL_COLUMNS = [
    "dataset_name", "request_index", "phase", "layer_boundary",
    "tier", "count",
    "raw_cosine_mean", "raw_cosine_min",
    "recon_cosine_mean", "recon_cosine_min",
    "mse_mean", "mse_max",
    "transfer_bytes",
]

TABLE_GROWTH_COLUMNS = [
    "dataset_name", "request_index", "phase",
    "num_trigrams", "num_bigrams", "memory_bytes",
    "trigram_coverage", "bigram_coverage",
    "new_trigrams_added", "new_bigrams_added", "update_time_ms",
    "evicted_count",
]

LATENCY_COLUMNS = [
    "dataset_name", "request_index", "phase",
    "prefill_ms", "classify_ms",
    "encode_delta_ms", "encode_self_ref_ms", "encode_unigram_ms",
    "decode_generate_ms", "table_update_ms", "total_ms",
    "seq_len",
]


# ===================================================================
# Core pipeline: encode/decode/measure one request
# ===================================================================
@torch.inference_mode()
def process_request(
    model,
    tokenizer,
    ngram_table: NgramTable,
    text: str,
    cfg: Dict[str, Any],
    device: torch.device,
    executor: ThreadPoolExecutor,
    dataset_name: str,
    req_idx: int,
    phase: str,
) -> Tuple[
    Optional[Dict], Optional[Dict], Dict, Optional[Dict],
    List[int],  # all_token_ids (prefill + decode)
    torch.Tensor,  # all hidden states for table update
]:
    """Process one request: prefill + encode/decode + generate.

    Returns (pipeline_quality_rec, tier_detail_recs, table_growth_rec, latency_rec,
             all_token_ids, all_hidden_states).
    For warmup phase, quality/tier/latency records are None.
    """
    layer_boundary = cfg["layer_boundary"]
    group_size = cfg["group_size"]
    top_k = cfg["top_k"]
    int8_group_size = cfg["int8_group_size"]
    int8_outlier_top_k = cfg["int8_outlier_top_k"]
    max_seq_len = cfg["max_seq_len"]
    decode_tokens = cfg["decode_tokens"]
    hidden_dim = model.config.hidden_size
    is_test = (phase == "test")

    t_total_start = time.perf_counter()

    # Tokenize (natural length, capped)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    seq_len = len(input_ids)

    # ------------------------------------------------------------------
    # Phase 1: Prefill
    # ------------------------------------------------------------------
    if is_test:
        # Launch classify on background CPU thread (reads existing table state)
        classify_future = executor.submit(
            ngram_table.classify_and_build_refs,
            input_ids, hidden_dim,
        )

    torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()
    batch = prefill(model, input_tensor, use_cache=True)
    torch.cuda.synchronize()
    t_prefill_ms = (time.perf_counter() - t_prefill_start) * 1000.0

    layer_idx = min(layer_boundary, batch.num_layers)
    prefill_hidden = batch.hidden_states[layer_idx].squeeze(0).to(torch.float16)  # (seq_len, hidden_dim)
    real_acts = prefill_hidden

    # ------------------------------------------------------------------
    # Online table update #1: immediately after prefill
    # This makes prefill trigrams available for decode-phase lookups.
    # ------------------------------------------------------------------
    t_update_start = time.perf_counter()
    total_new_tri, total_new_bi = 0, 0
    total_update_ms = 0.0

    new_tri, new_bi, upd_ms = ngram_table.update_from_hidden_states(
        input_ids, prefill_hidden,
    )
    total_new_tri += new_tri
    total_new_bi += new_bi
    total_update_ms += upd_ms

    # ------------------------------------------------------------------
    # Phase 2: Decode (generate tokens) with online table updates
    # After each decode step, update table so the next step benefits.
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_decode_gen_start = time.perf_counter()

    past_kv = batch.past_key_values
    next_tok = select_next_token(batch.last_logits, do_sample=False)
    del batch

    # Running sequence of all token IDs seen so far (prefill + generated)
    running_token_ids: List[int] = list(input_ids)
    decode_hiddens: List[torch.Tensor] = []
    decode_token_ids: List[int] = []

    for _step in range(decode_tokens):
        dbatch = decode_step(model, next_tok, past_kv)
        h = dbatch.hidden_states[layer_idx][0, 0].to(torch.float16)  # (hidden_dim,)
        decode_hiddens.append(h)
        tok_id = next_tok.item()
        decode_token_ids.append(tok_id)
        running_token_ids.append(tok_id)

        # Online table update: insert this step's trigram immediately
        # Need at least 3 tokens in running sequence to form a trigram
        if len(running_token_ids) >= 3:
            a = running_token_ids[-3]
            b = running_token_ids[-2]
            c = running_token_ids[-1]
            if not ngram_table.has_trigram(a, b, c):
                # Get the hidden state for bigram position (previous decode or last prefill)
                if len(decode_hiddens) >= 2:
                    bi_hidden = decode_hiddens[-2]
                else:
                    # Position b is the last prefill token or first decode
                    bi_pos = len(input_ids) + len(decode_hiddens) - 2
                    if bi_pos < len(input_ids):
                        bi_hidden = prefill_hidden[bi_pos]
                    else:
                        bi_hidden = decode_hiddens[bi_pos - len(input_ids)]
                node = ngram_table._get_or_create_node(a, b, bi_hidden.to(ngram_table.dtype).detach())
                if c not in node.suffixes:
                    node.suffixes[c] = h.to(ngram_table.dtype).detach()
                    ngram_table._num_trigrams += 1
                    total_new_tri += 1

        past_kv = dbatch.past_key_values
        next_tok = select_next_token(dbatch.last_logits, do_sample=False)
        del dbatch

    # Append final generated token (no hidden state for it)
    decode_token_ids.append(next_tok.item())

    torch.cuda.synchronize()
    t_decode_gen_ms = (time.perf_counter() - t_decode_gen_start) * 1000.0

    t_update_ms = (time.perf_counter() - t_update_start) * 1000.0

    del past_kv, next_tok

    # Build combined token IDs and hidden states (for records)
    all_token_ids = input_ids + decode_token_ids
    if decode_hiddens:
        decode_hidden_tensor = torch.stack(decode_hiddens)  # (decode_tokens, hidden_dim)
        all_hidden = torch.cat([prefill_hidden, decode_hidden_tensor], dim=0)
    else:
        all_hidden = prefill_hidden

    # ------------------------------------------------------------------
    # Warmup: just return table update info, no metrics
    # ------------------------------------------------------------------
    if not is_test:
        tbl_stats = ngram_table.stats
        table_rec = {
            "dataset_name": dataset_name, "request_index": req_idx, "phase": phase,
            "num_trigrams": tbl_stats["num_trigrams"],
            "num_bigrams": tbl_stats["num_bigrams"],
            "memory_bytes": tbl_stats["memory_bytes"],
            "trigram_coverage": 0.0, "bigram_coverage": 0.0,
            "new_trigrams_added": total_new_tri, "new_bigrams_added": total_new_bi,
            "update_time_ms": total_update_ms, "evicted_count": ngram_table._last_evicted,
        }
        return None, None, table_rec, None, all_token_ids, all_hidden

    # ------------------------------------------------------------------
    # Test phase: classify + encode + measure
    # ------------------------------------------------------------------
    t_classify_start = time.perf_counter()
    tiers, ref_acts, self_ref_sources, first_occ_map = classify_future.result()
    t_classify_ms = (time.perf_counter() - t_classify_start) * 1000.0

    trigram_indices = [i for i, t in enumerate(tiers) if t == "trigram"]
    bigram_indices = [i for i, t in enumerate(tiers) if t == "bigram"]
    self_ref_indices = [i for i, t in enumerate(tiers) if t == "self_ref"]
    unigram_indices = [i for i, t in enumerate(tiers) if t == "unigram"]

    reconstructed = torch.zeros_like(real_acts)
    transfer_bytes_by_tier: Dict[str, int] = {
        "trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0,
    }

    # ------------------------------------------------------------------
    # Encode TRIGRAM + BIGRAM
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_encode_delta_start = time.perf_counter()

    delta_indices = trigram_indices + bigram_indices
    if delta_indices:
        idx_t = torch.tensor(delta_indices, dtype=torch.long, device=device)
        real_batch = real_acts[idx_t]
        ref_batch = ref_acts[idx_t]

        scale, bias = compute_affine_params(real_batch, ref_batch)
        ref_t = apply_affine(ref_batch, scale, bias)
        delta = compute_delta(real_batch, ref_t)
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)

        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, hidden_dim)
        recon_batch = reconstruct_activation(dequant, ref_batch, scale, bias).to(torch.float16)
        reconstructed[idx_t] = recon_batch

        n_delta = len(delta_indices)
        packet = DeltaPacket(
            quantized_data=packed, scales=scales, zero_points=zeros,
            topk_values=tv, topk_indices=ti,
            affine_scale=scale.to(torch.float16),
            affine_bias=bias.to(torch.float16),
            ref_indices=torch.zeros(n_delta, dtype=torch.long, device=device),
            group_size=group_size, top_k=top_k,
        )
        total_delta_bytes = compute_transfer_size(packet)
        if n_delta > 0:
            per_pos = total_delta_bytes / n_delta
            transfer_bytes_by_tier["trigram"] = int(per_pos * len(trigram_indices))
            transfer_bytes_by_tier["bigram"] = int(per_pos * len(bigram_indices))

    torch.cuda.synchronize()
    t_encode_delta_ms = (time.perf_counter() - t_encode_delta_start) * 1000.0

    # ------------------------------------------------------------------
    # Encode UNIGRAM (Int8 + outliers)
    # ------------------------------------------------------------------
    t_encode_unigram_start = time.perf_counter()
    if unigram_indices:
        idx_u = torch.tensor(unigram_indices, dtype=torch.long, device=device)
        real_uni = real_acts[idx_u]
        int8_pkt = groupwise_int8_quantize_topk(real_uni, int8_group_size, int8_outlier_top_k)
        recon_uni = groupwise_int8_dequantize_topk(int8_pkt)
        transfer_bytes_by_tier["unigram"] = compute_transfer_size_int8_outlier(int8_pkt)
        reconstructed[idx_u] = recon_uni

    torch.cuda.synchronize()
    t_encode_unigram_ms = (time.perf_counter() - t_encode_unigram_start) * 1000.0

    # ------------------------------------------------------------------
    # Encode SELF_REF
    # ------------------------------------------------------------------
    t_encode_self_ref_start = time.perf_counter()
    if self_ref_indices:
        sorted_self_ref = sorted(self_ref_indices)
        idx_sr = torch.tensor(sorted_self_ref, dtype=torch.long, device=device)
        real_sr = real_acts[idx_sr]

        source_positions = [self_ref_sources[pos] for pos in sorted_self_ref]
        src_t = torch.tensor(source_positions, dtype=torch.long, device=device)
        ref_sr = reconstructed[src_t]
        ref_acts[idx_sr] = ref_sr

        scale, bias = compute_affine_params(real_sr, ref_sr)
        ref_t = apply_affine(ref_sr, scale, bias)
        delta = compute_delta(real_sr, ref_t)
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)

        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, hidden_dim)
        recon_sr = reconstruct_activation(dequant, ref_sr, scale, bias).to(torch.float16)
        reconstructed[idx_sr] = recon_sr

        n_sr = len(sorted_self_ref)
        pkt = DeltaPacket(
            quantized_data=packed, scales=scales, zero_points=zeros,
            topk_values=tv, topk_indices=ti,
            affine_scale=scale.to(torch.float16),
            affine_bias=bias.to(torch.float16),
            ref_indices=torch.zeros(n_sr, dtype=torch.long, device=device),
            group_size=group_size, top_k=top_k,
        )
        transfer_bytes_by_tier["self_ref"] = compute_transfer_size(pkt)

    torch.cuda.synchronize()
    t_encode_self_ref_ms = (time.perf_counter() - t_encode_self_ref_start) * 1000.0

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------
    base_meta = {
        "dataset_name": dataset_name,
        "request_index": req_idx,
        "phase": phase,
        "layer_boundary": layer_boundary,
    }

    tier_detail_records = []
    for tier_name, tier_indices in [
        ("trigram", trigram_indices),
        ("bigram", bigram_indices),
        ("self_ref", self_ref_indices),
        ("unigram", unigram_indices),
    ]:
        if not tier_indices:
            tier_detail_records.append({
                **base_meta,
                "tier": tier_name, "count": 0,
                "raw_cosine_mean": 0.0, "raw_cosine_min": 0.0,
                "recon_cosine_mean": 0.0, "recon_cosine_min": 0.0,
                "mse_mean": 0.0, "mse_max": 0.0,
                "transfer_bytes": 0,
            })
            continue

        idx_t = torch.tensor(tier_indices, dtype=torch.long, device=device)
        real_tier = real_acts[idx_t]
        ref_tier = ref_acts[idx_t]
        recon_tier = reconstructed[idx_t]

        if tier_name == "unigram":
            raw_cos = torch.zeros(len(tier_indices), device=device)
        else:
            raw_cos = F.cosine_similarity(real_tier.float(), ref_tier.float(), dim=-1)

        recon_cos = F.cosine_similarity(real_tier.float(), recon_tier.float(), dim=-1)
        mse = ((real_tier.float() - recon_tier.float()) ** 2).mean(dim=-1)

        tier_detail_records.append({
            **base_meta,
            "tier": tier_name,
            "count": len(tier_indices),
            "raw_cosine_mean": raw_cos.mean().item(),
            "raw_cosine_min": raw_cos.min().item() if raw_cos.numel() > 0 else 0.0,
            "recon_cosine_mean": recon_cos.mean().item(),
            "recon_cosine_min": recon_cos.min().item(),
            "mse_mean": mse.mean().item(),
            "mse_max": mse.max().item(),
            "transfer_bytes": transfer_bytes_by_tier[tier_name],
        })

    # Aggregate quality
    non_uni_indices = trigram_indices + bigram_indices + sorted(self_ref_indices)
    if non_uni_indices:
        idx_nu = torch.tensor(non_uni_indices, dtype=torch.long, device=device)
        raw_cos_all = F.cosine_similarity(
            real_acts[idx_nu].float(), ref_acts[idx_nu].float(), dim=-1,
        )
        raw_cosine_mean = raw_cos_all.mean().item()
        raw_cosine_min = raw_cos_all.min().item()
    else:
        raw_cosine_mean = 0.0
        raw_cosine_min = 0.0

    overall_cos = F.cosine_similarity(real_acts.float(), reconstructed.float(), dim=-1)
    overall_mse = ((real_acts.float() - reconstructed.float()) ** 2).mean(dim=-1)
    total_transfer = sum(transfer_bytes_by_tier.values())
    raw_fp16 = seq_len * hidden_dim * 2

    pipeline_rec = {
        **base_meta,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "group_size": group_size,
        "top_k": top_k,
        "int8_group_size": int8_group_size,
        "int8_outlier_top_k": int8_outlier_top_k,
        "num_trigram": len(trigram_indices),
        "num_bigram": len(bigram_indices),
        "num_self_ref": len(self_ref_indices),
        "num_unigram": len(unigram_indices),
        "pct_trigram": len(trigram_indices) / max(seq_len, 1) * 100,
        "pct_bigram": len(bigram_indices) / max(seq_len, 1) * 100,
        "pct_self_ref": len(self_ref_indices) / max(seq_len, 1) * 100,
        "pct_unigram": len(unigram_indices) / max(seq_len, 1) * 100,
        "raw_cosine_mean": raw_cosine_mean,
        "raw_cosine_min": raw_cosine_min,
        "cosine_similarity_mean": overall_cos.mean().item(),
        "cosine_similarity_min": overall_cos.min().item(),
        "mse_mean": overall_mse.mean().item(),
        "mse_max": overall_mse.max().item(),
        "total_transfer_bytes": total_transfer,
        "raw_fp16_bytes": raw_fp16,
        "compression_ratio": raw_fp16 / max(total_transfer, 1),
        "transfer_bytes_trigram": transfer_bytes_by_tier["trigram"],
        "transfer_bytes_bigram": transfer_bytes_by_tier["bigram"],
        "transfer_bytes_self_ref": transfer_bytes_by_tier["self_ref"],
        "transfer_bytes_unigram": transfer_bytes_by_tier["unigram"],
    }

    # Table update already done online (after prefill + each decode step)

    total_trigrams_in_request = max(seq_len - 2, 0)
    total_bigrams_in_request = max(seq_len - 1, 0)
    tri_hits = len(trigram_indices) + len(self_ref_indices)
    bi_hits = len(bigram_indices)

    tbl_stats = ngram_table.stats
    table_rec = {
        "dataset_name": dataset_name, "request_index": req_idx, "phase": phase,
        "num_trigrams": tbl_stats["num_trigrams"],
        "num_bigrams": tbl_stats["num_bigrams"],
        "memory_bytes": tbl_stats["memory_bytes"],
        "trigram_coverage": tri_hits / max(total_trigrams_in_request, 1),
        "bigram_coverage": bi_hits / max(total_bigrams_in_request, 1),
        "new_trigrams_added": total_new_tri, "new_bigrams_added": total_new_bi,
        "update_time_ms": total_update_ms, "evicted_count": ngram_table._last_evicted,
    }

    t_total_ms = (time.perf_counter() - t_total_start) * 1000.0
    latency_rec = {
        "dataset_name": dataset_name, "request_index": req_idx, "phase": phase,
        "prefill_ms": t_prefill_ms,
        "classify_ms": t_classify_ms,
        "encode_delta_ms": t_encode_delta_ms,
        "encode_self_ref_ms": t_encode_self_ref_ms,
        "encode_unigram_ms": t_encode_unigram_ms,
        "decode_generate_ms": t_decode_gen_ms,
        "table_update_ms": t_update_ms,
        "total_ms": t_total_ms,
        "seq_len": seq_len,
    }

    return pipeline_rec, tier_detail_records, table_rec, latency_rec, all_token_ids, all_hidden


# ===================================================================
# Main
# ===================================================================
def save_records_to_parquet(records: List[Dict], path: Path):
    """Save list of dicts to parquet via pyarrow."""
    import pyarrow as pa
    if not records:
        return
    tbl = pa.Table.from_pylist(records)
    pq.write_table(tbl, str(path))
    logger.info("Saved %d records to %s", len(records), path)


def run_dataset(
    model,
    tokenizer,
    dataset_name: str,
    cfg: Dict[str, Any],
    device: torch.device,
):
    """Run full warmup + test pipeline for one dataset."""
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("=" * 60)

    # Load texts
    t0 = time.perf_counter()
    texts = load_dataset_texts(dataset_name)
    logger.info("Loaded %d texts in %.1f s", len(texts), time.perf_counter() - t0)

    # Shuffle with fixed seed, split warmup/test
    import random
    rng = random.Random(cfg["seed"])
    rng.shuffle(texts)

    warmup_n = cfg["warmup_requests"]
    test_n = cfg["test_requests"]
    total_needed = warmup_n + test_n

    if len(texts) < total_needed:
        logger.warning(
            "Dataset %s has only %d texts, need %d. Recycling.",
            dataset_name, len(texts), total_needed,
        )
        # Cycle texts to fill
        while len(texts) < total_needed:
            texts.extend(texts[:total_needed - len(texts)])

    warmup_texts = texts[:warmup_n]
    test_texts = texts[warmup_n:warmup_n + test_n]

    # Create NgramTable
    ngram_table = NgramTable(device=device, dtype=torch.float16, max_entries=0)
    executor = ThreadPoolExecutor(max_workers=1)

    # Records
    pipeline_quality_records: List[Dict] = []
    tier_detail_records: List[Dict] = []
    table_growth_records: List[Dict] = []
    latency_records: List[Dict] = []

    # ------------------------------------------------------------------
    # Warmup phase
    # ------------------------------------------------------------------
    logger.info("--- Warmup phase: %d requests ---", warmup_n)
    for i, text in enumerate(warmup_texts):
        _, _, table_rec, _, _, _ = process_request(
            model, tokenizer, ngram_table, text, cfg, device, executor,
            dataset_name, i, "warmup",
        )
        table_growth_records.append(table_rec)

        if (i + 1) % 10 == 0:
            stats = ngram_table.stats
            logger.info(
                "Warmup %d/%d  table: %d tri, %d bi",
                i + 1, warmup_n, stats["num_trigrams"], stats["num_bigrams"],
            )

    logger.info(
        "Warmup done. Table: %d trigrams, %d bigrams",
        ngram_table.stats["num_trigrams"], ngram_table.stats["num_bigrams"],
    )

    # ------------------------------------------------------------------
    # Test phase
    # ------------------------------------------------------------------
    logger.info("--- Test phase: %d requests ---", test_n)
    for i, text in enumerate(test_texts):
        pipeline_rec, tier_recs, table_rec, latency_rec, _, _ = process_request(
            model, tokenizer, ngram_table, text, cfg, device, executor,
            dataset_name, i, "test",
        )

        if pipeline_rec is not None:
            pipeline_quality_records.append(pipeline_rec)
        if tier_recs is not None:
            if isinstance(tier_recs, list):
                tier_detail_records.extend(tier_recs)
            else:
                tier_detail_records.append(tier_recs)
        table_growth_records.append(table_rec)
        if latency_rec is not None:
            latency_records.append(latency_rec)

        if (i + 1) % 10 == 0:
            stats = ngram_table.stats
            cos = pipeline_rec["cosine_similarity_mean"] if pipeline_rec else 0
            ratio = pipeline_rec["compression_ratio"] if pipeline_rec else 0
            logger.info(
                "Test %d/%d  cos=%.4f  ratio=%.2fx  table=%d/%d",
                i + 1, test_n, cos, ratio,
                stats["num_trigrams"], stats["num_bigrams"],
            )

    executor.shutdown(wait=False)

    # Save results
    out_dir = Path(cfg["output_dir"]) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_records_to_parquet(pipeline_quality_records, out_dir / "pipeline_quality.parquet")
    save_records_to_parquet(tier_detail_records, out_dir / "tier_detail.parquet")
    save_records_to_parquet(table_growth_records, out_dir / "table_growth.parquet")
    save_records_to_parquet(latency_records, out_dir / "latency.parquet")

    logger.info("Results saved to %s", out_dir)

    # Cleanup
    del ngram_table
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="E12v2: Trigram pipeline with natural lengths + decode")
    parser.add_argument("--model", default=DEFAULTS["model_path"], help="Model path")
    parser.add_argument("--gpu", type=int, default=DEFAULTS["gpu"], help="GPU index")
    parser.add_argument("--warmup-requests", type=int, default=DEFAULTS["warmup_requests"])
    parser.add_argument("--test-requests", type=int, default=DEFAULTS["test_requests"])
    parser.add_argument("--decode-tokens", type=int, default=DEFAULTS["decode_tokens"])
    parser.add_argument("--layer-boundary", type=int, default=DEFAULTS["layer_boundary"])
    parser.add_argument("--group-size", type=int, default=DEFAULTS["group_size"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--max-seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help=f"Datasets to run. Options: {ALL_DATASETS}",
    )
    args = parser.parse_args()

    cfg = {
        "warmup_requests": args.warmup_requests,
        "test_requests": args.test_requests,
        "decode_tokens": args.decode_tokens,
        "layer_boundary": args.layer_boundary,
        "group_size": args.group_size,
        "top_k": args.top_k,
        "int8_group_size": DEFAULTS["int8_group_size"],
        "int8_outlier_top_k": DEFAULTS["int8_outlier_top_k"],
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }

    logger.info("Config: %s", json.dumps(cfg, indent=2))

    # Load model
    device = torch.device(f"cuda:{args.gpu}")
    logger.info("Loading model %s on %s...", args.model, device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded. hidden_size=%d, num_layers=%d",
                model.config.hidden_size, model.config.num_hidden_layers)

    # GPU warmup
    logger.info("GPU warmup pass...")
    warmup_ids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    _ = model(warmup_ids, output_hidden_states=True, use_cache=False)
    del warmup_ids, _
    torch.cuda.synchronize()

    # Run each dataset
    for ds_name in args.datasets:
        run_dataset(model, tokenizer, ds_name, cfg, device)

    logger.info("All done!")


if __name__ == "__main__":
    main()
