#!/usr/bin/env python3
"""E12v2-decode: Measure delta-coding quality on the DECODE phase.

For each request:
  1. Prefill → update table with prefill hidden states
  2. Decode 128 tokens, and at EACH decode step:
     a. Classify the new position (trigram/bigram/self-ref/unigram) against table
     b. If ref found: affine + Int4 delta encode/decode, measure cosine
     c. If unigram: Int8 + outlier encode/decode, measure cosine
     d. Update table with the new hidden state (online)
  3. Record per-step and per-request aggregate metrics

This answers: "How well does the trigram delta-coding pipeline compress
decode-phase activations (one token at a time)?"

Output: results_e12v2_decode/{dataset_name}/*.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

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
logger = logging.getLogger("e12v2_decode")

# ===================================================================
# Config
# ===================================================================
DEFAULTS = {
    "warmup_requests": 50,
    "test_requests": 50,
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
    "output_dir": "results_e12v2_decode",
}

# ===================================================================
# Dataset loading (reuse from run_e12v2)
# ===================================================================
DATASET_ROOT = Path("/root/share/dataset")


def load_dataset_texts(name: str) -> List[str]:
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
        return [t for t in tbl.column("text").to_pylist() if t and t.strip()]

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
                if not parts and "text" in cols:
                    val = tbl.column("text")[i].as_py()
                    if val:
                        parts.append(val)
                if parts:
                    texts.append("\n".join(parts))
        return texts

    raise ValueError(f"Unknown dataset: {name}")


ALL_DATASETS = ["cnn_dm", "sharegpt", "wikitext2", "gsm8k", "triviaqa", "alpaca"]


# ===================================================================
# Single-position encode/decode helper
# ===================================================================
def encode_decode_single(
    real_h: torch.Tensor,      # (1, hidden_dim)
    ref_h: torch.Tensor,       # (1, hidden_dim) or None
    tier: str,
    group_size: int,
    top_k: int,
    int8_group_size: int,
    int8_outlier_top_k: int,
    hidden_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Encode and decode a single position. Returns (reconstructed, transfer_bytes)."""

    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
        # Affine + Int4 delta
        scale, bias = compute_affine_params(real_h, ref_h)
        ref_t = apply_affine(ref_h, scale, bias)
        delta = compute_delta(real_h, ref_t)
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, hidden_dim)
        recon = reconstruct_activation(dequant, ref_h, scale, bias).to(torch.float16)

        pkt = DeltaPacket(
            quantized_data=packed, scales=scales, zero_points=zeros,
            topk_values=tv, topk_indices=ti,
            affine_scale=scale.to(torch.float16),
            affine_bias=bias.to(torch.float16),
            ref_indices=torch.zeros(1, dtype=torch.long, device=device),
            group_size=group_size, top_k=top_k,
        )
        transfer_bytes = compute_transfer_size(pkt)
        return recon, transfer_bytes
    else:
        # Int8 + outlier (unigram)
        int8_pkt = groupwise_int8_quantize_topk(real_h, int8_group_size, int8_outlier_top_k)
        recon = groupwise_int8_dequantize_topk(int8_pkt)
        transfer_bytes = compute_transfer_size_int8_outlier(int8_pkt)
        return recon, transfer_bytes


# ===================================================================
# Process one request
# ===================================================================
@torch.inference_mode()
def process_request(
    model,
    tokenizer,
    ngram_table: NgramTable,
    text: str,
    cfg: Dict[str, Any],
    device: torch.device,
    dataset_name: str,
    req_idx: int,
    phase: str,
) -> Tuple[Optional[List[Dict]], Optional[Dict], Dict]:
    """Process one request.

    Returns (step_records, aggregate_record, table_record).
    For warmup, step_records and aggregate_record are None.
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

    # Tokenize
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    seq_len = len(input_ids)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------
    batch = prefill(model, input_tensor, use_cache=True)
    layer_idx = min(layer_boundary, batch.num_layers)
    prefill_hidden = batch.hidden_states[layer_idx].squeeze(0).to(torch.float16)

    # Update table with prefill hidden states (online)
    ngram_table.update_from_hidden_states(input_ids, prefill_hidden)

    # ------------------------------------------------------------------
    # Decode with per-step classification + encoding
    # ------------------------------------------------------------------
    past_kv = batch.past_key_values
    next_tok = select_next_token(batch.last_logits, do_sample=False)
    del batch

    running_token_ids: List[int] = list(input_ids)
    # For self-ref: track first occurrence of each trigram in this request's decode
    first_occ_map: Dict[Tuple[int, int, int], int] = {}
    # Store reconstructed hiddens for self-ref lookups
    reconstructed_hiddens: Dict[int, torch.Tensor] = {}
    # Store real decode hiddens for table bigram node creation
    decode_hidden_by_pos: Dict[int, torch.Tensor] = {}

    step_records: List[Dict] = [] if is_test else None

    # Per-request counters
    tier_counts = {"trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0}
    tier_cosines = {"trigram": [], "bigram": [], "self_ref": [], "unigram": []}
    tier_raw_cosines = {"trigram": [], "bigram": [], "self_ref": [], "unigram": []}
    tier_bytes = {"trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0}
    total_transfer_bytes = 0
    total_raw_bytes = 0

    for step in range(decode_tokens):
        # --- Time: model forward pass ---
        torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        dbatch = decode_step(model, next_tok, past_kv)
        torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        decode_fwd_ms = (t_fwd_end - t_fwd_start) * 1000

        h = dbatch.hidden_states[layer_idx][0, 0].to(torch.float16)  # (hidden_dim,)
        tok_id = next_tok.item()
        running_token_ids.append(tok_id)
        decode_pos = len(running_token_ids) - 1  # absolute position
        decode_hidden_by_pos[decode_pos] = h

        # --- Time: classify ---
        t_classify_start = time.perf_counter()

        tier = "unigram"
        ref_h = None
        raw_cos = 0.0

        if len(running_token_ids) >= 3:
            a = running_token_ids[-3]
            b = running_token_ids[-2]
            c = running_token_ids[-1]
            trigram_key = (a, b, c)

            # 1. Trigram table lookup (highest priority)
            tri_ref = ngram_table.get_trigram(a, b, c)
            if tri_ref is not None:
                tier = "trigram"
                ref_h = tri_ref.unsqueeze(0).to(torch.float16)
                first_occ_map.setdefault(trigram_key, decode_pos)
            # 2. Self-ref: same trigram earlier in this request (priority over bigram)
            elif trigram_key in first_occ_map:
                src_pos = first_occ_map[trigram_key]
                if src_pos in reconstructed_hiddens:
                    tier = "self_ref"
                    ref_h = reconstructed_hiddens[src_pos].unsqueeze(0).to(torch.float16)
                    # Don't update first_occ_map — keep pointing to the original
            # 3. Bigram lookup (lower priority than self-ref)
            if tier == "unigram":
                if len(running_token_ids) >= 2:
                    b_tok = running_token_ids[-2]
                    c_tok = running_token_ids[-1]
                    bi_ref = ngram_table.get_bigram(b_tok, c_tok)
                    if bi_ref is not None:
                        tier = "bigram"
                        ref_h = bi_ref.unsqueeze(0).to(torch.float16)
                # Track first occurrence for bigram/unigram positions too
                first_occ_map.setdefault(trigram_key, decode_pos)

        # Compute raw cosine (ref vs real) for non-unigram
        real_h_2d = h.unsqueeze(0)  # (1, hidden_dim)
        if ref_h is not None:
            raw_cos = F.cosine_similarity(real_h_2d.float(), ref_h.float(), dim=-1).item()

        t_classify_end = time.perf_counter()
        classify_ms = (t_classify_end - t_classify_start) * 1000

        # --- Time: encode / decode ---
        torch.cuda.synchronize()
        t_encode_start = time.perf_counter()
        recon, xfer_bytes = encode_decode_single(
            real_h_2d, ref_h, tier,
            group_size, top_k, int8_group_size, int8_outlier_top_k,
            hidden_dim, device,
        )
        torch.cuda.synchronize()
        t_encode_end = time.perf_counter()
        encode_ms = (t_encode_end - t_encode_start) * 1000

        recon_cos = F.cosine_similarity(real_h_2d.float(), recon.float(), dim=-1).item()
        raw_fp16_bytes = hidden_dim * 2  # single position

        # Store reconstructed for self-ref
        reconstructed_hiddens[decode_pos] = recon.squeeze(0)

        # --- Time: online table update ---
        t_table_start = time.perf_counter()
        if len(running_token_ids) >= 3:
            a = running_token_ids[-3]
            b = running_token_ids[-2]
            c = running_token_ids[-1]
            if not ngram_table.has_trigram(a, b, c):
                # Get hidden state for position b (the bigram node)
                b_abs_pos = decode_pos - 1
                if b_abs_pos < len(input_ids):
                    bi_hidden = prefill_hidden[b_abs_pos]
                elif b_abs_pos in decode_hidden_by_pos:
                    bi_hidden = decode_hidden_by_pos[b_abs_pos]
                else:
                    bi_hidden = h  # fallback (should not happen)
                node = ngram_table._get_or_create_node(a, b, bi_hidden.to(ngram_table.dtype).detach())
                if c not in node.suffixes:
                    node.suffixes[c] = h.to(ngram_table.dtype).detach()
                    ngram_table._num_trigrams += 1
        t_table_end = time.perf_counter()
        table_update_ms = (t_table_end - t_table_start) * 1000

        # Record metrics
        tier_counts[tier] += 1
        tier_cosines[tier].append(recon_cos)
        tier_raw_cosines[tier].append(raw_cos)
        tier_bytes[tier] += xfer_bytes
        total_transfer_bytes += xfer_bytes
        total_raw_bytes += raw_fp16_bytes

        if is_test:
            step_records.append({
                "dataset_name": dataset_name,
                "request_index": req_idx,
                "decode_step": step,
                "tier": tier,
                "raw_cosine": raw_cos,
                "recon_cosine": recon_cos,
                "transfer_bytes": xfer_bytes,
                "raw_fp16_bytes": raw_fp16_bytes,
                "decode_forward_ms": decode_fwd_ms,
                "classify_ms": classify_ms,
                "encode_ms": encode_ms,
                "table_update_ms": table_update_ms,
            })

        past_kv = dbatch.past_key_values
        next_tok = select_next_token(dbatch.last_logits, do_sample=False)
        del dbatch

    del past_kv, next_tok

    # Table stats
    tbl_stats = ngram_table.stats
    table_rec = {
        "dataset_name": dataset_name, "request_index": req_idx, "phase": phase,
        "num_trigrams": tbl_stats["num_trigrams"],
        "num_bigrams": tbl_stats["num_bigrams"],
        "memory_bytes": tbl_stats["memory_bytes"],
    }

    if not is_test:
        return None, None, table_rec

    # Aggregate record
    all_cosines = []
    for t in tier_cosines.values():
        all_cosines.extend(t)

    agg_rec = {
        "dataset_name": dataset_name,
        "request_index": req_idx,
        "prefill_seq_len": seq_len,
        "decode_tokens": decode_tokens,
        "num_trigram": tier_counts["trigram"],
        "num_bigram": tier_counts["bigram"],
        "num_self_ref": tier_counts["self_ref"],
        "num_unigram": tier_counts["unigram"],
        "pct_trigram": tier_counts["trigram"] / max(decode_tokens, 1) * 100,
        "pct_bigram": tier_counts["bigram"] / max(decode_tokens, 1) * 100,
        "pct_self_ref": tier_counts["self_ref"] / max(decode_tokens, 1) * 100,
        "pct_unigram": tier_counts["unigram"] / max(decode_tokens, 1) * 100,
        "recon_cosine_mean": sum(all_cosines) / len(all_cosines) if all_cosines else 0,
        "recon_cosine_min": min(all_cosines) if all_cosines else 0,
        "total_transfer_bytes": total_transfer_bytes,
        "raw_fp16_bytes": total_raw_bytes,
        "compression_ratio": total_raw_bytes / max(total_transfer_bytes, 1),
        "transfer_bytes_trigram": tier_bytes["trigram"],
        "transfer_bytes_bigram": tier_bytes["bigram"],
        "transfer_bytes_self_ref": tier_bytes["self_ref"],
        "transfer_bytes_unigram": tier_bytes["unigram"],
    }

    # Add per-tier cosine stats
    for tier_name in ["trigram", "bigram", "self_ref", "unigram"]:
        cosines = tier_cosines[tier_name]
        raw_cosines = tier_raw_cosines[tier_name]
        if cosines:
            agg_rec[f"recon_cosine_{tier_name}_mean"] = sum(cosines) / len(cosines)
            agg_rec[f"recon_cosine_{tier_name}_min"] = min(cosines)
            agg_rec[f"raw_cosine_{tier_name}_mean"] = sum(raw_cosines) / len(raw_cosines)
        else:
            agg_rec[f"recon_cosine_{tier_name}_mean"] = 0.0
            agg_rec[f"recon_cosine_{tier_name}_min"] = 0.0
            agg_rec[f"raw_cosine_{tier_name}_mean"] = 0.0

    return step_records, agg_rec, table_rec


# ===================================================================
# Save helper
# ===================================================================
def save_records(records: List[Dict], path: Path):
    if not records:
        return
    tbl = pa.Table.from_pylist(records)
    pq.write_table(tbl, str(path))
    logger.info("Saved %d records to %s", len(records), path)


# ===================================================================
# Run one dataset
# ===================================================================
def run_dataset(model, tokenizer, dataset_name: str, cfg: Dict, device: torch.device):
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("=" * 60)

    texts = load_dataset_texts(dataset_name)
    logger.info("Loaded %d texts", len(texts))

    import random
    rng = random.Random(cfg["seed"])
    rng.shuffle(texts)

    warmup_n = cfg["warmup_requests"]
    test_n = cfg["test_requests"]
    total_needed = warmup_n + test_n
    while len(texts) < total_needed:
        texts.extend(texts[:total_needed - len(texts)])

    warmup_texts = texts[:warmup_n]
    test_texts = texts[warmup_n:warmup_n + test_n]

    ngram_table = NgramTable(device=device, dtype=torch.float16, max_entries=0)

    all_step_records: List[Dict] = []
    all_agg_records: List[Dict] = []
    all_table_records: List[Dict] = []

    # Warmup
    logger.info("--- Warmup: %d requests ---", warmup_n)
    for i, text in enumerate(warmup_texts):
        _, _, table_rec = process_request(
            model, tokenizer, ngram_table, text, cfg, device,
            dataset_name, i, "warmup",
        )
        all_table_records.append(table_rec)
        if (i + 1) % 10 == 0:
            s = ngram_table.stats
            logger.info("Warmup %d/%d  table: %d tri, %d bi",
                        i + 1, warmup_n, s["num_trigrams"], s["num_bigrams"])

    logger.info("Warmup done. Table: %d tri, %d bi",
                ngram_table.stats["num_trigrams"], ngram_table.stats["num_bigrams"])

    # Test
    logger.info("--- Test: %d requests ---", test_n)
    for i, text in enumerate(test_texts):
        step_recs, agg_rec, table_rec = process_request(
            model, tokenizer, ngram_table, text, cfg, device,
            dataset_name, i, "test",
        )
        if step_recs:
            all_step_records.extend(step_recs)
        if agg_rec:
            all_agg_records.append(agg_rec)
        all_table_records.append(table_rec)

        if (i + 1) % 10 == 0:
            cos = agg_rec["recon_cosine_mean"] if agg_rec else 0
            ratio = agg_rec["compression_ratio"] if agg_rec else 0
            cov = 100 - agg_rec["pct_unigram"] if agg_rec else 0
            logger.info("Test %d/%d  cos=%.4f  ratio=%.2fx  coverage=%.1f%%",
                        i + 1, test_n, cos, ratio, cov)

    # Save
    out_dir = Path(cfg["output_dir"]) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_records(all_step_records, out_dir / "decode_step_detail.parquet")
    save_records(all_agg_records, out_dir / "decode_aggregate.parquet")
    save_records(all_table_records, out_dir / "table_growth.parquet")

    logger.info("Results saved to %s", out_dir)

    del ngram_table
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="E12v2-decode: delta-coding quality on decode phase")
    parser.add_argument("--model", default=DEFAULTS["model_path"])
    parser.add_argument("--gpu", type=int, default=DEFAULTS["gpu"])
    parser.add_argument("--warmup-requests", type=int, default=DEFAULTS["warmup_requests"])
    parser.add_argument("--test-requests", type=int, default=DEFAULTS["test_requests"])
    parser.add_argument("--decode-tokens", type=int, default=DEFAULTS["decode_tokens"])
    parser.add_argument("--layer-boundary", type=int, default=DEFAULTS["layer_boundary"])
    parser.add_argument("--group-size", type=int, default=DEFAULTS["group_size"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--max-seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
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

    device = torch.device(f"cuda:{args.gpu}")
    logger.info("Loading model %s on %s...", args.model, device)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map={"": device}, trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded. hidden_size=%d", model.config.hidden_size)

    # GPU warmup
    wids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    _ = model(wids, output_hidden_states=True, use_cache=False)
    del wids, _
    torch.cuda.synchronize()

    for ds_name in args.datasets:
        run_dataset(model, tokenizer, ds_name, cfg, device)

    logger.info("All done!")


if __name__ == "__main__":
    main()
