#!/usr/bin/env python3
"""Large-sample phasewise compression evaluation.

This experiment is designed for larger-sample evaluation of the shortlisted
strategies only. It separates:

1. Delta path vs unigram path strategy changes.
2. LLM prefill positions vs LLM decode positions.
3. Compression / reconstruction / exact downstream drift / codec latency.

Compared with the earlier comprehensive experiment, this script:
- Removes sparse strategies.
- Uses more test requests.
- Supports exact phasewise drift for both prompt and generated tokens.
- Is intended to run one dataset per GPU and be merged afterwards.
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from activation_science.core.extraction import decode_step, prefill, select_next_token
from delta_coding_system.compression_experiment_comprehensive import (
    _blend_two_refs,
    _cosine,
    _encode_affine_only,
    _encode_delta_quantized,
    _encode_raw_reference,
    _init_timing,
    _quantize_direct_int4,
    _run_remaining_layers,
)
from delta_coding_system.run_experiment import load_dataset_texts
from delta_coding_system.table import NgramTable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compress_phasewise_large")

DELTA_CANDIDATES = [
    "baseline_current",
    "delta_noaffine_int4_k1",
    "delta_int2_k8_out4",
    "delta_noaffine_int2_k8_out4",
    "delta_int2_k8_out8_entropy",
    "ablate_delta_raw_ref",
    "ablate_delta_affine_only",
]

UNIGRAM_CANDIDATES = [
    "baseline_current",
    "unigram_int4_k4",
    "prev_int4_k2",
    "prev_gs256_k2",
    "prev_int2_k8_out4",
    "prev2_blend_int4_k2",
    "zero_affine_int4_k2",
    "ablate_prev_raw_ref",
    "ablate_prev_affine_only",
]

ALL_STRATEGIES = [
    "baseline_current",
    "delta_noaffine_int4_k1",
    "delta_int2_k8_out4",
    "delta_noaffine_int2_k8_out4",
    "delta_int2_k8_out8_entropy",
    "ablate_delta_raw_ref",
    "ablate_delta_affine_only",
    "unigram_int4_k4",
    "prev_int4_k2",
    "prev_gs256_k2",
    "prev_int2_k8_out4",
    "prev2_blend_int4_k2",
    "zero_affine_int4_k2",
    "ablate_prev_raw_ref",
    "ablate_prev_affine_only",
]
def _update_table_with_sequence(table: NgramTable, token_ids: torch.Tensor, hidden: torch.Tensor) -> None:
    if token_ids.shape[0] < 3:
        return
    trigrams = token_ids.unfold(0, 3, 1)
    for tri_idx in range(trigrams.shape[0]):
        tri = trigrams[tri_idx]
        table._get_or_create_node(
            tri[0].item(),
            tri[1].item(),
            hidden[tri_idx + 1].unsqueeze(0).to(torch.float16),
        )
        node = table._get_node(tri[0].item(), tri[1].item())
        if node is not None:
            node.suffixes[tri[2].item()] = hidden[tri_idx + 2].unsqueeze(0).to(torch.float16)


def _classify_positions(table: NgramTable, token_ids: torch.Tensor, device: torch.device) -> Tuple[List[str], List[Optional[torch.Tensor]]]:
    tiers: List[str] = []
    refs: List[Optional[torch.Tensor]] = []
    seq_len = token_ids.shape[0]
    for pos in range(seq_len):
        ref = None
        tier = "unigram"
        if pos >= 2:
            tri_ref = table.get_trigram(token_ids[pos - 2].item(), token_ids[pos - 1].item(), token_ids[pos].item())
            if tri_ref is not None:
                ref = tri_ref.to(torch.float16).to(device)
                tier = "trigram"
        if ref is None and pos >= 1:
            bi_ref = table.get_bigram(token_ids[pos - 1].item(), token_ids[pos].item())
            if bi_ref is not None:
                ref = bi_ref.to(torch.float16).to(device)
                tier = "bigram"
        tiers.append(tier)
        refs.append(ref)
    return tiers, refs


def _compute_drift_metrics(orig_logits: torch.Tensor, recon_logits: torch.Tensor) -> Dict[str, float]:
    seq_len = orig_logits.shape[1]
    logit_cos = F.cosine_similarity(orig_logits.float(), recon_logits.float(), dim=-1).squeeze(0)
    orig_top1 = orig_logits.argmax(dim=-1).squeeze(0)
    recon_top1 = recon_logits.argmax(dim=-1).squeeze(0)
    mismatch = orig_top1 != recon_top1
    first_top1 = int(torch.where(mismatch)[0][0].item()) if mismatch.any() else seq_len
    below = logit_cos < 0.999
    first_cos = int(torch.where(below)[0][0].item()) if below.any() else seq_len
    kl = F.kl_div(
        F.log_softmax(recon_logits.float(), dim=-1),
        F.softmax(orig_logits.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1).squeeze(0)
    return {
        "top1_match_rate": float((~mismatch).float().mean().item()),
        "first_top1_drift_pos": first_top1,
        "first_logit_cos_below_0_999": first_cos,
        "logit_cosine_mean": float(logit_cos.mean().item()),
        "logit_cosine_min": float(logit_cos.min().item()),
        "kl_mean": float(kl.mean().item()),
        "kl_max": float(kl.max().item()),
    }


class ShortlistStrategyRunner:
    def run_all(
        self,
        real_h: torch.Tensor,
        tier: str,
        table_ref: Optional[torch.Tensor],
        prev_h: Optional[torch.Tensor],
        prev2_h: Optional[torch.Tensor],
        history: List[torch.Tensor],
        global_mean_ref: torch.Tensor,
    ) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        zero_ref = torch.zeros_like(real_h)

        if table_ref is not None:
            results["baseline_current"] = _encode_delta_quantized(real_h, table_ref, 4, 1, outlier_bits=16, include_ref_idx=True, use_affine=True)
            results["delta_noaffine_int4_k1"] = _encode_delta_quantized(real_h, table_ref, 4, 1, outlier_bits=16, include_ref_idx=True, use_affine=False)
            results["delta_int2_k8_out4"] = _encode_delta_quantized(real_h, table_ref, 2, 8, outlier_bits=4, include_ref_idx=True, use_affine=True)
            results["delta_noaffine_int2_k8_out4"] = _encode_delta_quantized(real_h, table_ref, 2, 8, outlier_bits=4, include_ref_idx=True, use_affine=False)
            results["delta_int2_k8_out8_entropy"] = _encode_delta_quantized(real_h, table_ref, 2, 8, outlier_bits=8, include_ref_idx=True, use_affine=True, entropy_override=True)
            results["ablate_delta_raw_ref"] = _encode_raw_reference(real_h, table_ref, include_ref_idx=True)
            results["ablate_delta_affine_only"] = _encode_affine_only(real_h, table_ref, include_ref_idx=True)
            baseline = results["baseline_current"]
            for name in ALL_STRATEGIES:
                if name not in results:
                    results[name] = baseline
            return results

        results["baseline_current"] = _encode_int8_baseline(real_h)
        results["unigram_int4_k4"] = _quantize_direct_int4(real_h, top_k=4)

        if prev_h is not None:
            results["prev_int4_k2"] = _encode_delta_quantized(real_h, prev_h, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=True)
            results["prev_gs256_k2"] = _encode_delta_quantized(real_h, prev_h, 4, 2, outlier_bits=16, include_ref_idx=False, group_size=256, use_affine=True)
            results["prev_int2_k8_out4"] = _encode_delta_quantized(real_h, prev_h, 2, 8, outlier_bits=4, include_ref_idx=False, use_affine=True)
            results["ablate_prev_raw_ref"] = _encode_raw_reference(real_h, prev_h, include_ref_idx=False)
            results["ablate_prev_affine_only"] = _encode_affine_only(real_h, prev_h, include_ref_idx=False)
        else:
            for name in ["prev_int4_k2", "prev_gs256_k2", "prev_int2_k8_out4", "ablate_prev_raw_ref", "ablate_prev_affine_only"]:
                results[name] = results["baseline_current"]

        results["zero_affine_int4_k2"] = _encode_delta_quantized(real_h, zero_ref, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=True)

        if prev_h is not None and prev2_h is not None:
            blend_ref = _blend_two_refs(real_h, prev_h, prev2_h)
            results["prev2_blend_int4_k2"] = _encode_delta_quantized(real_h, blend_ref, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=False)
        else:
            results["prev2_blend_int4_k2"] = results["baseline_current"]

        baseline = results["baseline_current"]
        for name in ALL_STRATEGIES:
            if name not in results:
                results[name] = baseline
        return results


def _encode_int8_baseline(real_h: torch.Tensor) -> Dict[str, object]:
    from delta_coding_system.compression_experiment_comprehensive import _encode_int8_unigram

    return _encode_int8_unigram(real_h, top_k=1)


def _timed_prefill(model, input_ids: torch.Tensor, device: torch.device):
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    batch = prefill(model, input_ids, use_cache=True)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return batch, elapsed_ms


def _timed_decode_generate(model, prefill_batch, decode_tokens: int, eos_token_id: Optional[int], device: torch.device):
    generated: List[int] = []
    total_decode_ms = 0.0
    next_token = select_next_token(prefill_batch.last_logits, do_sample=False)
    past_key_values = prefill_batch.past_key_values

    for _ in range(decode_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        batch = decode_step(model, next_token, past_key_values)
        torch.cuda.synchronize(device)
        total_decode_ms += (time.perf_counter() - t0) * 1000.0
        past_key_values = batch.past_key_values
        next_token = select_next_token(batch.last_logits, do_sample=False)

    return generated, total_decode_ms


def run_experiment(args):
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    logger.info("Loading model on GPU %d...", args.gpu)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    _run_remaining_layers.start_layer = args.layer_boundary

    runner = ShortlistStrategyRunner()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_records = []
    drift_records = []
    position_records = []

    for ds_name in args.datasets:
        logger.info("%s", "=" * 60)
        logger.info("Dataset: %s", ds_name)
        texts = load_dataset_texts(ds_name)
        rng = random.Random(args.seed)
        rng.shuffle(texts)

        total_needed = args.warmup_requests + args.test_requests
        while len(texts) < total_needed:
            texts.extend(texts[: total_needed - len(texts)])

        table = NgramTable(device=device, dtype=torch.float16, max_entries=100000)
        mean_sum = None
        mean_count = 0

        logger.info("Warmup: %d requests...", args.warmup_requests)
        for warm_idx in range(args.warmup_requests):
            input_ids = tokenizer(texts[warm_idx], return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            mean_sum = h.sum(dim=0, keepdim=True).float() if mean_sum is None else mean_sum + h.sum(dim=0, keepdim=True).float()
            mean_count += h.shape[0]
            _update_table_with_sequence(table, token_ids, h)
            del out, h

        global_mean_ref = (mean_sum / max(mean_count, 1)).to(torch.float16)

        logger.info("Test: %d requests...", args.test_requests)
        for req_idx in range(args.test_requests):
            text = texts[args.warmup_requests + req_idx]
            input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            prefill_batch, prefill_forward_ms = _timed_prefill(model, input_ids, device)
            generated_tokens, decode_forward_ms = _timed_decode_generate(
                model,
                prefill_batch,
                args.decode_tokens,
                tokenizer.eos_token_id,
                device,
            )

            prompt_len = input_ids.shape[1]
            decode_len = len(generated_tokens)
            full_ids = input_ids if decode_len == 0 else torch.cat(
                [input_ids, torch.tensor([generated_tokens], dtype=torch.long, device=device)],
                dim=1,
            )
            full_attention_mask = torch.ones_like(full_ids)
            with torch.no_grad():
                full_out = model(full_ids, output_hidden_states=True, use_cache=False)

            full_hidden = full_out.hidden_states[args.layer_boundary].squeeze(0)
            full_logits = full_out.logits
            full_token_ids = full_ids.squeeze(0)
            seq_len = full_token_ids.shape[0]

            tiers, refs = _classify_positions(table, full_token_ids, device)
            strategy_accum = defaultdict(lambda: {
                "prefill": {
                    "bytes": 0,
                    "raw_bytes": 0,
                    "zlib_bytes": 0,
                    "ideal_bytes": 0.0,
                    "entropy_bpb_sum": 0.0,
                    "cos_sum": 0.0,
                    "cos_min": 1.0,
                    "recon_e_sum": 0.0,
                    "count": 0,
                    "timing_sum": _init_timing(),
                    "delta_count": 0,
                    "unigram_count": 0,
                },
                "decode": {
                    "bytes": 0,
                    "raw_bytes": 0,
                    "zlib_bytes": 0,
                    "ideal_bytes": 0.0,
                    "entropy_bpb_sum": 0.0,
                    "cos_sum": 0.0,
                    "cos_min": 1.0,
                    "recon_e_sum": 0.0,
                    "count": 0,
                    "timing_sum": _init_timing(),
                    "delta_count": 0,
                    "unigram_count": 0,
                },
            })
            recon_sequences = defaultdict(list)
            history: List[torch.Tensor] = []

            for pos in range(seq_len):
                phase = "prefill" if pos < prompt_len else "decode"
                real = full_hidden[pos:pos + 1].to(torch.float16)
                prev_h = full_hidden[pos - 1:pos].to(torch.float16) if pos > 0 else None
                prev2_h = full_hidden[pos - 2:pos - 1].to(torch.float16) if pos > 1 else None
                results = runner.run_all(real, tiers[pos], refs[pos], prev_h, prev2_h, history, global_mean_ref.to(device))

                for strategy_name, result in results.items():
                    recon = result["recon"]
                    recon_sequences[strategy_name].append(recon)
                    metrics = result["metrics"]
                    cos = _cosine(real, recon)
                    acc = strategy_accum[strategy_name][phase]
                    acc["bytes"] += result["bytes"]
                    acc["raw_bytes"] += result["raw_bytes"]
                    acc["zlib_bytes"] += result["zlib_bytes"]
                    acc["ideal_bytes"] += result["ideal_entropy_bytes"]
                    if result["entropy_bpb"] == result["entropy_bpb"]:
                        acc["entropy_bpb_sum"] += result["entropy_bpb"]
                    acc["cos_sum"] += cos
                    acc["cos_min"] = min(acc["cos_min"], cos)
                    acc["recon_e_sum"] += metrics["recon_energy_explained"]
                    acc["count"] += 1
                    if tiers[pos] == "unigram":
                        acc["unigram_count"] += 1
                    else:
                        acc["delta_count"] += 1
                    for key, value in result["timing"].items():
                        acc["timing_sum"][key] += value

                    if req_idx < args.detail_requests:
                        position_records.append({
                            "dataset": ds_name,
                            "request_index": req_idx,
                            "phase": phase,
                            "position": pos,
                            "phase_position": pos if phase == "prefill" else pos - prompt_len,
                            "tier": tiers[pos],
                            "strategy": strategy_name,
                            "prompt_len": prompt_len,
                            "decode_len": decode_len,
                            "transfer_bytes": result["bytes"],
                            "raw_packet_bytes": result["raw_bytes"],
                            "zlib_packet_bytes": result["zlib_bytes"],
                            "ideal_entropy_bytes": result["ideal_entropy_bytes"],
                            "cosine": cos,
                            "raw_ref_cosine": metrics["raw_ref_cosine"],
                            "affine_ref_cosine": metrics["affine_ref_cosine"],
                            "raw_ref_energy_explained": metrics["raw_ref_energy_explained"],
                            "affine_energy_explained": metrics["affine_energy_explained"],
                            "affine_gain": metrics["affine_gain"],
                            "recon_energy_explained": metrics["recon_energy_explained"],
                            "residual_coding_loss": metrics["residual_coding_loss"],
                            **{f"timing_{k}": v for k, v in result["timing"].items()},
                        })
                history.append(real)

            original_hidden_full = full_hidden.unsqueeze(0)
            for strategy_name in ALL_STRATEGIES:
                recon_seq = torch.cat(recon_sequences[strategy_name], dim=0)

                prefill_hidden = recon_seq[:prompt_len].unsqueeze(0)
                with torch.no_grad():
                    prefill_recon_logits = _run_remaining_layers(
                        model,
                        prefill_hidden,
                        torch.ones((1, prompt_len), dtype=torch.long, device=device),
                    )
                prefill_metrics = _compute_drift_metrics(full_logits[:, :prompt_len, :], prefill_recon_logits)
                drift_records.append({
                    "dataset": ds_name,
                    "request_index": req_idx,
                    "phase": "prefill",
                    "strategy": strategy_name,
                    "num_positions": prompt_len,
                    **prefill_metrics,
                })

                if decode_len > 0:
                    decode_hidden_full = original_hidden_full.clone()
                    decode_hidden_full[:, prompt_len:, :] = recon_seq[prompt_len:].unsqueeze(0)
                    with torch.no_grad():
                        decode_recon_logits_full = _run_remaining_layers(model, decode_hidden_full, full_attention_mask)
                    decode_metrics = _compute_drift_metrics(full_logits[:, prompt_len:, :], decode_recon_logits_full[:, prompt_len:, :])
                    drift_records.append({
                        "dataset": ds_name,
                        "request_index": req_idx,
                        "phase": "decode",
                        "strategy": strategy_name,
                        "num_positions": decode_len,
                        **decode_metrics,
                    })

            for strategy_name, phases in strategy_accum.items():
                for phase_name, acc in phases.items():
                    count = acc["count"]
                    if count == 0:
                        continue
                    raw_bytes = count * full_hidden.shape[-1] * 2
                    request_records.append({
                        "dataset": ds_name,
                        "request_index": req_idx,
                        "phase": phase_name,
                        "strategy": strategy_name,
                        "prompt_len": prompt_len,
                        "decode_len": decode_len,
                        "num_positions": count,
                        "num_delta_positions": acc["delta_count"],
                        "num_unigram_positions": acc["unigram_count"],
                        "raw_fp16_bytes": raw_bytes,
                        "total_transfer_bytes": acc["bytes"],
                        "total_raw_packet_bytes": acc["raw_bytes"],
                        "total_zlib_packet_bytes": acc["zlib_bytes"],
                        "total_ideal_entropy_bytes": acc["ideal_bytes"],
                        "compression_ratio": raw_bytes / max(acc["bytes"], 1),
                        "compression_ratio_raw_packet": raw_bytes / max(acc["raw_bytes"], 1),
                        "compression_ratio_zlib_packet": raw_bytes / max(acc["zlib_bytes"], 1),
                        "compression_ratio_ideal_entropy": raw_bytes / max(acc["ideal_bytes"], 1e-8),
                        "cosine_mean": acc["cos_sum"] / count,
                        "cosine_min": acc["cos_min"],
                        "recon_energy_explained_mean": acc["recon_e_sum"] / count,
                        "residual_coding_loss_mean": 1.0 - acc["recon_e_sum"] / count,
                        "packet_entropy_bits_per_byte_mean": acc["entropy_bpb_sum"] / count,
                        "prefill_forward_ms": prefill_forward_ms,
                        "decode_forward_ms": decode_forward_ms,
                        **{f"timing_{k}_sum": v for k, v in acc["timing_sum"].items()},
                        **{f"timing_{k}_mean": v / count for k, v in acc["timing_sum"].items()},
                    })

            _update_table_with_sequence(table, full_token_ids, full_hidden)
            del full_out, full_hidden, full_logits

            if (req_idx + 1) % 10 == 0:
                logger.info("  Test %d/%d complete", req_idx + 1, args.test_requests)

        gc.collect()
        torch.cuda.empty_cache()

    pq.write_table(pa.Table.from_pylist(request_records), str(output_dir / "phasewise_request_summary.parquet"))
    pq.write_table(pa.Table.from_pylist(drift_records), str(output_dir / "phasewise_drift.parquet"))
    if position_records:
        pq.write_table(pa.Table.from_pylist(position_records), str(output_dir / "phasewise_position_detail.parquet"))
    logger.info(
        "Saved %d request records, %d drift records, %d position records",
        len(request_records),
        len(drift_records),
        len(position_records),
    )


def main():
    parser = argparse.ArgumentParser(description="Large-sample phasewise compression evaluation")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=100)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--detail-requests", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_phasewise_large")
    parser.add_argument("--datasets", nargs="+", default=["wikitext2"])
    args = parser.parse_args()
    if args.warmup_requests < 20:
        raise ValueError("warmup_requests must be >= 20")
    run_experiment(args)


if __name__ == "__main__":
    main()