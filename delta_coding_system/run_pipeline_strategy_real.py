#!/usr/bin/env python3
"""Run real overlapped pipeline experiments for shortlisted strategy configs.

This script evaluates whole-pipeline behavior using the production
OverlappedPipeline implementation rather than the phasewise analysis scripts.

It compares two end-to-end configurations:
1. baseline_current: affine Int4 delta + Int8 unigram.
2. optimized_default: no-affine Int4 delta + direct Int4 unigram.

Measured outputs:
1. Real pipeline compression ratio.
2. Real overlapped communication critical-path latency with bandwidth models.
3. Decode hidden similarity.
4. Offline decode logit drift on the generated sequence.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.pipeline import OverlappedPipeline
from delta_coding_system.run_experiment import load_dataset_texts
from delta_coding_system.compression_experiment_comprehensive import _run_remaining_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline_strategy_real")

BANDWIDTHS_MBPS = [200, 500, 1000]

CONFIGS = [
    {
        "name": "baseline_current",
        "delta_strategy": "baseline_current",
        "unigram_strategy": "baseline_current",
    },
    {
        "name": "optimized_default",
        "delta_strategy": "delta_noaffine_int4_k1",
        "unigram_strategy": "unigram_int4_k4",
    },
]


def _network_ms(total_bytes: float, bandwidth_mbps: int) -> float:
    return float(total_bytes) * 8.0 / (float(bandwidth_mbps) * 1000.0)


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


def _save_records(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), str(path))
    logger.info("Saved %d records to %s", len(records), path)


def _prepare_texts(dataset_name: str, seed: int, warmup_requests: int, test_requests: int) -> tuple[list[str], list[str]]:
    texts = load_dataset_texts(dataset_name)
    rng = random.Random(seed)
    rng.shuffle(texts)
    total_needed = warmup_requests + test_requests
    while len(texts) < total_needed:
        texts.extend(texts[: total_needed - len(texts)])
    return texts[:warmup_requests], texts[warmup_requests:warmup_requests + test_requests]


def run_dataset(model, tokenizer, device: torch.device, dataset_name: str, args) -> None:
    warmup_texts, test_texts = _prepare_texts(dataset_name, args.seed, args.warmup_requests, args.test_requests)

    dataset_out = Path(args.output_dir) / dataset_name
    dataset_out.mkdir(parents=True, exist_ok=True)

    request_records: List[Dict[str, Any]] = []
    drift_records: List[Dict[str, Any]] = []

    pipelines = []

    for cfg in CONFIGS:
        logger.info("%s", "=" * 72)
        logger.info("Dataset=%s Config=%s", dataset_name, cfg["name"])

        pipeline = OverlappedPipeline(
            model=model,
            tokenizer=tokenizer,
            layer_boundary=args.layer_boundary,
            table_dtype=torch.float16,
            max_table_entries=args.max_table_entries,
            group_size=args.group_size,
            top_k=1,
            int8_group_size=args.group_size,
            int8_outlier_top_k=1,
            decode_tokens=args.max_decode_tokens,
            max_seq_len=args.max_seq_len,
            device=device,
            domain_aware=False,
            delta_strategy=cfg["delta_strategy"],
            unigram_strategy=cfg["unigram_strategy"],
        )
        pipelines.append((cfg, pipeline))

    for cfg, pipeline in pipelines:
        logger.info("Warmup %d requests for config=%s...", len(warmup_texts), cfg["name"])
        for text in warmup_texts:
            pipeline.process_request(text, phase="warmup")

    logger.info("Test %d requests across %d configs...", len(test_texts), len(pipelines))
    for request_index, text in enumerate(test_texts):
        tokenized_prompt = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        ).input_ids.to(device)

        cached_full_ids = None
        cached_full_mask = None
        cached_orig_logits = None

        for cfg, pipeline in pipelines:
            request_t0 = time.perf_counter()
            pipeline_t0 = time.perf_counter()
            prefill_res, decode_res, table_stats = pipeline.process_request(text, phase="test")
            pipeline_ms = (time.perf_counter() - pipeline_t0) * 1000.0

            prompt_len = prefill_res.seq_len
            decode_len = decode_res.decode_tokens
            full_ids = tokenized_prompt
            if decode_res.generated_token_ids:
                gen = torch.tensor([decode_res.generated_token_ids], dtype=torch.long, device=device)
                full_ids = torch.cat([full_ids, gen], dim=1)
            full_mask = torch.ones_like(full_ids)

            prefill_local_ms = prefill_res.classify_ms + prefill_res.encode_delta_ms + prefill_res.encode_self_ref_ms + prefill_res.encode_unigram_ms
            decode_local_ms = decode_res.total_classify_ms + decode_res.total_encode_ms

            request_row = {
                "dataset_name": dataset_name,
                "config_name": cfg["name"],
                "request_index": request_index,
                "prompt_len": prompt_len,
                "decode_len": decode_len,
                "prefill_transfer_bytes": prefill_res.total_transfer_bytes,
                "decode_transfer_bytes": decode_res.total_transfer_bytes,
                "total_transfer_bytes": prefill_res.total_transfer_bytes + decode_res.total_transfer_bytes,
                "prefill_raw_fp16_bytes": prefill_res.raw_fp16_bytes,
                "decode_raw_fp16_bytes": decode_res.raw_fp16_bytes,
                "total_raw_fp16_bytes": prefill_res.raw_fp16_bytes + decode_res.raw_fp16_bytes,
                "prefill_compression_ratio": prefill_res.compression_ratio,
                "decode_compression_ratio": decode_res.compression_ratio,
                "total_compression_ratio": (prefill_res.raw_fp16_bytes + decode_res.raw_fp16_bytes) / max(prefill_res.total_transfer_bytes + decode_res.total_transfer_bytes, 1),
                "prefill_recon_cosine_mean": prefill_res.recon_cosine_mean,
                "prefill_recon_cosine_min": prefill_res.recon_cosine_min,
                "decode_recon_cosine_mean": decode_res.recon_cosine_mean,
                "decode_recon_cosine_min": decode_res.recon_cosine_min,
                "prefill_forward_ms": prefill_res.prefill_fwd_ms,
                "prefill_classify_wait_ms": prefill_res.classify_ms,
                "prefill_encode_delta_ms": prefill_res.encode_delta_ms,
                "prefill_encode_self_ref_ms": prefill_res.encode_self_ref_ms,
                "prefill_encode_unigram_ms": prefill_res.encode_unigram_ms,
                "prefill_table_update_submit_ms": prefill_res.table_update_ms,
                "prefill_total_ms": prefill_res.total_ms,
                "decode_forward_ms": decode_res.total_fwd_ms,
                "decode_forward_per_token_ms": decode_res.total_fwd_ms / max(decode_len, 1),
                "decode_classify_wait_ms": decode_res.total_classify_ms,
                "decode_encode_ms": decode_res.total_encode_ms,
                "decode_table_wait_ms": decode_res.total_table_update_ms,
                "decode_total_ms": decode_res.total_ms,
                "pipeline_total_ms": prefill_res.total_ms + decode_res.total_ms,
                "table_num_trigrams": table_stats["num_trigrams"],
                "table_num_bigrams": table_stats["num_bigrams"],
            }

            for bw in BANDWIDTHS_MBPS:
                prefill_net = _network_ms(prefill_res.total_transfer_bytes, bw)
                decode_net_total = _network_ms(decode_res.total_transfer_bytes, bw)
                request_row[f"prefill_comm_e2e_{bw}mbps_ms"] = prefill_local_ms + prefill_net
                request_row[f"prefill_network_{bw}mbps_ms"] = prefill_net
                request_row[f"decode_comm_e2e_{bw}mbps_total_ms"] = decode_local_ms + decode_net_total
                request_row[f"decode_network_{bw}mbps_total_ms"] = decode_net_total
                request_row[f"decode_comm_e2e_{bw}mbps_per_token_ms"] = (decode_local_ms + decode_net_total) / max(decode_len, 1)

            request_records.append(request_row)

            drift_t0 = time.perf_counter()
            reuse_orig_logits = False
            if cached_full_ids is not None and cached_orig_logits is not None and torch.equal(cached_full_ids, full_ids):
                orig_logits = cached_orig_logits
                reuse_orig_logits = True
            else:
                with torch.no_grad():
                    full_out = model(full_ids, output_hidden_states=False, use_cache=False)
                orig_logits = full_out.logits
                cached_full_ids = full_ids.clone()
                cached_full_mask = full_mask.clone()
                cached_orig_logits = orig_logits

            recon_parts = [prefill_res.reconstructed_hidden]
            if decode_res.reconstructed_hidden is not None and decode_res.decode_tokens > 0:
                recon_parts.append(decode_res.reconstructed_hidden)
            recon_full = torch.cat(recon_parts, dim=0).unsqueeze(0)
            with torch.no_grad():
                recon_logits = _run_remaining_layers(model, recon_full, cached_full_mask if reuse_orig_logits else full_mask)

            if decode_len > 0:
                decode_metrics = _compute_drift_metrics(orig_logits[:, prompt_len:, :], recon_logits[:, prompt_len:, :])
            else:
                decode_metrics = {
                    "top1_match_rate": 1.0,
                    "first_top1_drift_pos": 0,
                    "first_logit_cos_below_0_999": 0,
                    "logit_cosine_mean": 1.0,
                    "logit_cosine_min": 1.0,
                    "kl_mean": 0.0,
                    "kl_max": 0.0,
                }

            drift_records.append({
                "dataset_name": dataset_name,
                "config_name": cfg["name"],
                "request_index": request_index,
                "prompt_len": prompt_len,
                "decode_len": decode_len,
                **decode_metrics,
            })

            drift_ms = (time.perf_counter() - drift_t0) * 1000.0
            total_request_ms = (time.perf_counter() - request_t0) * 1000.0

            if not reuse_orig_logits:
                del full_out
            del recon_logits, recon_full, full_ids, full_mask

            logger.info(
                "Dataset=%s Config=%s Request %d/%d pipeline=%.1fms drift=%.1fms total=%.1fms decode_tokens=%d decode_cos=%.6f orig_reuse=%s",
                dataset_name,
                cfg["name"],
                request_index + 1,
                len(test_texts),
                pipeline_ms,
                drift_ms,
                total_request_ms,
                decode_len,
                decode_res.recon_cosine_mean,
                reuse_orig_logits,
            )

    for _, pipeline in pipelines:
        pipeline.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    _save_records(request_records, dataset_out / "request_summary.parquet")
    _save_records(drift_records, dataset_out / "decode_drift.parquet")


def main():
    parser = argparse.ArgumentParser(description="Real pipeline strategy experiment")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=100)
    parser.add_argument("--max-decode-tokens", type=int, default=512)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_pipeline_strategy_real_run")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["wikitext2"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.clean_output:
        logger.info("Cleaning existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Config: %s", json.dumps(vars(args), indent=2))
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

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

    for dataset_name in args.datasets:
        run_dataset(model, tokenizer, device, dataset_name, args)


if __name__ == "__main__":
    main()