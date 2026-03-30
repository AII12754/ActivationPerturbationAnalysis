#!/usr/bin/env python3
"""Run the latency-first v1 benchmark path.

This entrypoint intentionally exposes only the final production design:
block-table storage, raw-FP16 decode transfer, and async cold paging.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.evaluation import compute_logit_drift_metrics, run_remaining_layers
from delta_coding_system.run_experiment import load_dataset_texts
from delta_coding_system.v1.pipeline import LatencyFirstPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("v1_latency_benchmark")

DEFAULT_BANDWIDTHS_MBPS = [200, 500, 1000]


def _network_ms(total_bytes: float, bandwidth_mbps: int) -> float:
    return float(total_bytes) * 8.0 / (float(bandwidth_mbps) * 1000.0)


def _save_records(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), str(path))
    logger.info("Saved %d records to %s", len(records), path)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * pct
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _build_latency_summary(dataset_name: str, request_records: List[Dict[str, Any]], bandwidths: List[int]) -> Dict[str, Any]:
    pipeline_values = [float(record["pipeline_total_ms"]) for record in request_records]
    summary: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "config_name": "v1_latency_first",
        "num_requests": len(request_records),
        "pipeline_total_ms_mean": _mean(pipeline_values),
        "pipeline_total_ms_p50": _percentile(pipeline_values, 0.50),
        "pipeline_total_ms_p95": _percentile(pipeline_values, 0.95),
    }
    op_keys = [
        "prefill_forward_ms",
        "prefill_prev_update_wait_ms",
        "prefill_classify_wait_ms",
        "prefill_encode_delta_ms",
        "prefill_encode_unigram_ms",
        "prefill_encode_self_ref_ms",
        "prefill_table_update_submit_ms",
        "prefill_total_ms",
        "decode_forward_ms",
        "decode_forward_per_token_ms",
        "decode_classify_wait_ms",
        "decode_lookup_wait_ms",
        "decode_local_comm_work_ms",
        "decode_encode_ms",
        "decode_transfer_prepare_ms",
        "decode_table_wait_ms",
        "decode_total_ms",
    ]
    for key in op_keys:
        values = [float(record[key]) for record in request_records]
        summary[f"{key}_mean"] = _mean(values)
        summary[f"{key}_p95"] = _percentile(values, 0.95)
    for bw in bandwidths:
        prefill_overlap_key = f"prefill_comm_e2e_{bw}mbps_ms"
        decode_overlap_key = f"decode_comm_e2e_{bw}mbps_total_ms"
        decode_overlap_per_token_key = f"decode_comm_e2e_{bw}mbps_per_token_ms"
        prefill_additive_key = f"prefill_comm_additive_{bw}mbps_ms"
        decode_additive_key = f"decode_comm_additive_{bw}mbps_total_ms"
        decode_additive_per_token_key = f"decode_comm_additive_{bw}mbps_per_token_ms"
        fp16_prefill_key = f"fp16_prefill_same_pp_{bw}mbps_ms"
        fp16_decode_key = f"fp16_decode_same_pp_{bw}mbps_total_ms"
        fp16_decode_per_token_key = f"fp16_decode_same_pp_{bw}mbps_per_token_ms"
        total_speedup_key = f"total_speedup_vs_fp16_same_pp_additive_{bw}mbps"

        prefill_overlap_values = [float(record[prefill_overlap_key]) for record in request_records]
        decode_overlap_values = [float(record[decode_overlap_key]) for record in request_records]
        decode_overlap_per_token_values = [float(record[decode_overlap_per_token_key]) for record in request_records]
        prefill_additive_values = [float(record[prefill_additive_key]) for record in request_records]
        decode_additive_values = [float(record[decode_additive_key]) for record in request_records]
        decode_additive_per_token_values = [float(record[decode_additive_per_token_key]) for record in request_records]
        fp16_prefill_values = [float(record[fp16_prefill_key]) for record in request_records]
        fp16_decode_values = [float(record[fp16_decode_key]) for record in request_records]
        fp16_decode_per_token_values = [float(record[fp16_decode_per_token_key]) for record in request_records]
        total_speedup_values = [float(record[total_speedup_key]) for record in request_records]
        total_additive_values = [
            float(record[prefill_additive_key]) + float(record[decode_additive_key])
            for record in request_records
        ]
        total_fp16_values = [
            float(record[fp16_prefill_key]) + float(record[fp16_decode_key])
            for record in request_records
        ]

        summary[f"prefill_comm_e2e_{bw}mbps_mean_ms"] = _mean(prefill_overlap_values)
        summary[f"prefill_comm_e2e_{bw}mbps_p95_ms"] = _percentile(prefill_overlap_values, 0.95)
        summary[f"decode_comm_e2e_{bw}mbps_total_mean_ms"] = _mean(decode_overlap_values)
        summary[f"decode_comm_e2e_{bw}mbps_total_p95_ms"] = _percentile(decode_overlap_values, 0.95)
        summary[f"decode_comm_e2e_{bw}mbps_per_token_mean_ms"] = _mean(decode_overlap_per_token_values)
        summary[f"decode_comm_e2e_{bw}mbps_per_token_p95_ms"] = _percentile(decode_overlap_per_token_values, 0.95)

        summary[f"prefill_comm_additive_{bw}mbps_mean_ms"] = _mean(prefill_additive_values)
        summary[f"prefill_comm_additive_{bw}mbps_p95_ms"] = _percentile(prefill_additive_values, 0.95)
        summary[f"decode_comm_additive_{bw}mbps_total_mean_ms"] = _mean(decode_additive_values)
        summary[f"decode_comm_additive_{bw}mbps_total_p95_ms"] = _percentile(decode_additive_values, 0.95)
        summary[f"decode_comm_additive_{bw}mbps_per_token_mean_ms"] = _mean(decode_additive_per_token_values)
        summary[f"decode_comm_additive_{bw}mbps_per_token_p95_ms"] = _percentile(decode_additive_per_token_values, 0.95)

        summary[f"fp16_prefill_same_pp_{bw}mbps_mean_ms"] = _mean(fp16_prefill_values)
        summary[f"fp16_prefill_same_pp_{bw}mbps_p95_ms"] = _percentile(fp16_prefill_values, 0.95)
        summary[f"fp16_decode_same_pp_{bw}mbps_total_mean_ms"] = _mean(fp16_decode_values)
        summary[f"fp16_decode_same_pp_{bw}mbps_total_p95_ms"] = _percentile(fp16_decode_values, 0.95)
        summary[f"fp16_decode_same_pp_{bw}mbps_per_token_mean_ms"] = _mean(fp16_decode_per_token_values)
        summary[f"fp16_decode_same_pp_{bw}mbps_per_token_p95_ms"] = _percentile(fp16_decode_per_token_values, 0.95)
        summary[f"total_comm_additive_{bw}mbps_mean_ms"] = _mean(total_additive_values)
        summary[f"total_comm_additive_{bw}mbps_p95_ms"] = _percentile(total_additive_values, 0.95)
        summary[f"fp16_total_same_pp_{bw}mbps_mean_ms"] = _mean(total_fp16_values)
        summary[f"fp16_total_same_pp_{bw}mbps_p95_ms"] = _percentile(total_fp16_values, 0.95)
        summary[f"total_speedup_vs_fp16_same_pp_additive_{bw}mbps_mean"] = _mean(total_speedup_values)
        summary[f"total_speedup_vs_fp16_same_pp_additive_{bw}mbps_p50"] = _percentile(total_speedup_values, 0.50)

        prefill_cpu_wait_share_values = [
            float(record["prefill_classify_wait_ms"]) / max(float(record[prefill_additive_key]), 1e-9)
            for record in request_records
        ]
        decode_cpu_wait_share_values = [
            float(record["decode_lookup_wait_ms"]) / max(float(record[decode_additive_key]), 1e-9)
            for record in request_records
        ]
        summary[f"prefill_cpu_wait_share_additive_{bw}mbps_mean"] = _mean(prefill_cpu_wait_share_values)
        summary[f"decode_cpu_wait_share_additive_{bw}mbps_mean"] = _mean(decode_cpu_wait_share_values)
    return summary


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
    decode_step_records: List[Dict[str, Any]] = []

    pipeline = LatencyFirstPipeline.from_legacy_args(model=model, tokenizer=tokenizer, args=args, device=device)
    logger.info("v1 policy: %s", json.dumps(pipeline.policy_summary(), ensure_ascii=False))

    logger.info("Warmup %d requests for dataset=%s", len(warmup_texts), dataset_name)
    for text in warmup_texts:
        pipeline.process_request_v1(
            text,
            phase="warmup",
            task_name=dataset_name,
        )

    logger.info("Test %d requests for dataset=%s", len(test_texts), dataset_name)
    for request_index, text in enumerate(test_texts):
        tokenized_prompt = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        ).input_ids.to(device)

        request_t0 = time.perf_counter()
        prefill_res, decode_res, table_stats = pipeline.process_request_v1(
            text,
            phase="test",
            task_name=dataset_name,
        )
        pipeline_ms = (time.perf_counter() - request_t0) * 1000.0

        full_ids = tokenized_prompt
        if decode_res.generated_token_ids:
            gen = torch.tensor([decode_res.generated_token_ids], dtype=torch.long, device=device)
            full_ids = torch.cat([full_ids, gen], dim=1)
        full_mask = torch.ones_like(full_ids)

        prefill_local_ms = prefill_res.classify_ms + prefill_res.encode_delta_ms + prefill_res.encode_self_ref_ms + prefill_res.encode_unigram_ms
        decode_local_ms = decode_res.total_classify_ms + decode_res.total_encode_ms

        request_row: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "config_name": "v1_latency_first",
            "request_index": request_index,
            "prompt_len": prefill_res.seq_len,
            "decode_len": decode_res.decode_tokens,
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
            "prefill_prev_update_wait_ms": prefill_res.prev_update_wait_ms,
            "prefill_classify_wait_ms": prefill_res.classify_ms,
            "prefill_cpu_cache_lookup_ms": prefill_res.classify_ms,
            "prefill_encode_delta_ms": prefill_res.encode_delta_ms,
            "prefill_encode_unigram_ms": prefill_res.encode_unigram_ms,
            "prefill_encode_self_ref_ms": prefill_res.encode_self_ref_ms,
            "prefill_table_update_submit_ms": prefill_res.table_update_ms,
            "prefill_encode_ms": prefill_local_ms,
            "prefill_total_ms": prefill_res.total_ms,
            "decode_forward_ms": decode_res.total_fwd_ms,
            "decode_forward_per_token_ms": decode_res.total_fwd_ms / max(decode_res.decode_tokens, 1),
            "decode_classify_wait_ms": decode_res.total_classify_ms,
            "decode_lookup_wait_ms": decode_res.total_classify_ms,
            "decode_cpu_cache_lookup_ms": decode_res.total_classify_ms,
            "decode_local_comm_work_ms": decode_local_ms,
            "decode_encode_ms": decode_res.total_encode_ms,
            "decode_transfer_prepare_ms": decode_res.total_encode_ms,
            "decode_table_wait_ms": decode_res.total_table_update_ms,
            "decode_total_ms": decode_res.total_ms,
            "pipeline_total_ms": pipeline_ms,
            "table_num_trigrams": table_stats["num_trigrams"],
            "table_num_bigrams": table_stats["num_bigrams"],
            "block_pager": json.dumps(table_stats.get("pager", {}), ensure_ascii=False),
        }

        for bw in args.bandwidths_mbps:
            prefill_network_ms = _network_ms(prefill_res.total_transfer_bytes, bw)
            decode_network_ms = _network_ms(decode_res.total_transfer_bytes, bw)
            fp16_prefill_ms = _network_ms(prefill_res.raw_fp16_bytes, bw)
            fp16_decode_ms = _network_ms(decode_res.raw_fp16_bytes, bw)
            prefill_comm_overlap_ms = max(prefill_local_ms, prefill_network_ms)
            decode_comm_overlap_ms = max(decode_local_ms, decode_network_ms)
            prefill_comm_additive_ms = prefill_local_ms + prefill_network_ms
            decode_comm_additive_ms = decode_local_ms + decode_network_ms
            fp16_total_ms = fp16_prefill_ms + fp16_decode_ms
            additive_total_ms = prefill_comm_additive_ms + decode_comm_additive_ms

            request_row[f"prefill_network_only_{bw}mbps_ms"] = prefill_network_ms
            request_row[f"decode_network_only_{bw}mbps_total_ms"] = decode_network_ms
            request_row[f"prefill_comm_e2e_{bw}mbps_ms"] = prefill_comm_overlap_ms
            request_row[f"decode_comm_e2e_{bw}mbps_total_ms"] = decode_comm_overlap_ms
            request_row[f"decode_comm_e2e_{bw}mbps_per_token_ms"] = decode_comm_overlap_ms / max(decode_res.decode_tokens, 1)
            request_row[f"prefill_comm_additive_{bw}mbps_ms"] = prefill_comm_additive_ms
            request_row[f"decode_comm_additive_{bw}mbps_total_ms"] = decode_comm_additive_ms
            request_row[f"decode_comm_additive_{bw}mbps_per_token_ms"] = decode_comm_additive_ms / max(decode_res.decode_tokens, 1)
            request_row[f"fp16_prefill_same_pp_{bw}mbps_ms"] = fp16_prefill_ms
            request_row[f"fp16_decode_same_pp_{bw}mbps_total_ms"] = fp16_decode_ms
            request_row[f"fp16_decode_same_pp_{bw}mbps_per_token_ms"] = fp16_decode_ms / max(decode_res.decode_tokens, 1)
            request_row[f"prefill_speedup_vs_fp16_same_pp_additive_{bw}mbps"] = fp16_prefill_ms / max(prefill_comm_additive_ms, 1e-9)
            request_row[f"decode_speedup_vs_fp16_same_pp_additive_{bw}mbps"] = fp16_decode_ms / max(decode_comm_additive_ms, 1e-9)
            request_row[f"total_speedup_vs_fp16_same_pp_additive_{bw}mbps"] = fp16_total_ms / max(additive_total_ms, 1e-9)

        request_records.append(request_row)

        for step_record in decode_res.step_records:
            step_row: Dict[str, Any] = {
                "dataset_name": dataset_name,
                "config_name": "v1_latency_first",
                "request_index": request_index,
                "step": step_record.step,
                "tier": step_record.tier,
                "raw_cosine": step_record.raw_cosine,
                "recon_cosine": step_record.recon_cosine,
                "transfer_bytes": step_record.transfer_bytes,
                "raw_fp16_bytes": step_record.raw_fp16_bytes,
                "fwd_ms": step_record.fwd_ms,
                "classify_ms": step_record.classify_ms,
                "lookup_wait_ms": step_record.classify_ms,
                "encode_ms": step_record.encode_ms,
                "transfer_prepare_ms": step_record.encode_ms,
                "table_update_ms": step_record.table_update_ms,
            }
            for bw in args.bandwidths_mbps:
                network_ms = _network_ms(step_record.transfer_bytes, bw)
                fp16_network_ms = _network_ms(step_record.raw_fp16_bytes, bw)
                local_ms = step_record.classify_ms + step_record.encode_ms
                step_row[f"comm_e2e_{bw}mbps_ms"] = max(local_ms, network_ms)
                step_row[f"comm_additive_{bw}mbps_ms"] = local_ms + network_ms
                step_row[f"network_only_{bw}mbps_ms"] = network_ms
                step_row[f"fp16_same_pp_{bw}mbps_ms"] = fp16_network_ms
                step_row[f"local_only_{bw}mbps_ms"] = local_ms
            decode_step_records.append(step_row)

        if not args.skip_drift:
            with torch.inference_mode():
                orig_out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
                orig_logits = orig_out.logits[:, -decode_res.decode_tokens - 1 :, :]
                recon_hidden = torch.cat([prefill_res.reconstructed_hidden, decode_res.reconstructed_hidden], dim=0)
                recon_hidden = recon_hidden.unsqueeze(0).to(device=device, dtype=torch.float16)
                recon_logits = run_remaining_layers(model, recon_hidden, full_mask, start_layer=pipeline.layer_boundary)
                recon_logits = recon_logits[:, -decode_res.decode_tokens - 1 :, :]
                drift_metrics = compute_logit_drift_metrics(orig_logits, recon_logits)
            drift_metrics.update(
                {
                    "dataset_name": dataset_name,
                    "config_name": "v1_latency_first",
                    "request_index": request_index,
                }
            )
            drift_records.append(drift_metrics)
            del orig_out, orig_logits, recon_hidden, recon_logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_records = [_build_latency_summary(dataset_name, request_records, args.bandwidths_mbps)]
    _save_records(request_records, dataset_out / "request_summary.parquet")
    _save_records(decode_step_records, dataset_out / "decode_step_breakdown.parquet")
    _save_records(drift_records, dataset_out / "decode_drift.parquet")
    _save_records(summary_records, dataset_out / "latency_summary.parquet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latency-first v1 benchmark")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--test-requests", type=int, default=30)
    parser.add_argument("--max-decode-tokens", type=int, default=128)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_v1_latency")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["wikitext2"])
    parser.add_argument("--table-placement", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--disable-pinned-cpu-table-copy", action="store_true")
    parser.add_argument("--disable-async-cpu-table-copy", action="store_true")
    parser.add_argument("--gpu-hot-cache-entries", type=int, default=0)
    parser.add_argument("--enable-disk-offload", action="store_true")
    parser.add_argument("--disk-offload-dir", default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-resident-blocks", type=int, default=64)
    parser.add_argument("--block-pager-workers", type=int, default=2)
    parser.add_argument("--pinned-block-budget", type=int, default=8)
    parser.add_argument("--bandwidths-mbps", nargs="+", type=int, default=DEFAULT_BANDWIDTHS_MBPS)
    parser.add_argument("--skip-drift", action="store_true", help="Skip logit drift computation for faster benchmark runs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.clean_output:
        logger.info("Cleaning existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Config: %s", json.dumps(vars(args), indent=2))
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    for dataset_name in args.datasets:
        run_dataset(model, tokenizer, device, dataset_name, args)


if __name__ == "__main__":
    main()