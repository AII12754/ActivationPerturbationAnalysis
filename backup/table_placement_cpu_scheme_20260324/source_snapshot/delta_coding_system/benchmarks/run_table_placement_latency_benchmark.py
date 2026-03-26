from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.benchmarks.run_longbench_benchmark import FP16_CONFIG_NAME
from delta_coding_system.benchmarks.run_ntp_loss_benchmark import (
    _build_pipeline_for_ntp,
    _load_model_and_tokenizer,
    _load_or_create_split_manifest,
    _wait_pending_update,
)

LOGGER = logging.getLogger("table_placement_latency")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPU-resident vs CPU-resident cache-table lookup latency")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config", default="optimized_default")
    parser.add_argument("--table-placement", choices=["gpu", "cpu"], required=True)
    parser.add_argument("--table-storage-format", choices=["raw", "int8"], default="raw")
    parser.add_argument("--cpu-transfer-mode", choices=["sync", "async_pinned"], default="async_pinned")
    parser.add_argument("--gpu-hot-cache-entries", type=int, default=0)
    parser.add_argument("--cpu-storage-format", choices=["raw", "int8"], default=None)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=100)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sequence_to_prompt(tokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": float(statistics.median(values)),
        "p95": float(_percentile(values, 0.95)),
        "max": float(max(values)),
    }


def main() -> None:
    args = _parse_args()
    if args.config == FP16_CONFIG_NAME:
        raise ValueError("FP16 baseline does not exercise cache-table lookup; choose a compressed config")
    if args.cpu_storage_format is not None:
        args.table_storage_format = args.cpu_storage_format

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    model, tokenizer = _load_model_and_tokenizer(args.model, device)
    pipeline = _build_pipeline_for_ntp(model, tokenizer, device, args)

    if args.table_storage_format == "int8":
        pipeline.table.enable_int8_storage(group_size=args.group_size, top_k=4)

    if args.table_placement == "cpu":
        pipeline.table.to_device(torch.device("cpu"))
        pipeline.table.pin_cpu_output_copy = (args.cpu_transfer_mode == "async_pinned")
        pipeline.table.enable_async_cpu_output_copy = (args.cpu_transfer_mode == "async_pinned")
        if args.gpu_hot_cache_entries > 0:
            pipeline.table.enable_gpu_hot_cache(device, args.gpu_hot_cache_entries)

    warmup_sequences, test_sequences, manifest_path = _load_or_create_split_manifest(tokenizer, args)

    for seq in warmup_sequences:
        prompt_ids = seq[:args.prompt_tokens]
        prompt = _sequence_to_prompt(tokenizer, prompt_ids)
        pipeline.process_prefill(prompt, phase="warmup")

    _wait_pending_update(pipeline)
    torch.cuda.empty_cache()

    per_request: List[Dict[str, float]] = []
    for idx, seq in enumerate(test_sequences):
        prompt_ids = seq[:args.prompt_tokens]
        prompt = _sequence_to_prompt(tokenizer, prompt_ids)
        result, _prefix_cache, _suffix_cache, _next_tok, _input_ids, _prefill_hidden, _local_prompt_refs = pipeline.process_prefill(
            prompt,
            phase="test",
        )
        classify_stats = pipeline.table.last_classify_stats
        per_request.append({
            "request_index": idx,
            "seq_len": float(result.seq_len),
            "total_ms": float(result.total_ms),
            "prefill_fwd_ms": float(result.prefill_fwd_ms),
            "classify_wait_ms": float(result.classify_ms),
            "encode_total_ms": float(result.encode_delta_ms + result.encode_unigram_ms + result.encode_self_ref_ms),
            "lookup_ms": float(classify_stats.get("lookup_ms", 0.0)),
            "materialize_ms": float(classify_stats.get("materialize_ms", 0.0)),
            "output_copy_ms": float(classify_stats.get("output_copy_ms", 0.0)),
            "pin_memory_ms": float(classify_stats.get("pin_memory_ms", 0.0)),
            "gpu_hot_hits": float(classify_stats.get("gpu_hot_hits", 0.0)),
            "num_refs": float(classify_stats.get("num_refs", 0)),
            "num_trigram": float(result.num_trigram),
            "num_bigram": float(result.num_bigram),
            "num_self_ref": float(result.num_self_ref),
            "num_unigram": float(result.num_unigram),
        })

    _wait_pending_update(pipeline)

    summary = {
        "total_ms": _summarize([row["total_ms"] for row in per_request]),
        "prefill_fwd_ms": _summarize([row["prefill_fwd_ms"] for row in per_request]),
        "classify_wait_ms": _summarize([row["classify_wait_ms"] for row in per_request]),
        "encode_total_ms": _summarize([row["encode_total_ms"] for row in per_request]),
        "lookup_ms": _summarize([row["lookup_ms"] for row in per_request]),
        "materialize_ms": _summarize([row["materialize_ms"] for row in per_request]),
        "output_copy_ms": _summarize([row["output_copy_ms"] for row in per_request]),
        "pin_memory_ms": _summarize([row["pin_memory_ms"] for row in per_request]),
        "gpu_hot_hits": _summarize([row["gpu_hot_hits"] for row in per_request]),
        "num_refs": _summarize([row["num_refs"] for row in per_request]),
        "num_trigram": _summarize([row["num_trigram"] for row in per_request]),
        "num_bigram": _summarize([row["num_bigram"] for row in per_request]),
        "num_self_ref": _summarize([row["num_self_ref"] for row in per_request]),
        "num_unigram": _summarize([row["num_unigram"] for row in per_request]),
    }

    payload = {
        "model": args.model,
        "gpu": args.gpu,
        "config": args.config,
        "table_placement": args.table_placement,
        "table_storage_format": args.table_storage_format,
        "cpu_transfer_mode": args.cpu_transfer_mode,
        "cpu_storage_format": args.cpu_storage_format,
        "gpu_hot_cache_entries": args.gpu_hot_cache_entries,
        "dataset": args.dataset,
        "warmup_requests": args.warmup_requests,
        "test_requests": args.test_requests,
        "prompt_tokens": args.prompt_tokens,
        "continuation_tokens": args.continuation_tokens,
        "max_seq_len": args.max_seq_len,
        "layer_boundary": args.layer_boundary,
        "group_size": args.group_size,
        "max_table_entries": args.max_table_entries,
        "seed": args.seed,
        "split_manifest": str(manifest_path),
        "final_table_stats": pipeline.table.stats,
        "summary": summary,
        "per_request": per_request,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    del pipeline
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()