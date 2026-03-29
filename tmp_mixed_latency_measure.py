from __future__ import annotations

import argparse
import json
import statistics
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_science.core.types import resolve_dtype
from delta_coding_system.pipeline import OverlappedPipeline
from delta_coding_system.run_experiment import load_dataset_texts


def _build_sequence(datasets: List[str], repeats: int) -> List[str]:
    order: List[str] = []
    for _ in range(repeats):
        order.extend(datasets)
    return order


def _pick_text(dataset_name: str, occurrence: int) -> str:
    texts = load_dataset_texts(dataset_name)
    if not texts:
        raise RuntimeError(f"No texts loaded for dataset {dataset_name}")
    return texts[occurrence % len(texts)]


def _summarize(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p90": ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.9) - 1))],
        "p99": ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.99) - 1))],
    }


def _safe_ratio_delta(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return (numerator / denominator - 1.0) * 100.0


def _parse_profile(spec: str) -> Dict[str, object]:
    profile: Dict[str, object] = {
        "name": spec,
        "table_backend": "trie",
        "enable_disk_offload": False,
        "enable_async_block_paging": False,
        "max_resident_blocks": 0,
        "block_pager_workers": 1,
        "block_size": 256,
        "pinned_block_budget": 4,
    }
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid profile token: {part}")
        key, value = part.split("=", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        if key in {"enable_disk_offload", "enable_async_block_paging"}:
            profile[key] = value.lower() in {"1", "true", "yes", "on"}
        elif key in {"max_resident_blocks", "block_pager_workers", "block_size", "pinned_block_budget"}:
            profile[key] = int(value)
        else:
            profile[key] = value
    return profile


def _list_page_files(run_dir: Path) -> List[str]:
    return sorted(str(path.relative_to(run_dir)) for path in run_dir.glob("**/*.pt"))


def _run_pass(
    model_path: str,
    device: torch.device,
    sequence: List[str],
    warmup_sequence: List[str],
    profile: Dict[str, object],
    working_dir: str,
    max_seq_len: int,
    decode_tokens: int,
    layer_boundary: int,
    group_size: int,
    top_k: int,
    max_table_entries: int,
    table_dtype: str,
) -> Dict[str, object]:
    run_dir = Path(working_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    pipeline = OverlappedPipeline(
        model=model,
        tokenizer=tokenizer,
        layer_boundary=layer_boundary,
        table_dtype=resolve_dtype(table_dtype, torch.float16),
        max_table_entries=max_table_entries,
        group_size=group_size,
        top_k=top_k,
        int8_group_size=128,
        int8_outlier_top_k=1,
        decode_tokens=decode_tokens,
        max_seq_len=max_seq_len,
        device=device,
        domain_aware=True,
        max_gpu_tables=1,
        max_active_tables_per_request=1,
        table_placement="cpu",
        gpu_hot_cache_entries=0,
        enable_disk_offload=bool(profile["enable_disk_offload"]),
        disk_offload_dir=str(run_dir),
        table_backend=str(profile["table_backend"]),
        block_size=int(profile["block_size"]),
        enable_async_block_paging=bool(profile["enable_async_block_paging"]),
        max_resident_blocks=int(profile["max_resident_blocks"]),
        block_pager_workers=int(profile["block_pager_workers"]),
        auto_topic_routing=True,
        delta_strategy="delta_noaffine_int4_k1",
        unigram_strategy="unigram_int4_k4",
        decode_use_raw_fp16=True,
        prefill_use_raw_fp16=False,
        pinned_block_budget=int(profile["pinned_block_budget"]),
    )

    occurrence_count: Dict[str, int] = {}
    for dataset_name in warmup_sequence:
        occurrence = occurrence_count.get(dataset_name, 0)
        occurrence_count[dataset_name] = occurrence + 1
        text = _pick_text(dataset_name, occurrence)
        pipeline.process_request(text, phase="warmup", task_name=dataset_name)

    rows: List[Dict[str, object]] = []
    for step, dataset_name in enumerate(sequence):
        occurrence = occurrence_count.get(dataset_name, 0)
        occurrence_count[dataset_name] = occurrence + 1
        text = _pick_text(dataset_name, occurrence)
        prefill_res, decode_res, _table_stats = pipeline.process_request(
            text,
            phase="test",
            task_name=dataset_name,
        )
        manager_stats = pipeline.table_manager.stats
        rows.append({
            "step": step,
            "dataset": dataset_name,
            "seq_len": prefill_res.seq_len,
            "prefill_ms": prefill_res.total_ms,
            "decode_ms": decode_res.total_ms,
            "e2e_ms": prefill_res.total_ms + decode_res.total_ms,
            "profile": profile["name"],
            "disk_resident": manager_stats.get("disk_resident", 0),
            "cpu_resident": manager_stats.get("cpu_resident", []),
            "pager": manager_stats.get("pager", {}),
            "files": _list_page_files(run_dir),
        })

    pipeline.release_domains()
    final_files = _list_page_files(run_dir)
    manager_stats = pipeline.table_manager.stats
    pipeline.shutdown()
    del model
    torch.cuda.empty_cache()

    prefill_values = [float(row["prefill_ms"]) for row in rows]
    decode_values = [float(row["decode_ms"]) for row in rows]
    e2e_values = [float(row["e2e_ms"]) for row in rows]
    return {
        "profile": profile,
        "rows": rows,
        "summary": {
            "prefill_ms": _summarize(prefill_values),
            "decode_ms": _summarize(decode_values),
            "e2e_ms": _summarize(e2e_values),
        },
        "manager_stats": manager_stats,
        "final_files": final_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure mixed-workload latency across storage profiles")
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--test-repeats", type=int, default=3)
    parser.add_argument("--working-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        help=(
            "Profile spec, e.g. name=trie_baseline,table_backend=trie or "
            "name=block_async,table_backend=block,enable_async_block_paging=true,"
            "max_resident_blocks=64,block_pager_workers=2,block_size=256,pinned_block_budget=8"
        ),
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--table-dtype", default="float16")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    warmup_sequence = _build_sequence(args.datasets, args.warmup_repeats)
    test_sequence = _build_sequence(args.datasets, args.test_repeats)

    root = Path(args.working_root)
    root.mkdir(parents=True, exist_ok=True)

    profiles = [_parse_profile(spec) for spec in args.profile]
    runs: List[Dict[str, object]] = []
    for profile in profiles:
        name = str(profile["name"])
        runs.append(
            _run_pass(
                model_path=args.model,
                device=device,
                sequence=test_sequence,
                warmup_sequence=warmup_sequence,
                profile=profile,
                working_dir=str(root / name),
                max_seq_len=args.max_seq_len,
                decode_tokens=args.decode_tokens,
                layer_boundary=args.layer_boundary,
                group_size=args.group_size,
                top_k=args.top_k,
                max_table_entries=args.max_table_entries,
                table_dtype=args.table_dtype,
            )
        )

    baseline = runs[0]
    comparisons: List[Dict[str, object]] = []
    baseline_name = str(baseline["profile"]["name"])
    for run in runs[1:]:
        run_name = str(run["profile"]["name"])
        comparisons.append({
            "baseline": baseline_name,
            "candidate": run_name,
            "prefill_mean_delta_ms": run["summary"]["prefill_ms"]["mean"] - baseline["summary"]["prefill_ms"]["mean"],
            "decode_mean_delta_ms": run["summary"]["decode_ms"]["mean"] - baseline["summary"]["decode_ms"]["mean"],
            "e2e_mean_delta_ms": run["summary"]["e2e_ms"]["mean"] - baseline["summary"]["e2e_ms"]["mean"],
            "prefill_p99_delta_ms": run["summary"]["prefill_ms"]["p99"] - baseline["summary"]["prefill_ms"]["p99"],
            "decode_p99_delta_ms": run["summary"]["decode_ms"]["p99"] - baseline["summary"]["decode_ms"]["p99"],
            "e2e_p99_delta_ms": run["summary"]["e2e_ms"]["p99"] - baseline["summary"]["e2e_ms"]["p99"],
            "prefill_mean_delta_pct": _safe_ratio_delta(run["summary"]["prefill_ms"]["mean"], baseline["summary"]["prefill_ms"]["mean"]),
            "decode_mean_delta_pct": _safe_ratio_delta(run["summary"]["decode_ms"]["mean"], baseline["summary"]["decode_ms"]["mean"]),
            "e2e_mean_delta_pct": _safe_ratio_delta(run["summary"]["e2e_ms"]["mean"], baseline["summary"]["e2e_ms"]["mean"]),
            "prefill_p99_delta_pct": _safe_ratio_delta(run["summary"]["prefill_ms"]["p99"], baseline["summary"]["prefill_ms"]["p99"]),
            "decode_p99_delta_pct": _safe_ratio_delta(run["summary"]["decode_ms"]["p99"], baseline["summary"]["decode_ms"]["p99"]),
            "e2e_p99_delta_pct": _safe_ratio_delta(run["summary"]["e2e_ms"]["p99"], baseline["summary"]["e2e_ms"]["p99"]),
        })

    payload = {
        "datasets": args.datasets,
        "warmup_sequence": warmup_sequence,
        "test_sequence": test_sequence,
        "profiles": profiles,
        "runs": runs,
        "comparisons": comparisons,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload["comparisons"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()