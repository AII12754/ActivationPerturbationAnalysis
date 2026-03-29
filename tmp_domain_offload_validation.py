from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_science.core.types import resolve_dtype
from delta_coding_system.pipeline import OverlappedPipeline
from delta_coding_system.run_experiment import load_dataset_texts


def _pick_text(dataset_name: str, index: int) -> str:
    texts = load_dataset_texts(dataset_name)
    if not texts:
        raise RuntimeError(f"No texts loaded for dataset {dataset_name}")
    return texts[index % len(texts)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate automatic domain-aware disk offload/reload")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-order", nargs="+", required=True)
    parser.add_argument("--disk-offload-dir", required=True)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--table-dtype", default="float16")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    pipeline = OverlappedPipeline(
        model=model,
        tokenizer=tokenizer,
        layer_boundary=args.layer_boundary,
        table_dtype=resolve_dtype(args.table_dtype, torch.float16),
        max_table_entries=args.max_table_entries,
        group_size=args.group_size,
        top_k=args.top_k,
        int8_group_size=128,
        int8_outlier_top_k=1,
        decode_tokens=args.decode_tokens,
        max_seq_len=args.max_seq_len,
        device=device,
        domain_aware=True,
        max_gpu_tables=1,
        max_active_tables_per_request=1,
        table_placement="cpu",
        gpu_hot_cache_entries=0,
        enable_disk_offload=True,
        disk_offload_dir=args.disk_offload_dir,
        auto_topic_routing=True,
        delta_strategy="delta_noaffine_int4_k1",
        unigram_strategy="unigram_int4_k4",
        decode_use_raw_fp16=True,
        prefill_use_raw_fp16=False,
    )

    root = Path(args.disk_offload_dir)
    root.mkdir(parents=True, exist_ok=True)

    for index, dataset_name in enumerate(args.dataset_order):
        text = _pick_text(dataset_name, index)
        prefill_res, decode_res, table_stats = pipeline.process_request(
            text,
            phase="test",
            task_name=dataset_name,
        )
        file_names = sorted(path.name for path in root.glob("*.pt"))
        manager_stats = pipeline.table_manager.stats
        print(json.dumps({
            "step": index,
            "dataset": dataset_name,
            "seq_len": prefill_res.seq_len,
            "prefill_total_ms": prefill_res.total_ms,
            "decode_total_ms": decode_res.total_ms,
            "active_domains": list(pipeline._current_domains),
            "write_domains": list(pipeline._current_write_domains),
            "routing": pipeline._last_routing_info,
            "disk_domains": manager_stats.get("disk_resident", []),
            "cpu_domains": manager_stats.get("cpu_resident", []),
            "gpu_domains": manager_stats.get("gpu_resident", []),
            "files": file_names,
            "table_stats": table_stats,
        }, ensure_ascii=False))

    pipeline.release_domains()
    file_names = sorted(path.name for path in root.glob("*.pt"))
    manager_stats = pipeline.table_manager.stats
    print(json.dumps({
        "event": "after_release_domains",
        "disk_domains": manager_stats.get("disk_resident", []),
        "cpu_domains": manager_stats.get("cpu_resident", []),
        "gpu_domains": manager_stats.get("gpu_resident", []),
        "files": file_names,
    }, ensure_ascii=False))

    pipeline.shutdown()


if __name__ == "__main__":
    main()