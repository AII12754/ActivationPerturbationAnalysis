#!/usr/bin/env python3
"""Run delta-coding system experiments across multiple datasets.

For each dataset:
  - 50 warmup requests (table filling, no metrics)
  - 50 test requests (prefill + 128-token decode, full metrics)
  - Output: results_delta_system/{dataset}/*.parquet

Usage:
  python -m delta_coding_system.run_experiment --gpu 1 --datasets gsm8k triviaqa
  python -m delta_coding_system.run_experiment --gpu 1  # all 6 datasets
    python -m delta_coding_system.run_experiment --gpu 1 --datasets /path/to/my_dataset.parquet
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.pipeline import OverlappedPipeline, PrefillResult, DecodeResult
from delta_coding_system.table import create_activation_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("delta_system")

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
    "table_dtype": "float8_e4m3fn",
    "max_table_entries": 100000,
    "model_path": "/root/share/models/Qwen2.5-32B-Instruct",
    "gpu": 0,
    "seed": 42,
    "output_dir": "results_delta_system",
    "delta_strategy": "delta_noaffine_int4_k1",
    "unigram_strategy": "unigram_int4_k4",
}

ALL_DATASETS = ["cnn_dm", "sharegpt", "wikitext2", "gsm8k", "triviaqa", "alpaca"]

# ===================================================================
# Dataset loading
# ===================================================================
DATASET_ROOT = Path("/root/share/dataset")


def _records_to_texts(records: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    preferred_groups = [
        ["instruction", "input", "output"],
        ["question", "answer"],
        ["prompt", "completion"],
    ]
    preferred_single = ["text", "prompt", "content", "article", "body", "question"]

    for record in records:
        if not isinstance(record, dict):
            if record is not None:
                value = str(record).strip()
                if value:
                    texts.append(value)
            continue

        joined = None
        for group in preferred_groups:
            parts = [str(record[key]).strip() for key in group if record.get(key)]
            if parts:
                joined = "\n".join(parts)
                break
        if joined is None:
            for key in preferred_single:
                value = record.get(key)
                if value:
                    joined = str(value).strip()
                    if joined:
                        break
        if joined is None:
            parts = []
            for key, value in record.items():
                if isinstance(value, (str, int, float)) and value:
                    parts.append(str(value).strip())
                if len(parts) >= 4:
                    break
            if parts:
                joined = "\n".join(parts)
        if joined:
            texts.append(joined)
    return texts


def _load_generic_path_texts(path_str: str) -> List[str]:
    path = Path(path_str)
    if not path.exists():
        raise ValueError(f"Unknown dataset or path does not exist: {path_str}")

    paths: List[Path]
    if path.is_dir():
        paths = sorted(
            list(path.rglob("*.parquet"))
            + list(path.rglob("*.jsonl"))
            + list(path.rglob("*.json"))
            + list(path.rglob("*.txt"))
        )
    else:
        paths = [path]

    texts: List[str] = []
    for item in paths:
        suffix = item.suffix.lower()
        if suffix == ".parquet":
            table = pq.read_table(item)
            records = table.to_pylist()
            texts.extend(_records_to_texts(records))
        elif suffix == ".jsonl":
            with open(item, "r") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            texts.extend(_records_to_texts(records))
        elif suffix == ".json":
            with open(item, "r") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                texts.extend(_records_to_texts(payload))
            else:
                texts.extend(_records_to_texts([payload]))
        elif suffix == ".txt":
            with open(item, "r") as handle:
                chunks = [chunk.strip() for chunk in handle.read().split("\n\n") if chunk.strip()]
            texts.extend(chunks)

    if not texts:
        raise ValueError(f"No usable text samples found in path: {path_str}")
    return texts


def load_dataset_texts(name: str) -> List[str]:
    """Load raw texts from each dataset."""

    if Path(name).exists():
        return _load_generic_path_texts(name)

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
def run_dataset(pipeline: OverlappedPipeline, dataset_name: str, cfg: Dict[str, Any]):
    logger.info("=" * 60)
    logger.info("Dataset: %s", dataset_name)
    logger.info("=" * 60)

    texts = load_dataset_texts(dataset_name)
    logger.info("Loaded %d texts", len(texts))

    rng = random.Random(cfg["seed"])
    rng.shuffle(texts)

    warmup_n = cfg["warmup_requests"]
    test_n = cfg["test_requests"]
    total_needed = warmup_n + test_n
    while len(texts) < total_needed:
        texts.extend(texts[:total_needed - len(texts)])

    warmup_texts = texts[:warmup_n]
    test_texts = texts[warmup_n:warmup_n + test_n]

    # Reset table for each dataset
    if pipeline.domain_aware:
        pipeline.select_domain(dataset_name)
    else:
        pipeline.table = create_activation_table(
            backend=pipeline.table_backend,
            device=pipeline.device,
            dtype=pipeline.table.dtype,
            max_entries=pipeline.table.max_entries,
            storage_format=pipeline.table.storage_format,
            int8_group_size=pipeline.table.int8_group_size,
            int8_top_k=pipeline.table.int8_top_k,
            pin_cpu_output_copy=pipeline.pin_cpu_output_copy,
            enable_async_cpu_output_copy=pipeline.enable_async_cpu_output_copy,
            gpu_hot_cache_entries=pipeline.gpu_hot_cache_entries,
            gpu_hot_cache_device=pipeline.device,
            block_size=pipeline.block_size,
            enable_async_paging=pipeline.enable_async_block_paging,
            page_directory=pipeline.disk_offload_dir,
            max_resident_blocks=pipeline.max_resident_blocks,
            pager_workers=pipeline.block_pager_workers,
            pinned_block_budget=pipeline.pinned_block_budget,
        )

    prefill_records: List[Dict] = []
    decode_step_records: List[Dict] = []
    decode_agg_records: List[Dict] = []
    table_records: List[Dict] = []
    tier_detail_records: List[Dict] = []

    # --- Warmup ---
    logger.info("--- Warmup: %d requests ---", warmup_n)
    for i, text in enumerate(warmup_texts):
        prefill_res, decode_res, tbl_stats = pipeline.process_request(
            text,
            phase="warmup",
            task_name=dataset_name,
        )
        table_records.append({
            "dataset_name": dataset_name, "request_index": i, "phase": "warmup",
            **tbl_stats,
        })
        if (i + 1) % 10 == 0:
            s = pipeline.table.stats
            logger.info("Warmup %d/%d  table: %d tri, %d bi",
                        i + 1, warmup_n, s["num_trigrams"], s["num_bigrams"])

    logger.info("Warmup done. Table: %d tri, %d bi",
                pipeline.table.stats["num_trigrams"], pipeline.table.stats["num_bigrams"])

    # --- Test ---
    logger.info("--- Test: %d requests ---", test_n)
    for i, text in enumerate(test_texts):
        prefill_res, decode_res, tbl_stats = pipeline.process_request(
            text,
            phase="test",
            task_name=dataset_name,
        )

        # Prefill record
        prefill_records.append({
            "dataset_name": dataset_name, "request_index": i, "phase": "test",
            "seq_len": prefill_res.seq_len,
            "num_trigram": prefill_res.num_trigram,
            "num_bigram": prefill_res.num_bigram,
            "num_self_ref": prefill_res.num_self_ref,
            "num_unigram": prefill_res.num_unigram,
            "pct_trigram": prefill_res.num_trigram / max(prefill_res.seq_len, 1) * 100,
            "pct_bigram": prefill_res.num_bigram / max(prefill_res.seq_len, 1) * 100,
            "pct_self_ref": prefill_res.num_self_ref / max(prefill_res.seq_len, 1) * 100,
            "pct_unigram": prefill_res.num_unigram / max(prefill_res.seq_len, 1) * 100,
            "raw_cosine_mean": prefill_res.raw_cosine_mean,
            "raw_cosine_min": prefill_res.raw_cosine_min,
            "recon_cosine_mean": prefill_res.recon_cosine_mean,
            "recon_cosine_min": prefill_res.recon_cosine_min,
            "mse_mean": prefill_res.mse_mean,
            "mse_max": prefill_res.mse_max,
            "total_transfer_bytes": prefill_res.total_transfer_bytes,
            "raw_fp16_bytes": prefill_res.raw_fp16_bytes,
            "compression_ratio": prefill_res.compression_ratio,
            "transfer_bytes_trigram": prefill_res.transfer_bytes_by_tier.get("trigram", 0),
            "transfer_bytes_bigram": prefill_res.transfer_bytes_by_tier.get("bigram", 0),
            "transfer_bytes_self_ref": prefill_res.transfer_bytes_by_tier.get("self_ref", 0),
            "transfer_bytes_unigram": prefill_res.transfer_bytes_by_tier.get("unigram", 0),
            "prefill_fwd_ms": prefill_res.prefill_fwd_ms,
            "classify_ms": prefill_res.classify_ms,
            "encode_delta_ms": prefill_res.encode_delta_ms,
            "encode_self_ref_ms": prefill_res.encode_self_ref_ms,
            "encode_unigram_ms": prefill_res.encode_unigram_ms,
            "table_update_ms": prefill_res.table_update_ms,
            "total_ms": prefill_res.total_ms,
        })

        # Tier detail records
        for td in prefill_res.tier_detail:
            tier_detail_records.append({
                "dataset_name": dataset_name, "request_index": i,
                "phase": "test", "source": "prefill",
                **td,
            })

        # Decode step records
        for sr in decode_res.step_records:
            decode_step_records.append({
                "dataset_name": dataset_name, "request_index": i,
                "decode_step": sr.step, "tier": sr.tier,
                "raw_cosine": sr.raw_cosine, "recon_cosine": sr.recon_cosine,
                "transfer_bytes": sr.transfer_bytes,
                "raw_fp16_bytes": sr.raw_fp16_bytes,
                "fwd_ms": sr.fwd_ms, "classify_ms": sr.classify_ms,
                "encode_ms": sr.encode_ms, "table_update_ms": sr.table_update_ms,
            })

        # Decode aggregate record
        dt = decode_res.decode_tokens
        decode_agg_records.append({
            "dataset_name": dataset_name, "request_index": i,
            "decode_tokens": dt,
            "num_trigram": decode_res.num_trigram,
            "num_bigram": decode_res.num_bigram,
            "num_self_ref": decode_res.num_self_ref,
            "num_unigram": decode_res.num_unigram,
            "pct_trigram": decode_res.num_trigram / max(dt, 1) * 100,
            "pct_bigram": decode_res.num_bigram / max(dt, 1) * 100,
            "pct_self_ref": decode_res.num_self_ref / max(dt, 1) * 100,
            "pct_unigram": decode_res.num_unigram / max(dt, 1) * 100,
            "recon_cosine_mean": decode_res.recon_cosine_mean,
            "recon_cosine_min": decode_res.recon_cosine_min,
            "total_transfer_bytes": decode_res.total_transfer_bytes,
            "raw_fp16_bytes": decode_res.raw_fp16_bytes,
            "compression_ratio": decode_res.compression_ratio,
            "total_fwd_ms": decode_res.total_fwd_ms,
            "total_classify_ms": decode_res.total_classify_ms,
            "total_encode_ms": decode_res.total_encode_ms,
            "total_ms": decode_res.total_ms,
        })

        # Table growth record
        table_records.append({
            "dataset_name": dataset_name, "request_index": i, "phase": "test",
            **tbl_stats,
        })

        if (i + 1) % 10 == 0:
            cos_p = prefill_res.recon_cosine_mean
            cos_d = decode_res.recon_cosine_mean
            ratio_p = prefill_res.compression_ratio
            ratio_d = decode_res.compression_ratio
            s = pipeline.table.stats
            logger.info(
                "Test %d/%d  prefill: cos=%.4f ratio=%.2fx  decode: cos=%.4f ratio=%.2fx  table=%d/%d",
                i + 1, test_n, cos_p, ratio_p, cos_d, ratio_d,
                s["num_trigrams"], s["num_bigrams"],
            )

    # Save
    out_dir = Path(cfg["output_dir"]) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_records(prefill_records, out_dir / "prefill_quality.parquet")
    save_records(tier_detail_records, out_dir / "tier_detail.parquet")
    save_records(decode_step_records, out_dir / "decode_step_detail.parquet")
    save_records(decode_agg_records, out_dir / "decode_aggregate.parquet")
    save_records(table_records, out_dir / "table_growth.parquet")

    logger.info("Results saved to %s", out_dir)

    # Release domain hint (domain-aware mode)
    if pipeline.domain_aware:
        pipeline.release_domain()

    gc.collect()
    torch.cuda.empty_cache()


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Delta-coding system experiment")
    parser.add_argument("--model", default=DEFAULTS["model_path"])
    parser.add_argument("--gpu", type=int, default=DEFAULTS["gpu"])
    parser.add_argument("--warmup-requests", type=int, default=DEFAULTS["warmup_requests"])
    parser.add_argument("--test-requests", type=int, default=DEFAULTS["test_requests"])
    parser.add_argument("--decode-tokens", type=int, default=DEFAULTS["decode_tokens"])
    parser.add_argument("--layer-boundary", type=int, default=DEFAULTS["layer_boundary"])
    parser.add_argument("--group-size", type=int, default=DEFAULTS["group_size"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--max-seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--table-dtype", default=DEFAULTS["table_dtype"])
    parser.add_argument("--max-table-entries", type=int, default=DEFAULTS["max_table_entries"])
    parser.add_argument("--domain-aware", action="store_true",
                        help="Use per-dataset domain-aware tables with GPU/CPU tiering")
    parser.add_argument("--max-gpu-tables", type=int, default=3,
                        help="Max domain tables to keep on GPU (domain-aware mode)")
    parser.add_argument("--enable-disk-offload", action="store_true",
                        help="Offload inactive domain tables to disk in domain-aware mode")
    parser.add_argument("--disk-offload-dir", default=None,
                        help="Directory for disk-offloaded domain tables")
    parser.add_argument("--table-backend", choices=["trie", "block"], default="trie")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--enable-async-block-paging", action="store_true")
    parser.add_argument("--max-resident-blocks", type=int, default=0)
    parser.add_argument("--block-pager-workers", type=int, default=1)
    parser.add_argument("--pinned-block-budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    args = parser.parse_args()

    # Resolve table dtype
    from activation_science.core.types import resolve_dtype
    table_dtype = resolve_dtype(args.table_dtype, torch.float16)

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
        "table_dtype": args.table_dtype,
        "max_table_entries": args.max_table_entries,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "delta_strategy": DEFAULTS["delta_strategy"],
        "unigram_strategy": DEFAULTS["unigram_strategy"],
    }

    logger.info("Config: %s", json.dumps(cfg, indent=2))

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
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

    # Create pipeline
    pipeline = OverlappedPipeline(
        model=model,
        tokenizer=tokenizer,
        layer_boundary=args.layer_boundary,
        table_dtype=table_dtype,
        max_table_entries=args.max_table_entries,
        group_size=args.group_size,
        top_k=args.top_k,
        int8_group_size=DEFAULTS["int8_group_size"],
        int8_outlier_top_k=DEFAULTS["int8_outlier_top_k"],
        decode_tokens=args.decode_tokens,
        max_seq_len=args.max_seq_len,
        device=device,
        domain_aware=args.domain_aware,
        max_gpu_tables=args.max_gpu_tables,
        enable_disk_offload=args.enable_disk_offload,
        disk_offload_dir=args.disk_offload_dir,
        table_backend=args.table_backend,
        block_size=args.block_size,
        enable_async_block_paging=args.enable_async_block_paging,
        max_resident_blocks=args.max_resident_blocks,
        block_pager_workers=args.block_pager_workers,
        pinned_block_budget=args.pinned_block_budget,
        delta_strategy=DEFAULTS["delta_strategy"],
        unigram_strategy=DEFAULTS["unigram_strategy"],
    )

    for ds_name in args.datasets:
        run_dataset(pipeline, ds_name, cfg)

    pipeline.shutdown()
    logger.info("All done!")


if __name__ == "__main__":
    main()
