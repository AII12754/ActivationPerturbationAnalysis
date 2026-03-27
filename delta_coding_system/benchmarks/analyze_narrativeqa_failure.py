#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.benchmarks.run_longbench_benchmark import (  # noqa: E402
    CONFIGS,
    _best_qa_f1,
    _build_pipeline,
    _candidate_seq_lens,
    _is_oom_error,
)
from delta_coding_system.benchmarks.run_longbench_official_benchmark import (  # noqa: E402
    LOCAL_DATASET_ROOT,
    OFFICIAL_CONFIG_DIR,
    _build_chat_prompt,
    _load_json,
    _load_model_and_tokenizer,
    _newline_stop_ids,
    _post_process,
    _truncate_in_middle,
)

LOGGER = logging.getLogger("narrativeqa_failure_analysis")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_narrativeqa_failure_analysis"
PROMPT_LEAK_PATTERNS = (
    "Question:",
    "Do not provide any explanation",
    "provide any explanation",
    "single phrase if possible",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic analysis for NarrativeQA failure modes")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS.keys()), required=True)
    parser.add_argument("--task", default="narrativeqa")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--table-placement", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-hot-cache-entries", type=int, default=0)
    parser.add_argument("--mode", choices=("shared", "reset-each-sample"), default="shared")
    parser.add_argument("--raw-table-storage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label", default="")
    return parser.parse_args()


def _load_records(task_name: str, limit: int) -> List[Dict[str, object]]:
    path = LOCAL_DATASET_ROOT / f"{task_name}.jsonl"
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _is_prompt_leak(prediction: str) -> bool:
    return any(marker in prediction for marker in PROMPT_LEAK_PATTERNS)


def _sanitize_label(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "run"


def _build_prompt(prompt_map: Dict[str, object], tokenizer, model_path: str, task_name: str, row: Dict[str, object], max_seq_len: int) -> str:
    raw_prompt = str(prompt_map[task_name]).format(**row)
    truncated_prompt = _truncate_in_middle(tokenizer, raw_prompt, max_seq_len)
    return _build_chat_prompt(tokenizer, truncated_prompt, model_path, task_name)


def _make_pipeline(model, tokenizer, device: torch.device, args: argparse.Namespace, max_new_tokens: int):
    pipeline_args = SimpleNamespace(
        layer_boundary=args.layer_boundary,
        max_table_entries=args.max_table_entries,
        group_size=args.group_size,
        max_new_tokens=max_new_tokens,
        max_seq_len=args.max_seq_len,
        score_only=True,
        table_placement=args.table_placement,
        gpu_hot_cache_entries=args.gpu_hot_cache_entries,
        disable_pinned_cpu_table_copy=False,
        disable_async_cpu_table_copy=False,
    )
    pipeline = _build_pipeline(model, tokenizer, device, pipeline_args, args.config)
    if args.raw_table_storage and pipeline.table is not None:
        pipeline.table.disable_int8_storage()
    return pipeline


def _run_request_with_retry(pipeline, prompt: str, task_name: str):
    original_budget = pipeline.max_seq_len
    last_error: BaseException | None = None
    for budget in _candidate_seq_lens(original_budget):
        pipeline.max_seq_len = budget
        try:
            return pipeline.process_request(prompt, phase="test", task_name=task_name)
        except BaseException as exc:  # noqa: BLE001
            if not _is_oom_error(exc):
                pipeline.max_seq_len = original_budget
                raise
            last_error = exc
            LOGGER.warning("OOM at max_seq_len=%d, retrying with smaller budget", budget)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pipeline.max_seq_len = original_budget
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop exited without result")


def _decode_prediction(pipeline, tokenizer, task_name: str, generated_token_ids: List[int]) -> str:
    original_stop_ids = set(pipeline.extra_stop_token_ids)
    try:
        pipeline.extra_stop_token_ids = set(_newline_stop_ids(tokenizer, task_name))
        prediction = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    finally:
        pipeline.extra_stop_token_ids = original_stop_ids
    return _post_process(prediction, str(pipeline.model.name_or_path))


def _shutdown_pipeline(pipeline) -> None:
    if pipeline is None:
        return
    pipeline.classify_executor.shutdown(wait=True)
    pipeline.update_executor.shutdown(wait=True)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    prompt_map = _load_json(OFFICIAL_CONFIG_DIR / "dataset2prompt.json")
    max_gen_map = _load_json(OFFICIAL_CONFIG_DIR / "dataset2maxlen.json")
    records = _load_records(args.task, args.samples)
    max_new_tokens = int(max_gen_map[args.task])

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    model, tokenizer = _load_model_and_tokenizer(args.model, device, tp_size=1)

    label_parts = [args.task, args.config, args.mode]
    if args.raw_table_storage:
        label_parts.append("rawtable")
    else:
        label_parts.append("int8table")
    if args.label:
        label_parts.append(args.label)
    run_label = _sanitize_label("_".join(label_parts))
    output_dir = Path(args.output_dir) / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    summary_path = output_dir / "summary.json"

    pipeline = None
    if args.mode == "shared":
        pipeline = _make_pipeline(model, tokenizer, device, args, max_new_tokens)

    total_score = 0.0
    prompt_leak_count = 0
    outputs: List[Dict[str, object]] = []

    try:
        for idx, row in enumerate(records, start=1):
            if args.mode == "reset-each-sample":
                pipeline = _make_pipeline(model, tokenizer, device, args, max_new_tokens)

            prompt = _build_prompt(prompt_map, tokenizer, args.model, args.task, row, args.max_seq_len)
            prefill_res, decode_res, table_stats = _run_request_with_retry(pipeline, prompt, args.task)
            prediction = _decode_prediction(pipeline, tokenizer, args.task, decode_res.generated_token_ids)
            score = float(_best_qa_f1(prediction, row.get("answers", [])))
            leak = _is_prompt_leak(prediction)

            total_score += score
            prompt_leak_count += int(leak)

            record = {
                "index": idx,
                "_id": row.get("_id"),
                "length": row.get("length"),
                "answers": row.get("answers"),
                "pred": prediction,
                "qa_f1": score,
                "prompt_leak": leak,
                "prefill": {
                    "seq_len": prefill_res.seq_len,
                    "num_trigram": prefill_res.num_trigram,
                    "num_bigram": prefill_res.num_bigram,
                    "num_self_ref": prefill_res.num_self_ref,
                    "num_unigram": prefill_res.num_unigram,
                    "raw_cosine_mean": prefill_res.raw_cosine_mean,
                    "raw_cosine_min": prefill_res.raw_cosine_min,
                    "recon_cosine_mean": prefill_res.recon_cosine_mean,
                    "recon_cosine_min": prefill_res.recon_cosine_min,
                    "mse_mean": prefill_res.mse_mean,
                    "mse_max": prefill_res.mse_max,
                    "tier_detail": prefill_res.tier_detail,
                },
                "decode": {
                    "decode_tokens": decode_res.decode_tokens,
                    "recon_cosine_mean": decode_res.recon_cosine_mean,
                    "recon_cosine_min": decode_res.recon_cosine_min,
                },
                "table_stats": table_stats,
            }
            outputs.append(record)
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

            LOGGER.info(
                "task=%s config=%s mode=%s sample=%d/%d score=%.4f leak=%s",
                args.task,
                args.config,
                args.mode,
                idx,
                len(records),
                score,
                leak,
            )

            if args.mode == "reset-each-sample":
                _shutdown_pipeline(pipeline)
                pipeline = None

        summary = {
            "task": args.task,
            "config": args.config,
            "mode": args.mode,
            "raw_table_storage": args.raw_table_storage,
            "gpu": args.gpu,
            "samples": len(outputs),
            "avg_qa_f1": total_score / max(len(outputs), 1),
            "prompt_leak_count": prompt_leak_count,
            "prompt_leak_rate": prompt_leak_count / max(len(outputs), 1),
            "exact_prediction_set_size": len({row["pred"] for row in outputs}),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("summary=%s", summary_path)
    finally:
        _shutdown_pipeline(pipeline)


if __name__ == "__main__":
    main()