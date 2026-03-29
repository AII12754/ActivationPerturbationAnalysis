#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

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
        _build_final_prompt,
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
MARKER_VARIANTS = {
    "story": ["Story:", "\nStory:", "\n\nStory:"],
    "question": ["Question:", "\nQuestion:", "\n\nQuestion:"],
    "answer": ["Answer:", "\nAnswer:", "\n\nAnswer:"],
}


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
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--analyze-token-errors", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--token-window-size", type=int, default=256)
    parser.add_argument("--normalized-bins", type=int, default=32)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label", default="")
    return parser.parse_args()


def _parse_sample_indices(text: str) -> List[int]:
    indices: List[int] = []
    if not text.strip():
        return indices
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 0:
            raise ValueError(f"sample index must be positive, got {value}")
        indices.append(value)
    return sorted(set(indices))


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
    return _build_final_prompt(
        tokenizer,
        str(prompt_map[task_name]),
        model_path,
        task_name,
        row,
        max_seq_len,
    )


def _segment_ranges_for_prompt(tokenizer, prompt_text: str, token_ids: List[int], task_name: str) -> List[Dict[str, int | str]]:
    seq_len = len(token_ids)
    if task_name != "narrativeqa":
        return [{"name": "full_prompt", "start": 0, "end": seq_len}]

    def _locate_marker_chars(variants: List[str]) -> Tuple[int, int]:
        best_start = -1
        best_len = 0
        for variant in variants:
            start = prompt_text.rfind(variant)
            if start > best_start:
                best_start = start
                best_len = len(variant)
        return best_start, best_len

    def _char_to_token_pos(char_pos: int) -> int:
        if char_pos <= 0:
            return 0
        token_pos = len(tokenizer.encode(prompt_text[:char_pos], add_special_tokens=False))
        return min(token_pos, seq_len)

    story_char, story_char_len = _locate_marker_chars(MARKER_VARIANTS["story"])
    question_char, _question_char_len = _locate_marker_chars(MARKER_VARIANTS["question"])
    answer_char, _answer_char_len = _locate_marker_chars(MARKER_VARIANTS["answer"])

    if story_char < 0 or question_char < 0 or answer_char < 0:
        return [{"name": "full_prompt", "start": 0, "end": seq_len}]

    story_body_start = _char_to_token_pos(story_char + story_char_len)
    question_start = _char_to_token_pos(question_char)
    answer_start = _char_to_token_pos(answer_char)
    ranges: List[Dict[str, int | str]] = [
        {"name": "instruction_prefix", "start": 0, "end": story_body_start},
        {"name": "context_body", "start": story_body_start, "end": question_start},
        {"name": "question_block", "start": question_start, "end": answer_start},
        {"name": "answer_cue", "start": answer_start, "end": seq_len},
    ]

    context_start = story_body_start
    context_end = question_start
    if context_end > context_start:
        context_len = context_end - context_start
        for part_idx in range(4):
            part_start = context_start + (context_len * part_idx) // 4
            part_end = context_start + (context_len * (part_idx + 1)) // 4
            ranges.append(
                {
                    "name": f"context_q{part_idx + 1}",
                    "start": part_start,
                    "end": part_end,
                }
            )
    return ranges


def _segment_name_for_pos(pos: int, ranges: List[Dict[str, int | str]]) -> str:
    for item in ranges:
        start = int(item["start"])
        end = int(item["end"])
        if start <= pos < end:
            return str(item["name"])
    return "full_prompt"


def _disjoint_window_stats(
    cosine: List[float],
    mse: List[float],
    segment_labels: List[str],
    window_size: int,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    seq_len = len(cosine)
    if seq_len == 0:
        return windows
    for start in range(0, seq_len, window_size):
        end = min(seq_len, start + window_size)
        window_cos = cosine[start:end]
        window_mse = mse[start:end]
        label_counts: Dict[str, int] = {}
        for label in segment_labels[start:end]:
            label_counts[label] = label_counts.get(label, 0) + 1
        dominant = max(label_counts.items(), key=lambda item: item[1])[0]
        windows.append(
            {
                "start": start,
                "end": end,
                "center": (start + end) / 2.0,
                "mean_recon_cosine": float(sum(window_cos) / max(len(window_cos), 1)),
                "mean_mse": float(sum(window_mse) / max(len(window_mse), 1)),
                "dominant_segment": dominant,
            }
        )
    return windows


def _normalized_bin_stats(values: List[float], start: int, end: int, bins: int) -> List[Dict[str, Any]]:
    if end <= start:
        return []
    stats: List[Dict[str, Any]] = []
    length = end - start
    for bin_idx in range(bins):
        bin_start = start + (length * bin_idx) // bins
        bin_end = start + (length * (bin_idx + 1)) // bins
        if bin_end <= bin_start:
            continue
        chunk = values[bin_start:bin_end]
        stats.append(
            {
                "bin": bin_idx,
                "start": bin_start,
                "end": bin_end,
                "mean": float(sum(chunk) / max(len(chunk), 1)),
            }
        )
    return stats


def _compute_token_error_analysis(
    tokenizer,
    task_name: str,
    prompt: str,
    max_seq_len: int,
    real_hidden: torch.Tensor,
    reconstructed_hidden: torch.Tensor,
    normalized_bins: int,
    token_window_size: int,
) -> Dict[str, Any]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)[:max_seq_len]
    seq_len = min(len(token_ids), int(real_hidden.shape[0]), int(reconstructed_hidden.shape[0]))
    token_ids = token_ids[:seq_len]

    real = real_hidden[:seq_len].float()
    recon = reconstructed_hidden[:seq_len].float()
    recon_cosine = F.cosine_similarity(real, recon, dim=-1).detach().cpu().tolist()
    mse = ((real - recon) ** 2).mean(dim=-1).detach().cpu().tolist()

    segment_ranges = _segment_ranges_for_prompt(tokenizer, prompt, token_ids, task_name)
    segment_labels = [_segment_name_for_pos(pos, segment_ranges) for pos in range(seq_len)]

    segment_stats: Dict[str, Dict[str, Any]] = {}
    for item in segment_ranges:
        name = str(item["name"])
        start = int(item["start"])
        end = min(int(item["end"]), seq_len)
        if end <= start:
            continue
        seg_cos = recon_cosine[start:end]
        seg_mse = mse[start:end]
        segment_stats[name] = {
            "start": start,
            "end": end,
            "tokens": end - start,
            "mean_recon_cosine": float(sum(seg_cos) / max(len(seg_cos), 1)),
            "min_recon_cosine": float(min(seg_cos)),
            "mean_mse": float(sum(seg_mse) / max(len(seg_mse), 1)),
            "max_mse": float(max(seg_mse)),
        }

    windows = _disjoint_window_stats(recon_cosine, mse, segment_labels, token_window_size)
    worst_cosine_windows = sorted(windows, key=lambda item: item["mean_recon_cosine"])[:8]
    worst_mse_windows = sorted(windows, key=lambda item: item["mean_mse"], reverse=True)[:8]

    context_ranges = [item for item in segment_ranges if str(item["name"]) == "context_body"]
    context_start = int(context_ranges[0]["start"]) if context_ranges else 0
    context_end = int(context_ranges[0]["end"]) if context_ranges else seq_len

    return {
        "seq_len": seq_len,
        "token_ids": token_ids,
        "recon_cosine": recon_cosine,
        "mse": mse,
        "segment_ranges": segment_ranges,
        "segment_labels": segment_labels,
        "segment_stats": segment_stats,
        "full_prompt_cosine_bins": _normalized_bin_stats(recon_cosine, 0, seq_len, normalized_bins),
        "full_prompt_mse_bins": _normalized_bin_stats(mse, 0, seq_len, normalized_bins),
        "context_cosine_bins": _normalized_bin_stats(recon_cosine, context_start, context_end, normalized_bins),
        "context_mse_bins": _normalized_bin_stats(mse, context_start, context_end, normalized_bins),
        "worst_cosine_windows": worst_cosine_windows,
        "worst_mse_windows": worst_mse_windows,
    }


def _aggregate_token_analysis(rows: List[Dict[str, Any]], normalized_bins: int) -> Dict[str, Any]:
    if not rows:
        return {}

    segment_bucket: Dict[str, Dict[str, float]] = {}
    full_cos_bins = [{"sum": 0.0, "count": 0} for _ in range(normalized_bins)]
    full_mse_bins = [{"sum": 0.0, "count": 0} for _ in range(normalized_bins)]
    ctx_cos_bins = [{"sum": 0.0, "count": 0} for _ in range(normalized_bins)]
    ctx_mse_bins = [{"sum": 0.0, "count": 0} for _ in range(normalized_bins)]
    worst_cos_windows: List[Dict[str, Any]] = []
    worst_mse_windows: List[Dict[str, Any]] = []

    for row in rows:
        token_analysis = row.get("token_analysis")
        if not token_analysis:
            continue
        for name, stats in token_analysis["segment_stats"].items():
            bucket = segment_bucket.setdefault(
                name,
                {
                    "tokens": 0.0,
                    "weighted_cosine": 0.0,
                    "weighted_mse": 0.0,
                    "min_recon_cosine": 1.0,
                    "max_mse": 0.0,
                },
            )
            tokens = float(stats["tokens"])
            bucket["tokens"] += tokens
            bucket["weighted_cosine"] += float(stats["mean_recon_cosine"]) * tokens
            bucket["weighted_mse"] += float(stats["mean_mse"]) * tokens
            bucket["min_recon_cosine"] = min(bucket["min_recon_cosine"], float(stats["min_recon_cosine"]))
            bucket["max_mse"] = max(bucket["max_mse"], float(stats["max_mse"]))

        for source, target in [
            (token_analysis["full_prompt_cosine_bins"], full_cos_bins),
            (token_analysis["full_prompt_mse_bins"], full_mse_bins),
            (token_analysis["context_cosine_bins"], ctx_cos_bins),
            (token_analysis["context_mse_bins"], ctx_mse_bins),
        ]:
            for item in source:
                idx = int(item["bin"])
                if idx >= normalized_bins:
                    continue
                target[idx]["sum"] += float(item["mean"])
                target[idx]["count"] += 1

        for item in token_analysis["worst_cosine_windows"]:
            tagged = dict(item)
            tagged["sample_index"] = row["index"]
            worst_cos_windows.append(tagged)
        for item in token_analysis["worst_mse_windows"]:
            tagged = dict(item)
            tagged["sample_index"] = row["index"]
            worst_mse_windows.append(tagged)

    segment_summary = {}
    for name, bucket in segment_bucket.items():
        tokens = max(bucket["tokens"], 1.0)
        segment_summary[name] = {
            "tokens": int(bucket["tokens"]),
            "mean_recon_cosine": bucket["weighted_cosine"] / tokens,
            "mean_mse": bucket["weighted_mse"] / tokens,
            "min_recon_cosine": bucket["min_recon_cosine"],
            "max_mse": bucket["max_mse"],
        }

    def _finalize_bins(items: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            if item["count"] == 0:
                continue
            result.append({"bin": idx, "mean": item["sum"] / item["count"], "count": item["count"]})
        return result

    return {
        "samples": len(rows),
        "segment_summary": segment_summary,
        "full_prompt_cosine_bins": _finalize_bins(full_cos_bins),
        "full_prompt_mse_bins": _finalize_bins(full_mse_bins),
        "context_cosine_bins": _finalize_bins(ctx_cos_bins),
        "context_mse_bins": _finalize_bins(ctx_mse_bins),
        "worst_cosine_windows": sorted(worst_cos_windows, key=lambda item: item["mean_recon_cosine"])[:12],
        "worst_mse_windows": sorted(worst_mse_windows, key=lambda item: item["mean_mse"], reverse=True)[:12],
    }


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


def _run_request_with_hidden_retry(pipeline, prompt: str, task_name: str):
    original_budget = pipeline.max_seq_len
    last_error: BaseException | None = None
    for budget in _candidate_seq_lens(original_budget):
        pipeline.max_seq_len = budget
        try:
            prefill_outputs = pipeline.process_prefill(prompt, phase="test", task_name=task_name)
            if len(prefill_outputs) == 6:
                prefill_res, prefix_cache, next_tok, input_ids, prefill_hidden, local_prompt_refs = prefill_outputs
                _suffix_logits, suffix_cache = pipeline._run_suffix_prefill(prefill_hidden)
            else:
                (
                    prefill_res,
                    prefix_cache,
                    suffix_cache,
                    next_tok,
                    input_ids,
                    prefill_hidden,
                    local_prompt_refs,
                ) = prefill_outputs

            decode_res = pipeline.process_decode(
                prefix_cache,
                suffix_cache,
                next_tok,
                input_ids,
                prefill_hidden,
                prefill_res.reconstructed_hidden,
                local_prompt_refs,
                phase="test",
            )
            table_stats = pipeline.table.stats if pipeline.table is not None else {}
            if pipeline.table is not None:
                table_stats["last_evicted"] = pipeline.table._last_evicted
            return prefill_res, decode_res, table_stats, prefill_hidden
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
    sample_indices = _parse_sample_indices(args.sample_indices)
    sample_index_set = set(sample_indices)

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
    token_dir = output_dir / "token_analysis"
    if args.analyze_token_errors:
        token_dir.mkdir(parents=True, exist_ok=True)

    pipeline = None
    if args.mode == "shared":
        pipeline = _make_pipeline(model, tokenizer, device, args, max_new_tokens)

    total_score = 0.0
    prompt_leak_count = 0
    outputs: List[Dict[str, object]] = []
    analyzed_token_rows: List[Dict[str, Any]] = []

    try:
        for idx, row in enumerate(records, start=1):
            if sample_index_set and idx not in sample_index_set:
                continue
            if args.mode == "reset-each-sample":
                pipeline = _make_pipeline(model, tokenizer, device, args, max_new_tokens)

            prompt = _build_prompt(prompt_map, tokenizer, args.model, args.task, row, args.max_seq_len)
            if args.analyze_token_errors:
                prefill_res, decode_res, table_stats, prefill_hidden = _run_request_with_hidden_retry(
                    pipeline,
                    prompt,
                    args.task,
                )
            else:
                prefill_res, decode_res, table_stats = _run_request_with_retry(pipeline, prompt, args.task)
                prefill_hidden = None
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

            if args.analyze_token_errors:
                token_analysis = _compute_token_error_analysis(
                    tokenizer,
                    args.task,
                    prompt,
                    pipeline.max_seq_len,
                    prefill_hidden,
                    prefill_res.reconstructed_hidden,
                    args.normalized_bins,
                    args.token_window_size,
                )
                token_path = token_dir / f"sample_{idx:03d}.json"
                token_path.write_text(json.dumps(token_analysis, ensure_ascii=False), encoding="utf-8")
                record["token_analysis"] = {
                    "path": str(token_path),
                    "segment_stats": token_analysis["segment_stats"],
                    "worst_cosine_windows": token_analysis["worst_cosine_windows"],
                    "worst_mse_windows": token_analysis["worst_mse_windows"],
                }
                analyzed_token_rows.append({"index": idx, "token_analysis": token_analysis})
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
            "sample_indices": sample_indices,
            "samples": len(outputs),
            "avg_qa_f1": total_score / max(len(outputs), 1),
            "prompt_leak_count": prompt_leak_count,
            "prompt_leak_rate": prompt_leak_count / max(len(outputs), 1),
            "exact_prediction_set_size": len({row["pred"] for row in outputs}),
        }
        if args.analyze_token_errors:
            summary["token_error_summary"] = _aggregate_token_analysis(analyzed_token_rows, args.normalized_bins)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("summary=%s", summary_path)
    finally:
        _shutdown_pipeline(pipeline)


if __name__ == "__main__":
    main()