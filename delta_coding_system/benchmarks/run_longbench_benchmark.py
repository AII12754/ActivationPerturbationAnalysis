#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.pipeline import OverlappedPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("longbench_benchmark")

_DIST_RANK = 0
_DIST_WORLD_SIZE = 1

DATASET_ROOT = Path("/root/share/dataset/Longbench/data")
FP16_CONFIG_NAME = "fp16_baseline"

CONFIGS = {
    "optimized_default": {
        "delta_strategy": "delta_noaffine_int4_k1",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int4_k1_unigram_int4": {
        "delta_strategy": "delta_noaffine_int4_k1",
        "unigram_strategy": "unigram_int4_k4",
    },
    "pure_int2": {
        "delta_strategy": "direct_int2",
        "unigram_strategy": "unigram_int2_k4",
    },
    "pure_int8": {
        "delta_strategy": "direct_int8",
        "unigram_strategy": "baseline_current",
    },
    "pure_int4": {
        "delta_strategy": "direct_int4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_baseline_unigram_int4": {
        "delta_strategy": "baseline_current",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k8_unigram_int4": {
        "delta_strategy": "delta_int2_k8_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_noaffine_int2_k8_unigram_int4": {
        "delta_strategy": "delta_noaffine_int2_k8_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k4_unigram_int4": {
        "delta_strategy": "delta_int2_k4_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_noaffine_int2_k4_unigram_int4": {
        "delta_strategy": "delta_noaffine_int2_k4_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k2_unigram_int4": {
        "delta_strategy": "delta_int2_k2_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_noaffine_int2_k2_unigram_int4": {
        "delta_strategy": "delta_noaffine_int2_k2_out4",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k8_entropy_unigram_int4": {
        "delta_strategy": "delta_int2_k8_out4_entropy",
        "unigram_strategy": "unigram_int4_k4",
    },
    "delta_int2_k4_entropy_unigram_int4": {
        "delta_strategy": "delta_int2_k4_out4_entropy",
        "unigram_strategy": "unigram_int4_k4",
    },
    "legacy_prev_int4_k2": {
        "delta_strategy": "delta_noaffine_int4_k1",
        "unigram_strategy": "prev_int4_k2",
    },
    "legacy_prev_gs256_k2": {
        "delta_strategy": "delta_noaffine_int4_k1",
        "unigram_strategy": "prev_gs256_k2",
    },
}

LEGACY_CONFIG_NAMES = (
    "legacy_prev_int4_k2",
    "legacy_prev_gs256_k2",
)

TASKS = {
    "narrativeqa": {"metric": "qa_f1"},
    "hotpotqa_e": {"metric": "qa_f1"},
    "2wikimqa_e": {"metric": "qa_f1"},
    "musique": {"metric": "qa_f1"},
    "qasper": {"metric": "qa_f1"},
    "triviaqa_e": {"metric": "qa_f1"},
    "qmsum": {"metric": "rouge_l"},
    "gov_report": {"metric": "rouge_l"},
    "passage_count": {"metric": "count_exact"},
    "passage_retrieval_en_e": {"metric": "retrieval_exact"},
    "passage_count_e": {"metric": "count_exact"},
}

ARTICLES = {"a", "an", "the"}
FALLBACK_MAX_SEQ_LENS = (8192, 6144, 4096, 3072, 2048, 1536, 1024, 768)


def _network_ms(total_bytes: float, bandwidth_mbps: int) -> float:
    return float(total_bytes) * 8.0 / (float(bandwidth_mbps) * 1000.0)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [tok for tok in text.split() if tok not in ARTICLES]
    return " ".join(tokens)


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: Dict[str, int] = defaultdict(int)
    gold_counts: Dict[str, int] = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in gold_tokens:
        gold_counts[token] += 1
    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_qa_f1(prediction: str, answers: Sequence[str]) -> float:
    return max((_token_f1(prediction, answer) for answer in answers), default=0.0)


def _lcs_length(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> int:
    if not left_tokens or not right_tokens:
        return 0
    prev = [0] * (len(right_tokens) + 1)
    for left_tok in left_tokens:
        curr = [0]
        for idx, right_tok in enumerate(right_tokens, start=1):
            if left_tok == right_tok:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]


def _rouge_l_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _best_rouge_l(prediction: str, answers: Sequence[str]) -> float:
    return max((_rouge_l_f1(prediction, answer) for answer in answers), default=0.0)


def _extract_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def _extract_paragraph_id(text: str) -> int | None:
    match = re.search(r"paragraph\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return _extract_first_int(text)


def _score_prediction(metric: str, prediction: str, answers: Sequence[str]) -> float:
    if metric == "qa_f1":
        return _best_qa_f1(prediction, answers)
    if metric == "rouge_l":
        return _best_rouge_l(prediction, answers)
    if metric == "retrieval_exact":
        pred_id = _extract_paragraph_id(prediction)
        gold_ids = {_extract_paragraph_id(answer) for answer in answers}
        return 1.0 if pred_id is not None and pred_id in gold_ids else 0.0
    if metric == "count_exact":
        pred_num = _extract_first_int(prediction)
        gold_nums = {_extract_first_int(answer) for answer in answers}
        return 1.0 if pred_num is not None and pred_num in gold_nums else 0.0
    raise ValueError(f"Unsupported metric: {metric}")


def _truncate_prediction(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    first_line = lines[0]
    return first_line[:512].strip()


def _decode_generated(tokenizer, generated_ids: List[int]) -> str:
    if not generated_ids:
        return ""
    return _truncate_prediction(tokenizer.decode(generated_ids, skip_special_tokens=True))


def _build_prompt(task_name: str, row: Dict[str, object]) -> str:
    context = str(row["context"]).strip()
    question = str(row["input"]).strip()
    instruction = (
        "Answer the question using only the provided context. "
        "Give a short final answer without explanation."
    )
    if task_name in {"qmsum", "gov_report"}:
        instruction = "Write a concise summary grounded only in the provided context."
    elif task_name in {"passage_retrieval_en_e", "passage_retrieval_en"}:
        instruction = (
            "Identify which paragraph best matches the query using only the provided context. "
            "Answer with the paragraph number only, for example: Paragraph 6."
        )
    elif task_name in {"passage_count_e", "passage_count"}:
        instruction = (
            "Count how many paragraphs satisfy the condition using only the provided context. "
            "Answer with the number only."
        )
    return f"{instruction}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"


def _context_length_chars(row: Dict[str, object]) -> int:
    return len(str(row.get("context", "")))


def _load_task_records(
    task_name: str,
    max_samples: int | None,
    min_context_chars: int,
    sort_by_context_length_desc: bool,
) -> List[Dict[str, object]]:
    file_path = DATASET_ROOT / f"{task_name}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"LongBench task file not found: {file_path}")
    records: List[Dict[str, object]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if _context_length_chars(row) < min_context_chars:
                continue
            records.append(row)
    if sort_by_context_length_desc:
        records.sort(key=_context_length_chars, reverse=True)
    if max_samples is not None:
        records = records[:max_samples]
    return records


def _build_pipeline(model, tokenizer, device: torch.device, args, config_name: str) -> OverlappedPipeline:
    cfg = CONFIGS[config_name]
    return OverlappedPipeline(
        model=model,
        tokenizer=tokenizer,
        layer_boundary=args.layer_boundary,
        table_dtype=torch.float16,
        max_table_entries=args.max_table_entries,
        group_size=args.group_size,
        top_k=1,
        int8_group_size=args.group_size,
        int8_outlier_top_k=1,
        decode_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        device=device,
        domain_aware=False,
        table_placement=getattr(args, "table_placement", "cpu"),
        pin_cpu_output_copy=not getattr(args, "disable_pinned_cpu_table_copy", False),
        enable_async_cpu_output_copy=not getattr(args, "disable_async_cpu_table_copy", False),
        gpu_hot_cache_entries=getattr(args, "gpu_hot_cache_entries", 0),
        delta_strategy=cfg["delta_strategy"],
        unigram_strategy=cfg["unigram_strategy"],
        track_transfer_bytes=not args.score_only,
    )


def _reduction_pct(compressed_bytes: int, raw_bytes: int) -> float:
    if raw_bytes <= 0:
        return 0.0
    return (1.0 - (compressed_bytes / raw_bytes)) * 100.0


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def _candidate_seq_lens(max_seq_len: int) -> List[int]:
    budgets = [max_seq_len]
    budgets.extend(length for length in FALLBACK_MAX_SEQ_LENS if length < max_seq_len)
    deduped: List[int] = []
    for length in budgets:
        if length not in deduped:
            deduped.append(length)
    return deduped


def _run_request_with_retry(
    pipeline: OverlappedPipeline,
    prompt: str,
    phase: str,
    config_name: str,
    task_name: str,
    sample_id: object,
) -> Tuple[object, object, object]:
    original_budget = pipeline.max_seq_len
    budgets = _candidate_seq_lens(original_budget)
    last_error: BaseException | None = None

    for budget in budgets:
        pipeline.max_seq_len = budget
        try:
            result = pipeline.process_request(prompt, phase=phase)
            if budget < original_budget:
                logger.warning(
                    "Config=%s task=%s sample=%s lowered max_seq_len from %d to %d after OOM",
                    config_name,
                    task_name,
                    sample_id,
                    original_budget,
                    budget,
                )
            return result
        except BaseException as exc:  # noqa: BLE001
            if not _is_oom_error(exc):
                pipeline.max_seq_len = original_budget
                raise
            last_error = exc
            logger.warning(
                "Config=%s task=%s sample=%s OOM at max_seq_len=%d, retrying with smaller budget",
                config_name,
                task_name,
                sample_id,
                budget,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pipeline.max_seq_len = original_budget
    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry loop exited without result")


def _run_fp16_request_with_retry(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
    task_name: str,
    sample_id: object,
) -> Tuple[str, int, int]:
    budgets = _candidate_seq_lens(max_seq_len)
    last_error: BaseException | None = None

    for budget in budgets:
        try:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=budget).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_ids = generated[0, encoded.input_ids.shape[1]:].tolist()
            if budget < max_seq_len:
                logger.warning(
                    "Config=%s task=%s sample=%s lowered max_seq_len from %d to %d after OOM",
                    FP16_CONFIG_NAME,
                    task_name,
                    sample_id,
                    max_seq_len,
                    budget,
                )
            return _decode_generated(tokenizer, generated_ids), int(encoded.input_ids.shape[1]), len(generated_ids)
        except BaseException as exc:  # noqa: BLE001
            if not _is_oom_error(exc):
                raise
            last_error = exc
            logger.warning(
                "Config=%s task=%s sample=%s OOM at max_seq_len=%d, retrying with smaller budget",
                FP16_CONFIG_NAME,
                task_name,
                sample_id,
                budget,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if last_error is not None:
        raise last_error
    raise RuntimeError("FP16 retry loop exited without result")


def _write_summary(
    output_path: str,
    args,
    results_by_config: Dict[str, Dict[str, object]],
) -> None:
    if _DIST_RANK != 0:
        return
    summary: Dict[str, object] = {
        "benchmark": "LongBench",
        "score_name": "average_task_score",
        "score_only": args.score_only,
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "configs": results_by_config,
    }
    if not args.score_only:
        summary["bandwidth_mbps"] = args.bandwidth_mbps

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def _setup_tp(tp_size: int) -> tuple[int, int, bool]:
    global _DIST_RANK, _DIST_WORLD_SIZE

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = tp_size > 1 or world_size > 1
    if distributed:
        if world_size <= 1:
            raise RuntimeError("Tensor parallel requires torchrun with WORLD_SIZE > 1")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        _DIST_RANK = rank
        _DIST_WORLD_SIZE = world_size
        if rank != 0:
            logger.setLevel(logging.WARNING)
        logger.info("Initialized TP rank=%d local_rank=%d world_size=%d", rank, local_rank, world_size)
    else:
        _DIST_RANK = 0
        _DIST_WORLD_SIZE = 1
    return rank, local_rank, distributed


def _cleanup_tp(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _effective_strategy_payload(config_name: str) -> Dict[str, object]:
    if config_name == FP16_CONFIG_NAME:
        return {
            "mode": "fp16",
            "delta_strategy": None,
            "unigram_strategy": None,
        }
    cfg = CONFIGS[config_name]
    return {
        "mode": "compressed",
        "delta_strategy": cfg["delta_strategy"],
        "unigram_strategy": cfg["unigram_strategy"],
    }


def _build_partial_config_result(
    config_name: str,
    state: Dict[str, object],
    args,
    completed: bool,
) -> Dict[str, object]:
    overall_count = int(state["overall_count"])
    overall_score_sum = float(state["overall_score_sum"])
    result: Dict[str, object] = {
        "score_name": "average_task_score",
        "score": overall_score_sum / max(overall_count, 1),
        "num_samples": overall_count,
        "tasks": dict(state["task_results"]),
        "effective_strategy": _effective_strategy_payload(config_name),
        "completed": completed,
    }
    if not args.score_only:
        compressed_prefill_bytes = int(state["compressed_prefill_bytes"])
        compressed_decode_bytes = int(state["compressed_decode_bytes"])
        compressed_prefill_raw_bytes = int(state["compressed_prefill_raw_bytes"])
        compressed_decode_raw_bytes = int(state["compressed_decode_raw_bytes"])
        compressed_total_bytes = compressed_prefill_bytes + compressed_decode_bytes
        compressed_total_raw_bytes = compressed_prefill_raw_bytes + compressed_decode_raw_bytes
        result.update({
            "prefill_comm_e2e_total_ms": float(state["compressed_prefill_comm_ms"]),
            "decode_comm_e2e_total_ms": float(state["compressed_decode_comm_ms"]),
            "prefill_transfer_bytes": compressed_prefill_bytes,
            "decode_transfer_bytes": compressed_decode_bytes,
            "total_transfer_bytes": compressed_total_bytes,
            "prefill_fp16_raw_bytes": compressed_prefill_raw_bytes,
            "decode_fp16_raw_bytes": compressed_decode_raw_bytes,
            "total_fp16_raw_bytes": compressed_total_raw_bytes,
            "prefill_comm_reduction_pct": _reduction_pct(compressed_prefill_bytes, compressed_prefill_raw_bytes),
            "decode_comm_reduction_pct": _reduction_pct(compressed_decode_bytes, compressed_decode_raw_bytes),
            "total_comm_reduction_pct": _reduction_pct(compressed_total_bytes, compressed_total_raw_bytes),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongBench benchmark for compressed pipeline")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--config", choices=sorted([*CONFIGS.keys(), FP16_CONFIG_NAME]), nargs="+", default=["optimized_default"])
    parser.add_argument("--tasks", choices=sorted(TASKS.keys()), nargs="+", default=list(TASKS.keys()))
    parser.add_argument("--samples-per-task", type=int, default=None)
    parser.add_argument("--min-context-chars", type=int, default=0)
    parser.add_argument("--sort-by-context-length-desc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--bandwidth-mbps", type=int, default=200)
    parser.add_argument("--output", default="results_longbench/summary.json")
    parser.add_argument("--save-per-sample", action="store_true")
    parser.add_argument("--score-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--interleave-configs-per-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()

    rank, local_rank, distributed = _setup_tp(args.tp_size)

    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    device_index = local_rank if distributed else args.gpu
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if distributed:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    hidden_dim = model.config.hidden_size

    task_records = {
        task_name: _load_task_records(
            task_name,
            args.samples_per_task,
            args.min_context_chars,
            args.sort_by_context_length_desc,
        )
        for task_name in args.tasks
    }

    results_by_config: Dict[str, Dict[str, object]] = {}
    total_samples = sum(len(records) for records in task_records.values())
    logger.info("Scoring %d LongBench samples across %d tasks for %d configs...", total_samples, len(args.tasks), len(args.config))

    if args.interleave_configs_per_sample:
        config_states: Dict[str, Dict[str, object]] = {}
        for config_name in args.config:
            config_states[config_name] = {
                "pipeline": None if config_name == FP16_CONFIG_NAME else _build_pipeline(model, tokenizer, device, args, config_name),
                "task_results": {},
                "task_score_sums": defaultdict(float),
                "overall_score_sum": 0.0,
                "overall_count": 0,
                "compressed_prefill_comm_ms": 0.0,
                "compressed_decode_comm_ms": 0.0,
                "compressed_prefill_bytes": 0,
                "compressed_decode_bytes": 0,
                "compressed_prefill_raw_bytes": 0,
                "compressed_decode_raw_bytes": 0,
                "per_sample": defaultdict(list),
            }

        for task_name in args.tasks:
            metric = TASKS[task_name]["metric"]
            records = task_records[task_name]
            warmup_records = records[: min(args.warmup_samples, len(records))]
            eval_records = records[min(args.warmup_samples, len(records)):]

            for config_name in args.config:
                state = config_states[config_name]
                pipeline = state["pipeline"]
                for row in warmup_records:
                    prompt = _build_prompt(task_name, row)
                    if pipeline is None:
                        _run_fp16_request_with_retry(
                            model,
                            tokenizer,
                            device,
                            prompt,
                            args.max_seq_len,
                            args.max_new_tokens,
                            task_name,
                            row.get("_id", "warmup"),
                        )
                    else:
                        pipeline.process_request(prompt, phase="warmup")

            for idx, row in enumerate(eval_records, start=1):
                prompt = _build_prompt(task_name, row)
                sample_id = row.get("_id", idx)
                answers = [str(answer) for answer in row["answers"]]

                for config_name in args.config:
                    state = config_states[config_name]
                    pipeline = state["pipeline"]

                    if pipeline is None:
                        prediction, prompt_len, decode_len = _run_fp16_request_with_retry(
                            model,
                            tokenizer,
                            device,
                            prompt,
                            args.max_seq_len,
                            args.max_new_tokens,
                            task_name,
                            sample_id,
                        )
                    else:
                        prefill_res, decode_res, _ = _run_request_with_retry(
                            pipeline,
                            prompt,
                            phase="test",
                            config_name=config_name,
                            task_name=task_name,
                            sample_id=sample_id,
                        )
                        prediction = _decode_generated(tokenizer, decode_res.generated_token_ids)

                    score = _score_prediction(metric, prediction, answers)
                    state["task_score_sums"][task_name] += score
                    state["overall_score_sum"] += score
                    state["overall_count"] += 1

                    if not args.score_only:
                        if pipeline is None:
                            fp16_prefill_bytes = int(prompt_len * hidden_dim * 2)
                            fp16_decode_bytes = int(decode_len * hidden_dim * 2)
                            state["compressed_prefill_comm_ms"] += _network_ms(fp16_prefill_bytes, args.bandwidth_mbps)
                            state["compressed_decode_comm_ms"] += _network_ms(fp16_decode_bytes, args.bandwidth_mbps)
                            state["compressed_prefill_bytes"] += fp16_prefill_bytes
                            state["compressed_decode_bytes"] += fp16_decode_bytes
                            state["compressed_prefill_raw_bytes"] += fp16_prefill_bytes
                            state["compressed_decode_raw_bytes"] += fp16_decode_bytes
                        else:
                            comp_prefill_local = prefill_res.classify_ms + prefill_res.encode_delta_ms + prefill_res.encode_self_ref_ms + prefill_res.encode_unigram_ms
                            comp_decode_local = decode_res.total_classify_ms + decode_res.total_encode_ms
                            state["compressed_prefill_comm_ms"] += comp_prefill_local + _network_ms(prefill_res.total_transfer_bytes, args.bandwidth_mbps)
                            state["compressed_decode_comm_ms"] += comp_decode_local + _network_ms(decode_res.total_transfer_bytes, args.bandwidth_mbps)
                            state["compressed_prefill_bytes"] += int(prefill_res.total_transfer_bytes)
                            state["compressed_decode_bytes"] += int(decode_res.total_transfer_bytes)
                            state["compressed_prefill_raw_bytes"] += int(prefill_res.raw_fp16_bytes)
                            state["compressed_decode_raw_bytes"] += int(decode_res.raw_fp16_bytes)

                    if args.save_per_sample:
                        state["per_sample"][task_name].append({
                            "task_id": sample_id,
                            "prediction": prediction,
                            "answers": answers,
                            "score": score,
                        })

                    task_result: Dict[str, object] = {
                        "metric": metric,
                        "score": float(state["task_score_sums"][task_name]) / max(idx, 1),
                        "num_samples": idx,
                        "completed": idx == len(eval_records),
                    }
                    if args.save_per_sample:
                        task_result["per_sample"] = list(state["per_sample"][task_name])
                    state["task_results"][task_name] = task_result
                    results_by_config[config_name] = _build_partial_config_result(
                        config_name,
                        state,
                        args,
                        completed=False,
                    )
                    logger.info(
                        "Task=%s sample=%d/%d config=%s avg_score=%.4f",
                        task_name,
                        idx,
                        len(eval_records),
                        config_name,
                        float(state["task_score_sums"][task_name]) / max(idx, 1),
                    )

                _write_summary(args.output, args, results_by_config)

        for config_name in args.config:
            state = config_states[config_name]
            results_by_config[config_name] = _build_partial_config_result(
                config_name,
                state,
                args,
                completed=True,
            )
            pipeline = state["pipeline"]
            if pipeline is not None:
                pipeline.shutdown()

        _write_summary(args.output, args, results_by_config)
        if rank == 0:
            logger.info("Saved summary to %s", Path(args.output))
        _cleanup_tp(distributed)
        return

    for config_name in args.config:
        pipeline = None if config_name == FP16_CONFIG_NAME else _build_pipeline(model, tokenizer, device, args, config_name)
        config_task_results: Dict[str, Dict[str, object]] = {}
        overall_score_sum = 0.0
        overall_count = 0
        compressed_prefill_comm_ms = 0.0
        compressed_decode_comm_ms = 0.0
        compressed_prefill_bytes = 0
        compressed_decode_bytes = 0
        compressed_prefill_raw_bytes = 0
        compressed_decode_raw_bytes = 0

        for task_name in args.tasks:
            metric = TASKS[task_name]["metric"]
            records = task_records[task_name]
            warmup_records = records[: min(args.warmup_samples, len(records))]
            eval_records = records[min(args.warmup_samples, len(records)):]

            for row in warmup_records:
                prompt = _build_prompt(task_name, row)
                if pipeline is None:
                    _run_fp16_request_with_retry(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        args.max_seq_len,
                        args.max_new_tokens,
                        task_name,
                        row.get("_id", "warmup"),
                    )
                else:
                    pipeline.process_request(prompt, phase="warmup")

            task_score_sum = 0.0
            per_sample: List[Dict[str, object]] = []

            for idx, row in enumerate(eval_records, start=1):
                prompt = _build_prompt(task_name, row)
                sample_id = row.get("_id", idx)
                if pipeline is None:
                    prediction, prompt_len, decode_len = _run_fp16_request_with_retry(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        args.max_seq_len,
                        args.max_new_tokens,
                        task_name,
                        sample_id,
                    )
                else:
                    prefill_res, decode_res, _ = _run_request_with_retry(
                        pipeline,
                        prompt,
                        phase="test",
                        config_name=config_name,
                        task_name=task_name,
                        sample_id=sample_id,
                    )
                    prediction = _decode_generated(tokenizer, decode_res.generated_token_ids)
                answers = [str(answer) for answer in row["answers"]]
                score = _score_prediction(metric, prediction, answers)
                task_score_sum += score
                overall_score_sum += score
                overall_count += 1

                if not args.score_only:
                    if pipeline is None:
                        fp16_prefill_bytes = int(prompt_len * hidden_dim * 2)
                        fp16_decode_bytes = int(decode_len * hidden_dim * 2)
                        compressed_prefill_comm_ms += _network_ms(fp16_prefill_bytes, args.bandwidth_mbps)
                        compressed_decode_comm_ms += _network_ms(fp16_decode_bytes, args.bandwidth_mbps)
                        compressed_prefill_bytes += fp16_prefill_bytes
                        compressed_decode_bytes += fp16_decode_bytes
                        compressed_prefill_raw_bytes += fp16_prefill_bytes
                        compressed_decode_raw_bytes += fp16_decode_bytes
                    else:
                        comp_prefill_local = prefill_res.classify_ms + prefill_res.encode_delta_ms + prefill_res.encode_self_ref_ms + prefill_res.encode_unigram_ms
                        comp_decode_local = decode_res.total_classify_ms + decode_res.total_encode_ms
                        compressed_prefill_comm_ms += comp_prefill_local + _network_ms(prefill_res.total_transfer_bytes, args.bandwidth_mbps)
                        compressed_decode_comm_ms += comp_decode_local + _network_ms(decode_res.total_transfer_bytes, args.bandwidth_mbps)
                        compressed_prefill_bytes += int(prefill_res.total_transfer_bytes)
                        compressed_decode_bytes += int(decode_res.total_transfer_bytes)
                        compressed_prefill_raw_bytes += int(prefill_res.raw_fp16_bytes)
                        compressed_decode_raw_bytes += int(decode_res.raw_fp16_bytes)

                if args.save_per_sample:
                    per_sample.append({
                        "task_id": sample_id,
                        "prediction": prediction,
                        "answers": answers,
                        "score": score,
                    })

                if idx % 10 == 0 or idx == len(eval_records):
                    config_task_results[task_name] = {
                        "metric": metric,
                        "score": task_score_sum / max(idx, 1),
                        "num_samples": idx,
                        "completed": idx == len(eval_records),
                    }
                    results_by_config[config_name] = {
                        "score_name": "average_task_score",
                        "score": overall_score_sum / max(overall_count, 1),
                        "num_samples": overall_count,
                        "tasks": config_task_results,
                        "completed": False,
                    }
                    _write_summary(args.output, args, results_by_config)
                    logger.info(
                        "Config=%s task=%s processed %d/%d samples avg_score=%.4f",
                        config_name,
                        task_name,
                        idx,
                        len(eval_records),
                        task_score_sum / max(idx, 1),
                    )

            task_result: Dict[str, object] = {
                "metric": metric,
                "score": task_score_sum / max(len(eval_records), 1),
                "num_samples": len(eval_records),
            }
            if args.save_per_sample:
                task_result["per_sample"] = per_sample
            config_task_results[task_name] = task_result

        config_result: Dict[str, object] = {
            "score_name": "average_task_score",
            "score": overall_score_sum / max(overall_count, 1),
            "num_samples": overall_count,
            "tasks": config_task_results,
            "effective_strategy": _effective_strategy_payload(config_name),
            "completed": True,
        }
        if not args.score_only:
            compressed_total_bytes = compressed_prefill_bytes + compressed_decode_bytes
            compressed_total_raw_bytes = compressed_prefill_raw_bytes + compressed_decode_raw_bytes
            config_result.update({
                "prefill_comm_e2e_total_ms": compressed_prefill_comm_ms,
                "decode_comm_e2e_total_ms": compressed_decode_comm_ms,
                "prefill_transfer_bytes": compressed_prefill_bytes,
                "decode_transfer_bytes": compressed_decode_bytes,
                "total_transfer_bytes": compressed_total_bytes,
                "prefill_fp16_raw_bytes": compressed_prefill_raw_bytes,
                "decode_fp16_raw_bytes": compressed_decode_raw_bytes,
                "total_fp16_raw_bytes": compressed_total_raw_bytes,
                "prefill_comm_reduction_pct": _reduction_pct(compressed_prefill_bytes, compressed_prefill_raw_bytes),
                "decode_comm_reduction_pct": _reduction_pct(compressed_decode_bytes, compressed_decode_raw_bytes),
                "total_comm_reduction_pct": _reduction_pct(compressed_total_bytes, compressed_total_raw_bytes),
            })
        results_by_config[config_name] = config_result
        _write_summary(args.output, args, results_by_config)
        if pipeline is not None:
            pipeline.shutdown()

    _write_summary(args.output, args, results_by_config)
    if rank == 0:
        logger.info("Saved summary to %s", Path(args.output))
    _cleanup_tp(distributed)


if __name__ == "__main__":
    main()