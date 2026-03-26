from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.benchmarks.run_longbench_benchmark import (  # noqa: E402
    CONFIGS,
    FP16_CONFIG_NAME,
    _build_pipeline,
    _cleanup_tp,
    _is_oom_error,
    _run_request_with_retry,
    _setup_tp,
)

LOGGER = logging.getLogger("longbench_official_benchmark")
OFFICIAL_ROOT = PROJECT_ROOT / "official_longbench" / "LongBench"
OFFICIAL_CONFIG_DIR = OFFICIAL_ROOT / "config"
LOCAL_DATASET_ROOT = Path("/root/share/dataset/Longbench/data")
NO_CHAT_WRAP_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
DEFAULT_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
]
DEFAULT_E_TASKS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official-style LongBench generation for local compression strategies")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", nargs="+", choices=sorted([FP16_CONFIG_NAME, *CONFIGS.keys()]), required=True)
    parser.add_argument("--task", nargs="+", default=None)
    parser.add_argument("--e", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--samples-per-task", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-table-entries", type=int, default=100000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--score-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-official-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label-prefix", default="")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_list(evaluate_e: bool, selected: Sequence[str] | None) -> List[str]:
    tasks = list(DEFAULT_E_TASKS if evaluate_e else DEFAULT_TASKS)
    return list(selected) if selected else tasks


def _load_records(dataset_name: str, limit: int | None) -> List[Dict[str, object]]:
    path = LOCAL_DATASET_ROOT / f"{dataset_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _truncate_in_middle(tokenizer, prompt: str, max_seq_len: int) -> str:
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) <= max_seq_len:
        return prompt
    half = max_seq_len // 2
    left = tokenizer.decode(tokenized[:half], skip_special_tokens=True)
    right = tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
    return left + right


def _build_chat_prompt(tokenizer, prompt: str, model_path: str, task_name: str) -> str:
    if task_name in NO_CHAT_WRAP_TASKS:
        return prompt
    if "qwen" in model_path.lower() and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _newline_stop_ids(tokenizer, task_name: str) -> List[int]:
    if task_name != "samsum":
        return []
    token_ids = tokenizer.encode("\n", add_special_tokens=False)
    return [token_ids[-1]] if token_ids else []


def _post_process(response: str, model_path: str) -> str:
    model_name = model_path.lower()
    if "xgen" in model_name:
        return response.strip().replace("Assistant:", "")
    if "internlm" in model_name:
        return response.split("<eoa>")[0]
    return response


def _load_model_and_tokenizer(model_path: str, device: torch.device, tp_size: int):
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if tp_size > 1:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def _build_official_pipeline(model, tokenizer, device: torch.device, args: argparse.Namespace, config_name: str, max_decode_tokens: int):
    pipeline_args = SimpleNamespace(
        layer_boundary=args.layer_boundary,
        max_table_entries=args.max_table_entries,
        group_size=args.group_size,
        max_new_tokens=max_decode_tokens,
        max_seq_len=args.max_seq_len,
        score_only=args.score_only,
    )
    return _build_pipeline(model, tokenizer, device, pipeline_args, config_name)


def _fp16_generate(model, tokenizer, device: torch.device, prompt: str, task_name: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = int(encoded.input_ids.shape[-1])
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "do_sample": False,
        "temperature": 1.0,
    }
    newline_stop_ids = _newline_stop_ids(tokenizer, task_name)
    if task_name == "samsum":
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        generate_kwargs["min_length"] = context_length + 1
        generate_kwargs["eos_token_id"] = eos_ids + newline_stop_ids
    with torch.no_grad():
        output = model.generate(**encoded, **generate_kwargs)[0]
    prediction = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return _post_process(prediction, str(model.name_or_path))


def _compressed_generate(pipeline, tokenizer, prompt: str, task_name: str, config_name: str, sample_id: object) -> str:
    original_stop_ids = set(pipeline.extra_stop_token_ids)
    try:
        pipeline.extra_stop_token_ids = set(_newline_stop_ids(tokenizer, task_name))
        _prefill, decode_res, _stats = _run_request_with_retry(
            pipeline,
            prompt,
            phase="test",
            config_name=config_name,
            task_name=task_name,
            sample_id=sample_id,
        )
    finally:
        pipeline.extra_stop_token_ids = original_stop_ids
    prediction = tokenizer.decode(decode_res.generated_token_ids, skip_special_tokens=True)
    return _post_process(prediction, str(pipeline.model.name_or_path))


def _existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(str(json.loads(line)["_id"]))
            except Exception:
                continue
    return seen


def _write_row(output_path: Path, row: Dict[str, object], prediction: str) -> None:
    payload = {
        "_id": row.get("_id"),
        "pred": prediction,
        "answers": row.get("answers"),
        "all_classes": row.get("all_classes"),
        "length": row.get("length"),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def _run_eval(model_label: str, evaluate_e: bool) -> None:
    cmd = [sys.executable, "eval.py", "--model", model_label]
    if evaluate_e:
        cmd.append("--e")
    subprocess.run(cmd, cwd=OFFICIAL_ROOT, check=True)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not OFFICIAL_ROOT.exists():
        raise FileNotFoundError(f"Official LongBench repo not found at {OFFICIAL_ROOT}")

    prompt_map = _load_json(OFFICIAL_CONFIG_DIR / "dataset2prompt.json")
    max_gen_map = _load_json(OFFICIAL_CONFIG_DIR / "dataset2maxlen.json")
    tasks = _task_list(args.e, args.task)
    max_decode_tokens = max(int(max_gen_map[task_name]) for task_name in tasks)

    rank, local_rank, distributed = _setup_tp(args.tp_size)
    device_index = local_rank if distributed else args.gpu
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    model, tokenizer = _load_model_and_tokenizer(args.model, device, args.tp_size)

    try:
        for config_name in args.config:
            model_label = f"{args.label_prefix}{config_name}"
            output_dir = OFFICIAL_ROOT / ("pred_e" if args.e else "pred") / model_label
            output_dir.mkdir(parents=True, exist_ok=True)

            pipeline = None if config_name == FP16_CONFIG_NAME else _build_official_pipeline(
                model,
                tokenizer,
                device,
                args,
                config_name,
                max_decode_tokens,
            )
            try:
                for task_name in tasks:
                    dataset_name = f"{task_name}_e" if args.e else task_name
                    records = _load_records(dataset_name, args.samples_per_task)
                    output_path = output_dir / f"{task_name}.jsonl"
                    if output_path.exists() and not args.resume:
                        output_path.unlink()
                    done_ids = _existing_ids(output_path) if args.resume else set()

                    for idx, row in enumerate(records, start=1):
                        row_id = str(row.get("_id", idx))
                        if row_id in done_ids:
                            continue

                        raw_prompt = str(prompt_map[task_name]).format(**row)
                        truncated_prompt = _truncate_in_middle(tokenizer, raw_prompt, args.max_seq_len)
                        final_prompt = _build_chat_prompt(tokenizer, truncated_prompt, args.model, task_name)
                        max_new_tokens = int(max_gen_map[task_name])

                        try:
                            if pipeline is None:
                                prediction = _fp16_generate(model, tokenizer, device, final_prompt, task_name, max_new_tokens)
                            else:
                                pipeline.decode_tokens = max_new_tokens
                                prediction = _compressed_generate(pipeline, tokenizer, final_prompt, task_name, config_name, row_id)
                        except BaseException as exc:  # noqa: BLE001
                            if _is_oom_error(exc):
                                LOGGER.error("OOM config=%s task=%s sample=%s max_seq_len=%d", config_name, task_name, row_id, args.max_seq_len)
                            raise

                        _write_row(output_path, row, prediction)
                        if idx % 10 == 0 or idx == len(records):
                            LOGGER.info("config=%s task=%s progress=%d/%d", config_name, task_name, idx, len(records))

                if args.run_official_eval and rank == 0:
                    _run_eval(model_label, args.e)
                    LOGGER.info("Official eval completed for %s", model_label)
            finally:
                if pipeline is not None:
                    pipeline.shutdown()
    finally:
        _cleanup_tp(distributed)


if __name__ == "__main__":
    main()