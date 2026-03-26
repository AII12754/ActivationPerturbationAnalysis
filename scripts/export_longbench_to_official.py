#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TASK_NAME_MAP = {
    "hotpotqa_e": "hotpotqa",
    "2wikimqa_e": "2wikimqa",
    "triviaqa_e": "triviaqa",
    "passage_retrieval_en_e": "passage_retrieval_en",
    "passage_count_e": "passage_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local LongBench per-sample results to official LongBench pred/pred_e layout",
    )
    parser.add_argument("--input", required=True, help="Path to local LongBench summary JSON")
    parser.add_argument("--config", required=True, help="Config name inside summary JSON")
    parser.add_argument("--official-root", default="official_longbench/LongBench", help="Official LongBench root directory")
    parser.add_argument("--dataset-root", default="/root/share/dataset/Longbench/data", help="Directory containing LongBench jsonl datasets")
    parser.add_argument("--model-label", required=True, help="Directory name to create under pred/ or pred_e/")
    parser.add_argument(
        "--mode",
        choices=["pred", "pred_e"],
        default="pred",
        help="Export to official pred or pred_e layout",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported jsonl files")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _official_task_name(task_name: str) -> str:
    return TASK_NAME_MAP.get(task_name, task_name)


def _source_dataset_name(task_name: str, mode: str) -> str:
    if mode == "pred_e":
        if task_name.endswith("_e"):
            return task_name
        candidate = f"{task_name}_e"
        return candidate
    return task_name


def _load_dataset_index(dataset_root: Path, dataset_name: str) -> Dict[str, Dict[str, object]]:
    dataset_path = dataset_root / f"{dataset_name}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    index: Dict[str, Dict[str, object]] = {}
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_id = str(row.get("_id"))
            index[row_id] = row
    return index


def _iter_task_exports(
    task_name: str,
    task_payload: Dict[str, object],
    dataset_index: Dict[str, Dict[str, object]],
    mode: str,
) -> Iterable[Dict[str, object]]:
    per_sample = task_payload.get("per_sample")
    if not isinstance(per_sample, list):
        raise ValueError(f"Task {task_name} does not contain per_sample results")

    for item in per_sample:
        task_id = str(item["task_id"])
        dataset_row = dataset_index.get(task_id)
        if dataset_row is None:
            raise KeyError(f"Task id {task_id} not found in source dataset for {task_name}")

        exported = {
            "pred": item["prediction"],
            "answers": item["answers"],
            "all_classes": dataset_row.get("all_classes"),
        }
        if mode == "pred_e":
            exported["length"] = dataset_row.get("length")
        yield exported


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    official_root = Path(args.official_root)
    dataset_root = Path(args.dataset_root)

    payload = _load_json(input_path)
    configs = payload.get("configs")
    if not isinstance(configs, dict) or args.config not in configs:
        raise KeyError(f"Config {args.config} not found in {input_path}")

    config_payload = configs[args.config]
    if not isinstance(config_payload, dict):
        raise ValueError(f"Malformed config payload for {args.config}")

    tasks = config_payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"Config {args.config} has no task results")

    output_root = official_root / args.mode / args.model_label
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_cache: Dict[str, Dict[str, Dict[str, object]]] = {}
    exported_files: List[Tuple[str, int, Path]] = []

    for task_name, task_payload in tasks.items():
        if not isinstance(task_payload, dict) or "per_sample" not in task_payload:
            continue

        source_dataset_name = _source_dataset_name(task_name, args.mode)
        if source_dataset_name not in dataset_cache:
            dataset_cache[source_dataset_name] = _load_dataset_index(dataset_root, source_dataset_name)
        dataset_index = dataset_cache[source_dataset_name]

        official_task_name = _official_task_name(task_name)
        output_path = output_root / f"{official_task_name}.jsonl"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

        rows = list(_iter_task_exports(task_name, task_payload, dataset_index, args.mode))
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        exported_files.append((official_task_name, len(rows), output_path))

    if not exported_files:
        raise ValueError("No tasks with per_sample data were exported")

    print(json.dumps(
        {
            "input": str(input_path),
            "config": args.config,
            "mode": args.mode,
            "model_label": args.model_label,
            "exported": [
                {"task": task, "rows": rows, "path": str(path)}
                for task, rows, path in exported_files
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()