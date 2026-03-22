#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded LongBench result files")
    parser.add_argument("inputs", nargs="+", help="Shard JSON files to merge")
    parser.add_argument("--output", required=True, help="Merged summary JSON path")
    args = parser.parse_args()

    merged_configs: Dict[str, object] = {}
    merged_tasks: List[str] = []
    benchmark = None
    score_name = None
    score_only = None
    samples_per_task = None
    sources: List[str] = []

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        payload = json.loads(input_path.read_text())
        sources.append(str(input_path))

        if benchmark is None:
            benchmark = payload.get("benchmark")
            score_name = payload.get("score_name")
            score_only = payload.get("score_only")
            samples_per_task = payload.get("samples_per_task")
        merged_tasks.extend(task for task in payload.get("tasks", []) if task not in merged_tasks)
        for config_name, config_result in payload.get("configs", {}).items():
            merged_configs[config_name] = config_result

    merged_summary = {
        "benchmark": benchmark or "LongBench",
        "score_name": score_name or "average_task_score",
        "score_only": True if score_only is None else score_only,
        "tasks": merged_tasks,
        "samples_per_task": samples_per_task,
        "num_configs": len(merged_configs),
        "sources": sources,
        "configs": merged_configs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()