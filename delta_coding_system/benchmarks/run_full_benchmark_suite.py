#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.benchmarks.run_longbench_benchmark import CONFIGS, FP16_CONFIG_NAME

PYTHON_BIN = "/usr/bin/python"
LOGGER = logging.getLogger("full_benchmark_suite")


@dataclass
class Job:
    benchmark: str
    config_name: str
    output_path: Path
    command: List[str]


def _strategy_names() -> List[str]:
    return [FP16_CONFIG_NAME, *CONFIGS.keys()]


def _is_completed_output(output_path: Path, config_name: str) -> bool:
    if not output_path.exists():
        return False
    payload = json.loads(output_path.read_text())
    cfg = payload.get("configs", {}).get(config_name)
    if not isinstance(cfg, dict):
        return False
    if not cfg.get("completed"):
        return False
    effective = cfg.get("effective_strategy", {})
    if config_name == FP16_CONFIG_NAME:
        return effective.get("mode") == "fp16"
    expected = CONFIGS[config_name]
    return (
        effective.get("mode") == "compressed"
        and effective.get("delta_strategy") == expected["delta_strategy"]
        and effective.get("unigram_strategy") == expected["unigram_strategy"]
    )


def _build_jobs(results_root: Path) -> List[Job]:
    jobs: List[Job] = []
    strategies = _strategy_names()

    for config_name in strategies:
        output = results_root / "humaneval" / f"{config_name}.json"
        command = [
            PYTHON_BIN,
            "delta_coding_system/benchmarks/run_humaneval_benchmark.py",
            "--gpu", "0",
            "--score-only",
            "--warmup-samples", "0",
            "--output", str(output),
            "--config", config_name,
        ]
        jobs.append(Job("HumanEval", config_name, output, command))

    for config_name in strategies:
        output = results_root / "longbench" / f"{config_name}.json"
        command = [
            PYTHON_BIN,
            "delta_coding_system/benchmarks/run_longbench_benchmark.py",
            "--gpu", "0",
            "--score-only",
            "--warmup-samples", "0",
            "--max-seq-len", "8192",
            "--output", str(output),
            "--config", config_name,
        ]
        jobs.append(Job("LongBench", config_name, output, command))

    for config_name in strategies:
        output = results_root / "longbench_v2" / f"{config_name}.json"
        command = [
            PYTHON_BIN,
            "delta_coding_system/benchmarks/run_longbench_v2_benchmark.py",
            "--gpu", "0",
            "--score-only",
            "--max-seq-len", "8192",
            "--output", str(output),
            "--config", config_name,
        ]
        jobs.append(Job("LongBenchV2", config_name, output, command))

    return jobs


def _worker(gpu_id: int, job_queue: "queue.Queue[Job]") -> None:
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return

        if _is_completed_output(job.output_path, job.config_name):
            LOGGER.info("GPU %d skipping completed %s %s", gpu_id, job.benchmark, job.config_name)
            job_queue.task_done()
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        LOGGER.info("GPU %d running %s %s", gpu_id, job.benchmark, job.config_name)
        completed = subprocess.run(job.command, cwd=PROJECT_ROOT, env=env, check=False)
        if completed.returncode != 0:
            LOGGER.error("GPU %d failed %s %s with exit code %d", gpu_id, job.benchmark, job.config_name, completed.returncode)
        elif not _is_completed_output(job.output_path, job.config_name):
            LOGGER.error("GPU %d finished %s %s but output is not completed/validated", gpu_id, job.benchmark, job.config_name)
        else:
            LOGGER.info("GPU %d completed %s %s", gpu_id, job.benchmark, job.config_name)
        job_queue.task_done()


def _merge_humaneval(results_root: Path) -> None:
    configs: Dict[str, object] = {}
    for file_path in sorted((results_root / "humaneval").glob("*.json")):
        payload = json.loads(file_path.read_text())
        for config_name, result in payload.get("configs", {}).items():
            configs[config_name] = result
    merged = {
        "benchmark": "HumanEval",
        "num_configs": len(configs),
        "configs": configs,
    }
    (results_root / "humaneval_merged.json").write_text(json.dumps(merged, indent=2))


def _merge_longbench_like(input_dir: Path, output_path: Path) -> None:
    files = sorted(str(path) for path in input_dir.glob("*.json"))
    if not files:
        return
    subprocess.run([
        PYTHON_BIN,
        "delta_coding_system/benchmarks/merge_longbench_results.py",
        *files,
        "--output", str(output_path),
    ], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HumanEval, LongBench, and LongBench v2 across all strategies on all available GPUs")
    parser.add_argument("--results-root", default="results_benchmark_suite")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results_root = PROJECT_ROOT / args.results_root
    for name in ["humaneval", "longbench", "longbench_v2"]:
        (results_root / name).mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(results_root)
    pending_jobs = [job for job in jobs if not _is_completed_output(job.output_path, job.config_name)]
    LOGGER.info("Prepared %d jobs, %d pending after resume check", len(jobs), len(pending_jobs))

    job_queue: "queue.Queue[Job]" = queue.Queue()
    for job in pending_jobs:
        job_queue.put(job)

    threads: List[threading.Thread] = []
    for gpu_id in args.gpus:
        thread = threading.Thread(target=_worker, args=(gpu_id, job_queue), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    _merge_humaneval(results_root)
    _merge_longbench_like(results_root / "longbench", results_root / "longbench_merged.json")
    _merge_longbench_like(results_root / "longbench_v2", results_root / "longbench_v2_merged.json")
    LOGGER.info("Benchmark suite finished. Merged results written under %s", results_root)


if __name__ == "__main__":
    main()