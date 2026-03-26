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
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.benchmarks.run_longbench_benchmark import CONFIGS, FP16_CONFIG_NAME, LEGACY_CONFIG_NAMES

PYTHON_BIN = "/usr/bin/python"
LOGGER = logging.getLogger("full_benchmark_suite")


@dataclass
class Job:
    benchmark: str
    config_name: str
    output_path: Path
    command: List[str]
    visible_gpus: List[int] | None = None


def _strategy_names(include_legacy_configs: bool) -> List[str]:
    strategies = [FP16_CONFIG_NAME]
    for name in CONFIGS.keys():
        if not include_legacy_configs and name in LEGACY_CONFIG_NAMES:
            continue
        strategies.append(name)
    return strategies


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


def _build_jobs(
    results_root: Path,
    model: str,
    include_legacy_configs: bool,
    benchmarks: Sequence[str],
    longbench_max_seq_len: int,
    longbench_max_new_tokens: int,
    longbench_v2_max_seq_len: int,
    tp_size: int,
    visible_gpus: Sequence[int] | None,
) -> List[Job]:
    jobs: List[Job] = []
    strategies = _strategy_names(include_legacy_configs)

    if "humaneval" in benchmarks:
        for config_name in strategies:
            output = results_root / "humaneval" / f"{config_name}.json"
            command = [
                *( ["torchrun", "--standalone", f"--nproc_per_node={tp_size}"] if tp_size > 1 else [PYTHON_BIN] ),
                "delta_coding_system/benchmarks/run_humaneval_benchmark.py",
                "--model", model,
                "--gpu", "0",
                "--score-only",
                "--warmup-samples", "0",
                "--output", str(output),
                "--config", config_name,
            ]
            jobs.append(Job("HumanEval", config_name, output, command, list(visible_gpus) if visible_gpus is not None else None))

    if "longbench" in benchmarks:
        for config_name in strategies:
            output = results_root / "longbench" / f"{config_name}.json"
            command = [
                *( ["torchrun", "--standalone", f"--nproc_per_node={tp_size}"] if tp_size > 1 else [PYTHON_BIN] ),
                "delta_coding_system/benchmarks/run_longbench_benchmark.py",
                "--model", model,
                "--gpu", "0",
                "--score-only",
                "--warmup-samples", "0",
                "--max-seq-len", str(longbench_max_seq_len),
                "--max-new-tokens", str(longbench_max_new_tokens),
                "--output", str(output),
                "--config", config_name,
                *( ["--tp-size", str(tp_size)] if tp_size > 1 else [] ),
            ]
            jobs.append(Job("LongBench", config_name, output, command, list(visible_gpus) if visible_gpus is not None else None))

    if "longbench_v2" in benchmarks:
        for config_name in strategies:
            output = results_root / "longbench_v2" / f"{config_name}.json"
            command = [
                *( ["torchrun", "--standalone", f"--nproc_per_node={tp_size}"] if tp_size > 1 else [PYTHON_BIN] ),
                "delta_coding_system/benchmarks/run_longbench_v2_benchmark.py",
                "--model", model,
                "--gpu", "0",
                "--score-only",
                "--max-seq-len", str(longbench_v2_max_seq_len),
                "--output", str(output),
                "--config", config_name,
                *( ["--tp-size", str(tp_size)] if tp_size > 1 else [] ),
            ]
            jobs.append(Job("LongBenchV2", config_name, output, command, list(visible_gpus) if visible_gpus is not None else None))

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
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in job.visible_gpus) if job.visible_gpus is not None else str(gpu_id)
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
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--include-legacy-configs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--benchmarks", nargs="+", choices=["humaneval", "longbench", "longbench_v2"], default=["humaneval", "longbench", "longbench_v2"])
    parser.add_argument("--longbench-max-seq-len", type=int, default=8192)
    parser.add_argument("--longbench-max-new-tokens", type=int, default=256)
    parser.add_argument("--longbench-v2-max-seq-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=1)
    args = parser.parse_args()

    if args.tp_size > 1 and len(args.gpus) != args.tp_size:
        raise ValueError("When tp-size > 1, --gpus must list exactly tp-size physical GPUs")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results_root = PROJECT_ROOT / args.results_root
    for name in ["humaneval", "longbench", "longbench_v2"]:
        (results_root / name).mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(
        results_root,
        args.model,
        args.include_legacy_configs,
        args.benchmarks,
        args.longbench_max_seq_len,
        args.longbench_max_new_tokens,
        args.longbench_v2_max_seq_len,
        args.tp_size,
        args.gpus if args.tp_size > 1 else None,
    )
    pending_jobs = [job for job in jobs if not _is_completed_output(job.output_path, job.config_name)]
    LOGGER.info("Prepared %d jobs, %d pending after resume check", len(jobs), len(pending_jobs))

    job_queue: "queue.Queue[Job]" = queue.Queue()
    for job in pending_jobs:
        job_queue.put(job)

    threads: List[threading.Thread] = []
    worker_gpu_ids = [args.gpus[0]] if args.tp_size > 1 else args.gpus
    for gpu_id in worker_gpu_ids:
        thread = threading.Thread(target=_worker, args=(gpu_id, job_queue), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    incomplete_jobs = [job for job in jobs if not _is_completed_output(job.output_path, job.config_name)]
    if incomplete_jobs:
        for job in incomplete_jobs:
            LOGGER.error("Incomplete output remains for %s %s at %s", job.benchmark, job.config_name, job.output_path)
        raise SystemExit(1)

    if "humaneval" in args.benchmarks:
        _merge_humaneval(results_root)
    if "longbench" in args.benchmarks:
        _merge_longbench_like(results_root / "longbench", results_root / "longbench_merged.json")
    if "longbench_v2" in args.benchmarks:
        _merge_longbench_like(results_root / "longbench_v2", results_root / "longbench_v2_merged.json")
    LOGGER.info("Benchmark suite finished. Merged results written under %s", results_root)


if __name__ == "__main__":
    main()