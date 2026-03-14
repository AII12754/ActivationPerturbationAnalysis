"""
Sweep orchestration for activation state detection experiments.

Generates the Cartesian product of (dataset x context_length x prompt_index),
then dispatches experiments sequentially or in parallel.
7 datasets x 3 context_lengths x 3 prompts = 63 jobs (256 decode steps each).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import torch

from .state_experiment import run_state_experiment
from .state_storage import StateResultStore
from .prompts import PromptGenerator
from .scheduler import (
    ExperimentJob, ExperimentScheduler, detect_gpus,
    estimate_gpus_per_experiment,
)

logger = logging.getLogger(__name__)


def _make_experiment_id(params: Dict[str, Any]) -> str:
    """Deterministic experiment id from the parameter dict."""
    key = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _resolve_dataset_list(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return the list of dataset specs to sweep over.

    Each entry is a dict with keys: name, config, split.
    If prompts.datasets is not set, falls back to the single dataset_name.
    """
    prompt_cfg = config.get("prompts", {})
    raw_list = prompt_cfg.get("datasets", [])

    if not raw_list:
        # Fallback: single dataset from the old-style config.
        return [{
            "name": prompt_cfg.get("dataset_name", "wikitext"),
            "config": prompt_cfg.get("dataset_config", "default"),
            "split": prompt_cfg.get("dataset_split", "train"),
        }]

    resolved: List[Dict[str, str]] = []
    for entry in raw_list:
        if isinstance(entry, str):
            resolved.append({
                "name": entry,
                "config": prompt_cfg.get("dataset_config", "default"),
                "split": prompt_cfg.get("dataset_split", "train"),
            })
        elif isinstance(entry, dict):
            resolved.append({
                "name": entry["name"],
                "config": entry.get("config", "default"),
                "split": entry.get("split", "train"),
            })
    return resolved


def build_state_sweep_jobs(config: Dict[str, Any]) -> List[ExperimentJob]:
    """Build the full list of state detection experiment jobs from config."""
    sweep = config["sweep"]
    context_lengths = sweep["context_lengths"]
    num_prompts = sweep.get("num_prompts_per_length", 3)

    dataset_specs = _resolve_dataset_list(config)

    jobs: List[ExperimentJob] = []
    for ds_spec, ctx_len, prompt_idx in product(
        dataset_specs,
        context_lengths,
        range(num_prompts),
    ):
        params = {
            "dataset_name": ds_spec["name"],
            "dataset_config": ds_spec["config"],
            "dataset_split": ds_spec["split"],
            "context_length": ctx_len,
            "prompt_index": prompt_idx,
        }
        job_id = _make_experiment_id(params)
        jobs.append(ExperimentJob(job_id=job_id, params=params))

    logger.info("Built %d state sweep jobs (%d datasets).", len(jobs), len(dataset_specs))
    return jobs


def run_state_sweep(config: Dict[str, Any]):
    """Sequential sweep: load model once, iterate all jobs."""
    # --- GPU detection ------------------------------------------------
    gpu_cfg = config.get("gpu", {})
    min_free = gpu_cfg.get("min_free_memory_gb", 20)
    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs meet the free-memory threshold. Will use CPU.")

    # --- Storage ------------------------------------------------------
    store_cfg = config.get("storage", {})
    store = StateResultStore(
        output_dir=store_cfg.get("output_dir", "./results_state"),
        format=store_cfg.get("format", "parquet"),
        checkpoint_every=store_cfg.get("checkpoint_every", 5),
    )

    # --- Model loading ------------------------------------------------
    from .model import load_model_and_tokenizer

    model_cfg = config["model"]
    max_mem_gb = model_cfg.get("max_memory_per_gpu_gb", 70)
    num_visible = torch.cuda.device_count()
    if num_visible > 0:
        max_memory = {i: f"{max_mem_gb}GiB" for i in range(num_visible)}
    else:
        max_memory = None

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_cfg["path"],
        dtype=model_cfg.get("dtype", "float16"),
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # --- Prompt generators (one per dataset, built lazily) -------------
    prompt_cfg = config.get("prompts", {})
    _prompt_gen_cache: Dict[str, PromptGenerator] = {}

    def _get_prompt_gen(ds_name: str, ds_config: str, ds_split: str) -> PromptGenerator:
        if ds_name not in _prompt_gen_cache:
            logger.info("Loading dataset: %s", ds_name)
            _prompt_gen_cache[ds_name] = PromptGenerator(
                tokenizer=tokenizer,
                source=prompt_cfg.get("source", "dataset"),
                dataset_name=ds_name,
                dataset_config=ds_config,
                dataset_split=ds_split,
                num_candidates=prompt_cfg.get("num_prompt_candidates", 50),
                seed=prompt_cfg.get("seed", 42),
            )
        return _prompt_gen_cache[ds_name]

    # --- Build jobs ---------------------------------------------------
    all_jobs = build_state_sweep_jobs(config)
    model_id = model_cfg["path"]

    total = len(all_jobs)
    logger.info("Starting state detection sweep: %d experiments.", total)
    t0 = time.time()

    for idx, job in enumerate(all_jobs):
        if store.is_completed(job.job_id):
            logger.info("[%d/%d] Skipping completed %s", idx + 1, total, job.job_id)
            continue

        params = job.params
        ds_label = os.path.basename(params["dataset_name"])
        logger.info(
            "[%d/%d] Running %s — dataset=%s ctx=%d prompt=%d",
            idx + 1,
            total,
            job.job_id,
            ds_label,
            params["context_length"],
            params["prompt_index"],
        )

        prompt_gen = _get_prompt_gen(
            params["dataset_name"],
            params["dataset_config"],
            params["dataset_split"],
        )
        prompt_text = prompt_gen.generate(
            target_length=params["context_length"],
            index=params["prompt_index"],
        )

        # Override num_decode_tokens in config for this job.
        job_config = dict(config)
        job_state = dict(config.get("state_detection", {}))
        job_config["state_detection"] = job_state

        try:
            trajectory_records = run_state_experiment(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                config=job_config,
            )
        except Exception:
            logger.exception("Experiment %s failed.", job.job_id)
            continue

        metadata = {
            "model_id": model_id,
            "dataset_name": params["dataset_name"],
            "prompt_id": params["prompt_index"],
            "context_length": params["context_length"],
        }
        for rec in trajectory_records:
            rec.update(metadata)

        store.add_trajectory_records(job.job_id, trajectory_records)
        store.mark_completed(job.job_id)

    store.flush()
    elapsed = time.time() - t0
    logger.info("State detection sweep complete. %d experiments in %.1f s.", total, elapsed)


# -------------------------------------------------------------------
# Parallel variant — module-level worker (must be picklable for spawn)
# -------------------------------------------------------------------
def _parallel_worker(
    jobs: List[ExperimentJob],
    gpu_indices: List[int],
) -> List[Tuple[str, Any]]:
    """Run a batch of experiments in one spawned subprocess.

    CUDA_VISIBLE_DEVICES is already set by the scheduler.  The model is
    loaded **once** and reused for all jobs.  Dataset PromptGenerators
    are cached lazily (one per dataset).

    Must live at module level to be picklable by the ``spawn`` context.
    """
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _logger = _logging.getLogger(__name__)

    from .state_experiment import run_state_experiment as _run
    from .model import load_model_and_tokenizer
    from .prompts import PromptGenerator

    # Config is the same for all jobs; grab from the first one.
    config = jobs[0].params["__config__"]
    model_cfg = config["model"]
    prompt_cfg = config.get("prompts", {})

    # --- Load model once for this GPU group --------------------------
    max_mem_gb = model_cfg.get("max_memory_per_gpu_gb", 70)
    num_visible = len(gpu_indices) if gpu_indices else 1
    max_memory = {i: f"{max_mem_gb}GiB" for i in range(num_visible)}

    _logger.info(
        "GPU group %s: loading model (visible devices: %s)",
        gpu_indices, os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
    )
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_cfg["path"],
        dtype=model_cfg.get("dtype", "float16"),
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # --- Dataset cache (one PromptGenerator per dataset) -------------
    _prompt_gen_cache: Dict[str, PromptGenerator] = {}

    def _get_prompt_gen(ds_name: str, ds_config: str, ds_split: str) -> PromptGenerator:
        if ds_name not in _prompt_gen_cache:
            _logger.info("Loading dataset: %s", ds_name)
            _prompt_gen_cache[ds_name] = PromptGenerator(
                tokenizer=tokenizer,
                source=prompt_cfg.get("source", "dataset"),
                dataset_name=ds_name,
                dataset_config=ds_config,
                dataset_split=ds_split,
                num_candidates=prompt_cfg.get("num_prompt_candidates", 50),
                seed=prompt_cfg.get("seed", 42),
            )
        return _prompt_gen_cache[ds_name]

    # --- Run all assigned jobs sequentially --------------------------
    results: List[Tuple[str, Any]] = []
    total = len(jobs)

    for idx, job in enumerate(jobs):
        params = job.params
        ds_label = os.path.basename(params["dataset_name"])
        _logger.info(
            "[%d/%d] GPU %s — %s dataset=%s ctx=%d prompt=%d",
            idx + 1, total, gpu_indices, job.job_id,
            ds_label, params["context_length"],
            params["prompt_index"],
        )

        prompt_gen = _get_prompt_gen(
            params["dataset_name"],
            params["dataset_config"],
            params["dataset_split"],
        )
        prompt_text = prompt_gen.generate(
            target_length=params["context_length"],
            index=params["prompt_index"],
        )

        job_config = dict(config)
        job_state = dict(config.get("state_detection", {}))
        job_config["state_detection"] = job_state

        try:
            trajectory_records = _run(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                config=job_config,
            )
        except Exception:
            _logger.exception("Job %s failed.", job.job_id)
            results.append((job.job_id, None))
            continue

        metadata = {
            "model_id": model_cfg["path"],
            "dataset_name": params["dataset_name"],
            "prompt_id": params["prompt_index"],
            "context_length": params["context_length"],
        }
        for rec in trajectory_records:
            rec.update(metadata)

        results.append((job.job_id, {"trajectory_records": trajectory_records}))

    import torch as _torch
    del model
    _torch.cuda.empty_cache()

    return results


def run_state_sweep_parallel(config: Dict[str, Any]):
    """Parallel sweep: auto-partition GPUs and run one model per group."""
    gpu_cfg = config.get("gpu", {})
    model_cfg = config["model"]
    min_free = gpu_cfg.get("min_free_memory_gb", 20)

    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs available; falling back to sequential sweep.")
        return run_state_sweep(config)

    # Auto-detect how many GPUs each model instance needs.
    gpus_per_exp = estimate_gpus_per_experiment(
        model_path=model_cfg["path"],
        dtype=model_cfg.get("dtype", "float16"),
        gpus=gpus,
        overhead_factor=gpu_cfg.get("overhead_factor", 1.25),
    )
    num_groups = len(gpus) // gpus_per_exp
    logger.info(
        "Parallel sweep: %d GPUs / %d per experiment = %d parallel groups.",
        len(gpus), gpus_per_exp, num_groups,
    )

    scheduler = ExperimentScheduler(
        available_gpus=gpus,
        max_parallel=num_groups,
        gpus_per_experiment=gpus_per_exp,
    )

    store_cfg = config.get("storage", {})
    store = StateResultStore(
        output_dir=store_cfg.get("output_dir", "./results_state"),
        format=store_cfg.get("format", "parquet"),
        checkpoint_every=store_cfg.get("checkpoint_every", 5),
    )

    all_jobs = build_state_sweep_jobs(config)
    pending_jobs = [j for j in all_jobs if not store.is_completed(j.job_id)]
    logger.info(
        "%d total jobs, %d already completed, %d pending.",
        len(all_jobs),
        len(all_jobs) - len(pending_jobs),
        len(pending_jobs),
    )

    # Attach config to each job so the spawned worker can access it
    # (closures are NOT picklable across spawn boundaries).
    for job in pending_jobs:
        job.params["__config__"] = config

    results = scheduler.run(pending_jobs, _parallel_worker)

    for job_id, result in results:
        if result and isinstance(result, dict) and "trajectory_records" in result:
            store.add_trajectory_records(job_id, result["trajectory_records"])
            store.mark_completed(job_id)

    store.flush()
    logger.info("Parallel state sweep complete. %d jobs processed.", len(results))
