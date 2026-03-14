"""
Sweep orchestration for cross-sequence alignment experiments.

Generates pairs of prompts for comparison:
  1. Within-dataset pairs: For each of 7 datasets, pair 3 prompts with each
     other at 3 context lengths = 7 x C(3,2) x 3 = 63 pairs.
  2. Cross-dataset pairs: For each pair of datasets (C(7,2)=21), compare
     prompt 0 at context_length 512 = 21 pairs. Plus a few at different
     context lengths to get ~30 cross pairs.
Total: ~93 pairs. Each pair requires 2 forward passes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import torch

from .cross_sequence_experiment import run_cross_sequence_experiment
from .cross_sequence_storage import CrossSequenceResultStore
from .prompts import PromptGenerator
from .scheduler import (
    ExperimentJob,
    ExperimentScheduler,
    detect_gpus,
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


def build_cross_sequence_sweep_jobs(config: Dict[str, Any]) -> List[ExperimentJob]:
    """Build the full list of cross-sequence alignment jobs from config."""
    sweep = config["sweep"]
    context_lengths = sweep["context_lengths"]
    num_prompts = sweep.get("num_prompts_per_length", 3)
    cross_ctx = config.get("cross_sequence", {}).get("context_length", 512)

    dataset_specs = _resolve_dataset_list(config)

    jobs: List[ExperimentJob] = []

    # 1. Within-dataset pairs: for each dataset, pair prompts at each ctx length.
    for ds_spec in dataset_specs:
        for ctx_len in context_lengths:
            for idx_a, idx_b in combinations(range(num_prompts), 2):
                params = {
                    "pair_type": "within",
                    "dataset_a": ds_spec["name"],
                    "dataset_config_a": ds_spec["config"],
                    "dataset_split_a": ds_spec["split"],
                    "dataset_b": ds_spec["name"],
                    "dataset_config_b": ds_spec["config"],
                    "dataset_split_b": ds_spec["split"],
                    "prompt_index_a": idx_a,
                    "prompt_index_b": idx_b,
                    "context_length": ctx_len,
                }
                job_id = _make_experiment_id(params)
                jobs.append(ExperimentJob(job_id=job_id, params=params))

    # 2. Cross-dataset pairs: pair prompt 0 at default cross context length.
    for ds_a, ds_b in combinations(dataset_specs, 2):
        params = {
            "pair_type": "cross",
            "dataset_a": ds_a["name"],
            "dataset_config_a": ds_a["config"],
            "dataset_split_a": ds_a["split"],
            "dataset_b": ds_b["name"],
            "dataset_config_b": ds_b["config"],
            "dataset_split_b": ds_b["split"],
            "prompt_index_a": 0,
            "prompt_index_b": 0,
            "context_length": cross_ctx,
        }
        job_id = _make_experiment_id(params)
        jobs.append(ExperimentJob(job_id=job_id, params=params))

    # 3. Additional cross-dataset pairs at different context lengths
    #    to reach ~30 cross pairs: add a few more at alternative lengths.
    extra_ctx_lengths = [cl for cl in context_lengths if cl != cross_ctx][:2]
    for extra_ctx in extra_ctx_lengths:
        # Use a subset of dataset pairs (first 5 combinations).
        for ds_a, ds_b in list(combinations(dataset_specs, 2))[:5]:
            params = {
                "pair_type": "cross",
                "dataset_a": ds_a["name"],
                "dataset_config_a": ds_a["config"],
                "dataset_split_a": ds_a["split"],
                "dataset_b": ds_b["name"],
                "dataset_config_b": ds_b["config"],
                "dataset_split_b": ds_b["split"],
                "prompt_index_a": 0,
                "prompt_index_b": 0,
                "context_length": extra_ctx,
            }
            job_id = _make_experiment_id(params)
            jobs.append(ExperimentJob(job_id=job_id, params=params))

    logger.info(
        "Built %d cross-sequence sweep jobs (%d datasets).",
        len(jobs),
        len(dataset_specs),
    )
    return jobs


def run_cross_sequence_sweep(config: Dict[str, Any]):
    """Sequential sweep: load model once, iterate all jobs."""
    # --- GPU detection ------------------------------------------------
    gpu_cfg = config.get("gpu", {})
    min_free = gpu_cfg.get("min_free_memory_gb", 20)
    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs meet the free-memory threshold. Will use CPU.")

    # --- Storage ------------------------------------------------------
    store_cfg = config.get("storage", {})
    store = CrossSequenceResultStore(
        output_dir=store_cfg.get("output_dir", "./results_cross_sequence"),
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
    all_jobs = build_cross_sequence_sweep_jobs(config)
    model_id = model_cfg["path"]

    total = len(all_jobs)
    logger.info("Starting cross-sequence sweep: %d experiments.", total)
    t0 = time.time()

    for idx, job in enumerate(all_jobs):
        if store.is_completed(job.job_id):
            logger.info("[%d/%d] Skipping completed %s", idx + 1, total, job.job_id)
            continue

        params = job.params
        ds_label_a = os.path.basename(params["dataset_a"])
        ds_label_b = os.path.basename(params["dataset_b"])
        logger.info(
            "[%d/%d] Running %s — %s pair ds_a=%s ds_b=%s ctx=%d prompt_a=%d prompt_b=%d",
            idx + 1,
            total,
            job.job_id,
            params["pair_type"],
            ds_label_a,
            ds_label_b,
            params["context_length"],
            params["prompt_index_a"],
            params["prompt_index_b"],
        )

        prompt_gen_a = _get_prompt_gen(
            params["dataset_a"],
            params["dataset_config_a"],
            params["dataset_split_a"],
        )
        prompt_text_a = prompt_gen_a.generate(
            target_length=params["context_length"],
            index=params["prompt_index_a"],
        )

        prompt_gen_b = _get_prompt_gen(
            params["dataset_b"],
            params["dataset_config_b"],
            params["dataset_split_b"],
        )
        prompt_text_b = prompt_gen_b.generate(
            target_length=params["context_length"],
            index=params["prompt_index_b"],
        )

        try:
            records = run_cross_sequence_experiment(
                model=model,
                tokenizer=tokenizer,
                prompt_text_a=prompt_text_a,
                prompt_text_b=prompt_text_b,
                config=config,
            )
        except Exception:
            logger.exception("Experiment %s failed.", job.job_id)
            continue

        metadata = {
            "model_id": model_id,
            "pair_type": params["pair_type"],
            "dataset_a": params["dataset_a"],
            "dataset_b": params["dataset_b"],
            "prompt_id_a": params["prompt_index_a"],
            "prompt_id_b": params["prompt_index_b"],
            "context_length": params["context_length"],
        }
        for rec in records:
            rec.update(metadata)

        store.add_alignment_records(job.job_id, records)
        store.mark_completed(job.job_id)

    store.flush()
    elapsed = time.time() - t0
    logger.info("Cross-sequence sweep complete. %d experiments in %.1f s.", total, elapsed)


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

    from .cross_sequence_experiment import run_cross_sequence_experiment as _run
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
        gpu_indices,
        os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
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
        ds_label_a = os.path.basename(params["dataset_a"])
        ds_label_b = os.path.basename(params["dataset_b"])
        _logger.info(
            "[%d/%d] GPU %s — %s %s pair ds_a=%s ds_b=%s ctx=%d",
            idx + 1,
            total,
            gpu_indices,
            job.job_id,
            params["pair_type"],
            ds_label_a,
            ds_label_b,
            params["context_length"],
        )

        prompt_gen_a = _get_prompt_gen(
            params["dataset_a"],
            params["dataset_config_a"],
            params["dataset_split_a"],
        )
        prompt_text_a = prompt_gen_a.generate(
            target_length=params["context_length"],
            index=params["prompt_index_a"],
        )

        prompt_gen_b = _get_prompt_gen(
            params["dataset_b"],
            params["dataset_config_b"],
            params["dataset_split_b"],
        )
        prompt_text_b = prompt_gen_b.generate(
            target_length=params["context_length"],
            index=params["prompt_index_b"],
        )

        try:
            records = _run(
                model=model,
                tokenizer=tokenizer,
                prompt_text_a=prompt_text_a,
                prompt_text_b=prompt_text_b,
                config=config,
            )
        except Exception:
            _logger.exception("Job %s failed.", job.job_id)
            results.append((job.job_id, None))
            continue

        metadata = {
            "model_id": model_cfg["path"],
            "pair_type": params["pair_type"],
            "dataset_a": params["dataset_a"],
            "dataset_b": params["dataset_b"],
            "prompt_id_a": params["prompt_index_a"],
            "prompt_id_b": params["prompt_index_b"],
            "context_length": params["context_length"],
        }
        for rec in records:
            rec.update(metadata)

        results.append((job.job_id, {"alignment_records": records}))

    import torch as _torch

    del model
    _torch.cuda.empty_cache()

    return results


def run_cross_sequence_sweep_parallel(config: Dict[str, Any]):
    """Parallel sweep: auto-partition GPUs and run one model per group."""
    gpu_cfg = config.get("gpu", {})
    model_cfg = config["model"]
    min_free = gpu_cfg.get("min_free_memory_gb", 20)

    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs available; falling back to sequential sweep.")
        return run_cross_sequence_sweep(config)

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
        len(gpus),
        gpus_per_exp,
        num_groups,
    )

    scheduler = ExperimentScheduler(
        available_gpus=gpus,
        max_parallel=num_groups,
        gpus_per_experiment=gpus_per_exp,
    )

    store_cfg = config.get("storage", {})
    store = CrossSequenceResultStore(
        output_dir=store_cfg.get("output_dir", "./results_cross_sequence"),
        format=store_cfg.get("format", "parquet"),
        checkpoint_every=store_cfg.get("checkpoint_every", 5),
    )

    all_jobs = build_cross_sequence_sweep_jobs(config)
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
        if result and isinstance(result, dict) and "alignment_records" in result:
            store.add_alignment_records(job_id, result["alignment_records"])
            store.mark_completed(job_id)

    store.flush()
    logger.info("Parallel cross-sequence sweep complete. %d jobs processed.", len(results))
