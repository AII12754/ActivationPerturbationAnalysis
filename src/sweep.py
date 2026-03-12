"""
Experiment sweep orchestration.

Generates the full Cartesian product of experimental conditions, creates
jobs, and dispatches them through the GPU scheduler.
"""

from __future__ import annotations

import hashlib
import logging
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from .experiment import run_single_experiment
from .prompts import PromptGenerator
from .scheduler import ExperimentJob, ExperimentScheduler, GPUInfo, detect_gpus
from .storage import ResultStore

logger = logging.getLogger(__name__)


def _make_experiment_id(params: Dict[str, Any]) -> str:
    """Deterministic experiment id from the parameter dict."""
    key = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_sweep_jobs(config: Dict[str, Any]) -> List[ExperimentJob]:
    """Build the full list of experiment jobs from config."""
    sweep = config["sweep"]
    jobs: List[ExperimentJob] = []

    for ctx_len, pos_frac, span_len, pert_type in product(
        sweep["context_lengths"],
        sweep["perturbation_positions"],
        sweep["perturbation_span_lengths"],
        sweep["perturbation_types"],
    ):
        # Skip impossible combinations:
        # span can't be larger than what remains after the perturbation position.
        max_possible_span = int(ctx_len * (1.0 - pos_frac))
        if span_len > max_possible_span:
            continue

        params = {
            "context_length": ctx_len,
            "perturbation_position_frac": pos_frac,
            "perturbation_span_length": span_len,
            "perturbation_type": pert_type,
        }
        job_id = _make_experiment_id(params)
        jobs.append(ExperimentJob(job_id=job_id, params=params))

    logger.info("Built %d experiment jobs.", len(jobs))
    return jobs


def run_sweep(config: Dict[str, Any]):
    """Entry point: configure resources, generate jobs, and run."""
    # --- GPU detection ------------------------------------------------
    gpu_cfg = config.get("gpu", {})
    min_free = gpu_cfg.get("min_free_memory_gb", 20)
    max_parallel = gpu_cfg.get("max_parallel_experiments", 4)

    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs meet the free-memory threshold. Will use CPU.")

    # --- Storage ------------------------------------------------------
    store_cfg = config.get("storage", {})
    store = ResultStore(
        output_dir=store_cfg.get("output_dir", "./results"),
        format=store_cfg.get("format", "parquet"),
        checkpoint_every=store_cfg.get("checkpoint_every", 10),
    )

    # --- Model loading (done once, shared across sequential jobs) ----
    from .model import load_model_and_tokenizer

    model_cfg = config["model"]

    # Build max_memory map from available GPUs.
    max_mem_gb = model_cfg.get("max_memory_per_gpu_gb", 70)
    max_memory = {g.index: f"{max_mem_gb}GiB" for g in gpus} if gpus else None

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_cfg["path"],
        dtype=model_cfg.get("dtype", "float16"),
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # --- Prompt generator ---------------------------------------------
    prompt_cfg = config.get("prompts", {})
    prompt_gen = PromptGenerator(
        tokenizer=tokenizer,
        source=prompt_cfg.get("source", "dataset"),
        dataset_name=prompt_cfg.get("dataset_name", "wikitext"),
        dataset_config=prompt_cfg.get("dataset_config", "wikitext-103-raw-v1"),
        dataset_split=prompt_cfg.get("dataset_split", "train"),
        num_candidates=prompt_cfg.get("num_prompt_candidates", 50),
        seed=prompt_cfg.get("seed", 42),
    )

    # Pre-tokenize a replacement pool for semantic_change strategy.
    _replacement_pool = tokenizer.encode(
        prompt_gen.generate(target_length=2048, index=99),
        add_special_tokens=False,
    )

    # --- Build jobs ---------------------------------------------------
    all_jobs = build_sweep_jobs(config)
    measurement_cfg = config.get("measurement", {})
    model_id = model_cfg["path"]

    # --- Execute (sequential; the scheduler is available for parallel
    #     runs when the model fits on 1 GPU and multiple copies can
    #     coexist, but the default safe mode is sequential) ------------
    total = len(all_jobs)
    logger.info("Starting sweep: %d experiments.", total)
    t0 = time.time()

    for idx, job in enumerate(all_jobs):
        if store.is_completed(job.job_id):
            logger.info("[%d/%d] Skipping completed %s", idx + 1, total, job.job_id)
            continue

        params = job.params
        logger.info(
            "[%d/%d] Running %s — ctx=%d pos=%.2f span=%d type=%s",
            idx + 1,
            total,
            job.job_id,
            params["context_length"],
            params["perturbation_position_frac"],
            params["perturbation_span_length"],
            params["perturbation_type"],
        )

        prompt_text = prompt_gen.generate(
            target_length=params["context_length"], index=idx
        )

        try:
            records = run_single_experiment(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                perturbation_type=params["perturbation_type"],
                perturbation_position_frac=params["perturbation_position_frac"],
                perturbation_span_length=params["perturbation_span_length"],
                measurement_cfg=measurement_cfg,
                seed=prompt_cfg.get("seed", 42),
                replacement_pool=_replacement_pool,
            )
        except Exception:
            logger.exception("Experiment %s failed.", job.job_id)
            continue

        metadata = {
            "model_id": model_id,
            "context_length": params["context_length"],
            "perturbation_type": params["perturbation_type"],
            "perturbation_position_frac": params["perturbation_position_frac"],
            "perturbation_position_abs": int(
                params["perturbation_position_frac"] * params["context_length"]
            ),
            "perturbation_span_length": params["perturbation_span_length"],
        }
        store.add_records(job.job_id, records, metadata=metadata)

    store.flush()
    elapsed = time.time() - t0
    logger.info("Sweep complete. %d experiments in %.1f s.", total, elapsed)


# -------------------------------------------------------------------
# Parallel variant (uses scheduler for multi-copy model serving)
# -------------------------------------------------------------------
def run_sweep_parallel(config: Dict[str, Any]):
    """Run the sweep using the GPU scheduler for parallelism.

    This variant loads a *separate model copy per GPU group* and runs
    experiments concurrently.  Suitable when the model fits on a single
    GPU with room to spare.
    """
    gpu_cfg = config.get("gpu", {})
    min_free = gpu_cfg.get("min_free_memory_gb", 20)
    max_parallel = gpu_cfg.get("max_parallel_experiments", 4)

    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs available; falling back to sequential sweep.")
        return run_sweep(config)

    scheduler = ExperimentScheduler(
        available_gpus=gpus,
        max_parallel=max_parallel,
        gpus_per_experiment=1,
    )

    store_cfg = config.get("storage", {})
    store = ResultStore(
        output_dir=store_cfg.get("output_dir", "./results"),
        format=store_cfg.get("format", "parquet"),
        checkpoint_every=store_cfg.get("checkpoint_every", 10),
    )

    all_jobs = build_sweep_jobs(config)
    # Filter already completed.
    pending_jobs = [j for j in all_jobs if not store.is_completed(j.job_id)]
    logger.info(
        "%d total jobs, %d already completed, %d pending.",
        len(all_jobs),
        len(all_jobs) - len(pending_jobs),
        len(pending_jobs),
    )

    def worker_fn(job: ExperimentJob, gpu_indices: List[int]):
        """Worker that loads a model copy and runs the experiment."""
        import os
        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))

        from .model import load_model_and_tokenizer
        from .prompts import PromptGenerator

        model_cfg = config["model"]
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_cfg["path"],
            dtype=model_cfg.get("dtype", "float16"),
            device_map="auto",
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )

        prompt_cfg = config.get("prompts", {})
        prompt_gen = PromptGenerator(
            tokenizer=tokenizer,
            source=prompt_cfg.get("source", "dataset"),
            dataset_name=prompt_cfg.get("dataset_name", "wikitext"),
            dataset_config=prompt_cfg.get("dataset_config", "wikitext-103-raw-v1"),
            dataset_split=prompt_cfg.get("dataset_split", "train"),
            num_candidates=prompt_cfg.get("num_prompt_candidates", 50),
            seed=prompt_cfg.get("seed", 42),
        )

        params = job.params
        prompt_text = prompt_gen.generate(target_length=params["context_length"])

        replacement_pool = tokenizer.encode(
            prompt_gen.generate(target_length=2048, index=99),
            add_special_tokens=False,
        )

        measurement_cfg = config.get("measurement", {})
        records = run_single_experiment(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            perturbation_type=params["perturbation_type"],
            perturbation_position_frac=params["perturbation_position_frac"],
            perturbation_span_length=params["perturbation_span_length"],
            measurement_cfg=measurement_cfg,
            seed=prompt_cfg.get("seed", 42),
            replacement_pool=replacement_pool,
        )

        # Clean up.
        del model
        torch.cuda.empty_cache()

        return {
            "records": records,
            "metadata": {
                "model_id": model_cfg["path"],
                "context_length": params["context_length"],
                "perturbation_type": params["perturbation_type"],
                "perturbation_position_frac": params["perturbation_position_frac"],
                "perturbation_position_abs": int(
                    params["perturbation_position_frac"] * params["context_length"]
                ),
                "perturbation_span_length": params["perturbation_span_length"],
            },
        }

    results = scheduler.run(pending_jobs, worker_fn)

    for job_id, result in results:
        if result and "records" in result:
            store.add_records(job_id, result["records"], metadata=result["metadata"])

    store.flush()
    logger.info("Parallel sweep complete. %d jobs processed.", len(results))
