"""
Sweep orchestration for decode-time similarity experiments.

Generates the Cartesian product of (context_length × prompt_index ×
num_decode_tokens), then dispatches experiments sequentially or in parallel.
"""

from __future__ import annotations

import hashlib
import logging
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import torch

from .decode_experiment import run_decode_experiment
from .decode_storage import DecodeResultStore
from .prompts import PromptGenerator
from .scheduler import ExperimentJob, ExperimentScheduler, detect_gpus

logger = logging.getLogger(__name__)


def _make_experiment_id(params: Dict[str, Any]) -> str:
    """Deterministic experiment id from the parameter dict."""
    key = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_decode_sweep_jobs(config: Dict[str, Any]) -> List[ExperimentJob]:
    """Build the full list of decode experiment jobs from config."""
    sweep = config["sweep"]
    context_lengths = sweep["context_lengths"]
    num_prompts = sweep.get("num_prompts_per_length", 3)
    num_decode_list = sweep.get("num_decode_tokens", [128])
    if isinstance(num_decode_list, int):
        num_decode_list = [num_decode_list]

    jobs: List[ExperimentJob] = []
    for ctx_len, prompt_idx, num_dec in product(
        context_lengths,
        range(num_prompts),
        num_decode_list,
    ):
        params = {
            "context_length": ctx_len,
            "prompt_index": prompt_idx,
            "num_decode_tokens": num_dec,
        }
        job_id = _make_experiment_id(params)
        jobs.append(ExperimentJob(job_id=job_id, params=params))

    logger.info("Built %d decode sweep jobs.", len(jobs))
    return jobs


def run_decode_sweep(config: Dict[str, Any]):
    """Sequential sweep: load model once, iterate all jobs."""
    # --- GPU detection ------------------------------------------------
    gpu_cfg = config.get("gpu", {})
    min_free = gpu_cfg.get("min_free_memory_gb", 20)
    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs meet the free-memory threshold. Will use CPU.")

    # --- Storage ------------------------------------------------------
    store_cfg = config.get("storage", {})
    store = DecodeResultStore(
        output_dir=store_cfg.get("output_dir", "./results_decode"),
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

    # --- Build jobs ---------------------------------------------------
    all_jobs = build_decode_sweep_jobs(config)
    model_id = model_cfg["path"]

    total = len(all_jobs)
    logger.info("Starting decode sweep: %d experiments.", total)
    t0 = time.time()

    for idx, job in enumerate(all_jobs):
        if store.is_completed(job.job_id):
            logger.info("[%d/%d] Skipping completed %s", idx + 1, total, job.job_id)
            continue

        params = job.params
        logger.info(
            "[%d/%d] Running %s — ctx=%d prompt=%d dec=%d",
            idx + 1,
            total,
            job.job_id,
            params["context_length"],
            params["prompt_index"],
            params["num_decode_tokens"],
        )

        prompt_text = prompt_gen.generate(
            target_length=params["context_length"],
            index=params["prompt_index"],
        )

        # Override num_decode_tokens in config for this job.
        job_config = dict(config)
        job_decode = dict(config.get("decode", {}))
        job_decode["num_decode_tokens"] = params["num_decode_tokens"]
        job_config["decode"] = job_decode

        try:
            topk_records, agg_records = run_decode_experiment(
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
            "prompt_id": params["prompt_index"],
            "context_length": params["context_length"],
            "num_decode_tokens": params["num_decode_tokens"],
        }
        for rec in topk_records:
            rec.update(metadata)
        for rec in agg_records:
            rec.update(metadata)

        store.add_topk_records(job.job_id, topk_records)
        store.add_aggregate_records(job.job_id, agg_records)
        store.mark_completed(job.job_id)

    store.flush()
    elapsed = time.time() - t0
    logger.info("Decode sweep complete. %d experiments in %.1f s.", total, elapsed)


# -------------------------------------------------------------------
# Parallel variant
# -------------------------------------------------------------------
def run_decode_sweep_parallel(config: Dict[str, Any]):
    """Parallel sweep using the GPU scheduler."""
    gpu_cfg = config.get("gpu", {})
    min_free = gpu_cfg.get("min_free_memory_gb", 20)
    max_parallel = gpu_cfg.get("max_parallel_experiments", 4)

    gpus = detect_gpus(min_free_gb=min_free)
    if not gpus:
        logger.warning("No GPUs available; falling back to sequential sweep.")
        return run_decode_sweep(config)

    scheduler = ExperimentScheduler(
        available_gpus=gpus,
        max_parallel=max_parallel,
        gpus_per_experiment=1,
    )

    store_cfg = config.get("storage", {})
    store = DecodeResultStore(
        output_dir=store_cfg.get("output_dir", "./results_decode"),
        format=store_cfg.get("format", "parquet"),
        checkpoint_every=store_cfg.get("checkpoint_every", 5),
    )

    all_jobs = build_decode_sweep_jobs(config)
    pending_jobs = [j for j in all_jobs if not store.is_completed(j.job_id)]
    logger.info(
        "%d total jobs, %d already completed, %d pending.",
        len(all_jobs),
        len(all_jobs) - len(pending_jobs),
        len(pending_jobs),
    )

    def worker_fn(job: ExperimentJob, gpu_indices: List[int]):
        import os
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
        prompt_text = prompt_gen.generate(
            target_length=params["context_length"],
            index=params["prompt_index"],
        )

        job_config = dict(config)
        job_decode = dict(config.get("decode", {}))
        job_decode["num_decode_tokens"] = params["num_decode_tokens"]
        job_config["decode"] = job_decode

        topk_records, agg_records = run_decode_experiment(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            config=job_config,
        )

        del model
        torch.cuda.empty_cache()

        metadata = {
            "model_id": model_cfg["path"],
            "prompt_id": params["prompt_index"],
            "context_length": params["context_length"],
            "num_decode_tokens": params["num_decode_tokens"],
        }
        for rec in topk_records:
            rec.update(metadata)
        for rec in agg_records:
            rec.update(metadata)

        return {"topk_records": topk_records, "agg_records": agg_records}

    results = scheduler.run(pending_jobs, worker_fn)

    for job_id, result in results:
        if result and "topk_records" in result:
            store.add_topk_records(job_id, result["topk_records"])
            store.add_aggregate_records(job_id, result["agg_records"])
            store.mark_completed(job_id)

    store.flush()
    logger.info("Parallel decode sweep complete. %d jobs processed.", len(results))
