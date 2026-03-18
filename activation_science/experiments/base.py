"""Base experiment ABC and generic sweep runner.

Provides ``BaseExperiment`` which all E0–E8 modules implement, and
``GenericSweepRunner`` that handles model loading, prompt generation,
checkpointing, and sequential/parallel dispatch.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list
from ..core.model import load_model_and_tokenizer
from ..core.datasets import PromptGenerator
from ..core.storage import ExperimentStore
from ..core.scheduler import (
    ExperimentScheduler, detect_gpus, estimate_gpus_per_experiment,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Base Experiment ABC
# ===================================================================
class BaseExperiment(ABC):
    """Abstract base class for all experiments."""

    experiment_id: str = ""         # e.g. "e0"
    experiment_name: str = ""       # e.g. "decode_similarity"

    @classmethod
    @abstractmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        """Return {table_name: [column_names]} for storage registration."""

    @classmethod
    @abstractmethod
    def default_config_section(cls) -> str:
        """Return the config key for this experiment (e.g. 'decode')."""

    @classmethod
    @abstractmethod
    def build_sweep_jobs(cls, config: Dict[str, Any]) -> List[ExperimentJob]:
        """Build the full list of experiment jobs from config."""

    @abstractmethod
    def run(
        self,
        model,
        tokenizer,
        prompt_text: str,
        config: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run a single experiment.

        Returns
        -------
        Dict mapping table_name -> list of record dicts.
        """

    def run_two_prompt(
        self,
        model,
        tokenizer,
        prompt_text_a: str,
        prompt_text_b: str,
        config: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run a two-prompt experiment (E4 cross-sequence).

        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError("This experiment does not support two-prompt mode.")

    @property
    def is_two_prompt(self) -> bool:
        """Whether this experiment requires two prompts (E4)."""
        return False


# ===================================================================
# Generic Sweep Runner
# ===================================================================
class GenericSweepRunner:
    """Unified sweep orchestrator for any BaseExperiment subclass."""

    def __init__(self, experiment_cls: Type[BaseExperiment]):
        self.experiment_cls = experiment_cls

    def run_sequential(self, config: Dict[str, Any]):
        """Load model once, iterate all jobs sequentially."""
        gpu_cfg = config.get("gpu", {})
        min_free = gpu_cfg.get("min_free_memory_gb", 20)
        gpus = detect_gpus(min_free_gb=min_free)
        if not gpus:
            logger.warning("No GPUs meet the free-memory threshold. Will use CPU.")

        # Storage
        store_cfg = config.get("storage", {})
        store = ExperimentStore(
            output_dir=store_cfg.get("output_dir", f"./results_{self.experiment_cls.experiment_name}"),
            checkpoint_every=store_cfg.get("checkpoint_every", 5),
        )
        for name, columns in self.experiment_cls.table_schemas().items():
            store.register_table(name, columns)

        # Model
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

        # Prompt generators (one per dataset, built lazily)
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

        # Build jobs
        all_jobs = self.experiment_cls.build_sweep_jobs(config)
        model_id = model_cfg["path"]
        experiment = self.experiment_cls()

        total = len(all_jobs)
        logger.info("Starting %s sweep: %d experiments.", self.experiment_cls.experiment_name, total)
        t0 = time.time()

        for idx, job in enumerate(all_jobs):
            if store.is_completed(job.job_id):
                logger.info("[%d/%d] Skipping completed %s", idx + 1, total, job.job_id)
                continue

            params = job.params
            ds_label = os.path.basename(params["dataset_name"])
            logger.info(
                "[%d/%d] Running %s — dataset=%s ctx=%d prompt=%d",
                idx + 1, total, job.job_id, ds_label,
                params["context_length"], params["prompt_index"],
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

            # Build per-job config overrides
            job_config = self._build_job_config(config, params)

            try:
                if experiment.is_two_prompt:
                    # Two-prompt experiment (E4)
                    prompt_gen_b = _get_prompt_gen(
                        params.get("dataset_name_b", params["dataset_name"]),
                        params.get("dataset_config_b", params["dataset_config"]),
                        params.get("dataset_split_b", params["dataset_split"]),
                    )
                    prompt_text_b = prompt_gen_b.generate(
                        target_length=params["context_length"],
                        index=params.get("prompt_index_b", params["prompt_index"] + 1),
                    )
                    results = experiment.run_two_prompt(
                        model=model, tokenizer=tokenizer,
                        prompt_text_a=prompt_text, prompt_text_b=prompt_text_b,
                        config=job_config,
                    )
                else:
                    results = experiment.run(
                        model=model, tokenizer=tokenizer,
                        prompt_text=prompt_text, config=job_config,
                    )
            except Exception:
                logger.exception("Experiment %s failed.", job.job_id)
                continue

            # Inject metadata and store
            metadata = self._build_metadata(model_id, params)
            for table_name, records in results.items():
                for rec in records:
                    rec.update(metadata)
                store.add_records(table_name, job.job_id, records)
            store.mark_completed(job.job_id)

        store.flush()
        elapsed = time.time() - t0
        logger.info(
            "%s sweep complete. %d experiments in %.1f s.",
            self.experiment_cls.experiment_name, total, elapsed,
        )

    def run_parallel(self, config: Dict[str, Any]):
        """Multi-GPU parallel sweep via ExperimentScheduler."""
        gpu_cfg = config.get("gpu", {})
        model_cfg = config["model"]
        min_free = gpu_cfg.get("min_free_memory_gb", 20)

        gpus = detect_gpus(min_free_gb=min_free)
        if not gpus:
            logger.warning("No GPUs available; falling back to sequential sweep.")
            return self.run_sequential(config)

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
        store = ExperimentStore(
            output_dir=store_cfg.get("output_dir", f"./results_{self.experiment_cls.experiment_name}"),
            checkpoint_every=store_cfg.get("checkpoint_every", 5),
        )
        for name, columns in self.experiment_cls.table_schemas().items():
            store.register_table(name, columns)

        all_jobs = self.experiment_cls.build_sweep_jobs(config)
        pending_jobs = [j for j in all_jobs if not store.is_completed(j.job_id)]
        logger.info(
            "%d total jobs, %d already completed, %d pending.",
            len(all_jobs), len(all_jobs) - len(pending_jobs), len(pending_jobs),
        )

        # Attach config + experiment class path to each job for the spawned worker
        for job in pending_jobs:
            job.params["__config__"] = config
            job.params["__experiment_module__"] = self.experiment_cls.__module__
            job.params["__experiment_class__"] = self.experiment_cls.__name__

        results = scheduler.run(pending_jobs, _parallel_worker)

        for job_id, result in results:
            if result and isinstance(result, dict):
                for table_name, records in result.items():
                    store.add_records(table_name, job_id, records)
                store.mark_completed(job_id)

        store.flush()
        logger.info(
            "Parallel %s sweep complete. %d jobs processed.",
            self.experiment_cls.experiment_name, len(results),
        )

    @staticmethod
    def _build_job_config(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Build per-job config with any parameter overrides."""
        job_config = dict(config)
        # Always propagate sweep params into job config
        if "context_length" in params:
            job_config["context_length"] = params["context_length"]
        if "dataset_name" in params:
            job_config["_dataset_name"] = params["dataset_name"]
            job_config["_dataset_config"] = params.get("dataset_config", "default")
            job_config["_dataset_split"] = params.get("dataset_split", "train")
        # Override num_decode_tokens if present in params
        if "num_decode_tokens" in params:
            for section_key in ["decode", "residual", "logit_lens", "state_detection",
                                "geometry", "cross_sequence", "causal", "perturbation", "token_type",
                                "delta_cache", "static_delta", "reference_strategies",
                                "trigram_pipeline"]:
                if section_key in config:
                    section = dict(config[section_key])
                    section["num_decode_tokens"] = params["num_decode_tokens"]
                    job_config[section_key] = section
        return job_config

    @staticmethod
    def _build_metadata(model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build metadata dict to inject into all records."""
        metadata = {
            "model_id": model_id,
            "dataset_name": params["dataset_name"],
            "prompt_id": params["prompt_index"],
            "context_length": params["context_length"],
        }
        if "num_decode_tokens" in params:
            metadata["num_decode_tokens"] = params["num_decode_tokens"]
        # Cross-sequence specific
        if "dataset_name_b" in params:
            metadata["dataset_b"] = params["dataset_name_b"]
            metadata["prompt_id_b"] = params.get("prompt_index_b")
            metadata["pair_type"] = params.get("pair_type", "cross_dataset")
        return metadata


# ===================================================================
# Module-level parallel worker (must be picklable for spawn)
# ===================================================================
def _parallel_worker(
    jobs: List[ExperimentJob],
    gpu_indices: List[int],
) -> List[Tuple[str, Any]]:
    """Run a batch of experiments in one spawned subprocess."""
    import importlib
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _logger = _logging.getLogger(__name__)

    config = jobs[0].params["__config__"]
    module_name = jobs[0].params["__experiment_module__"]
    class_name = jobs[0].params["__experiment_class__"]

    # Dynamically import the experiment class
    mod = importlib.import_module(module_name)
    experiment_cls = getattr(mod, class_name)

    model_cfg = config["model"]
    prompt_cfg = config.get("prompts", {})

    # Load model once for this GPU group
    max_mem_gb = model_cfg.get("max_memory_per_gpu_gb", 70)
    num_visible = len(gpu_indices) if gpu_indices else 1
    max_memory = {i: f"{max_mem_gb}GiB" for i in range(num_visible)}

    _logger.info(
        "GPU group %s: loading model (visible devices: %s)",
        gpu_indices, os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
    )

    from ..core.model import load_model_and_tokenizer
    from ..core.datasets import PromptGenerator

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_cfg["path"],
        dtype=model_cfg.get("dtype", "float16"),
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

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

    experiment = experiment_cls()
    results: List[Tuple[str, Any]] = []
    total = len(jobs)

    for idx, job in enumerate(jobs):
        params = job.params
        ds_label = os.path.basename(params["dataset_name"])
        _logger.info(
            "[%d/%d] GPU %s — %s dataset=%s ctx=%d prompt=%d",
            idx + 1, total, gpu_indices, job.job_id,
            ds_label, params["context_length"], params["prompt_index"],
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

        job_config = GenericSweepRunner._build_job_config(config, params)

        try:
            if experiment.is_two_prompt:
                prompt_gen_b = _get_prompt_gen(
                    params.get("dataset_name_b", params["dataset_name"]),
                    params.get("dataset_config_b", params["dataset_config"]),
                    params.get("dataset_split_b", params["dataset_split"]),
                )
                prompt_text_b = prompt_gen_b.generate(
                    target_length=params["context_length"],
                    index=params.get("prompt_index_b", params["prompt_index"] + 1),
                )
                table_results = experiment.run_two_prompt(
                    model=model, tokenizer=tokenizer,
                    prompt_text_a=prompt_text, prompt_text_b=prompt_text_b,
                    config=job_config,
                )
            else:
                table_results = experiment.run(
                    model=model, tokenizer=tokenizer,
                    prompt_text=prompt_text, config=job_config,
                )
        except Exception:
            _logger.exception("Job %s failed.", job.job_id)
            results.append((job.job_id, None))
            continue

        metadata = GenericSweepRunner._build_metadata(model_cfg["path"], params)
        for table_name, records in table_results.items():
            for rec in records:
                rec.update(metadata)

        results.append((job.job_id, table_results))

    del model
    torch.cuda.empty_cache()

    return results
