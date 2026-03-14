"""GPU resource detection and experiment scheduling.

Uses ``pynvml`` to query per-GPU memory and exposes a simple pool-based
scheduler that assigns experiment jobs to available GPUs.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from .types import ExperimentJob

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# GPU inspection
# -------------------------------------------------------------------
@dataclass
class GPUInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    used_mb: int


def detect_gpus(min_free_gb: float = 20.0) -> List[GPUInfo]:
    """Return a list of GPUs that have at least *min_free_gb* free VRAM."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as exc:
        logger.warning("pynvml unavailable (%s). Falling back to CUDA count.", exc)
        return _fallback_detect()

    count = pynvml.nvmlDeviceGetCount()
    gpus: List[GPUInfo] = []
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info = GPUInfo(
            index=i,
            name=name,
            total_mb=mem.total // (1024 * 1024),
            free_mb=mem.free // (1024 * 1024),
            used_mb=mem.used // (1024 * 1024),
        )
        if info.free_mb >= min_free_gb * 1024:
            gpus.append(info)
        else:
            logger.info(
                "GPU %d (%s): %d MB free — below threshold, skipping.",
                i, name, info.free_mb,
            )
    pynvml.nvmlShutdown()
    logger.info("Detected %d usable GPUs (min_free=%.1f GB).", len(gpus), min_free_gb)
    return gpus


def _fallback_detect() -> List[GPUInfo]:
    """Fallback when pynvml is not installed: rely on CUDA_VISIBLE_DEVICES."""
    import torch

    count = torch.cuda.device_count()
    return [
        GPUInfo(index=i, name=torch.cuda.get_device_name(i),
                total_mb=0, free_mb=0, used_mb=0)
        for i in range(count)
    ]


# -------------------------------------------------------------------
# Model VRAM estimation
# -------------------------------------------------------------------
def estimate_model_vram_gb(
    model_path: str,
    dtype: str = "float16",
    overhead_factor: float = 1.25,
) -> float:
    """Estimate the GPU VRAM needed to load a model for inference."""
    bytes_per_param = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1}
    target_bpp = bytes_per_param.get(dtype, 2)

    total_bytes = 0
    for fname in os.listdir(model_path):
        if fname.endswith((".safetensors", ".bin")):
            total_bytes += os.path.getsize(os.path.join(model_path, fname))

    if total_bytes == 0:
        logger.warning("No weight files found in %s; cannot estimate VRAM.", model_path)
        return 0.0

    disk_bpp = 2
    weight_gb = (total_bytes / disk_bpp * target_bpp) / (1024 ** 3)
    estimated = weight_gb * overhead_factor
    logger.info(
        "Model VRAM estimate: %.1f GB weights (dtype=%s) × %.2f overhead = %.1f GB",
        weight_gb, dtype, overhead_factor, estimated,
    )
    return estimated


def estimate_gpus_per_experiment(
    model_path: str,
    dtype: str,
    gpus: List[GPUInfo],
    overhead_factor: float = 1.25,
) -> int:
    """Auto-detect how many GPUs each experiment needs."""
    if not gpus:
        return 1
    model_vram = estimate_model_vram_gb(model_path, dtype, overhead_factor)
    if model_vram <= 0:
        logger.warning("Could not estimate model size; defaulting to 1 GPU per experiment.")
        return 1
    per_gpu_gb = min(g.free_mb for g in gpus) / 1024
    needed = max(1, math.ceil(model_vram / per_gpu_gb))
    needed = min(needed, len(gpus))
    num_groups = len(gpus) // needed
    logger.info(
        "Auto GPU partitioning: model needs ~%.1f GB, %.1f GB/GPU → "
        "%d GPU(s)/experiment, %d parallel group(s) from %d GPUs.",
        model_vram, per_gpu_gb, needed, num_groups, len(gpus),
    )
    return needed


# -------------------------------------------------------------------
# Simple experiment scheduler
# -------------------------------------------------------------------
class ExperimentScheduler:
    """A pool-based scheduler that maps jobs to available GPUs."""

    def __init__(
        self,
        available_gpus: List[GPUInfo],
        max_parallel: int = 4,
        gpus_per_experiment: int = 1,
    ):
        self.available_gpus = available_gpus
        self.gpus_per_experiment = gpus_per_experiment
        self.gpu_groups = self._partition_gpus()
        self.max_parallel = min(max_parallel, len(self.gpu_groups))
        logger.info(
            "Scheduler: %d GPU groups of size %d, max_parallel=%d",
            len(self.gpu_groups), gpus_per_experiment, self.max_parallel,
        )

    def _partition_gpus(self) -> List[List[int]]:
        """Split available GPUs into groups of *gpus_per_experiment*."""
        indices = [g.index for g in self.available_gpus]
        groups = []
        for i in range(0, len(indices), self.gpus_per_experiment):
            group = indices[i : i + self.gpus_per_experiment]
            if len(group) == self.gpus_per_experiment:
                groups.append(group)
        return groups

    def run(
        self,
        jobs: List[ExperimentJob],
        worker_fn: Callable[[List[ExperimentJob], List[int]], List[Tuple[str, Any]]],
    ) -> List[Tuple[str, Any]]:
        """Execute *jobs* in parallel across GPU groups."""
        results: List[Tuple[str, Any]] = []

        if self.max_parallel <= 0:
            logger.warning("No GPU groups available. Running jobs sequentially on CPU.")
            results = worker_fn(jobs, [])
            return results

        num_groups = len(self.gpu_groups)
        group_jobs: List[List[ExperimentJob]] = [[] for _ in range(num_groups)]
        for i, job in enumerate(jobs):
            group_jobs[i % num_groups].append(job)

        for g, gj in enumerate(group_jobs):
            gpu_indices = self.gpu_groups[g]
            logger.info(
                "GPU group %d (GPUs %s): assigned %d jobs.",
                g, gpu_indices, len(gj),
            )

        import multiprocessing as mp
        ctx = mp.get_context("spawn")

        import tempfile, pickle
        procs: List[Tuple[mp.Process, str, List[ExperimentJob]]] = []

        for g in range(num_groups):
            if not group_jobs[g]:
                continue
            gpu_indices = self.gpu_groups[g]
            result_path = tempfile.mktemp(suffix=f"_gpu{g}.pkl")
            proc = ctx.Process(
                target=_spawn_worker,
                args=(worker_fn, group_jobs[g], gpu_indices, result_path),
            )
            procs.append((proc, result_path, group_jobs[g]))
            proc.start()
            logger.info("Spawned process for GPU group %d (pid=%d).", g, proc.pid)

        for proc, result_path, gj in procs:
            proc.join()
            try:
                with open(result_path, "rb") as f:
                    group_results = pickle.load(f)
                results.extend(group_results)
            except (EOFError, FileNotFoundError, pickle.UnpicklingError):
                logger.error(
                    "GPU group process (pid=%d) exited with code %s and no results.",
                    proc.pid, proc.exitcode,
                )
                for job in gj:
                    results.append((job.job_id, None))
            finally:
                try:
                    os.unlink(result_path)
                except OSError:
                    pass

        return results


def _spawn_worker(worker_fn, jobs, gpu_indices, result_path):
    """Top-level target for spawned subprocesses."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))
    import pickle
    try:
        results = worker_fn(jobs, gpu_indices)
        with open(result_path, "wb") as f:
            pickle.dump(results, f)
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).exception(
            "GPU group (GPUs %s) failed: %s", gpu_indices, exc,
        )
        with open(result_path, "wb") as f:
            pickle.dump([(j.job_id, None) for j in jobs], f)
