"""
GPU resource detection and experiment scheduling.

Uses ``pynvml`` to query per-GPU memory and exposes a simple pool-based
scheduler that assigns experiment jobs to available GPUs.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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
                i,
                name,
                info.free_mb,
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
# Simple experiment scheduler
# -------------------------------------------------------------------
@dataclass
class ExperimentJob:
    """Describes a single experiment to be scheduled."""
    job_id: str
    params: Dict[str, Any]
    gpus_required: int = 1              # number of GPUs needed


class ExperimentScheduler:
    """A pool-based scheduler that maps jobs to available GPUs.

    For single-GPU jobs it launches one process per GPU.
    For multi-GPU jobs it batches GPUs together.
    """

    def __init__(
        self,
        available_gpus: List[GPUInfo],
        max_parallel: int = 4,
        gpus_per_experiment: int = 1,
    ):
        self.available_gpus = available_gpus
        self.max_parallel = min(max_parallel, len(available_gpus) // gpus_per_experiment)
        self.gpus_per_experiment = gpus_per_experiment
        # Partition GPUs into groups.
        self.gpu_groups = self._partition_gpus()
        logger.info(
            "Scheduler: %d GPU groups of size %d, max_parallel=%d",
            len(self.gpu_groups),
            gpus_per_experiment,
            self.max_parallel,
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
        worker_fn: Callable[[ExperimentJob, List[int]], Any],
    ) -> List[Tuple[str, Any]]:
        """Execute *jobs* in parallel across GPU groups.

        Parameters
        ----------
        jobs:
            List of experiment jobs.
        worker_fn:
            ``worker_fn(job, gpu_indices) -> result``.  Will be called in
            a subprocess with ``CUDA_VISIBLE_DEVICES`` set appropriately.

        Returns
        -------
        List of ``(job_id, result)`` tuples in completion order.
        """
        results: List[Tuple[str, Any]] = []

        if self.max_parallel <= 0:
            logger.warning("No GPU groups available. Running jobs sequentially on CPU.")
            for job in jobs:
                res = worker_fn(job, [])
                results.append((job.job_id, res))
            return results

        # Use a thread-based approach to manage GPU group allocation,
        # but run the actual work via worker_fn which should handle its
        # own torch device placement.
        from concurrent.futures import ThreadPoolExecutor

        gpu_group_semaphore: List[bool] = [True] * len(self.gpu_groups)

        def _acquire_group() -> Optional[int]:
            for idx, free in enumerate(gpu_group_semaphore):
                if free:
                    gpu_group_semaphore[idx] = False
                    return idx
            return None

        def _release_group(idx: int):
            gpu_group_semaphore[idx] = True

        def _run_job(job: ExperimentJob):
            # Busy-wait for a free GPU group (simple but effective).
            group_idx = None
            while group_idx is None:
                group_idx = _acquire_group()
                if group_idx is None:
                    time.sleep(1)
            gpu_indices = self.gpu_groups[group_idx]
            try:
                # Set env var so the worker uses the right GPUs.
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))
                result = worker_fn(job, gpu_indices)
                return job.job_id, result
            finally:
                _release_group(group_idx)

        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = {pool.submit(_run_job, job): job for job in jobs}
            for future in as_completed(futures):
                job_id, result = future.result()
                results.append((job_id, result))
                logger.info("Job %s completed.", job_id)

        return results
