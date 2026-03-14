"""Shared types, constants, and data classes used across the framework."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

# -------------------------------------------------------------------
# Dtype mapping — eliminates 6× duplication across experiments
# -------------------------------------------------------------------
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int8": torch.int8,
}


def resolve_dtype(name: str, default: torch.dtype = torch.float32) -> torch.dtype:
    """Resolve a string dtype name to a ``torch.dtype``."""
    return DTYPE_MAP.get(name, default)


# -------------------------------------------------------------------
# Experiment job descriptor (moved from src/scheduler.py)
# -------------------------------------------------------------------
@dataclass
class ExperimentJob:
    """Describes a single experiment to be scheduled."""
    job_id: str
    params: Dict[str, Any]
    gpus_required: int = 1


# -------------------------------------------------------------------
# Table schema descriptor
# -------------------------------------------------------------------
@dataclass
class TableSchema:
    """Describes a Parquet table's column layout."""
    name: str
    columns: List[str]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def make_experiment_id(params: Dict[str, Any]) -> str:
    """Deterministic experiment id from a parameter dict (SHA256, 16 hex chars)."""
    key = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def resolve_dataset_list(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return the list of dataset specs to sweep over.

    Each entry is a dict with keys: name, config, split.
    If ``prompts.datasets`` is not set, falls back to the single ``dataset_name``.
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
