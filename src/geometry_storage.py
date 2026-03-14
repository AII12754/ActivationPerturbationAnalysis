"""
Storage and checkpointing for representation geometry experiments.

Two separate Parquet table families:
  - summary_part_*.parquet  — one row per (experiment, layer)
  - spectrum_part_*.parquet — one row per (experiment, layer, sv_index)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "experiment_id",
    "model_id",
    "dataset_name",
    "prompt_id",
    "context_length",
    "layer",
    "effective_rank",
    "participation_ratio",
    "mean_pairwise_cosine",
    "norm_mean",
    "norm_std",
    "norm_min",
    "norm_max",
    "num_tokens",
    "top1_explained_var",
    "top5_explained_var",
    "top10_explained_var",
    "top20_explained_var",
    "top50_explained_var",
]

SPECTRUM_COLUMNS = [
    "experiment_id",
    "model_id",
    "dataset_name",
    "prompt_id",
    "context_length",
    "layer",
    "sv_index",
    "singular_value",
    "explained_variance_ratio",
    "cumulative_variance_ratio",
]


class GeometryResultStore:
    """Append-friendly storage for geometry analysis results."""

    def __init__(
        self,
        output_dir: str = "./results_geometry",
        format: str = "parquet",
        checkpoint_every: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.checkpoint_every = checkpoint_every

        self._summary_buffer: List[Dict[str, Any]] = []
        self._spectrum_buffer: List[Dict[str, Any]] = []
        self._summary_flushed = 0
        self._spectrum_flushed = 0

        self._checkpoint_path = self.output_dir / "checkpoint.json"
        self._completed_ids: set = self._load_checkpoint()

    # ---------------------------------------------------------------
    # Checkpoint management
    # ---------------------------------------------------------------
    def _load_checkpoint(self) -> set:
        if self._checkpoint_path.exists():
            data = json.loads(self._checkpoint_path.read_text())
            ids = set(data.get("completed_experiments", []))
            logger.info("Loaded checkpoint with %d completed experiments.", len(ids))
            return ids
        return set()

    def _save_checkpoint(self):
        data = {"completed_experiments": sorted(self._completed_ids)}
        self._checkpoint_path.write_text(json.dumps(data, indent=2))

    def is_completed(self, experiment_id: str) -> bool:
        return experiment_id in self._completed_ids

    # ---------------------------------------------------------------
    # Writing results
    # ---------------------------------------------------------------
    def add_summary_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        for rec in records:
            row = {col: None for col in SUMMARY_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(rec)
            self._summary_buffer.append(row)

    def add_spectrum_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        for rec in records:
            row = {col: None for col in SPECTRUM_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(rec)
            self._spectrum_buffer.append(row)

    def mark_completed(self, experiment_id: str):
        self._completed_ids.add(experiment_id)
        if len(self._completed_ids) % self.checkpoint_every == 0:
            self.flush()

    def flush(self):
        """Write buffered records to disk and update checkpoint."""
        if self._summary_buffer:
            df = pd.DataFrame(self._summary_buffer, columns=SUMMARY_COLUMNS)
            path = self.output_dir / f"summary_part_{self._summary_flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Flushed %d summary records to %s.", len(self._summary_buffer), path)
            self._summary_buffer.clear()
            self._summary_flushed += 1

        if self._spectrum_buffer:
            df = pd.DataFrame(self._spectrum_buffer, columns=SPECTRUM_COLUMNS)
            path = self.output_dir / f"spectrum_part_{self._spectrum_flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Flushed %d spectrum records to %s.", len(self._spectrum_buffer), path)
            self._spectrum_buffer.clear()
            self._spectrum_flushed += 1

        self._save_checkpoint()

    # ---------------------------------------------------------------
    # Reading results (for analysis)
    # ---------------------------------------------------------------
    @staticmethod
    def load_summary(output_dir: str = "./results_geometry") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("summary_part_*.parquet"))
        if not files:
            logger.warning("No summary files found in %s", output_dir)
            return pd.DataFrame(columns=SUMMARY_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d summary records from %s.", len(df), output_dir)
        return df

    @staticmethod
    def load_spectrum(output_dir: str = "./results_geometry") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("spectrum_part_*.parquet"))
        if not files:
            logger.warning("No spectrum files found in %s", output_dir)
            return pd.DataFrame(columns=SPECTRUM_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d spectrum records from %s.", len(df), output_dir)
        return df
