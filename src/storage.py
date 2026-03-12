"""
Result storage and checkpointing.

Stores experiment results as Parquet files using pandas + pyarrow.
Supports incremental writes and checkpoint/resume semantics.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Schema (column names and types)
# -------------------------------------------------------------------
RESULT_COLUMNS = [
    "experiment_id",
    "model_id",
    "context_length",
    "perturbation_type",
    "perturbation_position_frac",
    "perturbation_position_abs",
    "perturbation_span_length",
    "layer",
    "bucket",
    "offset_start",
    "offset_end",
    "mean_similarity",
    "min_similarity",
    "max_similarity",
    "std_similarity",
    "num_tokens",
]


class ResultStore:
    """Append-friendly result storage backed by Parquet files."""

    def __init__(
        self,
        output_dir: str = "./results",
        format: str = "parquet",
        checkpoint_every: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.checkpoint_every = checkpoint_every

        self._buffer: List[Dict[str, Any]] = []
        self._total_flushed = 0
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
        """Check whether an experiment was already completed (for resume)."""
        return experiment_id in self._completed_ids

    # ---------------------------------------------------------------
    # Writing results
    # ---------------------------------------------------------------
    def add_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Buffer records for one experiment."""
        meta = metadata or {}
        for rec in records:
            row = {col: None for col in RESULT_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(meta)
            row.update(rec)
            self._buffer.append(row)

        self._completed_ids.add(experiment_id)

        if len(self._completed_ids) % self.checkpoint_every == 0:
            self.flush()

    def flush(self):
        """Write buffered records to disk and update checkpoint."""
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer, columns=RESULT_COLUMNS)
        part_idx = self._total_flushed
        if self.format == "parquet":
            path = self.output_dir / f"results_part_{part_idx:06d}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = self.output_dir / f"results_part_{part_idx:06d}.csv"
            df.to_csv(path, index=False)
        logger.info("Flushed %d records to %s.", len(self._buffer), path)
        self._buffer.clear()
        self._total_flushed += 1
        self._save_checkpoint()

    # ---------------------------------------------------------------
    # Reading results (for analysis)
    # ---------------------------------------------------------------
    @staticmethod
    def load_all(output_dir: str = "./results") -> pd.DataFrame:
        """Load and concatenate all result files from *output_dir*."""
        p = Path(output_dir)
        parquet_files = sorted(p.glob("results_part_*.parquet"))
        csv_files = sorted(p.glob("results_part_*.csv"))

        frames = []
        for f in parquet_files:
            frames.append(pd.read_parquet(f))
        for f in csv_files:
            frames.append(pd.read_csv(f))

        if not frames:
            logger.warning("No result files found in %s", output_dir)
            return pd.DataFrame(columns=RESULT_COLUMNS)

        df = pd.concat(frames, ignore_index=True)
        logger.info("Loaded %d total records from %s.", len(df), output_dir)
        return df
