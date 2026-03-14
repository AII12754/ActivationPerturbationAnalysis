"""
Storage and checkpointing for residual stream decomposition experiments.

Single Parquet table family:
  - residual_part_*.parquet — one row per (experiment, phase, step, layer)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Schema
# -------------------------------------------------------------------
RESIDUAL_COLUMNS = [
    "experiment_id",
    "model_id",
    "dataset_name",
    "prompt_id",
    "context_length",
    "num_decode_tokens",
    "phase",
    "step",
    "layer",
    "delta_norm",
    "residual_norm",
    "delta_residual_ratio",
    "cos_delta_residual",
    "cos_delta_embedding",
    "cos_delta_prev_delta",
    "delta_norm_std",
    "residual_norm_std",
]


class ResidualResultStore:
    """Append-friendly storage for residual stream decomposition results."""

    def __init__(
        self,
        output_dir: str = "./results_residual",
        format: str = "parquet",
        checkpoint_every: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.checkpoint_every = checkpoint_every

        self._buffer: List[Dict[str, Any]] = []
        self._flushed = 0

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
    def add_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        for rec in records:
            row = {col: None for col in RESIDUAL_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(rec)
            self._buffer.append(row)

    def mark_completed(self, experiment_id: str):
        self._completed_ids.add(experiment_id)
        if len(self._completed_ids) % self.checkpoint_every == 0:
            self.flush()

    def flush(self):
        """Write buffered records to disk and update checkpoint."""
        if self._buffer:
            df = pd.DataFrame(self._buffer, columns=RESIDUAL_COLUMNS)
            path = self.output_dir / f"residual_part_{self._flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Flushed %d residual records to %s.", len(self._buffer), path)
            self._buffer.clear()
            self._flushed += 1

        self._save_checkpoint()

    # ---------------------------------------------------------------
    # Reading results (for analysis)
    # ---------------------------------------------------------------
    @staticmethod
    def load_results(output_dir: str = "./results_residual") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("residual_part_*.parquet"))
        if not files:
            logger.warning("No residual files found in %s", output_dir)
            return pd.DataFrame(columns=RESIDUAL_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d residual records from %s.", len(df), output_dir)
        return df
