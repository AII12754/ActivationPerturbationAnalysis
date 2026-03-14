"""
Storage and checkpointing for activation state detection experiments.

Single Parquet table family:
  - trajectory_part_*.parquet — one row per (experiment, decode_step, layer)
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
TRAJECTORY_COLUMNS = [
    "experiment_id",
    "model_id",
    "dataset_name",
    "prompt_id",
    "context_length",
    "decode_step",
    "layer",
    "token_id",
    "norm",
    "centroid_distance",
    "step_cosine",
    "pc0", "pc1", "pc2", "pc3", "pc4",
    "pc5", "pc6", "pc7", "pc8", "pc9",
    "pc10", "pc11", "pc12", "pc13", "pc14",
    "pc15", "pc16", "pc17", "pc18", "pc19",
]


class StateResultStore:
    """Append-friendly storage for state detection trajectory results."""

    def __init__(
        self,
        output_dir: str = "./results_state",
        format: str = "parquet",
        checkpoint_every: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.checkpoint_every = checkpoint_every

        self._trajectory_buffer: List[Dict[str, Any]] = []
        self._trajectory_flushed = 0

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
    def add_trajectory_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        for rec in records:
            row = {col: None for col in TRAJECTORY_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(rec)
            self._trajectory_buffer.append(row)

    def mark_completed(self, experiment_id: str):
        self._completed_ids.add(experiment_id)
        if len(self._completed_ids) % self.checkpoint_every == 0:
            self.flush()

    def flush(self):
        """Write buffered records to disk and update checkpoint."""
        if self._trajectory_buffer:
            df = pd.DataFrame(self._trajectory_buffer, columns=TRAJECTORY_COLUMNS)
            path = self.output_dir / f"trajectory_part_{self._trajectory_flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info(
                "Flushed %d trajectory records to %s.",
                len(self._trajectory_buffer),
                path,
            )
            self._trajectory_buffer.clear()
            self._trajectory_flushed += 1

        self._save_checkpoint()

    # ---------------------------------------------------------------
    # Reading results (for analysis)
    # ---------------------------------------------------------------
    @staticmethod
    def load_trajectories(output_dir: str = "./results_state") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("trajectory_part_*.parquet"))
        if not files:
            logger.warning("No trajectory files found in %s", output_dir)
            return pd.DataFrame(columns=TRAJECTORY_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d trajectory records from %s.", len(df), output_dir)
        return df
