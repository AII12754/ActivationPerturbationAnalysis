"""
Storage and checkpointing for cross-sequence alignment experiments.

Single Parquet table family:
  - alignment_part_*.parquet — one row per (experiment, layer)
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
ALIGNMENT_COLUMNS = [
    "experiment_id",
    "model_id",
    "pair_type",
    "dataset_a",
    "dataset_b",
    "prompt_id_a",
    "prompt_id_b",
    "context_length",
    "layer",
    "cka",
    "procrustes_distance",
    "subspace_overlap",
    "centroid_cosine",
    "shared_token_cosine",
    "num_shared_tokens",
    "seq_len_a",
    "seq_len_b",
]


class CrossSequenceResultStore:
    """Append-friendly storage for cross-sequence alignment results."""

    def __init__(
        self,
        output_dir: str = "./results_cross_sequence",
        format: str = "parquet",
        checkpoint_every: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.checkpoint_every = checkpoint_every

        self._alignment_buffer: List[Dict[str, Any]] = []
        self._alignment_flushed = 0

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
    def add_alignment_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        for rec in records:
            row = {col: None for col in ALIGNMENT_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(rec)
            self._alignment_buffer.append(row)

    def mark_completed(self, experiment_id: str):
        self._completed_ids.add(experiment_id)
        if len(self._completed_ids) % self.checkpoint_every == 0:
            self.flush()

    def flush(self):
        """Write buffered records to disk and update checkpoint."""
        if self._alignment_buffer:
            df = pd.DataFrame(self._alignment_buffer, columns=ALIGNMENT_COLUMNS)
            path = self.output_dir / f"alignment_part_{self._alignment_flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info(
                "Flushed %d alignment records to %s.",
                len(self._alignment_buffer),
                path,
            )
            self._alignment_buffer.clear()
            self._alignment_flushed += 1

        self._save_checkpoint()

    # ---------------------------------------------------------------
    # Reading results (for analysis)
    # ---------------------------------------------------------------
    @staticmethod
    def load_alignments(output_dir: str = "./results_cross_sequence") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("alignment_part_*.parquet"))
        if not files:
            logger.warning("No alignment files found in %s", output_dir)
            return pd.DataFrame(columns=ALIGNMENT_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d alignment records from %s.", len(df), output_dir)
        return df
