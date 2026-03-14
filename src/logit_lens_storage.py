"""
Storage and checkpointing for logit lens experiments.

Two separate Parquet table families:
  - perstep_part_*.parquet — one row per (experiment, decode_step, layer)
  - topk_part_*.parquet    — one row per (experiment, decode_step, layer, rank)
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
PERSTEP_COLUMNS = [
    "experiment_id",
    "model_id",
    "dataset_name",
    "prompt_id",
    "context_length",
    "decode_step",
    "layer",
    "entropy",
    "kl_from_final",
    "rank_of_correct",
    "cross_entropy_correct",
    "max_prob",
    "top1_token_id",
    "correct_token_id",
    "generated_token_id",
]

TOPK_COLUMNS = [
    "experiment_id",
    "model_id",
    "dataset_name",
    "prompt_id",
    "context_length",
    "decode_step",
    "layer",
    "rank",
    "token_id",
    "logit",
    "probability",
]


class LogitLensResultStore:
    """Append-friendly storage for logit lens results."""

    def __init__(
        self,
        output_dir: str = "./results_logit_lens",
        format: str = "parquet",
        checkpoint_every: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.checkpoint_every = checkpoint_every

        self._perstep_buffer: List[Dict[str, Any]] = []
        self._topk_buffer: List[Dict[str, Any]] = []
        self._perstep_flushed = 0
        self._topk_flushed = 0

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
    def add_perstep_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        for rec in records:
            row = {col: None for col in PERSTEP_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(rec)
            self._perstep_buffer.append(row)

    def add_topk_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        for rec in records:
            row = {col: None for col in TOPK_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(rec)
            self._topk_buffer.append(row)

    def mark_completed(self, experiment_id: str):
        self._completed_ids.add(experiment_id)
        if len(self._completed_ids) % self.checkpoint_every == 0:
            self.flush()

    def flush(self):
        """Write buffered records to disk and update checkpoint."""
        if self._perstep_buffer:
            df = pd.DataFrame(self._perstep_buffer, columns=PERSTEP_COLUMNS)
            path = self.output_dir / f"perstep_part_{self._perstep_flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Flushed %d perstep records to %s.", len(self._perstep_buffer), path)
            self._perstep_buffer.clear()
            self._perstep_flushed += 1

        if self._topk_buffer:
            df = pd.DataFrame(self._topk_buffer, columns=TOPK_COLUMNS)
            path = self.output_dir / f"topk_part_{self._topk_flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Flushed %d topk records to %s.", len(self._topk_buffer), path)
            self._topk_buffer.clear()
            self._topk_flushed += 1

        self._save_checkpoint()

    # ---------------------------------------------------------------
    # Reading results (for analysis)
    # ---------------------------------------------------------------
    @staticmethod
    def load_perstep(output_dir: str = "./results_logit_lens") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("perstep_part_*.parquet"))
        if not files:
            logger.warning("No perstep files found in %s", output_dir)
            return pd.DataFrame(columns=PERSTEP_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d perstep records from %s.", len(df), output_dir)
        return df

    @staticmethod
    def load_topk(output_dir: str = "./results_logit_lens") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("topk_part_*.parquet"))
        if not files:
            logger.warning("No topk files found in %s", output_dir)
            return pd.DataFrame(columns=TOPK_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d topk records from %s.", len(df), output_dir)
        return df
