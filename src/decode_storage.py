"""
Storage and checkpointing for decode-time similarity experiments.

Two separate Parquet table families:
  - topk_part_*.parquet  — one row per (experiment, decode_step, layer, rank)
  - agg_part_*.parquet   — one row per (experiment, decode_step, layer)
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
TOPK_COLUMNS = [
    "experiment_id",
    "model_id",
    "dataset_name",
    "prompt_id",
    "context_length",
    "num_decode_tokens",
    "decode_step",
    "token_id",
    "token_text",
    "layer",
    "ref_rank",
    "ref_token_position",
    "ref_token_id",
    "ref_token_text",
    "ref_token_distance",
    "ref_similarity",
]

AGG_COLUMNS = [
    "experiment_id",
    "model_id",
    "dataset_name",
    "prompt_id",
    "context_length",
    "num_decode_tokens",
    "decode_step",
    "token_id",
    "token_text",
    "layer",
    "mean_similarity",
    "median_similarity",
    "max_similarity",
    "std_similarity",
    "frac_above_090",
    "frac_above_095",
    "frac_above_098",
    "num_historical_tokens",
]


class DecodeResultStore:
    """Append-friendly storage for decode similarity results."""

    def __init__(
        self,
        output_dir: str = "./results_decode",
        format: str = "parquet",
        checkpoint_every: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.checkpoint_every = checkpoint_every

        self._topk_buffer: List[Dict[str, Any]] = []
        self._agg_buffer: List[Dict[str, Any]] = []
        self._topk_flushed = 0
        self._agg_flushed = 0

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

    def add_aggregate_records(
        self,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        for rec in records:
            row = {col: None for col in AGG_COLUMNS}
            row["experiment_id"] = experiment_id
            row.update(rec)
            self._agg_buffer.append(row)

    def mark_completed(self, experiment_id: str):
        self._completed_ids.add(experiment_id)
        if len(self._completed_ids) % self.checkpoint_every == 0:
            self.flush()

    def flush(self):
        """Write buffered records to disk and update checkpoint."""
        if self._topk_buffer:
            df = pd.DataFrame(self._topk_buffer, columns=TOPK_COLUMNS)
            path = self.output_dir / f"topk_part_{self._topk_flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Flushed %d topk records to %s.", len(self._topk_buffer), path)
            self._topk_buffer.clear()
            self._topk_flushed += 1

        if self._agg_buffer:
            df = pd.DataFrame(self._agg_buffer, columns=AGG_COLUMNS)
            path = self.output_dir / f"agg_part_{self._agg_flushed:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Flushed %d agg records to %s.", len(self._agg_buffer), path)
            self._agg_buffer.clear()
            self._agg_flushed += 1

        self._save_checkpoint()

    # ---------------------------------------------------------------
    # Reading results (for analysis)
    # ---------------------------------------------------------------
    @staticmethod
    def load_topk(output_dir: str = "./results_decode") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("topk_part_*.parquet"))
        if not files:
            logger.warning("No topk files found in %s", output_dir)
            return pd.DataFrame(columns=TOPK_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d topk records from %s.", len(df), output_dir)
        return df

    @staticmethod
    def load_aggregates(output_dir: str = "./results_decode") -> pd.DataFrame:
        p = Path(output_dir)
        files = sorted(p.glob("agg_part_*.parquet"))
        if not files:
            logger.warning("No agg files found in %s", output_dir)
            return pd.DataFrame(columns=AGG_COLUMNS)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d aggregate records from %s.", len(df), output_dir)
        return df
