"""Unified multi-table Parquet storage with checkpointing.

Replaces the 6 separate *_storage.py modules with a single ``ExperimentStore``
that supports registering N tables with different schemas.  Checkpoint format
is identical to the existing ``{"completed_experiments": [...]}`` for backward
compatibility with ``results_decode/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentStore:
    """Append-friendly, multi-table Parquet storage with checkpoint/resume.

    Usage
    -----
    >>> store = ExperimentStore("./results_decode")
    >>> store.register_table("topk", TOPK_COLUMNS)
    >>> store.register_table("agg", AGG_COLUMNS)
    >>> store.add_records("topk", experiment_id, records)
    >>> store.add_records("agg", experiment_id, records)
    >>> store.mark_completed(experiment_id)
    >>> store.flush()
    """

    def __init__(
        self,
        output_dir: str,
        checkpoint_every: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every

        # Per-table state
        self._tables: Dict[str, List[str]] = {}      # name -> columns
        self._buffers: Dict[str, List[Dict]] = {}     # name -> buffered rows
        self._flushed: Dict[str, int] = {}             # name -> file counter

        # Checkpoint
        self._checkpoint_path = self.output_dir / "checkpoint.json"
        self._completed_ids: set = self._load_checkpoint()

    # ---------------------------------------------------------------
    # Table registration
    # ---------------------------------------------------------------
    def register_table(self, name: str, columns: List[str]):
        """Register a table with the given column schema."""
        self._tables[name] = columns
        self._buffers[name] = []
        self._flushed[name] = self._count_existing_parts(name)

    def _count_existing_parts(self, name: str) -> int:
        """Count existing part files so we don't overwrite on resume."""
        existing = list(self.output_dir.glob(f"{name}_part_*.parquet"))
        return len(existing)

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
        table_name: str,
        experiment_id: str,
        records: List[Dict[str, Any]],
    ):
        """Buffer records for a table, filling missing columns with None."""
        columns = self._tables[table_name]
        buf = self._buffers[table_name]
        for rec in records:
            row = {col: None for col in columns}
            row["experiment_id"] = experiment_id
            row.update(rec)
            buf.append(row)

    def mark_completed(self, experiment_id: str):
        """Mark an experiment as completed and periodically flush."""
        self._completed_ids.add(experiment_id)
        if len(self._completed_ids) % self.checkpoint_every == 0:
            self.flush()

    def flush(self):
        """Write all buffered tables to disk and update checkpoint."""
        for name, buf in self._buffers.items():
            if not buf:
                continue
            columns = self._tables[name]
            df = pd.DataFrame(buf, columns=columns)
            idx = self._flushed[name]
            path = self.output_dir / f"{name}_part_{idx:06d}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Flushed %d %s records to %s.", len(buf), name, path)
            buf.clear()
            self._flushed[name] = idx + 1

        self._save_checkpoint()

    # ---------------------------------------------------------------
    # Reading results (for analysis) — static methods
    # ---------------------------------------------------------------
    @staticmethod
    def load_table(
        output_dir: str,
        table_name: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load all partitions of a table from disk.

        Parameters
        ----------
        output_dir:
            Directory containing the parquet files.
        table_name:
            Name prefix (e.g. ``"topk"``, ``"agg"``).
        columns:
            Optional column list for empty DataFrame fallback.
        """
        p = Path(output_dir)
        files = sorted(p.glob(f"{table_name}_part_*.parquet"))
        if not files:
            logger.warning("No %s files found in %s", table_name, output_dir)
            return pd.DataFrame(columns=columns or [])
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        logger.info("Loaded %d %s records from %s.", len(df), table_name, output_dir)
        return df
