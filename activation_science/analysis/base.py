"""Base analysis ABC.

Provides a common interface for per-experiment analysis modules.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from ..core.storage import ExperimentStore

logger = logging.getLogger(__name__)


class BaseAnalysis(ABC):
    """Abstract base for per-experiment analysis."""

    experiment_id: str = ""
    experiment_name: str = ""

    def __init__(self, results_dir: str, figures_dir: Optional[str] = None):
        self.results_dir = results_dir
        self.figures_dir = figures_dir or os.path.join(results_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_table(self, table_name: str, columns=None) -> pd.DataFrame:
        """Load a table using the unified ExperimentStore."""
        return ExperimentStore.load_table(self.results_dir, table_name, columns)

    @abstractmethod
    def run_all(self):
        """Run all analysis plots for this experiment."""
