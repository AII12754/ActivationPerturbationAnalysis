"""Cross-experiment analysis: Semantic bottleneck identification.

Uses E4 cross-sequence alignment (CKA) data to identify the layer where
representations are maximally aligned across different input prompts,
indicating a semantic bottleneck in the network.

Usage:
    from activation_science.analysis.cross_experiment.semantic_bottleneck import (
        SemanticBottleneckAnalysis,
    )
    analysis = SemanticBottleneckAnalysis(
        e4_results_dir="./results_e4",
        figures_dir="./results_cross/figures",
    )
    analysis.run_all()
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from ...core.storage import ExperimentStore

logger = logging.getLogger(__name__)


class SemanticBottleneckAnalysis:
    """Identify semantic bottleneck layers from cross-sequence CKA alignment."""

    def __init__(
        self,
        e4_results_dir: str,
        figures_dir: Optional[str] = None,
    ):
        self.e4_results_dir = e4_results_dir
        self.figures_dir = figures_dir or "./results_cross/figures"
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load E4 cross-sequence alignment data."""
        return ExperimentStore.load_table(self.e4_results_dir, "alignment")

    def find_peak_cka_layer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find the layer with maximum CKA similarity for each experiment pair.

        The peak CKA layer indicates where two different sequences are most
        similar in representation space — the semantic bottleneck.
        """
        required = {"experiment_id", "layer", "cka"}
        if not required.issubset(df.columns):
            logger.warning("E4 data missing required columns: %s", required - set(df.columns))
            return pd.DataFrame()

        results = []
        for exp_id, group in df.groupby("experiment_id"):
            profile = group.groupby("layer")["cka"].mean().sort_index()
            peak_idx = profile.values.argmax()
            results.append({
                "experiment_id": exp_id,
                "peak_cka_layer": profile.index[peak_idx],
                "peak_cka_value": profile.values[peak_idx],
                "mean_cka": profile.values.mean(),
                "std_cka": profile.values.std(),
            })
        return pd.DataFrame(results)

    def compute_bottleneck_statistics(self, peaks_df: pd.DataFrame) -> dict:
        """Aggregate bottleneck statistics across all experiment pairs."""
        if peaks_df.empty:
            return {}

        return {
            "mean_bottleneck_layer": peaks_df["peak_cka_layer"].mean(),
            "std_bottleneck_layer": peaks_df["peak_cka_layer"].std(),
            "median_bottleneck_layer": peaks_df["peak_cka_layer"].median(),
            "mean_peak_cka": peaks_df["peak_cka_value"].mean(),
            "n_experiments": len(peaks_df),
        }

    def run_all(self):
        """Run full semantic bottleneck identification pipeline."""
        logger.info("Running semantic bottleneck analysis...")
        try:
            df = self.load_data()
        except Exception as e:
            logger.error("Failed to load E4 data: %s", e)
            return

        peaks = self.find_peak_cka_layer(df)
        stats = self.compute_bottleneck_statistics(peaks)

        if peaks.empty:
            logger.warning("No bottleneck layers detected.")
            return

        out_path = os.path.join(self.figures_dir, "semantic_bottleneck.csv")
        peaks.to_csv(out_path, index=False)
        logger.info("Semantic bottleneck results saved to %s", out_path)
        logger.info("Bottleneck statistics: %s", stats)
