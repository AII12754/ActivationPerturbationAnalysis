"""Cross-experiment analysis: Phase boundary detection.

Correlates E1 residual update profiles with E2 logit lens crystallization
layers to identify phase boundaries in the transformer's processing pipeline.

Usage:
    from activation_science.analysis.cross_experiment.phase_boundary_detection import (
        PhaseBoundaryAnalysis,
    )
    analysis = PhaseBoundaryAnalysis(
        e1_results_dir="./results_e1",
        e2_results_dir="./results_e2",
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


class PhaseBoundaryAnalysis:
    """Detect phase boundaries by correlating residual updates with logit lens metrics."""

    def __init__(
        self,
        e1_results_dir: str,
        e2_results_dir: str,
        figures_dir: Optional[str] = None,
    ):
        self.e1_results_dir = e1_results_dir
        self.e2_results_dir = e2_results_dir
        self.figures_dir = figures_dir or "./results_cross/figures"
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load E1 residual and E2 logit lens data."""
        e1_df = ExperimentStore.load_table(self.e1_results_dir, "residual")
        e2_df = ExperimentStore.load_table(self.e2_results_dir, "perstep")
        return e1_df, e2_df

    def find_update_peaks(self, e1_df: pd.DataFrame) -> pd.DataFrame:
        """Identify layers where residual update magnitude peaks.

        Computes per-experiment layer-wise delta_norm profiles and finds
        local maxima that indicate phase transitions.
        """
        required = {"experiment_id", "layer", "delta_norm"}
        if not required.issubset(e1_df.columns):
            logger.warning("E1 data missing required columns: %s", required - set(e1_df.columns))
            return pd.DataFrame()

        peaks = []
        for exp_id, group in e1_df.groupby("experiment_id"):
            profile = group.groupby("layer")["delta_norm"].mean().sort_index()
            values = profile.values
            # Simple peak detection: local maxima
            for i in range(1, len(values) - 1):
                if values[i] > values[i - 1] and values[i] > values[i + 1]:
                    peaks.append({
                        "experiment_id": exp_id,
                        "peak_layer": profile.index[i],
                        "delta_norm": values[i],
                    })
        return pd.DataFrame(peaks)

    def find_crystallization_layers(self, e2_df: pd.DataFrame) -> pd.DataFrame:
        """Identify layers where logit lens predictions crystallize.

        Crystallization = layer where KL divergence from final drops below
        a threshold and stays low.
        """
        required = {"experiment_id", "layer", "kl_from_final"}
        if not required.issubset(e2_df.columns):
            logger.warning("E2 data missing required columns: %s", required - set(e2_df.columns))
            return pd.DataFrame()

        crystals = []
        for exp_id, group in e2_df.groupby("experiment_id"):
            profile = group.groupby("layer")["kl_from_final"].mean().sort_index()
            values = profile.values
            # Find first layer where KL drops below median and stays below
            median_kl = np.median(values)
            for i in range(len(values)):
                if values[i] < median_kl and all(v < median_kl for v in values[i:]):
                    crystals.append({
                        "experiment_id": exp_id,
                        "crystallization_layer": profile.index[i],
                        "kl_at_crystal": values[i],
                    })
                    break
        return pd.DataFrame(crystals)

    def correlate_boundaries(
        self, peaks_df: pd.DataFrame, crystals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Correlate update peaks with crystallization layers."""
        if peaks_df.empty or crystals_df.empty:
            logger.warning("Cannot correlate: one or both DataFrames are empty.")
            return pd.DataFrame()

        merged = peaks_df.merge(crystals_df, on="experiment_id", how="inner")
        if merged.empty:
            return merged

        merged["peak_crystal_gap"] = merged["peak_layer"] - merged["crystallization_layer"]
        return merged

    def run_all(self):
        """Run full phase boundary analysis pipeline."""
        logger.info("Running phase boundary detection analysis...")
        try:
            e1_df, e2_df = self.load_data()
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            return

        peaks = self.find_update_peaks(e1_df)
        crystals = self.find_crystallization_layers(e2_df)
        boundaries = self.correlate_boundaries(peaks, crystals)

        if not boundaries.empty:
            out_path = os.path.join(self.figures_dir, "phase_boundaries.csv")
            boundaries.to_csv(out_path, index=False)
            logger.info("Phase boundary results saved to %s", out_path)
        else:
            logger.warning("No phase boundaries detected.")
