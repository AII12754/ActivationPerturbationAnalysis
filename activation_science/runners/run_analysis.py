"""Unified analysis runner.

Usage:
    python -m activation_science.runners.run_analysis e0
    python -m activation_science.runners.run_analysis e0 --results ./results_decode --figures ./results_decode/figures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Callable, Dict, Type

import yaml

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Analysis registry
# -------------------------------------------------------------------
ANALYSIS_REGISTRY: Dict[str, Callable] = {}


def _register_analyses():
    """Lazily import and register all analysis runners."""
    if ANALYSIS_REGISTRY:
        return

    from ..analysis.per_experiment.e0_plots import run_all_decode_analyses
    from ..analysis.per_experiment.e1_plots import run_all_residual_analyses
    from ..analysis.per_experiment.e2_plots import run_all_logit_lens_analyses
    from ..analysis.per_experiment.e3_plots import run_all_geometry_analyses
    from ..analysis.per_experiment.e4_plots import run_all_cross_sequence_analyses
    from ..analysis.per_experiment.e5_plots import run_all_state_analyses
    from ..analysis.per_experiment.e6_plots import run_all_causal_analyses
    from ..analysis.per_experiment.e7_plots import run_all_perturbation_analyses
    from ..analysis.per_experiment.e8_plots import run_all_token_type_analyses

    ANALYSIS_REGISTRY["e0"] = run_all_decode_analyses
    ANALYSIS_REGISTRY["e1"] = run_all_residual_analyses
    ANALYSIS_REGISTRY["e2"] = run_all_logit_lens_analyses
    ANALYSIS_REGISTRY["e3"] = run_all_geometry_analyses
    ANALYSIS_REGISTRY["e4"] = run_all_cross_sequence_analyses
    ANALYSIS_REGISTRY["e5"] = run_all_state_analyses
    ANALYSIS_REGISTRY["e6"] = run_all_causal_analyses
    ANALYSIS_REGISTRY["e7"] = run_all_perturbation_analyses
    ANALYSIS_REGISTRY["e8"] = run_all_token_type_analyses


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    _register_analyses()

    parser = argparse.ArgumentParser(description="Run activation science analysis.")
    parser.add_argument(
        "experiment",
        choices=list(ANALYSIS_REGISTRY.keys()),
        help="Experiment ID.",
    )
    parser.add_argument("--results", default=None, help="Results directory path.")
    parser.add_argument("--figures", default=None, help="Figures output directory path.")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    analysis_fn = ANALYSIS_REGISTRY[args.experiment]

    # Determine default paths
    default_results = {
        "e0": "./results_decode_similarity",
        "e1": "./results_residual",
        "e2": "./results_logit_lens",
        "e3": "./results_geometry",
        "e4": "./results_cross_sequence",
        "e5": "./results_state",
        "e6": "./results_causal",
        "e7": "./results_perturbation",
        "e8": "./results_token_type",
    }
    results_dir = args.results or default_results.get(args.experiment, f"./results_{args.experiment}")
    figures_dir = args.figures or os.path.join(results_dir, "figures")

    logger.info("Running analysis for %s: results=%s, figures=%s", args.experiment, results_dir, figures_dir)
    analysis_fn(results_dir, figures_dir)


if __name__ == "__main__":
    main()
