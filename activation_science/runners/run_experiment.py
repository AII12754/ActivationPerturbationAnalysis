"""Unified experiment runner.

Usage:
    python -m activation_science.runners.run_experiment e0
    python -m activation_science.runners.run_experiment e0 --config path/to/config.yaml
    python -m activation_science.runners.run_experiment e0 --parallel
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Type

import yaml

from ..experiments.base import BaseExperiment, GenericSweepRunner

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Experiment registry
# -------------------------------------------------------------------
EXPERIMENT_REGISTRY: Dict[str, Type[BaseExperiment]] = {}


def _register_experiments():
    """Lazily import and register all experiment classes."""
    if EXPERIMENT_REGISTRY:
        return

    from ..experiments.e0_decode_similarity import DecodeSimilarityExperiment
    from ..experiments.e1_residual_decomposition import ResidualDecompositionExperiment
    from ..experiments.e2_logit_lens import LogitLensExperiment
    from ..experiments.e3_representation_geometry import RepresentationGeometryExperiment
    from ..experiments.e4_cross_sequence_alignment import CrossSequenceAlignmentExperiment
    from ..experiments.e5_state_detection import StateDetectionExperiment
    from ..experiments.e6_causal_intervention import CausalInterventionExperiment
    from ..experiments.e7_perturbation_sensitivity import PerturbationSensitivityExperiment
    from ..experiments.e8_token_type_analysis import TokenTypeAnalysisExperiment

    EXPERIMENT_REGISTRY["e0"] = DecodeSimilarityExperiment
    EXPERIMENT_REGISTRY["e1"] = ResidualDecompositionExperiment
    EXPERIMENT_REGISTRY["e2"] = LogitLensExperiment
    EXPERIMENT_REGISTRY["e3"] = RepresentationGeometryExperiment
    EXPERIMENT_REGISTRY["e4"] = CrossSequenceAlignmentExperiment
    EXPERIMENT_REGISTRY["e5"] = StateDetectionExperiment
    EXPERIMENT_REGISTRY["e6"] = CausalInterventionExperiment
    EXPERIMENT_REGISTRY["e7"] = PerturbationSensitivityExperiment
    EXPERIMENT_REGISTRY["e8"] = TokenTypeAnalysisExperiment


# -------------------------------------------------------------------
# Config loader with deep merge
# -------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(experiment_id: str, config_path: str = None) -> Dict[str, Any]:
    """Load config with deep merge: base <- experiment <- user override."""
    # Find the package config directory
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(pkg_dir, "config", "base.yaml")
    exp_path = os.path.join(pkg_dir, "config", "experiments", f"{experiment_id}.yaml")

    config = {}

    # Load base config
    if os.path.exists(base_path):
        with open(base_path) as f:
            config = yaml.safe_load(f) or {}

    # Merge experiment-specific config
    if os.path.exists(exp_path):
        with open(exp_path) as f:
            exp_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, exp_config)

    # Merge user-provided config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    return config


# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
def setup_logging(config: Dict[str, Any]):
    """Configure logging from config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    log_file = log_cfg.get("log_file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    _register_experiments()

    parser = argparse.ArgumentParser(description="Run an activation science experiment.")
    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENT_REGISTRY.keys()),
        help="Experiment ID (e0–e8).",
    )
    parser.add_argument("--config", default=None, help="Path to override config YAML.")
    parser.add_argument("--parallel", action="store_true", help="Use multi-GPU parallel mode.")
    args = parser.parse_args()

    config = load_config(args.experiment, args.config)
    setup_logging(config)

    experiment_cls = EXPERIMENT_REGISTRY[args.experiment]
    runner = GenericSweepRunner(experiment_cls)

    logger.info("Running experiment: %s (%s)", args.experiment, experiment_cls.experiment_name)

    if args.parallel:
        runner.run_parallel(config)
    else:
        runner.run_sequential(config)


if __name__ == "__main__":
    main()
