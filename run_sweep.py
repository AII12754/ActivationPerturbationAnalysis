#!/usr/bin/env python3
"""
run_sweep.py — Main entry point for running activation perturbation experiments.

Usage:
    python run_sweep.py --config config/default.yaml
    python run_sweep.py --config config/default.yaml --parallel
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def setup_logging(config: dict):
    """Configure logging from the config file."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run activation perturbation sweep experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel scheduler (loads one model copy per GPU).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded from %s", args.config)

    if args.parallel:
        from src.sweep import run_sweep_parallel
        run_sweep_parallel(config)
    else:
        from src.sweep import run_sweep
        run_sweep(config)


if __name__ == "__main__":
    main()
