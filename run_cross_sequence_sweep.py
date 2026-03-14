#!/usr/bin/env python3
"""
run_cross_sequence_sweep.py — Entry point for cross-sequence alignment experiments.

Usage:
    python run_cross_sequence_sweep.py --config config/cross_sequence.yaml
    python run_cross_sequence_sweep.py --config config/cross_sequence.yaml --parallel
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
        description="Run cross-sequence alignment experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/cross_sequence.yaml",
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
        from src.cross_sequence_sweep import run_cross_sequence_sweep_parallel
        run_cross_sequence_sweep_parallel(config)
    else:
        from src.cross_sequence_sweep import run_cross_sequence_sweep
        run_cross_sequence_sweep(config)


if __name__ == "__main__":
    main()
