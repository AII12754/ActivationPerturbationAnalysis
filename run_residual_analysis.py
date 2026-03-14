#!/usr/bin/env python3
"""
run_residual_analysis.py — Generate figures from residual stream decomposition results.

Usage:
    python run_residual_analysis.py --results ./results_residual
    python run_residual_analysis.py --results ./results_residual --figures ./results_residual/figures
"""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis figures from residual stream decomposition results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="./results_residual",
        help="Directory containing Parquet result files.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default=None,
        help="Directory to save figures (default: <results>/figures).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    from src.residual_analysis import run_all_residual_analyses

    run_all_residual_analyses(results_dir=args.results, figures_dir=args.figures)


if __name__ == "__main__":
    main()
