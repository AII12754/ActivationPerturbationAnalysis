#!/usr/bin/env python3
"""
run_analysis.py — Generate all figures from stored experiment results.

Usage:
    python run_analysis.py --results ./results
    python run_analysis.py --results ./results --figures ./results/figures
"""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis figures from experiment results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="./results",
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

    from src.analysis import run_all_analyses

    run_all_analyses(results_dir=args.results, figures_dir=args.figures)


if __name__ == "__main__":
    main()
