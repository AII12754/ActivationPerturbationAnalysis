#!/usr/bin/env python3
"""
run_cross_sequence_analysis.py — Generate figures from cross-sequence alignment results.

Usage:
    python run_cross_sequence_analysis.py --results ./results_cross_sequence
    python run_cross_sequence_analysis.py --results ./results_cross_sequence --figures ./results_cross_sequence/figures
"""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis figures from cross-sequence alignment results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="./results_cross_sequence",
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

    from src.cross_sequence_analysis import run_all_cross_sequence_analyses

    run_all_cross_sequence_analyses(results_dir=args.results, figures_dir=args.figures)


if __name__ == "__main__":
    main()
