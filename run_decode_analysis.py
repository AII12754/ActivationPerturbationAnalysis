#!/usr/bin/env python3
"""
run_decode_analysis.py — Generate figures from decode similarity results.

Usage:
    python run_decode_analysis.py --results ./results_decode
    python run_decode_analysis.py --results ./results_decode --figures ./results_decode/figures
"""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis figures from decode similarity results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="./results_decode",
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

    from src.decode_analysis import run_all_decode_analyses

    run_all_decode_analyses(results_dir=args.results, figures_dir=args.figures)


if __name__ == "__main__":
    main()
