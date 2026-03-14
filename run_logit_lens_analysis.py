#!/usr/bin/env python3
"""
run_logit_lens_analysis.py — Generate figures from logit lens results.

Usage:
    python run_logit_lens_analysis.py --results ./results_logit_lens
    python run_logit_lens_analysis.py --results ./results_logit_lens --figures ./results_logit_lens/figures
"""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate analysis figures from logit lens results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="./results_logit_lens",
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

    from src.logit_lens_analysis import run_all_logit_lens_analyses

    run_all_logit_lens_analyses(results_dir=args.results, figures_dir=args.figures)


if __name__ == "__main__":
    main()
