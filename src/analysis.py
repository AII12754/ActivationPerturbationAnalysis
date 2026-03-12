"""
Analysis and visualization module.

Reads stored results and produces publication-quality figures exploring
how perturbations propagate through the model's hidden representations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .storage import ResultStore

logger = logging.getLogger(__name__)

# Plotting defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (10, 6)


def load_data(results_dir: str = "./results") -> pd.DataFrame:
    """Load all experiment results into a single DataFrame."""
    return ResultStore.load_all(results_dir)


# ===================================================================
# 1. Similarity vs Distance Curves
# ===================================================================
def plot_similarity_vs_distance(
    df: pd.DataFrame,
    output_dir: str = "./results/figures",
    perturbation_type: Optional[str] = None,
    context_length: Optional[int] = None,
):
    """Line plot: mean cosine similarity vs offset from perturbation.

    Uses the ``periodic_*`` buckets to trace a curve along the sequence.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    periodic = df[df["bucket"].str.startswith("periodic_")].copy()
    if periodic.empty:
        logger.warning("No periodic buckets found; skipping similarity-vs-distance plot.")
        return

    # Extract numeric offset from bucket name.
    periodic["offset"] = periodic["bucket"].str.replace("periodic_", "").astype(int)

    # Filters.
    if perturbation_type:
        periodic = periodic[periodic["perturbation_type"] == perturbation_type]
    if context_length:
        periodic = periodic[periodic["context_length"] == context_length]

    # Group by layer and offset.
    grouped = (
        periodic.groupby(["layer", "offset"])["mean_similarity"]
        .mean()
        .reset_index()
    )

    # Select a few representative layers.
    all_layers = sorted(grouped["layer"].unique())
    if len(all_layers) > 8:
        step = max(1, len(all_layers) // 8)
        selected_layers = all_layers[::step]
    else:
        selected_layers = all_layers

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected_layers:
        sub = grouped[grouped["layer"] == layer].sort_values("offset")
        ax.plot(sub["offset"], sub["mean_similarity"], label=f"Layer {layer}", alpha=0.8)

    ax.set_xlabel("Token offset from perturbation end")
    ax.set_ylabel("Mean cosine similarity")
    title = "Similarity vs Distance"
    if perturbation_type:
        title += f" ({perturbation_type})"
    if context_length:
        title += f" [ctx={context_length}]"
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    fname = "similarity_vs_distance"
    if perturbation_type:
        fname += f"_{perturbation_type}"
    if context_length:
        fname += f"_ctx{context_length}"
    fig.savefig(out / f"{fname}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s.png", fname)


# ===================================================================
# 2. Layer-wise Similarity Heatmap
# ===================================================================
def plot_layer_heatmap(
    df: pd.DataFrame,
    output_dir: str = "./results/figures",
    bucket: str = "immediate",
):
    """Heatmap: layers × perturbation types → mean similarity."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sub = df[df["bucket"] == bucket]
    if sub.empty:
        logger.warning("No data for bucket '%s'; skipping heatmap.", bucket)
        return

    pivot = sub.pivot_table(
        values="mean_similarity",
        index="layer",
        columns="perturbation_type",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.3)))
    sns.heatmap(
        pivot,
        annot=False,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(f"Layer × Perturbation Type — {bucket} window")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Perturbation Type")
    fig.tight_layout()
    fig.savefig(out / f"heatmap_{bucket}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap_%s.png", bucket)


# ===================================================================
# 3. Comparison across Context Lengths
# ===================================================================
def plot_context_length_comparison(
    df: pd.DataFrame,
    output_dir: str = "./results/figures",
    bucket: str = "immediate",
):
    """Bar/line plot comparing similarity across context lengths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sub = df[df["bucket"] == bucket]
    if sub.empty:
        return

    grouped = (
        sub.groupby(["context_length", "perturbation_type"])["mean_similarity"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype in grouped["perturbation_type"].unique():
        psub = grouped[grouped["perturbation_type"] == ptype].sort_values("context_length")
        ax.plot(
            psub["context_length"],
            psub["mean_similarity"],
            marker="o",
            label=ptype,
        )

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(f"Similarity vs Context Length ({bucket} window)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / f"context_length_{bucket}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved context_length_%s.png", bucket)


# ===================================================================
# 4. Perturbation Position Effect
# ===================================================================
def plot_position_effect(
    df: pd.DataFrame,
    output_dir: str = "./results/figures",
    bucket: str = "immediate",
):
    """Show how perturbation position affects similarity."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sub = df[df["bucket"] == bucket]
    if sub.empty:
        return

    grouped = (
        sub.groupby(["perturbation_position_frac", "perturbation_type"])["mean_similarity"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype in grouped["perturbation_type"].unique():
        psub = grouped[grouped["perturbation_type"] == ptype].sort_values(
            "perturbation_position_frac"
        )
        ax.plot(
            psub["perturbation_position_frac"] * 100,
            psub["mean_similarity"],
            marker="s",
            label=ptype,
        )

    ax.set_xlabel("Perturbation Position (% of sequence)")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(f"Effect of Perturbation Position ({bucket} window)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / f"position_effect_{bucket}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved position_effect_%s.png", bucket)


# ===================================================================
# 5. Span Length Effect
# ===================================================================
def plot_span_length_effect(
    df: pd.DataFrame,
    output_dir: str = "./results/figures",
    bucket: str = "immediate",
):
    """Show how perturbation span length affects similarity."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sub = df[df["bucket"] == bucket]
    if sub.empty:
        return

    grouped = (
        sub.groupby(["perturbation_span_length", "perturbation_type"])["mean_similarity"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype in grouped["perturbation_type"].unique():
        psub = grouped[grouped["perturbation_type"] == ptype].sort_values(
            "perturbation_span_length"
        )
        ax.plot(
            psub["perturbation_span_length"],
            psub["mean_similarity"],
            marker="^",
            label=ptype,
        )

    ax.set_xlabel("Perturbation Span Length (tokens)")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(f"Effect of Span Length ({bucket} window)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / f"span_length_{bucket}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved span_length_%s.png", bucket)


# ===================================================================
# 6. Per-token Fine-grained Curve
# ===================================================================
def plot_per_token_curve(
    df: pd.DataFrame,
    output_dir: str = "./results/figures",
    perturbation_type: Optional[str] = None,
):
    """Fine-grained per-token similarity for the first 10 tokens."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sub = df[df["bucket"] == "per_token"].copy()
    if perturbation_type:
        sub = sub[sub["perturbation_type"] == perturbation_type]
    if sub.empty:
        return

    grouped = (
        sub.groupby(["layer", "offset_start"])["mean_similarity"]
        .mean()
        .reset_index()
    )

    all_layers = sorted(grouped["layer"].unique())
    if len(all_layers) > 8:
        step = max(1, len(all_layers) // 8)
        selected_layers = all_layers[::step]
    else:
        selected_layers = all_layers

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected_layers:
        lsub = grouped[grouped["layer"] == layer].sort_values("offset_start")
        ax.plot(lsub["offset_start"], lsub["mean_similarity"], marker="o",
                markersize=4, label=f"Layer {layer}")

    ax.set_xlabel("Token position after perturbation")
    ax.set_ylabel("Cosine Similarity")
    title = "Per-Token Similarity (first 10 tokens)"
    if perturbation_type:
        title += f" — {perturbation_type}"
    ax.set_title(title)
    ax.set_xticks(range(10))
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    fname = "per_token"
    if perturbation_type:
        fname += f"_{perturbation_type}"
    fig.savefig(out / f"{fname}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s.png", fname)


# ===================================================================
# 7. Distribution Plots
# ===================================================================
def plot_similarity_distributions(
    df: pd.DataFrame,
    output_dir: str = "./results/figures",
    bucket: str = "immediate",
):
    """Violin/box plot of similarity distributions by perturbation type."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sub = df[df["bucket"] == bucket]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(
        data=sub,
        x="perturbation_type",
        y="mean_similarity",
        ax=ax,
    )
    ax.set_xlabel("Perturbation Type")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(f"Similarity Distribution ({bucket} window)")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / f"distribution_{bucket}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved distribution_%s.png", bucket)


# ===================================================================
# Master routine
# ===================================================================
def run_all_analyses(
    results_dir: str = "./results",
    figures_dir: Optional[str] = None,
):
    """Run the full analysis suite and save all figures."""
    if figures_dir is None:
        figures_dir = str(Path(results_dir) / "figures")

    df = load_data(results_dir)
    if df.empty:
        logger.error("No data found in %s. Run experiments first.", results_dir)
        return

    logger.info("Loaded %d records. Generating figures…", len(df))

    # Global plots.
    for bucket in ["immediate", "short", "all_remaining"]:
        plot_layer_heatmap(df, figures_dir, bucket=bucket)
        plot_context_length_comparison(df, figures_dir, bucket=bucket)
        plot_position_effect(df, figures_dir, bucket=bucket)
        plot_span_length_effect(df, figures_dir, bucket=bucket)
        plot_similarity_distributions(df, figures_dir, bucket=bucket)

    # Per-perturbation-type distance curves.
    for ptype in df["perturbation_type"].dropna().unique():
        plot_similarity_vs_distance(df, figures_dir, perturbation_type=ptype)
        plot_per_token_curve(df, figures_dir, perturbation_type=ptype)

    # Overall distance curve.
    plot_similarity_vs_distance(df, figures_dir)

    # Per context-length distance curves.
    for ctx in df["context_length"].dropna().unique():
        plot_similarity_vs_distance(df, figures_dir, context_length=int(ctx))

    logger.info("All figures saved to %s.", figures_dir)
