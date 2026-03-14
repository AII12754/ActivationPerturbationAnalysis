"""
Visualization module for decode-time activation similarity experiments.

Reads stored topk and aggregate results and produces figures exploring
activation reuse patterns during autoregressive decoding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .decode_storage import DecodeResultStore

logger = logging.getLogger(__name__)

# Plotting defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (10, 6)


def _select_layers(all_layers, max_show: int = 8):
    """Pick a representative subset of layers for line plots."""
    if len(all_layers) <= max_show:
        return all_layers
    step = max(1, len(all_layers) // max_show)
    return all_layers[::step]


# ===================================================================
# 1. Similarity vs Distance (rank-1 reference)
# ===================================================================
def plot_similarity_vs_distance(
    df_topk: pd.DataFrame,
    output_dir: str = "./results_decode/figures",
):
    """Rank-1 ref_similarity vs ref_token_distance, by layer."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rank1 = df_topk[df_topk["ref_rank"] == 0]
    if rank1.empty:
        logger.warning("No rank-1 topk data; skipping similarity_vs_distance.")
        return

    all_layers = sorted(rank1["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = rank1[rank1["layer"] == layer]
        binned = sub.groupby(pd.cut(sub["ref_token_distance"], bins=20))["ref_similarity"].mean()
        mids = [(b.left + b.right) / 2 for b in binned.index]
        ax.plot(mids, binned.values, label=f"Layer {layer}", alpha=0.8)

    ax.set_xlabel("Distance to reference token (positions)")
    ax.set_ylabel("Cosine similarity (rank-1)")
    ax.set_title("Rank-1 Reference Similarity vs Distance")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "similarity_vs_distance.png", dpi=150)
    plt.close(fig)
    logger.info("Saved similarity_vs_distance.png")


# ===================================================================
# 2. Layer × Decode-Step Heatmap
# ===================================================================
def plot_layer_decode_step_heatmap(
    df_agg: pd.DataFrame,
    output_dir: str = "./results_decode/figures",
):
    """Heatmap: X=decode_step, Y=layer, color=max_similarity."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df_agg.empty:
        return

    pivot = df_agg.pivot_table(
        values="max_similarity",
        index="layer",
        columns="decode_step",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    sns.heatmap(pivot, cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Max Similarity: Layer × Decode Step")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Decode Step")
    fig.tight_layout()
    fig.savefig(out / "layer_decode_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved layer_decode_heatmap.png")


# ===================================================================
# 3. Max-Similarity Distribution
# ===================================================================
def plot_max_similarity_distribution(
    df_agg: pd.DataFrame,
    output_dir: str = "./results_decode/figures",
):
    """Histogram/KDE of max_similarity, faceted by layer groups."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df_agg.empty:
        return

    all_layers = sorted(df_agg["layer"].unique())
    # Group layers into early / middle / late.
    n = len(all_layers)
    if n >= 3:
        groups = {
            "early": all_layers[: n // 3],
            "middle": all_layers[n // 3: 2 * n // 3],
            "late": all_layers[2 * n // 3:],
        }
    else:
        groups = {"all": all_layers}

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    for ax, (name, layers) in zip(axes[0], groups.items()):
        sub = df_agg[df_agg["layer"].isin(layers)]
        ax.hist(sub["max_similarity"].dropna(), bins=50, alpha=0.7, density=True)
        ax.set_title(f"Max Similarity — {name} layers")
        ax.set_xlabel("Max cosine similarity")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out / "max_similarity_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved max_similarity_distribution.png")


# ===================================================================
# 4. Threshold Fraction vs Decode Step
# ===================================================================
def plot_threshold_fraction(
    df_agg: pd.DataFrame,
    output_dir: str = "./results_decode/figures",
):
    """Fraction above thresholds vs decode_step, by layer."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df_agg.empty:
        return

    all_layers = sorted(df_agg["layer"].unique())
    selected = _select_layers(all_layers, max_show=4)

    thresholds = [("frac_above_090", "0.90"), ("frac_above_095", "0.95"), ("frac_above_098", "0.98")]
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5), squeeze=False)

    for ax, (col, label) in zip(axes[0], thresholds):
        for layer in selected:
            sub = df_agg[df_agg["layer"] == layer].sort_values("decode_step")
            grouped = sub.groupby("decode_step")[col].mean()
            ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
        ax.set_title(f"Frac > {label} vs Decode Step")
        ax.set_xlabel("Decode Step")
        ax.set_ylabel(f"Fraction above {label}")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out / "threshold_fraction.png", dpi=150)
    plt.close(fig)
    logger.info("Saved threshold_fraction.png")


# ===================================================================
# 5. Temporal Drift
# ===================================================================
def plot_temporal_drift(
    df_agg: pd.DataFrame,
    output_dir: str = "./results_decode/figures",
):
    """Max similarity vs decode_step, by layer."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df_agg.empty:
        return

    all_layers = sorted(df_agg["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df_agg[df_agg["layer"] == layer].sort_values("decode_step")
        grouped = sub.groupby("decode_step")["max_similarity"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)

    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Max Cosine Similarity")
    ax.set_title("Temporal Drift: Max Similarity over Decoding")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "temporal_drift.png", dpi=150)
    plt.close(fig)
    logger.info("Saved temporal_drift.png")


# ===================================================================
# 6. Context Length Effect
# ===================================================================
def plot_context_length_effect(
    df_agg: pd.DataFrame,
    output_dir: str = "./results_decode/figures",
):
    """Box plot: context_length vs max_similarity."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df_agg.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(
        data=df_agg,
        x="context_length",
        y="max_similarity",
        ax=ax,
    )
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Max Cosine Similarity")
    ax.set_title("Effect of Context Length on Max Similarity")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "context_length_effect.png", dpi=150)
    plt.close(fig)
    logger.info("Saved context_length_effect.png")


# ===================================================================
# 7. Reference Position Distribution
# ===================================================================
def plot_reference_position_distribution(
    df_topk: pd.DataFrame,
    output_dir: str = "./results_decode/figures",
):
    """Histogram of where rank-1 references come from (relative position)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rank1 = df_topk[df_topk["ref_rank"] == 0]
    if rank1.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(rank1["ref_token_distance"].dropna(), bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Distance to rank-1 reference (positions)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Rank-1 Reference Token Distance")
    fig.tight_layout()
    fig.savefig(out / "reference_position_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved reference_position_distribution.png")


# ===================================================================
# 8. Top-K Similarity Decay
# ===================================================================
def plot_topk_similarity_decay(
    df_topk: pd.DataFrame,
    output_dir: str = "./results_decode/figures",
):
    """Similarity vs rank (1 through k), by layer."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if df_topk.empty:
        return

    all_layers = sorted(df_topk["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df_topk[df_topk["layer"] == layer]
        grouped = sub.groupby("ref_rank")["ref_similarity"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", markersize=4,
                label=f"Layer {layer}", alpha=0.8)

    ax.set_xlabel("Reference Rank")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Top-K Similarity Decay by Layer")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "topk_similarity_decay.png", dpi=150)
    plt.close(fig)
    logger.info("Saved topk_similarity_decay.png")


# ===================================================================
# Master routine
# ===================================================================
def run_all_decode_analyses(
    results_dir: str = "./results_decode",
    figures_dir: Optional[str] = None,
):
    """Run the full decode analysis suite and save all figures."""
    if figures_dir is None:
        figures_dir = str(Path(results_dir) / "figures")

    df_topk = DecodeResultStore.load_topk(results_dir)
    df_agg = DecodeResultStore.load_aggregates(results_dir)

    if df_topk.empty and df_agg.empty:
        logger.error("No data found in %s. Run decode experiments first.", results_dir)
        return

    logger.info(
        "Loaded %d topk records, %d aggregate records. Generating figures…",
        len(df_topk),
        len(df_agg),
    )

    # Topk-based plots.
    plot_similarity_vs_distance(df_topk, figures_dir)
    plot_reference_position_distribution(df_topk, figures_dir)
    plot_topk_similarity_decay(df_topk, figures_dir)

    # Aggregate-based plots.
    plot_layer_decode_step_heatmap(df_agg, figures_dir)
    plot_max_similarity_distribution(df_agg, figures_dir)
    plot_threshold_fraction(df_agg, figures_dir)
    plot_temporal_drift(df_agg, figures_dir)
    plot_context_length_effect(df_agg, figures_dir)

    logger.info("All decode figures saved to %s.", figures_dir)
