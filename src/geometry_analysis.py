"""
Visualization module for representation geometry experiments.

Reads stored summary and spectrum results and produces figures exploring
the geometry of hidden-state representations across layers.

Analysis suite (10 plots):
  1. Effective rank by layer
  2. Participation ratio by layer
  3. Anisotropy profile (mean pairwise cosine by layer)
  4. Top-k explained variance by layer
  5. Singular value spectrum (log-scale, selected layers)
  6. Norm statistics by layer
  7. Effective rank by dataset comparison
  8. Isotropy by context length
  9. Cumulative variance curves by layer
  10. Effective rank vs context length interaction
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from .geometry_storage import GeometryResultStore

logger = logging.getLogger(__name__)

# Plotting defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (10, 10)
DPI = 150


def _select_layers(all_layers, max_show: int = 8):
    """Pick a representative subset of layers for line plots."""
    if len(all_layers) <= max_show:
        return all_layers
    step = max(1, len(all_layers) // max_show)
    return all_layers[::step]


def _layer_groups(all_layers) -> Dict[str, list]:
    """Split layers into early / middle / late thirds."""
    n = len(all_layers)
    if n >= 3:
        return {
            "early": all_layers[: n // 3],
            "middle": all_layers[n // 3 : 2 * n // 3],
            "late": all_layers[2 * n // 3 :],
        }
    return {"all": all_layers}


def _ds_label(name: str) -> str:
    """Short label from a dataset path."""
    return os.path.basename(str(name)) if name else "unknown"


def _save(fig, path, name):
    fig.tight_layout()
    fig.savefig(path / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", name)


# ===================================================================
# 1. Effective Rank by Layer
# ===================================================================
def plot_effective_rank_by_layer(df_summary, output_dir):
    """Effective rank (exp of entropy of normalized SVs) per layer."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_summary.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    profile = df_summary.groupby("layer")["effective_rank"].agg(["mean", "std"]).reset_index()
    ax.plot(profile["layer"], profile["mean"], "b-", linewidth=2)
    ax.fill_between(
        profile["layer"],
        profile["mean"] - profile["std"],
        profile["mean"] + profile["std"],
        alpha=0.2,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank by Layer Depth")
    _save(fig, out, "01_effective_rank_by_layer.png")


# ===================================================================
# 2. Participation Ratio by Layer
# ===================================================================
def plot_participation_ratio_by_layer(df_summary, output_dir):
    """Participation ratio (sum(sv)^2 / sum(sv^2)) per layer."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_summary.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    profile = df_summary.groupby("layer")["participation_ratio"].agg(["mean", "std"]).reset_index()
    ax.plot(profile["layer"], profile["mean"], "g-", linewidth=2)
    ax.fill_between(
        profile["layer"],
        profile["mean"] - profile["std"],
        profile["mean"] + profile["std"],
        alpha=0.2, color="green",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("Participation Ratio by Layer Depth")
    _save(fig, out, "02_participation_ratio_by_layer.png")


# ===================================================================
# 3. Anisotropy Profile (Mean Pairwise Cosine by Layer)
# ===================================================================
def plot_anisotropy_profile(df_summary, output_dir):
    """Mean pairwise cosine similarity (isotropy measure) per layer.

    High values indicate anisotropic (narrow cone) representations;
    values near 0 indicate isotropic (uniformly spread) representations.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_summary.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    profile = df_summary.groupby("layer")["mean_pairwise_cosine"].agg(["mean", "std"]).reset_index()
    ax.plot(profile["layer"], profile["mean"], "r-", linewidth=2)
    ax.fill_between(
        profile["layer"],
        profile["mean"] - profile["std"],
        profile["mean"] + profile["std"],
        alpha=0.2, color="red",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Pairwise Cosine Similarity")
    ax.set_title("Anisotropy Profile by Layer Depth\n(higher = more anisotropic)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    _save(fig, out, "03_anisotropy_profile.png")


# ===================================================================
# 4. Top-k Explained Variance by Layer
# ===================================================================
def plot_topk_explained_variance(df_summary, output_dir):
    """Fraction of variance explained by the top k singular values."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_summary.empty:
        return

    k_cols = [
        ("top1_explained_var", "Top 1"),
        ("top5_explained_var", "Top 5"),
        ("top10_explained_var", "Top 10"),
        ("top20_explained_var", "Top 20"),
        ("top50_explained_var", "Top 50"),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col, label in k_cols:
        if col not in df_summary.columns:
            continue
        profile = df_summary.groupby("layer")[col].mean().sort_index()
        ax.plot(profile.index, profile.values, label=label, alpha=0.8, linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Top-k Explained Variance by Layer")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    _save(fig, out, "04_topk_explained_variance.png")


# ===================================================================
# 5. Singular Value Spectrum (Log-Scale, Selected Layers)
# ===================================================================
def plot_singular_value_spectrum(df_spectrum, output_dir):
    """Log-scale singular value spectrum for selected layers."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_spectrum.empty:
        return

    all_layers = sorted(df_spectrum["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df_spectrum[df_spectrum["layer"] == layer].sort_values("sv_index")
        ax.semilogy(sub["sv_index"], sub["singular_value"], label=f"Layer {layer}", alpha=0.8)

    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title("Singular Value Spectrum by Layer")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out, "05_singular_value_spectrum.png")


# ===================================================================
# 6. Norm Statistics by Layer
# ===================================================================
def plot_norm_statistics(df_summary, output_dir):
    """Mean, std, min, max of hidden-state norms per layer."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_summary.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Mean norm with std band
    profile = df_summary.groupby("layer").agg(
        norm_mean_avg=("norm_mean", "mean"),
        norm_std_avg=("norm_std", "mean"),
    ).reset_index()
    axes[0].plot(profile["layer"], profile["norm_mean_avg"], "b-", linewidth=2)
    axes[0].fill_between(
        profile["layer"],
        profile["norm_mean_avg"] - profile["norm_std_avg"],
        profile["norm_mean_avg"] + profile["norm_std_avg"],
        alpha=0.2,
    )
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Hidden State Norm")
    axes[0].set_title("Mean Norm by Layer (with std band)")

    # Min/Max range
    minmax = df_summary.groupby("layer").agg(
        norm_min_avg=("norm_min", "mean"),
        norm_max_avg=("norm_max", "mean"),
    ).reset_index()
    axes[1].fill_between(
        minmax["layer"],
        minmax["norm_min_avg"],
        minmax["norm_max_avg"],
        alpha=0.3, color="orange", label="Min-Max range",
    )
    axes[1].plot(minmax["layer"], minmax["norm_min_avg"], "r--", alpha=0.7, label="Mean of min")
    axes[1].plot(minmax["layer"], minmax["norm_max_avg"], "g--", alpha=0.7, label="Mean of max")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Hidden State Norm")
    axes[1].set_title("Norm Range by Layer")
    axes[1].legend(fontsize=8)
    _save(fig, out, "06_norm_statistics.png")


# ===================================================================
# 7. Effective Rank by Dataset Comparison
# ===================================================================
def plot_effective_rank_by_dataset(df_summary, output_dir):
    """Compare effective rank profiles across datasets."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_summary.empty or "dataset_name" not in df_summary.columns:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in df_summary["dataset_name"].unique():
        sub = df_summary[df_summary["dataset_name"] == ds]
        profile = sub.groupby("layer")["effective_rank"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank by Layer — Dataset Comparison")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "07_effective_rank_by_dataset.png")


# ===================================================================
# 8. Isotropy by Context Length
# ===================================================================
def plot_isotropy_by_context_length(df_summary, output_dir):
    """How does anisotropy change with context length?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_summary.empty:
        return

    ctx_lengths = sorted(df_summary["context_length"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cl in ctx_lengths:
        sub = df_summary[df_summary["context_length"] == cl]
        profile = sub.groupby("layer")["mean_pairwise_cosine"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Pairwise Cosine")
    ax.set_title("Anisotropy by Context Length")
    ax.legend(fontsize=8)
    _save(fig, out, "08_isotropy_by_context_length.png")


# ===================================================================
# 9. Cumulative Variance Curves by Layer
# ===================================================================
def plot_cumulative_variance(df_spectrum, output_dir):
    """Cumulative explained variance ratio for selected layers."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_spectrum.empty:
        return

    all_layers = sorted(df_spectrum["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df_spectrum[df_spectrum["layer"] == layer]
        profile = sub.groupby("sv_index")["cumulative_variance_ratio"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=f"Layer {layer}", alpha=0.8)

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Ratio")
    ax.set_title("Cumulative Explained Variance by Layer")
    ax.axhline(0.9, color="gray", ls="--", alpha=0.5, label="90% threshold")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    _save(fig, out, "09_cumulative_variance.png")


# ===================================================================
# 10. Effective Rank vs Context Length Interaction
# ===================================================================
def plot_effective_rank_vs_context(df_summary, output_dir):
    """How does effective rank change with context length across layers?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_summary.empty:
        return

    all_layers = sorted(df_summary["layer"].unique())
    groups = _layer_groups(all_layers)

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    for ax, (name, layers) in zip(axes[0], groups.items()):
        sub = df_summary[df_summary["layer"].isin(layers)]
        grouped = sub.groupby("context_length")["effective_rank"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            grouped["context_length"], grouped["mean"], yerr=grouped["std"],
            marker="o", capsize=4, linewidth=2,
        )
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Effective Rank")
        ax.set_title(f"Effective Rank vs Context Length\n({name} layers)")
    _save(fig, out, "10_effective_rank_vs_context.png")


# ===================================================================
# Master routine
# ===================================================================
def run_all_geometry_analyses(
    results_dir: str = "./results_geometry",
    figures_dir: Optional[str] = None,
):
    """Run the full geometry analysis suite and save all figures."""
    if figures_dir is None:
        figures_dir = str(Path(results_dir) / "figures")

    df_summary = GeometryResultStore.load_summary(results_dir)
    df_spectrum = GeometryResultStore.load_spectrum(results_dir)

    if df_summary.empty and df_spectrum.empty:
        logger.error("No data found in %s. Run geometry experiments first.", results_dir)
        return

    logger.info(
        "Loaded %d summary records, %d spectrum records. Generating figures...",
        len(df_summary), len(df_spectrum),
    )

    # --- Geometry metrics (1-6) ---
    plot_effective_rank_by_layer(df_summary, figures_dir)
    plot_participation_ratio_by_layer(df_summary, figures_dir)
    plot_anisotropy_profile(df_summary, figures_dir)
    plot_topk_explained_variance(df_summary, figures_dir)
    plot_singular_value_spectrum(df_spectrum, figures_dir)
    plot_norm_statistics(df_summary, figures_dir)

    # --- Comparisons (7-10) ---
    plot_effective_rank_by_dataset(df_summary, figures_dir)
    plot_isotropy_by_context_length(df_summary, figures_dir)
    plot_cumulative_variance(df_spectrum, figures_dir)
    plot_effective_rank_vs_context(df_summary, figures_dir)

    logger.info("All %d figures saved to %s.", 10, figures_dir)
