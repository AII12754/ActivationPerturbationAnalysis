"""E3: Representation geometry analysis — 10 plot functions.

Ported from ``src/geometry_analysis.py``.  Analyses the geometry of
hidden-state representations across layers — effective rank, isotropy,
singular value spectra, and norm statistics.

Plots
-----
01. effective_rank_by_layer       — Line: effective_rank +/- std
02. participation_ratio_by_layer  — Line: participation_ratio
03. anisotropy_profile            — Line: mean_pairwise_cosine
04. topk_explained_variance       — Multi-line: top1/5/10/20/50
05. singular_value_spectrum       — Log line: SV by index (selected layers)
06. norm_statistics               — 2-panel: norm stats by layer
07. effective_rank_by_dataset     — Line per dataset
08. isotropy_by_context_length    — Line per context_length
09. cumulative_variance           — Line per layer: cumulative var ratio
10. effective_rank_vs_context     — Multi-panel errorbar
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ...core.storage import ExperimentStore

logger = logging.getLogger(__name__)

# Plotting defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
FIGSIZE = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (10, 10)
DPI = 150


def _select_layers(all_layers, max_show: int = 8):
    if len(all_layers) <= max_show:
        return all_layers
    step = max(1, len(all_layers) // max_show)
    return all_layers[::step]


def _layer_groups(all_layers) -> Dict[str, list]:
    n = len(all_layers)
    if n >= 3:
        return {
            "early": all_layers[: n // 3],
            "middle": all_layers[n // 3 : 2 * n // 3],
            "late": all_layers[2 * n // 3 :],
        }
    return {"all": all_layers}


def _savefig(fig, figures_dir, name, idx=None):
    prefix = f"{idx:02d}_" if idx else ""
    path = os.path.join(figures_dir, f"{prefix}{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _ds_label(name: str) -> str:
    return os.path.basename(str(name)) if name else "unknown"


def _load_data(results_dir: str):
    summary = ExperimentStore.load_table(results_dir, "summary")
    spectrum = ExperimentStore.load_table(results_dir, "spectrum")
    return summary, spectrum


# ===================================================================
# Plot 1: Effective Rank by Layer
# ===================================================================
def plot_effective_rank_by_layer(df_summary, figures_dir):
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
    _savefig(fig, figures_dir, "effective_rank_by_layer", 1)


# ===================================================================
# Plot 2: Participation Ratio by Layer
# ===================================================================
def plot_participation_ratio_by_layer(df_summary, figures_dir):
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
    _savefig(fig, figures_dir, "participation_ratio_by_layer", 2)


# ===================================================================
# Plot 3: Anisotropy Profile
# ===================================================================
def plot_anisotropy_profile(df_summary, figures_dir):
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
    _savefig(fig, figures_dir, "anisotropy_profile", 3)


# ===================================================================
# Plot 4: Top-k Explained Variance
# ===================================================================
def plot_topk_explained_variance(df_summary, figures_dir):
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
    _savefig(fig, figures_dir, "topk_explained_variance", 4)


# ===================================================================
# Plot 5: Singular Value Spectrum
# ===================================================================
def plot_singular_value_spectrum(df_spectrum, figures_dir):
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
    _savefig(fig, figures_dir, "singular_value_spectrum", 5)


# ===================================================================
# Plot 6: Norm Statistics
# ===================================================================
def plot_norm_statistics(df_summary, figures_dir):
    if df_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

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
    _savefig(fig, figures_dir, "norm_statistics", 6)


# ===================================================================
# Plot 7: Effective Rank by Dataset
# ===================================================================
def plot_effective_rank_by_dataset(df_summary, figures_dir):
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
    _savefig(fig, figures_dir, "effective_rank_by_dataset", 7)


# ===================================================================
# Plot 8: Isotropy by Context Length
# ===================================================================
def plot_isotropy_by_context_length(df_summary, figures_dir):
    if df_summary.empty or "context_length" not in df_summary.columns:
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
    _savefig(fig, figures_dir, "isotropy_by_context_length", 8)


# ===================================================================
# Plot 9: Cumulative Variance
# ===================================================================
def plot_cumulative_variance(df_spectrum, figures_dir):
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
    _savefig(fig, figures_dir, "cumulative_variance", 9)


# ===================================================================
# Plot 10: Effective Rank vs Context Length
# ===================================================================
def plot_effective_rank_vs_context(df_summary, figures_dir):
    if df_summary.empty or "context_length" not in df_summary.columns:
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
    _savefig(fig, figures_dir, "effective_rank_vs_context", 10)


# ===================================================================
# Master function
# ===================================================================
def run_all_geometry_analyses(results_dir: str, figures_dir: str):
    """Run all 10 representation geometry analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading geometry data from %s", results_dir)

    df_summary, df_spectrum = _load_data(results_dir)
    if df_summary.empty and df_spectrum.empty:
        logger.warning("No geometry data found in %s. Skipping.", results_dir)
        return

    logger.info("Loaded %d summary and %d spectrum records.", len(df_summary), len(df_spectrum))

    for name, fn, args in [
        ("effective_rank_by_layer", plot_effective_rank_by_layer, (df_summary, figures_dir)),
        ("participation_ratio_by_layer", plot_participation_ratio_by_layer, (df_summary, figures_dir)),
        ("anisotropy_profile", plot_anisotropy_profile, (df_summary, figures_dir)),
        ("topk_explained_variance", plot_topk_explained_variance, (df_summary, figures_dir)),
        ("singular_value_spectrum", plot_singular_value_spectrum, (df_spectrum, figures_dir)),
        ("norm_statistics", plot_norm_statistics, (df_summary, figures_dir)),
        ("effective_rank_by_dataset", plot_effective_rank_by_dataset, (df_summary, figures_dir)),
        ("isotropy_by_context_length", plot_isotropy_by_context_length, (df_summary, figures_dir)),
        ("cumulative_variance", plot_cumulative_variance, (df_spectrum, figures_dir)),
        ("effective_rank_vs_context", plot_effective_rank_vs_context, (df_summary, figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info("All 10 geometry plots complete. Figures saved to %s", figures_dir)
