"""E8: Token type analysis — 10 plot functions.

New design for token type analysis experiments that compare
representation properties (norm, rank, isotropy, separability)
across different token types (content, function, punctuation, etc.).

Plots
-----
01. norm_by_token_type              — Line: norm_mean by layer per token_type
02. effective_rank_by_token_type    — Line: effective_rank per token_type
03. isotropy_by_token_type          — Line: mean_pairwise_cosine per token_type
04. fisher_discriminant_by_layer    — Line: fisher_discriminant by layer
05. centroid_norm_by_token_type     — Grouped bar at early/mid/late layers
06. token_type_composition          — Stacked bar: num_tokens proportions
07. fisher_by_dataset               — Line per dataset
08. separability_by_context_length  — Line per context_length
09. token_type_radar                — Multi-panel radar chart
10. metric_token_type_heatmap       — Heatmap: metric x token_type across layers
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
    token_type = ExperimentStore.load_table(results_dir, "token_type")
    separability = ExperimentStore.load_table(results_dir, "separability")
    return token_type, separability


# ===================================================================
# Plot 1: Norm by Token Type
# ===================================================================
def plot_norm_by_token_type(df, figures_dir):
    if df.empty or "token_type" not in df.columns or "norm_mean" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for tt in sorted(df["token_type"].unique()):
        sub = df[df["token_type"] == tt]
        profile = sub.groupby("layer")["norm_mean"].mean().sort_index()
        ax.plot(profile.index, profile.values, linewidth=2, label=tt)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Norm")
    ax.set_title("Hidden State Norm by Layer per Token Type")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "norm_by_token_type", 1)


# ===================================================================
# Plot 2: Effective Rank by Token Type
# ===================================================================
def plot_effective_rank_by_token_type(df, figures_dir):
    if df.empty or "token_type" not in df.columns or "effective_rank" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for tt in sorted(df["token_type"].unique()):
        sub = df[df["token_type"] == tt]
        profile = sub.groupby("layer")["effective_rank"].mean().sort_index()
        ax.plot(profile.index, profile.values, linewidth=2, label=tt)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank by Layer per Token Type")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "effective_rank_by_token_type", 2)


# ===================================================================
# Plot 3: Isotropy by Token Type
# ===================================================================
def plot_isotropy_by_token_type(df, figures_dir):
    if df.empty or "token_type" not in df.columns or "mean_pairwise_cosine" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for tt in sorted(df["token_type"].unique()):
        sub = df[df["token_type"] == tt]
        profile = sub.groupby("layer")["mean_pairwise_cosine"].mean().sort_index()
        ax.plot(profile.index, profile.values, linewidth=2, label=tt)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Pairwise Cosine")
    ax.set_title("Anisotropy by Layer per Token Type")
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    _savefig(fig, figures_dir, "isotropy_by_token_type", 3)


# ===================================================================
# Plot 4: Fisher Discriminant by Layer
# ===================================================================
def plot_fisher_discriminant_by_layer(df_sep, figures_dir):
    if df_sep.empty or "fisher_discriminant" not in df_sep.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    profile = df_sep.groupby("layer")["fisher_discriminant"].agg(["mean", "std"]).reset_index()
    ax.plot(profile["layer"], profile["mean"], "b-", linewidth=2)
    ax.fill_between(
        profile["layer"],
        (profile["mean"] - profile["std"]).clip(lower=0),
        profile["mean"] + profile["std"],
        alpha=0.2,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fisher Discriminant Ratio")
    ax.set_title("Token Type Separability by Layer (Fisher Discriminant)")
    _savefig(fig, figures_dir, "fisher_discriminant_by_layer", 4)


# ===================================================================
# Plot 5: Centroid Norm by Token Type (grouped bar)
# ===================================================================
def plot_centroid_norm_by_token_type(df, figures_dir):
    if df.empty or "token_type" not in df.columns or "centroid_norm" not in df.columns:
        return
    all_layers = sorted(df["layer"].unique())
    groups = _layer_groups(all_layers)

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    for ax, (gname, layers) in zip(axes[0], groups.items()):
        sub = df[df["layer"].isin(layers)]
        pivot = sub.groupby("token_type")["centroid_norm"].mean()
        ax.bar(range(len(pivot)), pivot.values, tick_label=pivot.index)
        ax.set_xlabel("Token Type")
        ax.set_ylabel("Centroid Norm")
        ax.set_title(f"Centroid Norm ({gname} layers)")
        ax.tick_params(axis="x", rotation=45)
    _savefig(fig, figures_dir, "centroid_norm_by_token_type", 5)


# ===================================================================
# Plot 6: Token Type Composition
# ===================================================================
def plot_token_type_composition(df, figures_dir):
    if df.empty or "token_type" not in df.columns or "num_tokens" not in df.columns:
        return
    # Aggregate over all layers (composition is per-experiment).
    totals = df.groupby("token_type")["num_tokens"].sum()
    if totals.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    proportions = totals / totals.sum()
    colors = plt.cm.Set2(np.linspace(0, 1, len(proportions)))
    ax.bar(range(len(proportions)), proportions.values, color=colors, edgecolor="gray")
    ax.set_xticks(range(len(proportions)))
    ax.set_xticklabels(proportions.index, rotation=45, ha="right")
    ax.set_xlabel("Token Type")
    ax.set_ylabel("Proportion")
    ax.set_title("Token Type Composition")
    _savefig(fig, figures_dir, "token_type_composition", 6)


# ===================================================================
# Plot 7: Fisher by Dataset
# ===================================================================
def plot_fisher_by_dataset(df_sep, figures_dir):
    if df_sep.empty or "dataset_name" not in df_sep.columns or "fisher_discriminant" not in df_sep.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in df_sep["dataset_name"].unique():
        sub = df_sep[df_sep["dataset_name"] == ds]
        profile = sub.groupby("layer")["fisher_discriminant"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fisher Discriminant Ratio")
    ax.set_title("Token Type Separability by Layer — Dataset Comparison")
    ax.legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "fisher_by_dataset", 7)


# ===================================================================
# Plot 8: Separability by Context Length
# ===================================================================
def plot_separability_by_context_length(df_sep, figures_dir):
    if df_sep.empty or "context_length" not in df_sep.columns or "fisher_discriminant" not in df_sep.columns:
        return
    ctx_lengths = sorted(df_sep["context_length"].unique())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cl in ctx_lengths:
        sub = df_sep[df_sep["context_length"] == cl]
        profile = sub.groupby("layer")["fisher_discriminant"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fisher Discriminant Ratio")
    ax.set_title("Token Type Separability by Context Length")
    ax.legend(fontsize=8)
    _savefig(fig, figures_dir, "separability_by_context_length", 8)


# ===================================================================
# Plot 9: Token Type Radar Chart
# ===================================================================
def plot_token_type_radar(df, figures_dir):
    if df.empty or "token_type" not in df.columns:
        return
    metrics = ["norm_mean", "effective_rank", "mean_pairwise_cosine", "centroid_norm"]
    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        return

    token_types = sorted(df["token_type"].unique())
    all_layers = sorted(df["layer"].unique())
    groups = _layer_groups(all_layers)

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 6),
                              subplot_kw=dict(polar=True), squeeze=False)
    angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
    angles += angles[:1]

    for ax, (gname, layers) in zip(axes[0], groups.items()):
        sub = df[df["layer"].isin(layers)]
        for tt in token_types:
            tt_data = sub[sub["token_type"] == tt]
            values = [tt_data[m].mean() for m in available]
            # Normalize to [0, 1] for radar.
            max_vals = [sub[m].max() for m in available]
            norm_values = [v / mx if mx > 0 else 0 for v, mx in zip(values, max_vals)]
            norm_values += norm_values[:1]
            ax.plot(angles, norm_values, label=tt, alpha=0.8)
            ax.fill(angles, norm_values, alpha=0.05)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available, fontsize=7)
        ax.set_title(f"{gname} layers", fontsize=10, y=1.1)
        ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.0))

    fig.suptitle("Token Type Profiles (Radar)", fontsize=14, y=1.05)
    _savefig(fig, figures_dir, "token_type_radar", 9)


# ===================================================================
# Plot 10: Metric x Token Type Heatmap
# ===================================================================
def plot_metric_token_type_heatmap(df, figures_dir):
    if df.empty or "token_type" not in df.columns:
        return
    metrics = ["norm_mean", "effective_rank", "mean_pairwise_cosine", "centroid_norm"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        return

    # Average over all layers to get a metric x token_type matrix.
    token_types = sorted(df["token_type"].unique())
    matrix = np.zeros((len(available), len(token_types)))
    for i, m in enumerate(available):
        for j, tt in enumerate(token_types):
            vals = df[df["token_type"] == tt][m]
            matrix[i, j] = vals.mean() if len(vals) > 0 else 0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(
        matrix, xticklabels=token_types, yticklabels=available,
        cmap="viridis", annot=True, fmt=".2f", ax=ax,
    )
    ax.set_title("Mean Metric Values: Metric x Token Type (all layers)")
    ax.set_xlabel("Token Type")
    ax.set_ylabel("Metric")
    _savefig(fig, figures_dir, "metric_token_type_heatmap", 10)


# ===================================================================
# Master function
# ===================================================================
def run_all_token_type_analyses(results_dir: str, figures_dir: str):
    """Run all 10 token type analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading token type data from %s", results_dir)

    df_tt, df_sep = _load_data(results_dir)
    if df_tt.empty and df_sep.empty:
        logger.warning("No token type data found in %s. Skipping.", results_dir)
        return

    logger.info("Loaded %d token_type and %d separability records.", len(df_tt), len(df_sep))

    for name, fn, args in [
        ("norm_by_token_type", plot_norm_by_token_type, (df_tt, figures_dir)),
        ("effective_rank_by_token_type", plot_effective_rank_by_token_type, (df_tt, figures_dir)),
        ("isotropy_by_token_type", plot_isotropy_by_token_type, (df_tt, figures_dir)),
        ("fisher_discriminant_by_layer", plot_fisher_discriminant_by_layer, (df_sep, figures_dir)),
        ("centroid_norm_by_token_type", plot_centroid_norm_by_token_type, (df_tt, figures_dir)),
        ("token_type_composition", plot_token_type_composition, (df_tt, figures_dir)),
        ("fisher_by_dataset", plot_fisher_by_dataset, (df_sep, figures_dir)),
        ("separability_by_context_length", plot_separability_by_context_length, (df_sep, figures_dir)),
        ("token_type_radar", plot_token_type_radar, (df_tt, figures_dir)),
        ("metric_token_type_heatmap", plot_metric_token_type_heatmap, (df_tt, figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info("All 10 token type plots complete. Figures saved to %s", figures_dir)
