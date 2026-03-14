"""E4: Cross-sequence alignment analysis — 10 plot functions.

Ported from ``src/cross_sequence_analysis.py``.  Analyses representation
similarity across different prompts and datasets using CKA, Procrustes
distance, subspace overlap, and centroid cosine metrics.

Plots
-----
01. cka_by_layer                   — Line: CKA (within vs cross)
02. procrustes_by_layer            — Line: Procrustes distance
03. subspace_overlap_by_layer      — Line: subspace_overlap
04. centroid_cosine_by_layer       — Line: centroid_cosine
05. cka_dataset_heatmap            — Symmetric heatmap: CKA
06. cka_layer_profile_by_pair      — Line per top-10 pairs
07. shared_token_cosine_by_layer   — Line: shared_token_cosine
08. context_length_effect_cka      — Box + line
09. within_dataset_cka_variability — Line with +/-std
10. cross_dataset_alignment_matrix — 2x2 heatmaps
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

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
    return ExperimentStore.load_table(results_dir, "alignment")


# ===================================================================
# Plot 1: CKA by Layer
# ===================================================================
def plot_cka_by_layer(df, figures_dir):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype, color, label in [("within", "steelblue", "Within-dataset"),
                                 ("cross", "coral", "Cross-dataset")]:
        sub = df[df["pair_type"] == ptype]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["cka"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            color=color, alpha=0.2,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title("CKA by Layer: Within-Dataset vs Cross-Dataset Pairs")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    _savefig(fig, figures_dir, "cka_by_layer", 1)


# ===================================================================
# Plot 2: Procrustes Distance by Layer
# ===================================================================
def plot_procrustes_by_layer(df, figures_dir):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype, color, label in [("within", "steelblue", "Within-dataset"),
                                 ("cross", "coral", "Cross-dataset")]:
        sub = df[df["pair_type"] == ptype]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["procrustes_distance"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(
            profile["layer"],
            (profile["mean"] - profile["std"]).clip(lower=0),
            profile["mean"] + profile["std"],
            color=color, alpha=0.2,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Procrustes Distance")
    ax.set_title("Procrustes Distance by Layer: Within vs Cross")
    ax.legend(fontsize=10)
    _savefig(fig, figures_dir, "procrustes_by_layer", 2)


# ===================================================================
# Plot 3: Subspace Overlap by Layer
# ===================================================================
def plot_subspace_overlap_by_layer(df, figures_dir):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype, color, label in [("within", "steelblue", "Within-dataset"),
                                 ("cross", "coral", "Cross-dataset")]:
        sub = df[df["pair_type"] == ptype]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["subspace_overlap"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(
            profile["layer"],
            (profile["mean"] - profile["std"]).clip(lower=0),
            profile["mean"] + profile["std"],
            color=color, alpha=0.2,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Subspace Overlap")
    ax.set_title("Subspace Overlap by Layer: Within vs Cross")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    _savefig(fig, figures_dir, "subspace_overlap_by_layer", 3)


# ===================================================================
# Plot 4: Centroid Cosine by Layer
# ===================================================================
def plot_centroid_cosine_by_layer(df, figures_dir):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype, color, label in [("within", "steelblue", "Within-dataset"),
                                 ("cross", "coral", "Cross-dataset")]:
        sub = df[df["pair_type"] == ptype]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["centroid_cosine"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            color=color, alpha=0.2,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Centroid Cosine Similarity")
    ax.set_title("Centroid Cosine by Layer: Within vs Cross")
    ax.legend(fontsize=10)
    _savefig(fig, figures_dir, "centroid_cosine_by_layer", 4)


# ===================================================================
# Plot 5: CKA Dataset Heatmap
# ===================================================================
def plot_cka_dataset_heatmap(df, figures_dir):
    if df.empty:
        return
    df_avg = df.groupby(["dataset_a", "dataset_b"])["cka"].mean().reset_index()
    all_datasets = sorted(set(df_avg["dataset_a"].unique()) | set(df_avg["dataset_b"].unique()))
    labels = [_ds_label(d) for d in all_datasets]
    n = len(all_datasets)
    matrix = np.full((n, n), np.nan)
    ds_to_idx = {d: i for i, d in enumerate(all_datasets)}
    for _, row in df_avg.iterrows():
        i = ds_to_idx.get(row["dataset_a"])
        j = ds_to_idx.get(row["dataset_b"])
        if i is not None and j is not None:
            matrix[i, j] = row["cka"]
            matrix[j, i] = row["cka"]
    np.fill_diagonal(matrix, 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix, xticklabels=labels, yticklabels=labels,
        cmap="RdYlGn", vmin=0, vmax=1, annot=True, fmt=".2f", square=True, ax=ax,
    )
    ax.set_title("Mean CKA: Dataset x Dataset")
    ax.set_xlabel("Dataset B")
    ax.set_ylabel("Dataset A")
    _savefig(fig, figures_dir, "cka_dataset_heatmap", 5)


# ===================================================================
# Plot 6: CKA Layer Profile by Pair
# ===================================================================
def plot_cka_layer_profile_by_pair(df, figures_dir):
    cross = df[df["pair_type"] == "cross"]
    if cross.empty:
        return
    pairs = cross.groupby(["dataset_a", "dataset_b"]).size().reset_index()[["dataset_a", "dataset_b"]]
    if len(pairs) == 0:
        return
    if len(pairs) > 10:
        pairs = pairs.head(10)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for _, pair_row in pairs.iterrows():
        da, db = pair_row["dataset_a"], pair_row["dataset_b"]
        sub = cross[(cross["dataset_a"] == da) & (cross["dataset_b"] == db)]
        profile = sub.groupby("layer")["cka"].mean().sort_index()
        label = f"{_ds_label(da)} vs {_ds_label(db)}"
        ax.plot(profile.index, profile.values, label=label, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title("CKA Layer Profile by Cross-Dataset Pair")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_ylim(0, 1.05)
    _savefig(fig, figures_dir, "cka_layer_profile_by_pair", 6)


# ===================================================================
# Plot 7: Shared Token Cosine by Layer
# ===================================================================
def plot_shared_token_cosine_by_layer(df, figures_dir):
    if df.empty:
        return
    valid = df.dropna(subset=["shared_token_cosine"])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype, color, label in [("within", "steelblue", "Within-dataset"),
                                 ("cross", "coral", "Cross-dataset")]:
        sub = valid[valid["pair_type"] == ptype]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["shared_token_cosine"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            color=color, alpha=0.2,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Shared Token Cosine Similarity")
    ax.set_title("Shared Token Cosine by Layer: Within vs Cross")
    ax.legend(fontsize=10)
    _savefig(fig, figures_dir, "shared_token_cosine_by_layer", 7)


# ===================================================================
# Plot 8: Context Length Effect on CKA
# ===================================================================
def plot_context_length_effect_cka(df, figures_dir):
    within = df[df["pair_type"] == "within"]
    if within.empty or "context_length" not in within.columns:
        return
    ctx_lengths = sorted(within["context_length"].unique())
    if len(ctx_lengths) <= 1:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    sns.boxplot(data=within, x="context_length", y="cka", ax=axes[0])
    axes[0].set_xlabel("Context Length (tokens)")
    axes[0].set_ylabel("Linear CKA")
    axes[0].set_title("CKA Distribution by Context Length")
    axes[0].set_ylim(0, 1.05)
    for cl in ctx_lengths:
        sub = within[within["context_length"] == cl]
        profile = sub.groupby("layer")["cka"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Linear CKA")
    axes[1].set_title("CKA Layer Profile by Context Length")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)
    _savefig(fig, figures_dir, "context_length_effect_cka", 8)


# ===================================================================
# Plot 9: Within-Dataset CKA Variability
# ===================================================================
def plot_within_dataset_cka_variability(df, figures_dir):
    within = df[df["pair_type"] == "within"]
    if within.empty:
        return
    datasets = within["dataset_a"].unique()
    if len(datasets) == 0:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in datasets:
        sub = within[within["dataset_a"] == ds]
        profile = sub.groupby("layer")["cka"].agg(["mean", "std"]).reset_index()
        label = _ds_label(ds)
        ax.plot(profile["layer"], profile["mean"], label=label, alpha=0.8)
        ax.fill_between(
            profile["layer"],
            (profile["mean"] - profile["std"]).clip(lower=0),
            (profile["mean"] + profile["std"]).clip(upper=1),
            alpha=0.1,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Within-Dataset CKA Variability Across Prompt Pairs")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    _savefig(fig, figures_dir, "within_dataset_cka_variability", 9)


# ===================================================================
# Plot 10: Cross-Dataset Alignment Matrix
# ===================================================================
def plot_cross_dataset_alignment_matrix(df, figures_dir):
    cross = df[df["pair_type"] == "cross"]
    if cross.empty:
        return
    metrics = ["cka", "procrustes_distance", "subspace_overlap", "centroid_cosine"]
    titles = ["CKA", "Procrustes Distance", "Subspace Overlap", "Centroid Cosine"]
    cmaps = ["RdYlGn", "RdYlGn_r", "RdYlGn", "RdYlGn"]
    all_datasets = sorted(set(cross["dataset_a"].unique()) | set(cross["dataset_b"].unique()))
    labels = [_ds_label(d) for d in all_datasets]
    n = len(all_datasets)
    ds_to_idx = {d: i for i, d in enumerate(all_datasets)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, metric, title, cmap in zip(axes.flat, metrics, titles, cmaps):
        df_avg = cross.groupby(["dataset_a", "dataset_b"])[metric].mean().reset_index()
        matrix = np.full((n, n), np.nan)
        for _, row in df_avg.iterrows():
            i = ds_to_idx.get(row["dataset_a"])
            j = ds_to_idx.get(row["dataset_b"])
            if i is not None and j is not None:
                matrix[i, j] = row[metric]
                matrix[j, i] = row[metric]
        if metric == "procrustes_distance":
            np.fill_diagonal(matrix, 0.0)
        else:
            np.fill_diagonal(matrix, 1.0)
        sns.heatmap(
            matrix, xticklabels=labels, yticklabels=labels,
            cmap=cmap, annot=True, fmt=".2f", square=True, ax=ax,
        )
        ax.set_title(f"Cross-Dataset {title}")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    _savefig(fig, figures_dir, "cross_dataset_alignment_matrix", 10)


# ===================================================================
# Master function
# ===================================================================
def run_all_cross_sequence_analyses(results_dir: str, figures_dir: str):
    """Run all 10 cross-sequence alignment analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading alignment data from %s", results_dir)

    df = _load_data(results_dir)
    if df.empty:
        logger.warning("No alignment data found in %s. Skipping.", results_dir)
        return

    logger.info("Loaded %d alignment records.", len(df))

    for name, fn, args in [
        ("cka_by_layer", plot_cka_by_layer, (df, figures_dir)),
        ("procrustes_by_layer", plot_procrustes_by_layer, (df, figures_dir)),
        ("subspace_overlap_by_layer", plot_subspace_overlap_by_layer, (df, figures_dir)),
        ("centroid_cosine_by_layer", plot_centroid_cosine_by_layer, (df, figures_dir)),
        ("cka_dataset_heatmap", plot_cka_dataset_heatmap, (df, figures_dir)),
        ("cka_layer_profile_by_pair", plot_cka_layer_profile_by_pair, (df, figures_dir)),
        ("shared_token_cosine_by_layer", plot_shared_token_cosine_by_layer, (df, figures_dir)),
        ("context_length_effect_cka", plot_context_length_effect_cka, (df, figures_dir)),
        ("within_dataset_cka_variability", plot_within_dataset_cka_variability, (df, figures_dir)),
        ("cross_dataset_alignment_matrix", plot_cross_dataset_alignment_matrix, (df, figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info("All 10 cross-sequence plots complete. Figures saved to %s", figures_dir)
