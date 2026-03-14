"""
Visualization module for cross-sequence alignment experiments.

Reads stored alignment results and produces figures exploring
representation similarity across different prompts and datasets.

Analysis suite (10 plots):
  1. CKA by layer (within-dataset vs cross-dataset)
  2. Procrustes distance by layer (within vs cross)
  3. Subspace overlap by layer
  4. Centroid cosine by layer
  5. CKA heatmap: dataset_a x dataset_b (averaged over layers)
  6. CKA layer profile by dataset pair
  7. Shared token cosine by layer
  8. Context length effect on CKA
  9. Within-dataset CKA variability
  10. Cross-dataset alignment matrix
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

from .cross_sequence_storage import CrossSequenceResultStore

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
# 1. CKA by Layer (within vs cross)
# ===================================================================
def plot_cka_by_layer(df, output_dir):
    """CKA similarity by layer, split by within-dataset vs cross-dataset pairs."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
    _save(fig, out, "01_cka_by_layer.png")


# ===================================================================
# 2. Procrustes Distance by Layer (within vs cross)
# ===================================================================
def plot_procrustes_by_layer(df, output_dir):
    """Procrustes distance by layer, split by pair type."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
    _save(fig, out, "02_procrustes_by_layer.png")


# ===================================================================
# 3. Subspace Overlap by Layer
# ===================================================================
def plot_subspace_overlap_by_layer(df, output_dir):
    """Subspace overlap by layer, split by pair type."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
    _save(fig, out, "03_subspace_overlap_by_layer.png")


# ===================================================================
# 4. Centroid Cosine by Layer
# ===================================================================
def plot_centroid_cosine_by_layer(df, output_dir):
    """Centroid cosine similarity by layer, split by pair type."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
    _save(fig, out, "04_centroid_cosine_by_layer.png")


# ===================================================================
# 5. CKA Heatmap: Dataset A x Dataset B
# ===================================================================
def plot_cka_dataset_heatmap(df, output_dir):
    """CKA averaged over layers, shown as a dataset-by-dataset heatmap."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    # Average CKA over all layers for each (dataset_a, dataset_b) pair.
    df_avg = df.groupby(["dataset_a", "dataset_b"])["cka"].mean().reset_index()

    # Build a symmetric matrix.
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

    # Fill diagonal with 1.0 (self-similarity).
    np.fill_diagonal(matrix, 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
    )
    ax.set_title("Mean CKA: Dataset x Dataset")
    ax.set_xlabel("Dataset B")
    ax.set_ylabel("Dataset A")
    _save(fig, out, "05_cka_dataset_heatmap.png")


# ===================================================================
# 6. CKA Layer Profile by Dataset Pair
# ===================================================================
def plot_cka_layer_profile_by_pair(df, output_dir):
    """CKA layer profile for each unique dataset pair (cross-dataset)."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    cross = df[df["pair_type"] == "cross"]
    if cross.empty:
        return

    # Get unique dataset pairs.
    pairs = cross.groupby(["dataset_a", "dataset_b"]).size().reset_index()[["dataset_a", "dataset_b"]]
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    # Select a subset if too many.
    max_show = 10
    if n_pairs > max_show:
        pairs = pairs.head(max_show)

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
    _save(fig, out, "06_cka_layer_profile_by_pair.png")


# ===================================================================
# 7. Shared Token Cosine by Layer
# ===================================================================
def plot_shared_token_cosine_by_layer(df, output_dir):
    """Shared token cosine similarity by layer, split by pair type."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    # Drop NaN shared_token_cosine rows.
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
    _save(fig, out, "07_shared_token_cosine_by_layer.png")


# ===================================================================
# 8. Context Length Effect on CKA
# ===================================================================
def plot_context_length_effect_cka(df, output_dir):
    """How does context length affect CKA for within-dataset pairs?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    within = df[df["pair_type"] == "within"]
    if within.empty:
        return

    ctx_lengths = sorted(within["context_length"].unique())
    if len(ctx_lengths) <= 1:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Box plot: CKA by context length (all layers pooled).
    sns.boxplot(data=within, x="context_length", y="cka", ax=axes[0])
    axes[0].set_xlabel("Context Length (tokens)")
    axes[0].set_ylabel("Linear CKA")
    axes[0].set_title("CKA Distribution by Context Length")
    axes[0].set_ylim(0, 1.05)

    # Layer profiles per context length.
    for cl in ctx_lengths:
        sub = within[within["context_length"] == cl]
        profile = sub.groupby("layer")["cka"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Linear CKA")
    axes[1].set_title("CKA Layer Profile by Context Length")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.05)
    _save(fig, out, "08_context_length_effect_cka.png")


# ===================================================================
# 9. Within-Dataset CKA Variability
# ===================================================================
def plot_within_dataset_cka_variability(df, output_dir):
    """How much does CKA vary across prompt pairs within the same dataset?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
    _save(fig, out, "09_within_dataset_cka_variability.png")


# ===================================================================
# 10. Cross-Dataset Alignment Matrix
# ===================================================================
def plot_cross_dataset_alignment_matrix(df, output_dir):
    """Which datasets have the most similar representations?
    Shows a multi-metric heatmap averaged over all layers.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
            matrix,
            xticklabels=labels,
            yticklabels=labels,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            square=True,
            ax=ax,
        )
        ax.set_title(f"Cross-Dataset {title}")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    _save(fig, out, "10_cross_dataset_alignment_matrix.png")


# ===================================================================
# Master routine
# ===================================================================
def run_all_cross_sequence_analyses(
    results_dir: str = "./results_cross_sequence",
    figures_dir: Optional[str] = None,
):
    """Run the full cross-sequence analysis suite and save all figures."""
    if figures_dir is None:
        figures_dir = str(Path(results_dir) / "figures")

    df = CrossSequenceResultStore.load_alignments(results_dir)

    if df.empty:
        logger.error("No data found in %s. Run cross-sequence experiments first.", results_dir)
        return

    logger.info(
        "Loaded %d alignment records. Generating figures...",
        len(df),
    )

    plot_cka_by_layer(df, figures_dir)
    plot_procrustes_by_layer(df, figures_dir)
    plot_subspace_overlap_by_layer(df, figures_dir)
    plot_centroid_cosine_by_layer(df, figures_dir)
    plot_cka_dataset_heatmap(df, figures_dir)
    plot_cka_layer_profile_by_pair(df, figures_dir)
    plot_shared_token_cosine_by_layer(df, figures_dir)
    plot_context_length_effect_cka(df, figures_dir)
    plot_within_dataset_cka_variability(df, figures_dir)
    plot_cross_dataset_alignment_matrix(df, figures_dir)

    logger.info("All %d figures saved to %s.", 10, figures_dir)
