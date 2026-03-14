"""E5: Activation state detection analysis — 10 plot functions.

Ported from ``src/state_analysis.py``.  Analyses hidden-state
trajectories in PCA space during autoregressive decoding — regime
stability, drift from prefill, and change-point detection.

Plots
-----
01. pca_trajectory              — Multi-panel scatter: PC1 vs PC2
02. step_cosine                 — Line: step_cosine by decode_step
03. centroid_distance           — Line: centroid_distance by decode_step
04. norm_trajectory             — Line: norm by decode_step
05. step_cosine_heatmap         — Heatmap: step_cosine (layer x step)
06. centroid_distance_heatmap   — Heatmap: centroid_distance
07. change_point_detection      — 2-panel: derivatives
08. trajectory_speed            — Line: PCA displacement per step
09. dataset_centroid_comparison — 2-panel by dataset
10. pca_variance_by_layer       — 2-panel: variance by layer
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
    return ExperimentStore.load_table(results_dir, "trajectory")


# ===================================================================
# Plot 1: PCA Trajectory
# ===================================================================
def plot_pca_trajectory(df, figures_dir):
    if df.empty or "pc0" not in df.columns or "pc1" not in df.columns:
        return
    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers, max_show=6)
    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    exp_ids = df["experiment_id"].unique()
    sub_df = df[df["experiment_id"] == exp_ids[0]] if len(exp_ids) > 0 else df

    for i, layer in enumerate(selected):
        ax = axes_flat[i]
        layer_data = sub_df[sub_df["layer"] == layer].sort_values("decode_step")
        if layer_data.empty:
            continue
        sc = ax.scatter(
            layer_data["pc0"], layer_data["pc1"],
            c=layer_data["decode_step"], cmap="viridis", s=8, alpha=0.7,
        )
        ax.plot(layer_data["pc0"].values, layer_data["pc1"].values,
                "k-", alpha=0.15, linewidth=0.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Layer {layer}")
        plt.colorbar(sc, ax=ax, label="Decode Step")

    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("PCA Trajectory: PC1 vs PC2 by Layer", fontsize=14, y=1.02)
    _savefig(fig, figures_dir, "pca_trajectory", 1)


# ===================================================================
# Plot 2: Step Cosine
# ===================================================================
def plot_step_cosine(df, figures_dir):
    if df.empty:
        return
    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df[df["layer"] == layer].sort_values("decode_step")
        grouped = sub.groupby("decode_step")["step_cosine"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Step-to-Step Cosine Similarity")
    ax.set_title("Regime Stability: Step-to-Step Cosine over Decoding")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.1, 1.05)
    _savefig(fig, figures_dir, "step_cosine", 2)


# ===================================================================
# Plot 3: Centroid Distance
# ===================================================================
def plot_centroid_distance(df, figures_dir):
    if df.empty:
        return
    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df[df["layer"] == layer].sort_values("decode_step")
        grouped = sub.groupby("decode_step")["centroid_distance"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Distance from Prefill Centroid (PCA)")
    ax.set_title("Drift from Prefill: Centroid Distance over Decoding")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "centroid_distance", 3)


# ===================================================================
# Plot 4: Norm Trajectory
# ===================================================================
def plot_norm_trajectory(df, figures_dir):
    if df.empty:
        return
    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df[df["layer"] == layer].sort_values("decode_step")
        grouped = sub.groupby("decode_step")["norm"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Hidden State Norm")
    ax.set_title("Norm Trajectory over Decoding")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "norm_trajectory", 4)


# ===================================================================
# Plot 5: Step Cosine Heatmap
# ===================================================================
def plot_step_cosine_heatmap(df, figures_dir):
    if df.empty:
        return
    pivot = df.pivot_table(
        values="step_cosine", index="layer", columns="decode_step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.06), max(6, len(pivot) * 0.5))
    )
    sns.heatmap(pivot, cmap="RdYlGn", vmin=-0.2, vmax=1, ax=ax)
    ax.set_title("Step Cosine: Layer x Decode Step")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Decode Step")
    _savefig(fig, figures_dir, "step_cosine_heatmap", 5)


# ===================================================================
# Plot 6: Centroid Distance Heatmap
# ===================================================================
def plot_centroid_distance_heatmap(df, figures_dir):
    if df.empty:
        return
    pivot = df.pivot_table(
        values="centroid_distance", index="layer", columns="decode_step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.06), max(6, len(pivot) * 0.5))
    )
    sns.heatmap(pivot, cmap="magma", ax=ax)
    ax.set_title("Centroid Distance: Layer x Decode Step (Drift from Prefill)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Decode Step")
    _savefig(fig, figures_dir, "centroid_distance_heatmap", 6)


# ===================================================================
# Plot 7: Change-Point Detection
# ===================================================================
def plot_change_point_detection(df, figures_dir):
    if df.empty:
        return
    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers, max_show=4)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for layer in selected:
        sub = df[df["layer"] == layer]
        profile = sub.groupby("decode_step")["centroid_distance"].mean().sort_index()
        if len(profile) < 3:
            continue
        values = profile.values
        steps = profile.index.values
        grad1 = np.gradient(values)
        axes[0].plot(steps, grad1, label=f"Layer {layer}", alpha=0.8)
        grad2 = np.gradient(grad1)
        axes[1].plot(steps, grad2, label=f"Layer {layer}", alpha=0.8)
    axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("d(centroid_dist)/d(step)")
    axes[0].set_title("First Derivative — Drift Rate")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[1].set_xlabel("Decode Step")
    axes[1].set_ylabel("d\u00b2(centroid_dist)/d(step)\u00b2")
    axes[1].set_title("Second Derivative — Regime Shift Detection\n(sharp peaks = change points)")
    axes[1].legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "change_point_detection", 7)


# ===================================================================
# Plot 8: Trajectory Speed
# ===================================================================
def plot_trajectory_speed(df, figures_dir):
    if df.empty or "pc0" not in df.columns:
        return
    pc_cols = [c for c in df.columns if c.startswith("pc") and c[2:].isdigit()]
    if not pc_cols:
        return
    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df[df["layer"] == layer].sort_values(["experiment_id", "decode_step"])
        speeds = []
        step_indices = []
        for exp_id in sub["experiment_id"].unique():
            exp_data = sub[sub["experiment_id"] == exp_id].sort_values("decode_step")
            coords = exp_data[pc_cols].values
            if len(coords) < 2:
                continue
            diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            steps = exp_data["decode_step"].values[1:]
            speeds.extend(diffs)
            step_indices.extend(steps)
        if not speeds:
            continue
        speed_df = pd.DataFrame({"decode_step": step_indices, "speed": speeds})
        grouped = speed_df.groupby("decode_step")["speed"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Decode Step")
    ax.set_ylabel("Trajectory Speed (PCA L2 distance)")
    ax.set_title("Trajectory Speed: ||proj[t+1] - proj[t]|| over Decoding")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "trajectory_speed", 8)


# ===================================================================
# Plot 9: Dataset Centroid Comparison
# ===================================================================
def plot_dataset_centroid_comparison(df, figures_dir):
    if df.empty or "dataset_name" not in df.columns:
        return
    datasets = df["dataset_name"].unique()
    if len(datasets) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for ds in datasets:
        sub = df[df["dataset_name"] == ds]
        by_step = sub.groupby("decode_step")["centroid_distance"].mean().sort_index()
        axes[0].plot(by_step.index, by_step.values, label=_ds_label(ds), alpha=0.8)
    axes[0].set_xlabel("Decode Step")
    axes[0].set_ylabel("Centroid Distance")
    axes[0].set_title("Centroid Distance by Dataset (averaged over layers)")
    axes[0].legend(fontsize=7, ncol=2)
    for ds in datasets:
        sub = df[df["dataset_name"] == ds]
        by_step = sub.groupby("decode_step")["step_cosine"].mean().sort_index()
        axes[1].plot(by_step.index, by_step.values, label=_ds_label(ds), alpha=0.8)
    axes[1].set_xlabel("Decode Step")
    axes[1].set_ylabel("Step Cosine")
    axes[1].set_title("Step Cosine by Dataset (averaged over layers)")
    axes[1].legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "dataset_centroid_comparison", 9)


# ===================================================================
# Plot 10: PCA Variance by Layer
# ===================================================================
def plot_pca_variance_by_layer(df, figures_dir):
    if df.empty:
        return
    pc_cols = [c for c in df.columns if c.startswith("pc") and c[2:].isdigit()]
    if not pc_cols:
        return
    all_layers = sorted(df["layer"].unique())
    var_data = []
    for layer in all_layers:
        sub = df[df["layer"] == layer]
        variances = sub[pc_cols].var().values
        total_var = variances.sum()
        if total_var > 0:
            frac_top1 = variances[0] / total_var
            frac_top3 = variances[:3].sum() / total_var if len(variances) >= 3 else 1.0
            frac_top5 = variances[:5].sum() / total_var if len(variances) >= 5 else 1.0
        else:
            frac_top1 = frac_top3 = frac_top5 = 0.0
        var_data.append({
            "layer": layer,
            "frac_top1": frac_top1,
            "frac_top3": frac_top3,
            "frac_top5": frac_top5,
            "total_variance": total_var,
        })
    var_df = pd.DataFrame(var_data)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    axes[0].plot(var_df["layer"], var_df["frac_top1"], "b-", label="PC1", linewidth=2)
    axes[0].plot(var_df["layer"], var_df["frac_top3"], "g-", label="PC1-3", linewidth=2)
    axes[0].plot(var_df["layer"], var_df["frac_top5"], "r-", label="PC1-5", linewidth=2)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Fraction of Total Variance")
    axes[0].set_title("PCA Variance Concentration by Layer")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[1].plot(var_df["layer"], var_df["total_variance"], "k-", linewidth=2)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Total PCA Variance")
    axes[1].set_title("Total Trajectory Variance by Layer")
    _savefig(fig, figures_dir, "pca_variance_by_layer", 10)


# ===================================================================
# Master function
# ===================================================================
def run_all_state_analyses(results_dir: str, figures_dir: str):
    """Run all 10 state detection analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading trajectory data from %s", results_dir)

    df = _load_data(results_dir)
    if df.empty:
        logger.warning("No trajectory data found in %s. Skipping.", results_dir)
        return

    logger.info("Loaded %d trajectory records.", len(df))

    for name, fn, args in [
        ("pca_trajectory", plot_pca_trajectory, (df, figures_dir)),
        ("step_cosine", plot_step_cosine, (df, figures_dir)),
        ("centroid_distance", plot_centroid_distance, (df, figures_dir)),
        ("norm_trajectory", plot_norm_trajectory, (df, figures_dir)),
        ("step_cosine_heatmap", plot_step_cosine_heatmap, (df, figures_dir)),
        ("centroid_distance_heatmap", plot_centroid_distance_heatmap, (df, figures_dir)),
        ("change_point_detection", plot_change_point_detection, (df, figures_dir)),
        ("trajectory_speed", plot_trajectory_speed, (df, figures_dir)),
        ("dataset_centroid_comparison", plot_dataset_centroid_comparison, (df, figures_dir)),
        ("pca_variance_by_layer", plot_pca_variance_by_layer, (df, figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info("All 10 state detection plots complete. Figures saved to %s", figures_dir)
