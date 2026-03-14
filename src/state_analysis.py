"""
Visualization module for activation state detection experiments.

Reads stored trajectory results and produces figures exploring
hidden-state trajectories in PCA space during autoregressive decoding.

Analysis suite (10 plots):
  1. PCA trajectory (PC1 vs PC2) colored by decode step
  2. Step-to-step cosine by decode step (regime stability)
  3. Centroid distance by decode step (drift from prefill)
  4. Norm trajectory by decode step
  5. Step cosine heatmap: layer x decode_step
  6. Centroid distance heatmap: layer x decode_step
  7. Change-point detection: second derivative of centroid distance
  8. Trajectory speed: ||projected[t+1] - projected[t]|| over time
  9. Dataset comparison: centroid distance profiles
  10. PCA variance explained by layer (from trajectory data)
  + Post-hoc KMeans clustering on PCA coordinates
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

from .state_storage import StateResultStore

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
# 1. PCA Trajectory Plots: PC1 vs PC2 colored by decode step
# ===================================================================
def plot_pca_trajectory(df, output_dir):
    """PC1 vs PC2 colored by decode step for selected layers."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty or "pc0" not in df.columns or "pc1" not in df.columns:
        return

    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers, max_show=6)

    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    # Use first experiment for a clean trajectory
    exp_ids = df["experiment_id"].unique()
    sub_df = df[df["experiment_id"] == exp_ids[0]] if len(exp_ids) > 0 else df

    for i, layer in enumerate(selected):
        ax = axes_flat[i]
        layer_data = sub_df[sub_df["layer"] == layer].sort_values("decode_step")
        if layer_data.empty:
            continue
        sc = ax.scatter(
            layer_data["pc0"], layer_data["pc1"],
            c=layer_data["decode_step"], cmap="viridis",
            s=8, alpha=0.7,
        )
        ax.plot(layer_data["pc0"].values, layer_data["pc1"].values,
                "k-", alpha=0.15, linewidth=0.5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"Layer {layer}")
        plt.colorbar(sc, ax=ax, label="Decode Step")

    # Hide unused axes.
    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("PCA Trajectory: PC1 vs PC2 by Layer", fontsize=14, y=1.02)
    _save(fig, out, "01_pca_trajectory.png")


# ===================================================================
# 2. Step-to-Step Cosine by Decode Step (regime stability)
# ===================================================================
def plot_step_cosine(df, output_dir):
    """Step-to-step cosine similarity over decode steps by layer."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df[df["layer"] == layer].sort_values("decode_step")
        grouped = sub.groupby("decode_step")["step_cosine"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Decode Step"); ax.set_ylabel("Step-to-Step Cosine Similarity")
    ax.set_title("Regime Stability: Step-to-Step Cosine over Decoding")
    ax.legend(fontsize=8, ncol=2); ax.set_ylim(-0.1, 1.05)
    _save(fig, out, "02_step_cosine.png")


# ===================================================================
# 3. Centroid Distance by Decode Step (drift from prefill)
# ===================================================================
def plot_centroid_distance(df, output_dir):
    """Centroid distance over decode steps by layer."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df[df["layer"] == layer].sort_values("decode_step")
        grouped = sub.groupby("decode_step")["centroid_distance"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Decode Step"); ax.set_ylabel("Distance from Prefill Centroid (PCA)")
    ax.set_title("Drift from Prefill: Centroid Distance over Decoding")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out, "03_centroid_distance.png")


# ===================================================================
# 4. Norm Trajectory by Decode Step
# ===================================================================
def plot_norm_trajectory(df, output_dir):
    """Hidden state norm over decode steps by layer."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df[df["layer"] == layer].sort_values("decode_step")
        grouped = sub.groupby("decode_step")["norm"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Decode Step"); ax.set_ylabel("Hidden State Norm")
    ax.set_title("Norm Trajectory over Decoding")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out, "04_norm_trajectory.png")


# ===================================================================
# 5. Step Cosine Heatmap: layer x decode_step
# ===================================================================
def plot_step_cosine_heatmap(df, output_dir):
    """Heatmap of step-to-step cosine similarity."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
    ax.set_ylabel("Layer"); ax.set_xlabel("Decode Step")
    _save(fig, out, "05_step_cosine_heatmap.png")


# ===================================================================
# 6. Centroid Distance Heatmap: layer x decode_step
# ===================================================================
def plot_centroid_distance_heatmap(df, output_dir):
    """Heatmap of centroid distance from prefill."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
    ax.set_ylabel("Layer"); ax.set_xlabel("Decode Step")
    _save(fig, out, "06_centroid_distance_heatmap.png")


# ===================================================================
# 7. Change-Point Detection: second derivative of centroid distance
# ===================================================================
def plot_change_point_detection(df, output_dir):
    """Second derivative of centroid distance — sharp changes indicate
    regime shifts in the model's hidden-state trajectory.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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

        # First derivative.
        grad1 = np.gradient(values)
        axes[0].plot(steps, grad1, label=f"Layer {layer}", alpha=0.8)

        # Second derivative — change-point signal.
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

    _save(fig, out, "07_change_point_detection.png")


# ===================================================================
# 8. Trajectory Speed: ||projected[t+1] - projected[t]|| over time
# ===================================================================
def plot_trajectory_speed(df, output_dir):
    """Speed of the PCA trajectory: Euclidean distance between consecutive
    projected vectors over decode steps.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
            coords = exp_data[pc_cols].values  # (n_steps, n_components)
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

    ax.set_xlabel("Decode Step"); ax.set_ylabel("Trajectory Speed (PCA L2 distance)")
    ax.set_title("Trajectory Speed: ||proj[t+1] - proj[t]|| over Decoding")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out, "08_trajectory_speed.png")


# ===================================================================
# 9. Dataset Comparison: Centroid Distance Profiles
# ===================================================================
def plot_dataset_centroid_comparison(df, output_dir):
    """Compare centroid distance profiles across datasets."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty or "dataset_name" not in df.columns:
        return

    datasets = df["dataset_name"].unique()
    if len(datasets) < 2:
        return

    # Average across layers for a cleaner signal.
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Panel 1: centroid distance by decode step, per dataset.
    for ds in datasets:
        sub = df[df["dataset_name"] == ds]
        by_step = sub.groupby("decode_step")["centroid_distance"].mean().sort_index()
        axes[0].plot(by_step.index, by_step.values, label=_ds_label(ds), alpha=0.8)
    axes[0].set_xlabel("Decode Step"); axes[0].set_ylabel("Centroid Distance")
    axes[0].set_title("Centroid Distance by Dataset (averaged over layers)")
    axes[0].legend(fontsize=7, ncol=2)

    # Panel 2: step cosine by decode step, per dataset.
    for ds in datasets:
        sub = df[df["dataset_name"] == ds]
        by_step = sub.groupby("decode_step")["step_cosine"].mean().sort_index()
        axes[1].plot(by_step.index, by_step.values, label=_ds_label(ds), alpha=0.8)
    axes[1].set_xlabel("Decode Step"); axes[1].set_ylabel("Step Cosine")
    axes[1].set_title("Step Cosine by Dataset (averaged over layers)")
    axes[1].legend(fontsize=7, ncol=2)

    _save(fig, out, "09_dataset_centroid_comparison.png")


# ===================================================================
# 10. PCA Variance Explained by Layer (from trajectory data)
# ===================================================================
def plot_pca_variance_by_layer(df, output_dir):
    """Estimate effective PCA variance from trajectory coordinates.

    Compute the variance of each PC coordinate across decode steps and
    show how variance distributes across components per layer.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    pc_cols = [c for c in df.columns if c.startswith("pc") and c[2:].isdigit()]
    if not pc_cols:
        return

    all_layers = sorted(df["layer"].unique())

    # Compute variance per PC per layer.
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

    # Panel 1: cumulative variance fraction.
    axes[0].plot(var_df["layer"], var_df["frac_top1"], "b-", label="PC1", linewidth=2)
    axes[0].plot(var_df["layer"], var_df["frac_top3"], "g-", label="PC1-3", linewidth=2)
    axes[0].plot(var_df["layer"], var_df["frac_top5"], "r-", label="PC1-5", linewidth=2)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Fraction of Total Variance")
    axes[0].set_title("PCA Variance Concentration by Layer")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1.05)

    # Panel 2: total variance.
    axes[1].plot(var_df["layer"], var_df["total_variance"], "k-", linewidth=2)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Total PCA Variance")
    axes[1].set_title("Total Trajectory Variance by Layer")

    _save(fig, out, "10_pca_variance_by_layer.png")


# ===================================================================
# Bonus: KMeans Clustering on PCA coordinates
# ===================================================================
def plot_kmeans_clustering(df, output_dir, n_clusters: int = 4):
    """Post-hoc KMeans clustering on PCA coordinates.

    Clusters decode steps and shows cluster membership over time.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    pc_cols = [c for c in df.columns if c.startswith("pc") and c[2:].isdigit()]
    if not pc_cols:
        return

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.warning("scikit-learn not installed; skipping KMeans clustering plot.")
        return

    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers, max_show=4)

    # Use first experiment for clean clustering.
    exp_ids = df["experiment_id"].unique()
    sub_df = df[df["experiment_id"] == exp_ids[0]] if len(exp_ids) > 0 else df

    ncols = min(2, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, layer in enumerate(selected):
        ax = axes_flat[i]
        layer_data = sub_df[sub_df["layer"] == layer].sort_values("decode_step")
        if len(layer_data) < n_clusters:
            continue

        coords = layer_data[pc_cols].dropna().values
        if len(coords) < n_clusters:
            continue

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(coords)

        steps = layer_data["decode_step"].values[:len(labels)]
        sc = ax.scatter(
            layer_data["pc0"].values[:len(labels)],
            layer_data["pc1"].values[:len(labels)],
            c=labels, cmap="tab10", s=12, alpha=0.8,
        )
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"Layer {layer} — {n_clusters} Clusters")
        plt.colorbar(sc, ax=ax, label="Cluster ID")

    for j in range(len(selected), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"KMeans Clustering (k={n_clusters}) on PCA Coordinates", fontsize=14, y=1.02)
    _save(fig, out, "11_kmeans_clustering.png")

    # Also show cluster membership over time.
    fig2, axes2 = plt.subplots(len(selected), 1,
                               figsize=(12, 3 * len(selected)), squeeze=False)
    for i, layer in enumerate(selected):
        ax = axes2[i, 0]
        layer_data = sub_df[sub_df["layer"] == layer].sort_values("decode_step")
        if len(layer_data) < n_clusters:
            continue

        coords = layer_data[pc_cols].dropna().values
        if len(coords) < n_clusters:
            continue

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(coords)
        steps = layer_data["decode_step"].values[:len(labels)]

        ax.scatter(steps, labels, c=labels, cmap="tab10", s=6, alpha=0.7)
        ax.set_ylabel(f"Cluster (L{layer})")
        ax.set_yticks(range(n_clusters))
        if i == len(selected) - 1:
            ax.set_xlabel("Decode Step")

    fig2.suptitle("Cluster Membership over Decode Steps", fontsize=14)
    _save(fig2, out, "12_kmeans_membership.png")


# ===================================================================
# Master routine
# ===================================================================
def run_all_state_analyses(
    results_dir: str = "./results_state",
    figures_dir: Optional[str] = None,
):
    """Run the full state detection analysis suite and save all figures."""
    if figures_dir is None:
        figures_dir = str(Path(results_dir) / "figures")

    df = StateResultStore.load_trajectories(results_dir)

    if df.empty:
        logger.error("No data found in %s. Run state experiments first.", results_dir)
        return

    logger.info(
        "Loaded %d trajectory records. Generating figures...",
        len(df),
    )

    # --- Core trajectory plots (1-4) ---
    plot_pca_trajectory(df, figures_dir)
    plot_step_cosine(df, figures_dir)
    plot_centroid_distance(df, figures_dir)
    plot_norm_trajectory(df, figures_dir)

    # --- Heatmaps (5-6) ---
    plot_step_cosine_heatmap(df, figures_dir)
    plot_centroid_distance_heatmap(df, figures_dir)

    # --- Regime detection (7-8) ---
    plot_change_point_detection(df, figures_dir)
    plot_trajectory_speed(df, figures_dir)

    # --- Comparisons (9-10) ---
    plot_dataset_centroid_comparison(df, figures_dir)
    plot_pca_variance_by_layer(df, figures_dir)

    # --- Post-hoc clustering (bonus) ---
    plot_kmeans_clustering(df, figures_dir)

    logger.info("All figures saved to %s.", figures_dir)
