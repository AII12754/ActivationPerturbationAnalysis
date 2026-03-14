"""E7: Perturbation sensitivity analysis — 10 plot functions.

New design for perturbation sensitivity experiments that measure how
adding noise along SVD or random directions affects model predictions.

Plots
-----
01. kl_by_layer_svd_vs_random      — Line: KL by layer (SVD vs random)
02. sensitivity_by_magnitude       — Line: KL vs magnitude per layer
03. direction_index_effect         — Line: KL vs direction_index (SVD)
04. kl_heatmap_layer_magnitude     — Heatmap: KL (layer x magnitude)
05. svd_vs_random_comparison       — 2-panel: box + ratio line
06. prediction_flip_profile        — 2-panel: flip rate by layer + by magnitude
07. dataset_comparison             — Line per dataset
08. context_length_effect          — Line per context_length
09. direction_magnitude_interaction — Heatmap: KL (direction x magnitude)
10. sensitivity_spectrum           — Line: KL by direction_index per layer
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
    return ExperimentStore.load_table(results_dir, "perturbation")


# ===================================================================
# Plot 1: KL by Layer — SVD vs Random
# ===================================================================
def plot_kl_by_layer_svd_vs_random(df, figures_dir):
    if df.empty or "perturbation_type" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ptype in sorted(df["perturbation_type"].unique()):
        sub = df[df["perturbation_type"] == ptype]
        profile = sub.groupby("layer")["kl_divergence"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], linewidth=2, label=ptype)
        ax.fill_between(
            profile["layer"],
            (profile["mean"] - profile["std"]).clip(lower=0),
            profile["mean"] + profile["std"],
            alpha=0.15,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence by Layer: SVD vs Random Perturbation")
    ax.legend(fontsize=10)
    _savefig(fig, figures_dir, "kl_by_layer_svd_vs_random", 1)


# ===================================================================
# Plot 2: Sensitivity by Magnitude
# ===================================================================
def plot_sensitivity_by_magnitude(df, figures_dir):
    if df.empty or "magnitude" not in df.columns:
        return
    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers, max_show=6)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df[df["layer"] == layer]
        profile = sub.groupby("magnitude")["kl_divergence"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=f"Layer {layer}", alpha=0.8, marker="o", markersize=4)
    ax.set_xlabel("Perturbation Magnitude")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Sensitivity vs Perturbation Magnitude by Layer")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "sensitivity_by_magnitude", 2)


# ===================================================================
# Plot 3: Direction Index Effect (SVD)
# ===================================================================
def plot_direction_index_effect(df, figures_dir):
    if df.empty or "direction_index" not in df.columns:
        return
    svd = df[df["perturbation_type"] == "svd"] if "perturbation_type" in df.columns else df
    if svd.empty:
        return
    all_layers = sorted(svd["layer"].unique())
    selected = _select_layers(all_layers, max_show=6)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = svd[svd["layer"] == layer]
        profile = sub.groupby("direction_index")["kl_divergence"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("SVD Direction Index (0 = top SV)")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Sensitivity Along SVD Directions by Layer")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "direction_index_effect", 3)


# ===================================================================
# Plot 4: KL Heatmap (layer x magnitude)
# ===================================================================
def plot_kl_heatmap_layer_magnitude(df, figures_dir):
    if df.empty or "magnitude" not in df.columns:
        return
    pivot = df.pivot_table(
        values="kl_divergence", index="layer", columns="magnitude", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_title("KL Divergence: Layer x Perturbation Magnitude")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Magnitude")
    _savefig(fig, figures_dir, "kl_heatmap_layer_magnitude", 4)


# ===================================================================
# Plot 5: SVD vs Random Comparison
# ===================================================================
def plot_svd_vs_random_comparison(df, figures_dir):
    if df.empty or "perturbation_type" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Box plot
    sns.boxplot(data=df, x="perturbation_type", y="kl_divergence", ax=axes[0])
    axes[0].set_xlabel("Perturbation Type")
    axes[0].set_ylabel("KL Divergence")
    axes[0].set_title("KL Distribution: SVD vs Random")

    # Ratio line
    svd_profile = df[df["perturbation_type"] == "svd"].groupby("layer")["kl_divergence"].mean()
    rand_profile = df[df["perturbation_type"] == "random"].groupby("layer")["kl_divergence"].mean()
    common_layers = svd_profile.index.intersection(rand_profile.index)
    if len(common_layers) > 0:
        ratio = svd_profile.loc[common_layers] / rand_profile.loc[common_layers].clip(lower=1e-8)
        axes[1].plot(common_layers, ratio.values, "b-", linewidth=2, marker=".")
        axes[1].axhline(1, color="gray", ls="--", alpha=0.5)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("SVD / Random KL Ratio")
        axes[1].set_title("SVD-to-Random Sensitivity Ratio by Layer")
    _savefig(fig, figures_dir, "svd_vs_random_comparison", 5)


# ===================================================================
# Plot 6: Prediction Flip Profile
# ===================================================================
def plot_prediction_flip_profile(df, figures_dir):
    if df.empty or "prediction_flip" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # By layer
    for ptype in sorted(df["perturbation_type"].unique()) if "perturbation_type" in df.columns else ["all"]:
        sub = df[df["perturbation_type"] == ptype] if "perturbation_type" in df.columns else df
        profile = sub.groupby("layer")["prediction_flip"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, linewidth=2, label=ptype)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Flip Rate")
    axes[0].set_title("Prediction Flip Rate by Layer")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)

    # By magnitude
    if "magnitude" in df.columns:
        for ptype in sorted(df["perturbation_type"].unique()) if "perturbation_type" in df.columns else ["all"]:
            sub = df[df["perturbation_type"] == ptype] if "perturbation_type" in df.columns else df
            profile = sub.groupby("magnitude")["prediction_flip"].mean().sort_index()
            axes[1].plot(profile.index, profile.values, linewidth=2, label=ptype, marker="o", markersize=4)
        axes[1].set_xlabel("Magnitude")
        axes[1].set_ylabel("Flip Rate")
        axes[1].set_title("Prediction Flip Rate by Magnitude")
        axes[1].legend(fontsize=9)
    _savefig(fig, figures_dir, "prediction_flip_profile", 6)


# ===================================================================
# Plot 7: Dataset Comparison
# ===================================================================
def plot_dataset_comparison(df, figures_dir):
    if df.empty or "dataset_name" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in df["dataset_name"].unique():
        sub = df[df["dataset_name"] == ds]
        profile = sub.groupby("layer")["kl_divergence"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Perturbation Sensitivity by Layer — Dataset Comparison")
    ax.legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "dataset_comparison", 7)


# ===================================================================
# Plot 8: Context Length Effect
# ===================================================================
def plot_context_length_effect(df, figures_dir):
    if df.empty or "context_length" not in df.columns:
        return
    ctx_lengths = sorted(df["context_length"].unique())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cl in ctx_lengths:
        sub = df[df["context_length"] == cl]
        profile = sub.groupby("layer")["kl_divergence"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Perturbation Sensitivity by Context Length")
    ax.legend(fontsize=8)
    _savefig(fig, figures_dir, "context_length_effect", 8)


# ===================================================================
# Plot 9: Direction x Magnitude Interaction
# ===================================================================
def plot_direction_magnitude_interaction(df, figures_dir):
    if df.empty or "direction_index" not in df.columns or "magnitude" not in df.columns:
        return
    svd = df[df["perturbation_type"] == "svd"] if "perturbation_type" in df.columns else df
    if svd.empty:
        return
    pivot = svd.pivot_table(
        values="kl_divergence", index="direction_index", columns="magnitude", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_title("KL Divergence: SVD Direction x Magnitude")
    ax.set_ylabel("Direction Index")
    ax.set_xlabel("Magnitude")
    _savefig(fig, figures_dir, "direction_magnitude_interaction", 9)


# ===================================================================
# Plot 10: Sensitivity Spectrum
# ===================================================================
def plot_sensitivity_spectrum(df, figures_dir):
    if df.empty or "direction_index" not in df.columns:
        return
    svd = df[df["perturbation_type"] == "svd"] if "perturbation_type" in df.columns else df
    if svd.empty:
        return
    all_layers = sorted(svd["layer"].unique())
    selected = _select_layers(all_layers, max_show=6)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = svd[svd["layer"] == layer]
        profile = sub.groupby("direction_index")["kl_divergence"].mean().sort_index()
        ax.plot(profile.index, profile.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("SVD Direction Index")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Sensitivity Spectrum: KL by Direction Index per Layer")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "sensitivity_spectrum", 10)


# ===================================================================
# Master function
# ===================================================================
def run_all_perturbation_analyses(results_dir: str, figures_dir: str):
    """Run all 10 perturbation sensitivity analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading perturbation data from %s", results_dir)

    df = _load_data(results_dir)
    if df.empty:
        logger.warning("No perturbation data found in %s. Skipping.", results_dir)
        return

    logger.info("Loaded %d perturbation records.", len(df))

    for name, fn, args in [
        ("kl_by_layer_svd_vs_random", plot_kl_by_layer_svd_vs_random, (df, figures_dir)),
        ("sensitivity_by_magnitude", plot_sensitivity_by_magnitude, (df, figures_dir)),
        ("direction_index_effect", plot_direction_index_effect, (df, figures_dir)),
        ("kl_heatmap_layer_magnitude", plot_kl_heatmap_layer_magnitude, (df, figures_dir)),
        ("svd_vs_random_comparison", plot_svd_vs_random_comparison, (df, figures_dir)),
        ("prediction_flip_profile", plot_prediction_flip_profile, (df, figures_dir)),
        ("dataset_comparison", plot_dataset_comparison, (df, figures_dir)),
        ("context_length_effect", plot_context_length_effect, (df, figures_dir)),
        ("direction_magnitude_interaction", plot_direction_magnitude_interaction, (df, figures_dir)),
        ("sensitivity_spectrum", plot_sensitivity_spectrum, (df, figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info("All 10 perturbation plots complete. Figures saved to %s", figures_dir)
