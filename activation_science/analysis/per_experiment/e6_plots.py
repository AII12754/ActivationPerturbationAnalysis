"""E6: Causal intervention analysis — 10 plot functions.

New design for causal intervention experiments that measure how
zeroing-out or replacing layer activations affects downstream predictions.

Plots
-----
01. kl_by_layer_per_intervention   — Line: KL by layer per intervention_type
02. prediction_flip_by_layer       — Line: flip rate by layer per intervention_type
03. js_divergence_by_layer         — Line: JS divergence by layer
04. kl_heatmap                     — Heatmap: KL (layer x intervention_type)
05. critical_layers                — Horizontal bar: top layers by KL
06. context_length_sensitivity     — 2-panel: KL and flip by context_length
07. dataset_comparison             — 2-panel: KL by layer per dataset
08. layer_importance_ranking       — Grouped bar: KL per layer by intervention
09. intervention_type_comparison   — Box: KL per intervention_type
10. cumulative_prediction_flip     — Line: cumulative flip fraction by layer
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
    return ExperimentStore.load_table(results_dir, "causal")


# ===================================================================
# Plot 1: KL by Layer per Intervention Type
# ===================================================================
def plot_kl_by_layer_per_intervention(df, figures_dir):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for itype in sorted(df["intervention_type"].unique()):
        sub = df[df["intervention_type"] == itype]
        profile = sub.groupby("layer")["kl_divergence"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], linewidth=2, label=itype)
        ax.fill_between(
            profile["layer"],
            (profile["mean"] - profile["std"]).clip(lower=0),
            profile["mean"] + profile["std"],
            alpha=0.15,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence by Layer per Intervention Type")
    ax.legend(fontsize=9)
    _savefig(fig, figures_dir, "kl_by_layer_per_intervention", 1)


# ===================================================================
# Plot 2: Prediction Flip by Layer
# ===================================================================
def plot_prediction_flip_by_layer(df, figures_dir):
    if df.empty or "prediction_flip" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for itype in sorted(df["intervention_type"].unique()):
        sub = df[df["intervention_type"] == itype]
        profile = sub.groupby("layer")["prediction_flip"].mean().sort_index()
        ax.plot(profile.index, profile.values, linewidth=2, label=itype, marker=".", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Prediction Flip Rate")
    ax.set_title("Prediction Flip Rate by Layer per Intervention Type")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    _savefig(fig, figures_dir, "prediction_flip_by_layer", 2)


# ===================================================================
# Plot 3: JS Divergence by Layer
# ===================================================================
def plot_js_divergence_by_layer(df, figures_dir):
    if df.empty or "js_divergence" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for itype in sorted(df["intervention_type"].unique()):
        sub = df[df["intervention_type"] == itype]
        profile = sub.groupby("layer")["js_divergence"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], linewidth=2, label=itype)
        ax.fill_between(
            profile["layer"],
            (profile["mean"] - profile["std"]).clip(lower=0),
            profile["mean"] + profile["std"],
            alpha=0.15,
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("JS Divergence")
    ax.set_title("Jensen-Shannon Divergence by Layer")
    ax.legend(fontsize=9)
    _savefig(fig, figures_dir, "js_divergence_by_layer", 3)


# ===================================================================
# Plot 4: KL Heatmap (layer x intervention_type)
# ===================================================================
def plot_kl_heatmap(df, figures_dir):
    if df.empty:
        return
    pivot = df.pivot_table(
        values="kl_divergence", index="layer", columns="intervention_type", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, annot=len(pivot.columns) <= 8)
    ax.set_title("KL Divergence: Layer x Intervention Type")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Intervention Type")
    _savefig(fig, figures_dir, "kl_heatmap", 4)


# ===================================================================
# Plot 5: Critical Layers (top by KL)
# ===================================================================
def plot_critical_layers(df, figures_dir):
    if df.empty:
        return
    ranking = df.groupby("layer")["kl_divergence"].mean().sort_values(ascending=False)
    top_n = min(20, len(ranking))
    ranking = ranking.head(top_n)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(ranking)))
    ax.barh(range(len(ranking)), ranking.values, color=colors, edgecolor="gray", linewidth=0.3)
    ax.set_yticks(range(len(ranking)))
    ax.set_yticklabels([f"L{int(l)}" for l in ranking.index], fontsize=7)
    ax.set_xlabel("Mean KL Divergence")
    ax.set_title("Critical Layers Ranked by Causal Impact (KL)")
    ax.invert_yaxis()
    _savefig(fig, figures_dir, "critical_layers", 5)


# ===================================================================
# Plot 6: Context Length Sensitivity
# ===================================================================
def plot_context_length_sensitivity(df, figures_dir):
    if df.empty or "context_length" not in df.columns:
        return
    ctx_lengths = sorted(df["context_length"].unique())
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for cl in ctx_lengths:
        sub = df[df["context_length"] == cl]
        profile = sub.groupby("layer")["kl_divergence"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("KL Divergence")
    axes[0].set_title("KL by Layer x Context Length")
    axes[0].legend(fontsize=8)

    if "prediction_flip" in df.columns:
        for cl in ctx_lengths:
            sub = df[df["context_length"] == cl]
            profile = sub.groupby("layer")["prediction_flip"].mean().sort_index()
            axes[1].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Flip Rate")
        axes[1].set_title("Prediction Flip by Layer x Context Length")
        axes[1].legend(fontsize=8)
    _savefig(fig, figures_dir, "context_length_sensitivity", 6)


# ===================================================================
# Plot 7: Dataset Comparison
# ===================================================================
def plot_dataset_comparison(df, figures_dir):
    if df.empty or "dataset_name" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for ds in df["dataset_name"].unique():
        sub = df[df["dataset_name"] == ds]
        profile = sub.groupby("layer")["kl_divergence"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("KL Divergence")
    axes[0].set_title("KL by Layer — Dataset Comparison")
    axes[0].legend(fontsize=7, ncol=2)

    if "prediction_flip" in df.columns:
        for ds in df["dataset_name"].unique():
            sub = df[df["dataset_name"] == ds]
            profile = sub.groupby("layer")["prediction_flip"].mean().sort_index()
            axes[1].plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Flip Rate")
        axes[1].set_title("Prediction Flip by Layer — Dataset Comparison")
        axes[1].legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "dataset_comparison", 7)


# ===================================================================
# Plot 8: Layer Importance Ranking (grouped bar)
# ===================================================================
def plot_layer_importance_ranking(df, figures_dir):
    if df.empty:
        return
    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers, max_show=12)
    sub = df[df["layer"].isin(selected)]
    pivot = sub.pivot_table(
        values="kl_divergence", index="layer", columns="intervention_type", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    pivot.plot(kind="bar", ax=ax, edgecolor="gray", linewidth=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean KL Divergence")
    ax.set_title("Layer Importance by Intervention Type")
    ax.legend(fontsize=8, title="Intervention")
    ax.tick_params(axis="x", rotation=0)
    _savefig(fig, figures_dir, "layer_importance_ranking", 8)


# ===================================================================
# Plot 9: Intervention Type Comparison (box)
# ===================================================================
def plot_intervention_type_comparison(df, figures_dir):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=df, x="intervention_type", y="kl_divergence", ax=ax)
    ax.set_xlabel("Intervention Type")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence Distribution by Intervention Type")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, figures_dir, "intervention_type_comparison", 9)


# ===================================================================
# Plot 10: Cumulative Prediction Flip
# ===================================================================
def plot_cumulative_prediction_flip(df, figures_dir):
    if df.empty or "prediction_flip" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for itype in sorted(df["intervention_type"].unique()):
        sub = df[df["intervention_type"] == itype]
        profile = sub.groupby("layer")["prediction_flip"].mean().sort_index()
        cumulative = profile.cumsum() / profile.sum() if profile.sum() > 0 else profile.cumsum()
        ax.plot(cumulative.index, cumulative.values, linewidth=2, label=itype)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cumulative Flip Fraction")
    ax.set_title("Cumulative Prediction Flip by Layer")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="50%")
    _savefig(fig, figures_dir, "cumulative_prediction_flip", 10)


# ===================================================================
# Master function
# ===================================================================
def run_all_causal_analyses(results_dir: str, figures_dir: str):
    """Run all 10 causal intervention analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading causal data from %s", results_dir)

    df = _load_data(results_dir)
    if df.empty:
        logger.warning("No causal data found in %s. Skipping.", results_dir)
        return

    logger.info("Loaded %d causal records.", len(df))

    for name, fn, args in [
        ("kl_by_layer_per_intervention", plot_kl_by_layer_per_intervention, (df, figures_dir)),
        ("prediction_flip_by_layer", plot_prediction_flip_by_layer, (df, figures_dir)),
        ("js_divergence_by_layer", plot_js_divergence_by_layer, (df, figures_dir)),
        ("kl_heatmap", plot_kl_heatmap, (df, figures_dir)),
        ("critical_layers", plot_critical_layers, (df, figures_dir)),
        ("context_length_sensitivity", plot_context_length_sensitivity, (df, figures_dir)),
        ("dataset_comparison", plot_dataset_comparison, (df, figures_dir)),
        ("layer_importance_ranking", plot_layer_importance_ranking, (df, figures_dir)),
        ("intervention_type_comparison", plot_intervention_type_comparison, (df, figures_dir)),
        ("cumulative_prediction_flip", plot_cumulative_prediction_flip, (df, figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info("All 10 causal plots complete. Figures saved to %s", figures_dir)
