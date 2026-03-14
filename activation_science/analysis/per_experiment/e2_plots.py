"""E2: Logit lens analysis — 10 plot functions.

Ported from ``src/logit_lens_analysis.py``.  Analyses how intermediate
layers predict the final token — when the model "decides" on its output
and how certainty builds across layers.

Plots
-----
01. crystallization_depth        — Histogram: first layer where rank=0
02. entropy_waterfall            — 2-panel: entropy by layer
03. kl_from_final                — 2-panel: KL by layer (log scale)
04. rank_trajectory_heatmap      — Heatmap: log1p(rank) (layer x step)
05. cross_entropy_heatmap        — Heatmap: cross_entropy (layer x step)
06. max_probability_by_layer     — Line + heatmap: max_prob
07. top1_accuracy_by_layer       — 2-panel: accuracy by layer + step
08. entropy_by_dataset           — 2-panel: entropy & KL per dataset
09. crystallization_by_dataset   — Box: crystallization depth per dataset
10. kl_by_context_length         — 2-panel: KL & entropy per context_length
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
    perstep = ExperimentStore.load_table(results_dir, "perstep")
    topk = ExperimentStore.load_table(results_dir, "topk")
    return perstep, topk


# ===================================================================
# Plot 1: Crystallization Depth
# ===================================================================
def plot_crystallization_depth(df_perstep, figures_dir):
    if df_perstep.empty:
        return

    def _first_zero_rank(group):
        zero_rank = group[group["rank_of_correct"] == 0]
        if zero_rank.empty:
            return np.nan
        return zero_rank["layer"].min()

    crystal = df_perstep.groupby(
        ["experiment_id", "decode_step"]
    ).apply(_first_zero_rank).reset_index(name="crystal_layer")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    valid = crystal["crystal_layer"].dropna()
    axes[0].hist(valid, bins=50, alpha=0.7, edgecolor="black", density=True)
    if len(valid) > 0:
        axes[0].axvline(valid.median(), color="red", ls="--",
                        label=f"Median={valid.median():.0f}")
    axes[0].set_xlabel("Crystallization Layer")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of Crystallization Depth\n(first layer where rank=0)")
    axes[0].legend(fontsize=8)

    by_step = crystal.groupby("decode_step")["crystal_layer"].mean().sort_index()
    axes[1].plot(by_step.index, by_step.values, "b-", linewidth=2)
    axes[1].set_xlabel("Decode Step")
    axes[1].set_ylabel("Mean Crystallization Layer")
    axes[1].set_title("Crystallization Depth over Decoding")
    _savefig(fig, figures_dir, "crystallization_depth", 1)


# ===================================================================
# Plot 2: Entropy Waterfall
# ===================================================================
def plot_entropy_waterfall(df_perstep, figures_dir):
    if df_perstep.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    profile = df_perstep.groupby("layer")["entropy"].agg(["mean", "std"]).reset_index()
    axes[0].plot(profile["layer"], profile["mean"], "b-", linewidth=2)
    axes[0].fill_between(
        profile["layer"],
        profile["mean"] - profile["std"],
        profile["mean"] + profile["std"],
        alpha=0.2,
    )
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].set_title("Entropy Waterfall: Certainty Building Across Layers")

    all_steps = sorted(df_perstep["decode_step"].unique())
    selected_steps = _select_layers(all_steps, max_show=6)
    for step in selected_steps:
        sub = df_perstep[df_perstep["decode_step"] == step]
        by_layer = sub.groupby("layer")["entropy"].mean().sort_index()
        axes[1].plot(by_layer.index, by_layer.values, label=f"Step {step}", alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Entropy (nats)")
    axes[1].set_title("Entropy by Layer for Selected Decode Steps")
    axes[1].legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "entropy_waterfall", 2)


# ===================================================================
# Plot 3: KL from Final Layer
# ===================================================================
def plot_kl_from_final(df_perstep, figures_dir):
    if df_perstep.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    profile = df_perstep.groupby("layer")["kl_from_final"].agg(["mean", "std"]).reset_index()
    axes[0].plot(profile["layer"], profile["mean"], "r-", linewidth=2)
    axes[0].fill_between(
        profile["layer"],
        (profile["mean"] - profile["std"]).clip(lower=0),
        profile["mean"] + profile["std"],
        alpha=0.2, color="red",
    )
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("KL Divergence (nats)")
    axes[0].set_title("KL Divergence from Final Layer\n(convergence to final prediction)")
    axes[0].set_yscale("log")

    all_steps = sorted(df_perstep["decode_step"].unique())
    selected_steps = _select_layers(all_steps, max_show=6)
    for step in selected_steps:
        sub = df_perstep[df_perstep["decode_step"] == step]
        by_layer = sub.groupby("layer")["kl_from_final"].mean().sort_index()
        axes[1].plot(by_layer.index, by_layer.values, label=f"Step {step}", alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("KL Divergence (nats)")
    axes[1].set_title("KL from Final Layer for Selected Decode Steps")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "kl_from_final", 3)


# ===================================================================
# Plot 4: Rank Trajectory Heatmap
# ===================================================================
def plot_rank_trajectory_heatmap(df_perstep, figures_dir):
    if df_perstep.empty:
        return
    pivot = df_perstep.pivot_table(
        values="rank_of_correct", index="layer", columns="decode_step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    sns.heatmap(np.log1p(pivot), cmap="YlOrRd_r", ax=ax)
    ax.set_title("Rank of Correct Token (log1p): Layer x Decode Step\n(darker = lower rank = better)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Decode Step")
    _savefig(fig, figures_dir, "rank_trajectory_heatmap", 4)


# ===================================================================
# Plot 5: Cross-Entropy Heatmap
# ===================================================================
def plot_cross_entropy_heatmap(df_perstep, figures_dir):
    if df_perstep.empty:
        return
    pivot = df_perstep.pivot_table(
        values="cross_entropy_correct", index="layer", columns="decode_step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    sns.heatmap(pivot, cmap="RdYlGn_r", ax=ax)
    ax.set_title("Cross-Entropy of Correct Token: Layer x Decode Step\n(lower = more confident)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Decode Step")
    _savefig(fig, figures_dir, "cross_entropy_heatmap", 5)


# ===================================================================
# Plot 6: Max Probability by Layer
# ===================================================================
def plot_max_probability_by_layer(df_perstep, figures_dir):
    if df_perstep.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    profile = df_perstep.groupby("layer")["max_prob"].agg(["mean", "std"]).reset_index()
    axes[0].plot(profile["layer"], profile["mean"], "g-", linewidth=2)
    axes[0].fill_between(
        profile["layer"],
        (profile["mean"] - profile["std"]).clip(lower=0),
        (profile["mean"] + profile["std"]).clip(upper=1),
        alpha=0.2, color="green",
    )
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Max Probability")
    axes[0].set_title("Max Probability by Layer\n(confidence building)")
    axes[0].set_ylim(0, 1.05)

    pivot = df_perstep.pivot_table(
        values="max_prob", index="layer", columns="decode_step", aggfunc="mean",
    )
    sns.heatmap(pivot, cmap="Greens", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Max Probability: Layer x Decode Step")
    axes[1].set_ylabel("Layer")
    axes[1].set_xlabel("Decode Step")
    _savefig(fig, figures_dir, "max_probability_by_layer", 6)


# ===================================================================
# Plot 7: Top-1 Accuracy by Layer
# ===================================================================
def plot_top1_accuracy_by_layer(df_perstep, figures_dir):
    if df_perstep.empty:
        return
    df = df_perstep.copy()
    df["top1_correct"] = (df["top1_token_id"] == df["correct_token_id"]).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    by_layer = df.groupby("layer")["top1_correct"].mean().sort_index()
    axes[0].plot(by_layer.index, by_layer.values * 100, "b-", linewidth=2, marker=".", markersize=4)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Logit Lens Accuracy by Layer\n(% of steps matching final layer)")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(100, color="gray", ls="--", alpha=0.3)

    all_layers = sorted(df["layer"].unique())
    selected = _select_layers(all_layers)
    for layer in selected:
        sub = df[df["layer"] == layer]
        by_step = sub.groupby("decode_step")["top1_correct"].mean().sort_index()
        axes[1].plot(by_step.index, by_step.values * 100, label=f"Layer {layer}", alpha=0.8)
    axes[1].set_xlabel("Decode Step")
    axes[1].set_ylabel("Top-1 Accuracy (%)")
    axes[1].set_title("Top-1 Accuracy over Decoding by Layer")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_ylim(0, 105)
    _savefig(fig, figures_dir, "top1_accuracy_by_layer", 7)


# ===================================================================
# Plot 8: Entropy by Dataset
# ===================================================================
def plot_entropy_by_dataset(df_perstep, figures_dir):
    if df_perstep.empty or "dataset_name" not in df_perstep.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for ds in df_perstep["dataset_name"].unique():
        sub = df_perstep[df_perstep["dataset_name"] == ds]
        profile = sub.groupby("layer")["entropy"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].set_title("Entropy by Layer — Dataset Comparison")
    axes[0].legend(fontsize=7, ncol=2)

    for ds in df_perstep["dataset_name"].unique():
        sub = df_perstep[df_perstep["dataset_name"] == ds]
        profile = sub.groupby("layer")["kl_from_final"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("KL Divergence (nats)")
    axes[1].set_title("KL from Final — Dataset Comparison")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "entropy_by_dataset", 8)


# ===================================================================
# Plot 9: Crystallization by Dataset
# ===================================================================
def plot_crystallization_by_dataset(df_perstep, figures_dir):
    if df_perstep.empty or "dataset_name" not in df_perstep.columns:
        return

    def _first_zero_rank(group):
        zero_rank = group[group["rank_of_correct"] == 0]
        if zero_rank.empty:
            return np.nan
        return zero_rank["layer"].min()

    crystal = df_perstep.groupby(
        ["experiment_id", "dataset_name", "decode_step"]
    ).apply(_first_zero_rank).reset_index(name="crystal_layer")

    crystal["ds_label"] = crystal["dataset_name"].apply(_ds_label)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    sns.boxplot(data=crystal, x="ds_label", y="crystal_layer", ax=axes[0])
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Crystallization Layer")
    axes[0].set_title("Crystallization Depth by Dataset")
    axes[0].tick_params(axis="x", rotation=45)

    for ds in crystal["dataset_name"].unique():
        sub = crystal[crystal["dataset_name"] == ds]
        by_step = sub.groupby("decode_step")["crystal_layer"].mean().sort_index()
        axes[1].plot(by_step.index, by_step.values, label=_ds_label(ds), alpha=0.8)
    axes[1].set_xlabel("Decode Step")
    axes[1].set_ylabel("Mean Crystallization Layer")
    axes[1].set_title("Crystallization Depth over Decoding by Dataset")
    axes[1].legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "crystallization_by_dataset", 9)


# ===================================================================
# Plot 10: KL by Context Length
# ===================================================================
def plot_kl_by_context_length(df_perstep, figures_dir):
    if df_perstep.empty or "context_length" not in df_perstep.columns:
        return
    ctx_lengths = sorted(df_perstep["context_length"].unique())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for cl in ctx_lengths:
        sub = df_perstep[df_perstep["context_length"] == cl]
        profile = sub.groupby("layer")["kl_from_final"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("KL Divergence (nats)")
    axes[0].set_title("KL from Final by Context Length")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    for cl in ctx_lengths:
        sub = df_perstep[df_perstep["context_length"] == cl]
        profile = sub.groupby("layer")["entropy"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Entropy (nats)")
    axes[1].set_title("Entropy by Layer x Context Length")
    axes[1].legend(fontsize=8)
    _savefig(fig, figures_dir, "kl_by_context_length", 10)


# ===================================================================
# Master function
# ===================================================================
def run_all_logit_lens_analyses(results_dir: str, figures_dir: str):
    """Run all 10 logit lens analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading logit lens data from %s", results_dir)

    df_perstep, df_topk = _load_data(results_dir)
    if df_perstep.empty and df_topk.empty:
        logger.warning("No logit lens data found in %s. Skipping.", results_dir)
        return

    logger.info("Loaded %d perstep and %d topk records.", len(df_perstep), len(df_topk))

    for name, fn, args in [
        ("crystallization_depth", plot_crystallization_depth, (df_perstep, figures_dir)),
        ("entropy_waterfall", plot_entropy_waterfall, (df_perstep, figures_dir)),
        ("kl_from_final", plot_kl_from_final, (df_perstep, figures_dir)),
        ("rank_trajectory_heatmap", plot_rank_trajectory_heatmap, (df_perstep, figures_dir)),
        ("cross_entropy_heatmap", plot_cross_entropy_heatmap, (df_perstep, figures_dir)),
        ("max_probability_by_layer", plot_max_probability_by_layer, (df_perstep, figures_dir)),
        ("top1_accuracy_by_layer", plot_top1_accuracy_by_layer, (df_perstep, figures_dir)),
        ("entropy_by_dataset", plot_entropy_by_dataset, (df_perstep, figures_dir)),
        ("crystallization_by_dataset", plot_crystallization_by_dataset, (df_perstep, figures_dir)),
        ("kl_by_context_length", plot_kl_by_context_length, (df_perstep, figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info("All 10 logit lens plots complete. Figures saved to %s", figures_dir)
