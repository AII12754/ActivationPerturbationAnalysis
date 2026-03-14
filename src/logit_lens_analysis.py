"""
Visualization module for logit lens experiments.

Reads stored per-step and top-k results and produces figures exploring
how intermediate layers predict the final token — when the model "decides"
on its output and how certainty builds across layers.

Analysis suite (10 plots):
  1. Crystallization depth: at which layer does rank_of_correct drop to 0
  2. Entropy waterfall: entropy by layer (certainty building)
  3. KL from final layer by layer (convergence to final prediction)
  4. Rank trajectory heatmap (layer x decode_step)
  5. Cross-entropy heatmap (layer x decode_step)
  6. Max probability by layer
  7. Top-1 accuracy by layer (fraction of steps where top1 matches final)
  8. Entropy by layer x dataset comparison
  9. Crystallization depth by dataset
  10. KL divergence by context length
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

from .logit_lens_storage import LogitLensResultStore

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
# 1. Crystallization Depth
# ===================================================================
def plot_crystallization_depth(df_perstep, output_dir):
    """At which layer does rank_of_correct first drop to 0?

    This is the "crystallization depth" — the earliest layer where the
    model's intermediate prediction matches the final layer's prediction.
    Shows how early the model "decides" on the correct token.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty:
        return

    # For each (experiment_id, decode_step), find the first layer where rank == 0
    def _first_zero_rank(group):
        zero_rank = group[group["rank_of_correct"] == 0]
        if zero_rank.empty:
            return np.nan
        return zero_rank["layer"].min()

    crystal = df_perstep.groupby(
        ["experiment_id", "decode_step"]
    ).apply(_first_zero_rank).reset_index(name="crystal_layer")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Distribution of crystallization depth
    valid = crystal["crystal_layer"].dropna()
    axes[0].hist(valid, bins=50, alpha=0.7, edgecolor="black", density=True)
    if len(valid) > 0:
        axes[0].axvline(valid.median(), color="red", ls="--",
                        label=f"Median={valid.median():.0f}")
    axes[0].set_xlabel("Crystallization Layer")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of Crystallization Depth\n(first layer where rank=0)")
    axes[0].legend(fontsize=8)

    # Crystallization depth over decode steps
    by_step = crystal.groupby("decode_step")["crystal_layer"].mean().sort_index()
    axes[1].plot(by_step.index, by_step.values, "b-", linewidth=2)
    axes[1].set_xlabel("Decode Step")
    axes[1].set_ylabel("Mean Crystallization Layer")
    axes[1].set_title("Crystallization Depth over Decoding")
    _save(fig, out, "01_crystallization_depth.png")


# ===================================================================
# 2. Entropy Waterfall
# ===================================================================
def plot_entropy_waterfall(df_perstep, output_dir):
    """Entropy by layer — shows certainty building through the network.

    Theory: early layers have high entropy (uniform distribution over vocab),
    and entropy decreases as the model narrows down the prediction.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Mean entropy by layer
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

    # Entropy by layer for selected decode steps
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
    _save(fig, out, "02_entropy_waterfall.png")


# ===================================================================
# 3. KL from Final Layer
# ===================================================================
def plot_kl_from_final(df_perstep, output_dir):
    """KL divergence from each layer's distribution to the final layer.

    Shows how quickly intermediate representations converge to the final
    prediction. Later layers should have lower KL divergence.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Mean KL by layer
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

    # KL by layer for selected decode steps
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
    _save(fig, out, "03_kl_from_final.png")


# ===================================================================
# 4. Rank Trajectory Heatmap
# ===================================================================
def plot_rank_trajectory_heatmap(df_perstep, output_dir):
    """Heatmap of rank_of_correct: layer x decode_step.

    Shows the trajectory of how the correct token's rank improves
    (decreases) through the layers at each decode step.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty:
        return

    pivot = df_perstep.pivot_table(
        values="rank_of_correct", index="layer", columns="decode_step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    # Use log scale for rank since it can span a wide range
    sns.heatmap(
        np.log1p(pivot), cmap="YlOrRd_r", ax=ax,
    )
    ax.set_title("Rank of Correct Token (log1p): Layer x Decode Step\n(darker = lower rank = better)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Decode Step")
    _save(fig, out, "04_rank_trajectory_heatmap.png")


# ===================================================================
# 5. Cross-Entropy Heatmap
# ===================================================================
def plot_cross_entropy_heatmap(df_perstep, output_dir):
    """Heatmap of cross_entropy_correct: layer x decode_step.

    Shows the cross-entropy loss of the correct token at each layer
    and decode step. Should decrease through layers.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
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
    _save(fig, out, "05_cross_entropy_heatmap.png")


# ===================================================================
# 6. Max Probability by Layer
# ===================================================================
def plot_max_probability_by_layer(df_perstep, output_dir):
    """Max probability in each layer's softmax distribution.

    Shows how the model's confidence builds across layers.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Mean max probability by layer
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

    # Heatmap
    pivot = df_perstep.pivot_table(
        values="max_prob", index="layer", columns="decode_step", aggfunc="mean",
    )
    sns.heatmap(pivot, cmap="Greens", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Max Probability: Layer x Decode Step")
    axes[1].set_ylabel("Layer")
    axes[1].set_xlabel("Decode Step")
    _save(fig, out, "06_max_probability_by_layer.png")


# ===================================================================
# 7. Top-1 Accuracy by Layer
# ===================================================================
def plot_top1_accuracy_by_layer(df_perstep, output_dir):
    """Fraction of decode steps where top-1 prediction matches the
    final layer's prediction, by layer.

    This is the "logit lens accuracy" — how often each layer would
    predict the same token as the final layer.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty:
        return

    df = df_perstep.copy()
    df["top1_correct"] = (df["top1_token_id"] == df["correct_token_id"]).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Accuracy by layer
    by_layer = df.groupby("layer")["top1_correct"].mean().sort_index()
    axes[0].plot(by_layer.index, by_layer.values * 100, "b-", linewidth=2, marker=".", markersize=4)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Logit Lens Accuracy by Layer\n(% of steps matching final layer)")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(100, color="gray", ls="--", alpha=0.3)

    # Accuracy by decode step (averaged over layers)
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
    _save(fig, out, "07_top1_accuracy_by_layer.png")


# ===================================================================
# 8. Entropy by Layer x Dataset Comparison
# ===================================================================
def plot_entropy_by_dataset(df_perstep, output_dir):
    """Compare entropy profiles across datasets.

    Different text domains may produce different entropy signatures —
    factual text might crystallize earlier than creative text.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty or "dataset_name" not in df_perstep.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Entropy by layer for each dataset
    for ds in df_perstep["dataset_name"].unique():
        sub = df_perstep[df_perstep["dataset_name"] == ds]
        profile = sub.groupby("layer")["entropy"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Entropy (nats)")
    axes[0].set_title("Entropy by Layer — Dataset Comparison")
    axes[0].legend(fontsize=7, ncol=2)

    # KL divergence by layer for each dataset
    for ds in df_perstep["dataset_name"].unique():
        sub = df_perstep[df_perstep["dataset_name"] == ds]
        profile = sub.groupby("layer")["kl_from_final"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("KL Divergence (nats)")
    axes[1].set_title("KL from Final — Dataset Comparison")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, ncol=2)
    _save(fig, out, "08_entropy_by_dataset.png")


# ===================================================================
# 9. Crystallization Depth by Dataset
# ===================================================================
def plot_crystallization_by_dataset(df_perstep, output_dir):
    """Compare crystallization depth across datasets.

    Theory: mathematical or structured text may crystallize later
    (requires more computation) than predictable narrative text.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty or "dataset_name" not in df_perstep.columns:
        return

    # Compute crystallization depth per (experiment, decode_step)
    def _first_zero_rank(group):
        zero_rank = group[group["rank_of_correct"] == 0]
        if zero_rank.empty:
            return np.nan
        return zero_rank["layer"].min()

    crystal = df_perstep.groupby(
        ["experiment_id", "dataset_name", "decode_step"]
    ).apply(_first_zero_rank).reset_index(name="crystal_layer")

    # Box plot by dataset
    crystal["ds_label"] = crystal["dataset_name"].apply(_ds_label)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    sns.boxplot(data=crystal, x="ds_label", y="crystal_layer", ax=axes[0])
    axes[0].set_xlabel("Dataset")
    axes[0].set_ylabel("Crystallization Layer")
    axes[0].set_title("Crystallization Depth by Dataset")
    axes[0].tick_params(axis="x", rotation=45)

    # Mean crystallization depth over decode steps by dataset
    for ds in crystal["dataset_name"].unique():
        sub = crystal[crystal["dataset_name"] == ds]
        by_step = sub.groupby("decode_step")["crystal_layer"].mean().sort_index()
        axes[1].plot(by_step.index, by_step.values, label=_ds_label(ds), alpha=0.8)
    axes[1].set_xlabel("Decode Step")
    axes[1].set_ylabel("Mean Crystallization Layer")
    axes[1].set_title("Crystallization Depth over Decoding by Dataset")
    axes[1].legend(fontsize=7, ncol=2)
    _save(fig, out, "09_crystallization_by_dataset.png")


# ===================================================================
# 10. KL Divergence by Context Length
# ===================================================================
def plot_kl_by_context_length(df_perstep, output_dir):
    """How does context length affect the convergence of intermediate
    layers to the final prediction?

    Theory: longer contexts may require more layers of processing
    to integrate, delaying convergence.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_perstep.empty:
        return

    ctx_lengths = sorted(df_perstep["context_length"].unique())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # KL profile by context length
    for cl in ctx_lengths:
        sub = df_perstep[df_perstep["context_length"] == cl]
        profile = sub.groupby("layer")["kl_from_final"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("KL Divergence (nats)")
    axes[0].set_title("KL from Final by Context Length")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    # Entropy profile by context length
    for cl in ctx_lengths:
        sub = df_perstep[df_perstep["context_length"] == cl]
        profile = sub.groupby("layer")["entropy"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Entropy (nats)")
    axes[1].set_title("Entropy by Layer x Context Length")
    axes[1].legend(fontsize=8)
    _save(fig, out, "10_kl_by_context_length.png")


# ===================================================================
# Master routine
# ===================================================================
def run_all_logit_lens_analyses(
    results_dir: str = "./results_logit_lens",
    figures_dir: Optional[str] = None,
):
    """Run the full logit lens analysis suite and save all figures."""
    if figures_dir is None:
        figures_dir = str(Path(results_dir) / "figures")

    df_perstep = LogitLensResultStore.load_perstep(results_dir)
    df_topk = LogitLensResultStore.load_topk(results_dir)

    if df_perstep.empty and df_topk.empty:
        logger.error("No data found in %s. Run logit lens experiments first.", results_dir)
        return

    logger.info(
        "Loaded %d perstep records, %d topk records. Generating figures...",
        len(df_perstep), len(df_topk),
    )

    # --- Core logit lens plots (1-7) ---
    plot_crystallization_depth(df_perstep, figures_dir)
    plot_entropy_waterfall(df_perstep, figures_dir)
    plot_kl_from_final(df_perstep, figures_dir)
    plot_rank_trajectory_heatmap(df_perstep, figures_dir)
    plot_cross_entropy_heatmap(df_perstep, figures_dir)
    plot_max_probability_by_layer(df_perstep, figures_dir)
    plot_top1_accuracy_by_layer(df_perstep, figures_dir)

    # --- Comparison plots (8-10) ---
    plot_entropy_by_dataset(df_perstep, figures_dir)
    plot_crystallization_by_dataset(df_perstep, figures_dir)
    plot_kl_by_context_length(df_perstep, figures_dir)

    logger.info("All %d figures saved to %s.", 10, figures_dir)
