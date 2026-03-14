"""E0: Decode similarity analysis — 27 plot functions.

Wraps the original ``src/decode_analysis.py`` visualization suite with
backward-compatible data loading via ``ExperimentStore.load_table()``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    """Pick a representative subset of layers for line plots."""
    if len(all_layers) <= max_show:
        return all_layers
    step = max(1, len(all_layers) // max_show)
    return all_layers[::step]


def _load_data(results_dir: str):
    """Load topk and agg data using ExperimentStore for backward compat."""
    topk = ExperimentStore.load_table(results_dir, "topk")
    agg = ExperimentStore.load_table(results_dir, "agg")
    return topk, agg


def _savefig(fig, figures_dir, name, idx=None):
    prefix = f"{idx:02d}_" if idx else ""
    path = os.path.join(figures_dir, f"{prefix}{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ===================================================================
# Plot 1: Similarity vs distance
# ===================================================================
def plot_similarity_vs_distance(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(df["layer"].unique())
    show_layers = _select_layers(layers)
    for layer in show_layers:
        sub = df[df["layer"] == layer]
        ax.scatter(sub["ref_token_distance"], sub["ref_similarity"],
                   alpha=0.3, s=5, label=f"L{layer}")
    ax.set_xlabel("Reference token distance")
    ax.set_ylabel("Cosine similarity (rank-1)")
    ax.set_title("Rank-1 Reference Similarity vs Distance")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "similarity_vs_distance", 1)


# ===================================================================
# Plot 2: Layer × decode step heatmap
# ===================================================================
def plot_layer_decode_step_heatmap(agg, figures_dir):
    if agg.empty:
        return
    pivot = agg.pivot_table(
        values="max_similarity", index="layer", columns="decode_step", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Max Similarity: Layer × Decode Step")
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Layer")
    _savefig(fig, figures_dir, "layer_decode_heatmap", 2)


# ===================================================================
# Plot 3: Max similarity distribution
# ===================================================================
def plot_max_similarity_distribution(agg, figures_dir):
    if agg.empty:
        return
    layers = sorted(agg["layer"].unique())
    n_groups = min(4, len(layers))
    group_size = max(1, len(layers) // n_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=FIGSIZE_WIDE, sharey=True)
    if n_groups == 1:
        axes = [axes]
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, len(layers))
        group_layers = layers[start:end]
        sub = agg[agg["layer"].isin(group_layers)]
        axes[g].hist(sub["max_similarity"], bins=50, alpha=0.7)
        axes[g].set_title(f"Layers {group_layers[0]}-{group_layers[-1]}")
        axes[g].set_xlabel("Max similarity")
    axes[0].set_ylabel("Count")
    fig.suptitle("Max Similarity Distribution by Layer Group")
    fig.tight_layout()
    _savefig(fig, figures_dir, "max_similarity_distribution", 3)


# ===================================================================
# Plot 4: Threshold fraction
# ===================================================================
def plot_threshold_fraction(agg, figures_dir):
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col, label in [("frac_above_090", ">0.90"), ("frac_above_095", ">0.95"),
                        ("frac_above_098", ">0.98")]:
        if col in agg.columns:
            means = agg.groupby("decode_step")[col].mean()
            ax.plot(means.index, means.values, label=label)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Fraction of tokens above threshold")
    ax.set_title("Fraction Above Similarity Thresholds")
    ax.legend()
    _savefig(fig, figures_dir, "threshold_fraction", 4)


# ===================================================================
# Plot 5: Temporal drift
# ===================================================================
def plot_temporal_drift(agg, figures_dir):
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(agg["layer"].unique())
    show_layers = _select_layers(layers)
    for layer in show_layers:
        sub = agg[agg["layer"] == layer].groupby("decode_step")["max_similarity"].mean()
        ax.plot(sub.index, sub.values, label=f"L{layer}", alpha=0.8)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Mean max similarity")
    ax.set_title("Temporal Drift in Max Similarity")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "temporal_drift", 5)


# ===================================================================
# Plot 6: Context length effect
# ===================================================================
def plot_context_length_effect(agg, figures_dir):
    if agg.empty or "context_length" not in agg.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    agg.boxplot(column="max_similarity", by="context_length", ax=ax)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Max similarity")
    ax.set_title("Max Similarity by Context Length")
    fig.suptitle("")
    _savefig(fig, figures_dir, "context_length_effect", 6)


# ===================================================================
# Plot 7: Reference position distribution
# ===================================================================
def plot_reference_position_distribution(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(df["ref_token_distance"], bins=100, alpha=0.7)
    ax.set_xlabel("Reference token distance")
    ax.set_ylabel("Count")
    ax.set_title("Rank-1 Reference Position Distribution")
    ax.set_yscale("log")
    _savefig(fig, figures_dir, "reference_position_distribution", 7)


# ===================================================================
# Plot 8: Top-k similarity decay
# ===================================================================
def plot_topk_similarity_decay(topk, figures_dir):
    if topk.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(topk["layer"].unique())
    show_layers = _select_layers(layers, 4)
    for layer in show_layers:
        sub = topk[topk["layer"] == layer].groupby("ref_rank")["ref_similarity"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=f"L{layer}")
    ax.set_xlabel("Reference rank")
    ax.set_ylabel("Mean similarity")
    ax.set_title("Top-k Similarity Decay by Layer")
    ax.legend()
    _savefig(fig, figures_dir, "topk_similarity_decay", 8)


# ===================================================================
# Plot 9: Layer similarity profile
# ===================================================================
def plot_layer_similarity_profile(agg, figures_dir):
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    profile = agg.groupby("layer").agg(
        mean_sim=("mean_similarity", "mean"),
        max_sim=("max_similarity", "mean"),
        std_sim=("std_similarity", "mean"),
    )
    ax.plot(profile.index, profile["mean_sim"], label="Mean", marker=".")
    ax.plot(profile.index, profile["max_sim"], label="Max", marker=".")
    ax.fill_between(profile.index,
                     profile["mean_sim"] - profile["std_sim"],
                     profile["mean_sim"] + profile["std_sim"],
                     alpha=0.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Similarity")
    ax.set_title("Layer Similarity Profile (U-shape)")
    ax.legend()
    _savefig(fig, figures_dir, "layer_similarity_profile", 9)


# ===================================================================
# Plot 10: Layer phase transitions
# ===================================================================
def plot_layer_phase_transitions(agg, figures_dir):
    if agg.empty:
        return
    profile = agg.groupby("layer")["max_similarity"].mean()
    values = profile.values
    if len(values) < 3:
        return
    first_deriv = np.gradient(values)
    second_deriv = np.gradient(first_deriv)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax1.plot(profile.index, values, marker=".")
    ax1.set_ylabel("Max similarity")
    ax1.set_title("Layer Profile + Derivatives (Phase Transitions)")
    ax2.plot(profile.index, first_deriv, marker=".", color="orange")
    ax2.set_ylabel("1st derivative")
    ax2.axhline(0, color="gray", ls="--", lw=0.5)
    ax3.plot(profile.index, second_deriv, marker=".", color="red")
    ax3.set_ylabel("2nd derivative")
    ax3.set_xlabel("Layer")
    ax3.axhline(0, color="gray", ls="--", lw=0.5)
    fig.tight_layout()
    _savefig(fig, figures_dir, "layer_phase_transitions", 10)


# ===================================================================
# Plot 11: Inter-layer correlation
# ===================================================================
def plot_inter_layer_correlation(agg, figures_dir):
    if agg.empty:
        return
    pivot = agg.pivot_table(
        values="max_similarity", index=["decode_step"], columns="layer", aggfunc="mean"
    )
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    ax.set_title("Inter-Layer Correlation (max similarity)")
    _savefig(fig, figures_dir, "inter_layer_correlation", 11)


# ===================================================================
# Plot 12: Similarity concentration
# ===================================================================
def plot_similarity_concentration(agg, figures_dir):
    if agg.empty:
        return
    agg_c = agg.copy()
    agg_c["concentration"] = agg_c["mean_similarity"] / agg_c["max_similarity"].clip(lower=1e-8)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(agg_c["layer"].unique())
    show_layers = _select_layers(layers)
    for layer in show_layers:
        sub = agg_c[agg_c["layer"] == layer].groupby("decode_step")["concentration"].mean()
        ax.plot(sub.index, sub.values, label=f"L{layer}", alpha=0.8)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Mean / Max similarity ratio")
    ax.set_title("Similarity Concentration (retrieval sharpness)")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "similarity_concentration", 12)


# ===================================================================
# Plot 13-14: Mean and Std similarity heatmaps
# ===================================================================
def plot_mean_similarity_heatmap(agg, figures_dir):
    if agg.empty:
        return
    pivot = agg.pivot_table(
        values="mean_similarity", index="layer", columns="decode_step", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(pivot, ax=ax, cmap="viridis")
    ax.set_title("Mean Similarity: Layer × Decode Step")
    _savefig(fig, figures_dir, "mean_similarity_heatmap", 13)


def plot_std_similarity_heatmap(agg, figures_dir):
    if agg.empty:
        return
    pivot = agg.pivot_table(
        values="std_similarity", index="layer", columns="decode_step", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(pivot, ax=ax, cmap="magma")
    ax.set_title("Std Similarity: Layer × Decode Step")
    _savefig(fig, figures_dir, "std_similarity_heatmap", 14)


# ===================================================================
# Plot 15: Prompt vs decoded reference
# ===================================================================
def plot_prompt_vs_decoded_reference(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty or "context_length" not in df.columns:
        return
    df["from_prompt"] = df["ref_token_position"] < df["context_length"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(df["layer"].unique())
    show_layers = _select_layers(layers)
    for layer in show_layers:
        sub = df[df["layer"] == layer].groupby("decode_step")["from_prompt"].mean()
        ax.plot(sub.index, sub.values, label=f"L{layer}", alpha=0.8)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Fraction of refs from prompt")
    ax.set_title("Prompt vs Decoded Token References")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "prompt_vs_decoded_reference", 15)


# ===================================================================
# Plot 16: Reference source by context
# ===================================================================
def plot_reference_source_by_context(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty or "context_length" not in df.columns:
        return
    df["from_prompt"] = df["ref_token_position"] < df["context_length"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    summary = df.groupby("context_length")["from_prompt"].mean()
    ax.bar(summary.index.astype(str), summary.values)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Fraction of refs from prompt")
    ax.set_title("Prompt Anchoring by Context Length")
    _savefig(fig, figures_dir, "reference_source_by_context", 16)


# ===================================================================
# Plot 17: Self-token reference
# ===================================================================
def plot_self_token_reference(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty or "token_id" not in df.columns or "ref_token_id" not in df.columns:
        return
    df["is_self"] = df["token_id"] == df["ref_token_id"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(df["layer"].unique())
    show_layers = _select_layers(layers)
    for layer in show_layers:
        sub = df[df["layer"] == layer].groupby("decode_step")["is_self"].mean()
        ax.plot(sub.index, sub.values, label=f"L{layer}", alpha=0.8)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Fraction self-token refs")
    ax.set_title("Self-Token Reference Rate")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "self_token_reference", 17)


# ===================================================================
# Plot 18: Reference distance by layer
# ===================================================================
def plot_reference_distance_by_layer(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(df["layer"].unique())
    n_groups = min(4, len(layers))
    group_size = max(1, len(layers) // n_groups)
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, len(layers))
        group_layers = layers[start:end]
        sub = df[df["layer"].isin(group_layers)]
        ax.hist(sub["ref_token_distance"], bins=100, alpha=0.5,
                label=f"L{group_layers[0]}-{group_layers[-1]}")
    ax.set_xlabel("Reference distance")
    ax.set_ylabel("Count")
    ax.set_title("Reference Distance Distribution by Layer Group")
    ax.set_yscale("log")
    ax.legend()
    _savefig(fig, figures_dir, "reference_distance_by_layer", 18)


# ===================================================================
# Plot 19: Top-k reference spread
# ===================================================================
def plot_topk_reference_spread(topk, figures_dir):
    if topk.empty:
        return
    spread = topk.groupby(["layer", "decode_step"])["ref_token_position"].agg(["min", "max"])
    spread["range"] = spread["max"] - spread["min"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(topk["layer"].unique())
    show_layers = _select_layers(layers, 4)
    for layer in show_layers:
        sub = spread.loc[layer] if layer in spread.index.get_level_values(0) else pd.DataFrame()
        if not sub.empty:
            ax.plot(sub.index, sub["range"], label=f"L{layer}", alpha=0.8)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Top-k reference spread (positions)")
    ax.set_title("Spatial Spread of Top-k References")
    ax.legend()
    _savefig(fig, figures_dir, "topk_reference_spread", 19)


# ===================================================================
# Plot 20-23: Dataset comparison
# ===================================================================
def plot_dataset_layer_profiles(agg, figures_dir):
    if agg.empty or "dataset_name" not in agg.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in agg["dataset_name"].unique():
        sub = agg[agg["dataset_name"] == ds].groupby("layer")["max_similarity"].mean()
        label = os.path.basename(str(ds))
        ax.plot(sub.index, sub.values, label=label, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean max similarity")
    ax.set_title("Layer Profile by Dataset")
    ax.legend(fontsize=7)
    _savefig(fig, figures_dir, "dataset_layer_profiles", 20)


def plot_dataset_temporal_drift(agg, figures_dir):
    if agg.empty or "dataset_name" not in agg.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in agg["dataset_name"].unique():
        sub = agg[agg["dataset_name"] == ds].groupby("decode_step")["max_similarity"].mean()
        label = os.path.basename(str(ds))
        ax.plot(sub.index, sub.values, label=label, alpha=0.8)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Mean max similarity")
    ax.set_title("Temporal Drift by Dataset")
    ax.legend(fontsize=7)
    _savefig(fig, figures_dir, "dataset_temporal_drift", 21)


def plot_dataset_self_reference(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty or "dataset_name" not in df.columns:
        return
    if "token_id" not in df.columns or "ref_token_id" not in df.columns:
        return
    df["is_self"] = df["token_id"] == df["ref_token_id"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    summary = df.groupby("dataset_name")["is_self"].mean()
    labels = [os.path.basename(str(x)) for x in summary.index]
    ax.bar(labels, summary.values)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Self-reference rate")
    ax.set_title("Self-Reference Rate by Dataset")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    _savefig(fig, figures_dir, "dataset_self_reference", 22)


def plot_dataset_prompt_anchoring(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty or "context_length" not in df.columns or "dataset_name" not in df.columns:
        return
    df["from_prompt"] = df["ref_token_position"] < df["context_length"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in df["dataset_name"].unique():
        sub = df[df["dataset_name"] == ds].groupby("decode_step")["from_prompt"].mean()
        label = os.path.basename(str(ds))
        ax.plot(sub.index, sub.values, label=label, alpha=0.8)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Fraction from prompt")
    ax.set_title("Prompt Anchoring Decay by Dataset")
    ax.legend(fontsize=7)
    _savefig(fig, figures_dir, "dataset_prompt_anchoring", 23)


# ===================================================================
# Plot 24-27: Information-theoretic
# ===================================================================
def plot_similarity_entropy(agg, figures_dir):
    if agg.empty:
        return
    agg_c = agg.copy()
    agg_c["cv"] = agg_c["std_similarity"] / agg_c["mean_similarity"].clip(lower=1e-8)
    pivot = agg_c.pivot_table(values="cv", index="layer", columns="decode_step", aggfunc="mean")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(pivot, ax=ax, cmap="coolwarm")
    ax.set_title("Coefficient of Variation: Layer × Decode Step")
    _savefig(fig, figures_dir, "similarity_entropy", 24)


def plot_layer_redundancy(agg, figures_dir):
    if agg.empty:
        return
    pivot = agg.pivot_table(
        values="max_similarity", index="decode_step", columns="layer", aggfunc="mean"
    )
    if pivot.shape[1] < 2:
        return
    layers = pivot.columns.tolist()
    adj_corr = [pivot[layers[i]].corr(pivot[layers[i + 1]]) for i in range(len(layers) - 1)]
    skip_corr = [pivot[layers[i]].corr(pivot[layers[min(i + 2, len(layers) - 1)]])
                 for i in range(len(layers) - 1)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(range(len(adj_corr)), adj_corr, label="Adjacent-layer", marker=".")
    ax.plot(range(len(skip_corr)), skip_corr, label="Skip-layer (L+2)", marker=".", alpha=0.7)
    ax.set_xlabel("Layer pair index")
    ax.set_ylabel("Correlation")
    ax.set_title("Layer Redundancy")
    ax.legend()
    _savefig(fig, figures_dir, "layer_redundancy", 25)


def plot_attention_sink(topk, figures_dir):
    df = topk[topk["ref_rank"] == 0].copy()
    if df.empty:
        return
    df["is_sink"] = df["ref_token_position"] < 4
    fig, ax = plt.subplots(figsize=FIGSIZE)
    layers = sorted(df["layer"].unique())
    show_layers = _select_layers(layers)
    for layer in show_layers:
        sub = df[df["layer"] == layer].groupby("decode_step")["is_sink"].mean()
        ax.plot(sub.index, sub.values, label=f"L{layer}", alpha=0.8)
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Fraction referencing positions 0-3")
    ax.set_title("Attention Sink Effect")
    ax.legend(fontsize=8, ncol=2)
    _savefig(fig, figures_dir, "attention_sink", 26)


def plot_context_layer_interaction(agg, figures_dir):
    if agg.empty or "context_length" not in agg.columns:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ctx in sorted(agg["context_length"].unique()):
        sub = agg[agg["context_length"] == ctx].groupby("layer")["max_similarity"].mean()
        ax.plot(sub.index, sub.values, label=f"ctx={ctx}", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean max similarity")
    ax.set_title("Layer Profile by Context Length")
    ax.legend()
    _savefig(fig, figures_dir, "context_layer_interaction", 27)


# ===================================================================
# Master function
# ===================================================================
def run_all_decode_analyses(results_dir: str, figures_dir: str):
    """Run all 27 decode similarity analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading data from %s", results_dir)

    topk, agg = _load_data(results_dir)
    if topk.empty and agg.empty:
        logger.warning("No data found in %s. Skipping analysis.", results_dir)
        return

    logger.info("Loaded %d topk and %d agg records.", len(topk), len(agg))

    # Run all 27 plots
    plot_similarity_vs_distance(topk, figures_dir)
    plot_layer_decode_step_heatmap(agg, figures_dir)
    plot_max_similarity_distribution(agg, figures_dir)
    plot_threshold_fraction(agg, figures_dir)
    plot_temporal_drift(agg, figures_dir)
    plot_context_length_effect(agg, figures_dir)
    plot_reference_position_distribution(topk, figures_dir)
    plot_topk_similarity_decay(topk, figures_dir)
    plot_layer_similarity_profile(agg, figures_dir)
    plot_layer_phase_transitions(agg, figures_dir)
    plot_inter_layer_correlation(agg, figures_dir)
    plot_similarity_concentration(agg, figures_dir)
    plot_mean_similarity_heatmap(agg, figures_dir)
    plot_std_similarity_heatmap(agg, figures_dir)
    plot_prompt_vs_decoded_reference(topk, figures_dir)
    plot_reference_source_by_context(topk, figures_dir)
    plot_self_token_reference(topk, figures_dir)
    plot_reference_distance_by_layer(topk, figures_dir)
    plot_topk_reference_spread(topk, figures_dir)
    plot_dataset_layer_profiles(agg, figures_dir)
    plot_dataset_temporal_drift(agg, figures_dir)
    plot_dataset_self_reference(topk, figures_dir)
    plot_dataset_prompt_anchoring(topk, figures_dir)
    plot_similarity_entropy(agg, figures_dir)
    plot_layer_redundancy(agg, figures_dir)
    plot_attention_sink(topk, figures_dir)
    plot_context_layer_interaction(agg, figures_dir)

    logger.info("All 27 plots complete. Figures saved to %s", figures_dir)
