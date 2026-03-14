"""
Visualization module for decode-time activation similarity experiments.

Reads stored topk and aggregate results and produces figures exploring
activation reuse patterns during autoregressive decoding.

Analysis suite (27 plots):
  Basic (1-8):    Original descriptive statistics
  Structural (9-14):  Layer geometry, phase transitions, inter-layer coupling
  Reference (15-19):  Memory provenance — where does the model "look back" to
  Dataset (20-23):     Domain-dependent activation reuse signatures
  Information (24-27): Entropy, effective rank, attention sink, redundancy
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
from scipy import stats as sp_stats

from .decode_storage import DecodeResultStore

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
# 1. Similarity vs Distance (rank-1 reference)
# ===================================================================
def plot_similarity_vs_distance(df_topk, output_dir):
    """Rank-1 ref_similarity vs ref_token_distance, by layer."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0]
    if rank1.empty:
        return

    all_layers = sorted(rank1["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = rank1[rank1["layer"] == layer]
        binned = sub.groupby(pd.cut(sub["ref_token_distance"], bins=20))["ref_similarity"].mean()
        mids = [(b.left + b.right) / 2 for b in binned.index]
        ax.plot(mids, binned.values, label=f"Layer {layer}", alpha=0.8)

    ax.set_xlabel("Distance to reference token (positions)")
    ax.set_ylabel("Cosine similarity (rank-1)")
    ax.set_title("Rank-1 Reference Similarity vs Distance")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    _save(fig, out, "01_similarity_vs_distance.png")


# ===================================================================
# 2. Layer x Decode-Step Heatmap
# ===================================================================
def plot_layer_decode_step_heatmap(df_agg, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return
    pivot = df_agg.pivot_table(
        values="max_similarity", index="layer", columns="decode_step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    sns.heatmap(pivot, cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Max Similarity: Layer x Decode Step")
    ax.set_ylabel("Layer"); ax.set_xlabel("Decode Step")
    _save(fig, out, "02_layer_decode_heatmap.png")


# ===================================================================
# 3. Max-Similarity Distribution
# ===================================================================
def plot_max_similarity_distribution(df_agg, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return
    all_layers = sorted(df_agg["layer"].unique())
    groups = _layer_groups(all_layers)

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    for ax, (name, layers) in zip(axes[0], groups.items()):
        sub = df_agg[df_agg["layer"].isin(layers)]
        ax.hist(sub["max_similarity"].dropna(), bins=50, alpha=0.7, density=True)
        ax.set_title(f"Max Similarity — {name} layers")
        ax.set_xlabel("Max cosine similarity"); ax.set_ylabel("Density")
        ax.set_xlim(0, 1.05)
    _save(fig, out, "03_max_similarity_distribution.png")


# ===================================================================
# 4. Threshold Fraction vs Decode Step
# ===================================================================
def plot_threshold_fraction(df_agg, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return
    all_layers = sorted(df_agg["layer"].unique())
    selected = _select_layers(all_layers, max_show=4)

    thresholds = [("frac_above_090", "0.90"), ("frac_above_095", "0.95"), ("frac_above_098", "0.98")]
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5 * len(thresholds), 5), squeeze=False)
    for ax, (col, label) in zip(axes[0], thresholds):
        for layer in selected:
            sub = df_agg[df_agg["layer"] == layer].sort_values("decode_step")
            grouped = sub.groupby("decode_step")[col].mean()
            ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
        ax.set_title(f"Frac > {label} vs Decode Step")
        ax.set_xlabel("Decode Step"); ax.set_ylabel(f"Fraction above {label}")
        ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
    _save(fig, out, "04_threshold_fraction.png")


# ===================================================================
# 5. Temporal Drift
# ===================================================================
def plot_temporal_drift(df_agg, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return
    all_layers = sorted(df_agg["layer"].unique())
    selected = _select_layers(all_layers)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df_agg[df_agg["layer"] == layer].sort_values("decode_step")
        grouped = sub.groupby("decode_step")["max_similarity"].mean()
        ax.plot(grouped.index, grouped.values, label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Decode Step"); ax.set_ylabel("Max Cosine Similarity")
    ax.set_title("Temporal Drift: Max Similarity over Decoding")
    ax.legend(fontsize=8, ncol=2); ax.set_ylim(0, 1.05)
    _save(fig, out, "05_temporal_drift.png")


# ===================================================================
# 6. Context Length Effect
# ===================================================================
def plot_context_length_effect(df_agg, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=df_agg, x="context_length", y="max_similarity", ax=ax)
    ax.set_xlabel("Context Length (tokens)"); ax.set_ylabel("Max Cosine Similarity")
    ax.set_title("Effect of Context Length on Max Similarity")
    ax.set_ylim(0, 1.05)
    _save(fig, out, "06_context_length_effect.png")


# ===================================================================
# 7. Reference Position Distribution
# ===================================================================
def plot_reference_position_distribution(df_topk, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0]
    if rank1.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(rank1["ref_token_distance"].dropna(), bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Distance to rank-1 reference (positions)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Rank-1 Reference Token Distance")
    _save(fig, out, "07_reference_position_distribution.png")


# ===================================================================
# 8. Top-K Similarity Decay
# ===================================================================
def plot_topk_similarity_decay(df_topk, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_topk.empty:
        return
    all_layers = sorted(df_topk["layer"].unique())
    selected = _select_layers(all_layers)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for layer in selected:
        sub = df_topk[df_topk["layer"] == layer]
        grouped = sub.groupby("ref_rank")["ref_similarity"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", markersize=4,
                label=f"Layer {layer}", alpha=0.8)
    ax.set_xlabel("Reference Rank"); ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Top-K Similarity Decay by Layer")
    ax.legend(fontsize=8, ncol=2); ax.set_ylim(0, 1.05)
    _save(fig, out, "08_topk_similarity_decay.png")


# ===================================================================
# 9. Layer-wise Similarity Profile (the "U-shape" or "V-shape")
# ===================================================================
def plot_layer_similarity_profile(df_agg, output_dir):
    """Mean/max/std similarity as a function of layer depth.

    Theory: early layers encode surface-level features (high similarity),
    middle layers perform complex reasoning (low similarity, high diversity),
    late layers reconverge toward the output distribution. This reveals
    the computational "bottleneck" depth.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title in zip(
        axes,
        ["mean_similarity", "max_similarity", "std_similarity"],
        ["Mean Similarity", "Max Similarity", "Std Similarity"],
    ):
        profile = df_agg.groupby("layer")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], "b-", linewidth=2)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            alpha=0.2,
        )
        ax.set_xlabel("Layer"); ax.set_ylabel(title)
        ax.set_title(f"{title} vs Layer Depth")

        # Mark the minimum (bottleneck layer).
        min_idx = profile["mean"].idxmin()
        ax.axvline(profile.loc[min_idx, "layer"], color="red", ls="--", alpha=0.5,
                    label=f"Min at layer {profile.loc[min_idx, 'layer']}")
        ax.legend(fontsize=8)
    _save(fig, out, "09_layer_similarity_profile.png")


# ===================================================================
# 10. Layer Phase Transition Detection
# ===================================================================
def plot_layer_phase_transitions(df_agg, output_dir):
    """Numerical derivative of mean_similarity across layers.

    Sharp changes in d(similarity)/d(layer) indicate phase transitions —
    boundaries between qualitatively different processing regimes.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return

    profile = df_agg.groupby("layer")["mean_similarity"].mean().sort_index()
    gradient = np.gradient(profile.values)
    curvature = np.gradient(gradient)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    axes[0].plot(profile.index, gradient, "b-", linewidth=1.5)
    axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("d(mean_similarity)/d(layer)")
    axes[0].set_title("First Derivative — Processing Regime Changes")
    # Mark significant transitions.
    threshold = np.std(gradient) * 1.5
    trans_layers = profile.index[np.abs(gradient) > threshold]
    for tl in trans_layers:
        axes[0].axvline(tl, color="red", alpha=0.3, linewidth=0.8)

    axes[1].plot(profile.index, curvature, "r-", linewidth=1.5)
    axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("d²(mean_similarity)/d(layer)²")
    axes[1].set_title("Second Derivative — Phase Transition Sharpness")
    _save(fig, out, "10_layer_phase_transitions.png")


# ===================================================================
# 11. Inter-layer Correlation Matrix
# ===================================================================
def plot_inter_layer_correlation(df_agg, output_dir):
    """Correlation of max_similarity across layers.

    Reveals which layers form correlated "processing blocks" vs which
    are informationally independent. Block-diagonal structure indicates
    modular computation stages.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return

    pivot = df_agg.pivot_table(
        values="max_similarity",
        index=["experiment_id", "decode_step"],
        columns="layer",
    )
    corr = pivot.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-0.3, vmax=1,
                square=True, ax=ax,
                xticklabels=5, yticklabels=5)
    ax.set_title("Inter-Layer Max-Similarity Correlation Matrix")
    ax.set_xlabel("Layer"); ax.set_ylabel("Layer")
    _save(fig, out, "11_inter_layer_correlation.png")


# ===================================================================
# 12. Similarity Concentration (Effective Neighbor Count)
# ===================================================================
def plot_similarity_concentration(df_agg, output_dir):
    """Ratio of mean/max similarity per layer — measures how concentrated
    the similarity is around the top match vs. spread across history.

    A ratio near 1 means all tokens are equally similar (uniform attention).
    A low ratio means a few tokens dominate (sharp retrieval).
    This is related to the "effective number of neighbors" concept.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return

    df = df_agg.copy()
    df["concentration"] = df["mean_similarity"] / df["max_similarity"].clip(lower=1e-8)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # By layer.
    profile = df.groupby("layer")["concentration"].mean().sort_index()
    axes[0].plot(profile.index, profile.values, "b-", linewidth=2)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Mean / Max Similarity")
    axes[0].set_title("Similarity Concentration by Layer\n(lower = sharper retrieval)")
    axes[0].set_ylim(0, 1)

    # By decode step (averaged over layers).
    step_profile = df.groupby("decode_step")["concentration"].mean().sort_index()
    axes[1].plot(step_profile.index, step_profile.values, "r-", linewidth=2)
    axes[1].set_xlabel("Decode Step"); axes[1].set_ylabel("Mean / Max Similarity")
    axes[1].set_title("Similarity Concentration over Decoding")
    axes[1].set_ylim(0, 1)
    _save(fig, out, "12_similarity_concentration.png")


# ===================================================================
# 13. Mean Similarity Heatmap (complement to max-similarity heatmap)
# ===================================================================
def plot_mean_similarity_heatmap(df_agg, output_dir):
    """Heatmap of mean similarity — reveals where the model's
    representation is broadly similar to history (high background)
    vs. where it is diverging into new territory (low background).
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return
    pivot = df_agg.pivot_table(
        values="mean_similarity", index="layer", columns="decode_step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    ax.set_title("Mean Similarity: Layer x Decode Step")
    ax.set_ylabel("Layer"); ax.set_xlabel("Decode Step")
    _save(fig, out, "13_mean_similarity_heatmap.png")


# ===================================================================
# 14. Std Similarity Heatmap (uncertainty/diversity map)
# ===================================================================
def plot_std_similarity_heatmap(df_agg, output_dir):
    """Where is the model uncertain about what to attend to?

    High std_similarity means the similarity distribution is wide —
    some tokens are very similar, others very different. This is a
    proxy for representational diversity or "decision uncertainty."
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return
    pivot = df_agg.pivot_table(
        values="std_similarity", index="layer", columns="decode_step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    sns.heatmap(pivot, cmap="magma", ax=ax)
    ax.set_title("Std Similarity: Layer x Decode Step (Representational Diversity)")
    ax.set_ylabel("Layer"); ax.set_xlabel("Decode Step")
    _save(fig, out, "14_std_similarity_heatmap.png")


# ===================================================================
# 15. Prompt vs Decoded Reference Source by Layer
# ===================================================================
def plot_prompt_vs_decoded_reference(df_topk, output_dir):
    """For each layer, what fraction of rank-1 references come from the
    original prompt vs. previously decoded tokens?

    Theory: early layers may preserve more surface-level prompt features,
    while deeper layers increasingly attend to the model's own outputs
    (the "autoregressive drift" hypothesis).
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0].copy()
    if rank1.empty:
        return

    rank1["is_prompt_ref"] = rank1["ref_token_position"] < rank1["context_length"]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # By layer.
    layer_pct = rank1.groupby("layer")["is_prompt_ref"].mean().sort_index()
    axes[0].bar(layer_pct.index, layer_pct.values * 100, color="steelblue", alpha=0.8, width=1)
    axes[0].axhline(50, color="red", ls="--", alpha=0.5, label="50% line")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("% References from Prompt")
    axes[0].set_title("Prompt Reference Rate by Layer")
    axes[0].legend()

    # By decode step (all layers aggregated).
    step_pct = rank1.groupby("decode_step")["is_prompt_ref"].mean().sort_index()
    axes[1].plot(step_pct.index, step_pct.values * 100, "b-", linewidth=2)
    axes[1].axhline(50, color="red", ls="--", alpha=0.5)
    axes[1].set_xlabel("Decode Step"); axes[1].set_ylabel("% References from Prompt")
    axes[1].set_title("Prompt Reference Rate over Decoding\n(autoregressive drift)")
    _save(fig, out, "15_prompt_vs_decoded_reference.png")


# ===================================================================
# 16. Reference Source by Layer AND Context Length
# ===================================================================
def plot_reference_source_by_context(df_topk, output_dir):
    """Does a longer prompt "anchor" the model's representations more?

    With more context, do references increasingly come from the prompt?
    This tests the hypothesis that context length modulates the balance
    between grounding in input vs. autoregressive self-reference.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0].copy()
    if rank1.empty:
        return

    rank1["is_prompt_ref"] = rank1["ref_token_position"] < rank1["context_length"]
    ctx_lengths = sorted(rank1["context_length"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for cl in ctx_lengths:
        sub = rank1[rank1["context_length"] == cl]
        by_step = sub.groupby("decode_step")["is_prompt_ref"].mean()
        ax.plot(by_step.index, by_step.values * 100, label=f"ctx={cl}", alpha=0.8)
    ax.set_xlabel("Decode Step"); ax.set_ylabel("% References from Prompt")
    ax.set_title("Prompt Anchoring Effect: Context Length Modulation")
    ax.legend(fontsize=8); ax.set_ylim(0, 105)
    _save(fig, out, "16_reference_source_by_context.png")


# ===================================================================
# 17. Self-Token Reference Rate
# ===================================================================
def plot_self_token_reference(df_topk, output_dir):
    """How often does the most similar historical token share the same
    token ID as the newly decoded token?

    High self-reference rate suggests the model strongly reuses
    representations of identical tokens — a "type-level" memory.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0].copy()
    if rank1.empty:
        return

    rank1["is_self_ref"] = rank1["token_id"] == rank1["ref_token_id"]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # By layer.
    by_layer = rank1.groupby("layer")["is_self_ref"].mean().sort_index()
    axes[0].bar(by_layer.index, by_layer.values * 100, color="coral", alpha=0.8, width=1)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("% Self-Token References")
    axes[0].set_title("Self-Token Reference Rate by Layer\n(same token ID as decoded)")

    # By decode step.
    by_step = rank1.groupby("decode_step")["is_self_ref"].mean().sort_index()
    axes[1].plot(by_step.index, by_step.values * 100, "r-", linewidth=2)
    axes[1].set_xlabel("Decode Step"); axes[1].set_ylabel("% Self-Token References")
    axes[1].set_title("Self-Token Reference over Decoding")
    _save(fig, out, "17_self_token_reference.png")


# ===================================================================
# 18. Reference Distance Distribution by Layer Group
# ===================================================================
def plot_reference_distance_by_layer(df_topk, output_dir):
    """How far back does each layer "look" for its nearest neighbor?

    Theory: early layers might reference nearby tokens (local syntax),
    while deeper layers may reach further back (long-range semantics).
    Or the opposite — testing both hypotheses.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0]
    if rank1.empty:
        return

    all_layers = sorted(rank1["layer"].unique())
    groups = _layer_groups(all_layers)

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), squeeze=False)
    for ax, (name, layers) in zip(axes[0], groups.items()):
        sub = rank1[rank1["layer"].isin(layers)]
        distances = sub["ref_token_distance"].dropna()
        ax.hist(distances, bins=50, alpha=0.7, density=True, edgecolor="black")
        median_d = distances.median()
        ax.axvline(median_d, color="red", ls="--", label=f"Median={median_d:.0f}")
        ax.set_title(f"Reference Distance — {name} layers")
        ax.set_xlabel("Distance (positions)"); ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    _save(fig, out, "18_reference_distance_by_layer.png")


# ===================================================================
# 19. Top-K Reference Diversity (spatial spread)
# ===================================================================
def plot_topk_reference_spread(df_topk, output_dir):
    """How spread out are the top-k references? If they cluster near
    the same position, the model has a narrow attention focus.
    If they span the full context, it's performing broad retrieval.

    Measured as std of ref_token_position within each (step, layer).
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_topk.empty:
        return

    spread = df_topk.groupby(
        ["experiment_id", "decode_step", "layer"]
    )["ref_token_position"].std().reset_index(name="position_std")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # By layer.
    by_layer = spread.groupby("layer")["position_std"].mean().sort_index()
    axes[0].plot(by_layer.index, by_layer.values, "b-", linewidth=2)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Std of Top-K Positions")
    axes[0].set_title("Top-K Reference Spatial Spread by Layer")

    # By decode step.
    by_step = spread.groupby("decode_step")["position_std"].mean().sort_index()
    axes[1].plot(by_step.index, by_step.values, "r-", linewidth=2)
    axes[1].set_xlabel("Decode Step"); axes[1].set_ylabel("Std of Top-K Positions")
    axes[1].set_title("Top-K Reference Spread over Decoding")
    _save(fig, out, "19_topk_reference_spread.png")


# ===================================================================
# 20. Dataset Comparison: Layer Profiles
# ===================================================================
def plot_dataset_layer_profiles(df_agg, output_dir):
    """Do different text domains produce different layer-wise similarity
    signatures? Conversational data (ShareGPT) vs. factual (Wikipedia)
    vs. mathematical (GSM8K) may activate very different layer patterns.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty or "dataset_name" not in df_agg.columns:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title in zip(
        axes,
        ["mean_similarity", "max_similarity", "std_similarity"],
        ["Mean Similarity", "Max Similarity", "Std Similarity"],
    ):
        for ds in df_agg["dataset_name"].unique():
            sub = df_agg[df_agg["dataset_name"] == ds]
            profile = sub.groupby("layer")[metric].mean().sort_index()
            ax.plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
        ax.set_xlabel("Layer"); ax.set_ylabel(title)
        ax.set_title(f"{title} by Layer — Dataset Comparison")
        ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "20_dataset_layer_profiles.png")


# ===================================================================
# 21. Dataset Comparison: Temporal Drift
# ===================================================================
def plot_dataset_temporal_drift(df_agg, output_dir):
    """Does activation similarity decay at different rates for different
    domains? Math reasoning might diverge faster than narrative text.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty or "dataset_name" not in df_agg.columns:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in df_agg["dataset_name"].unique():
        sub = df_agg[df_agg["dataset_name"] == ds]
        by_step = sub.groupby("decode_step")["max_similarity"].mean().sort_index()
        ax.plot(by_step.index, by_step.values, label=_ds_label(ds), alpha=0.8)
    ax.set_xlabel("Decode Step"); ax.set_ylabel("Max Similarity")
    ax.set_title("Temporal Drift by Dataset")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    _save(fig, out, "21_dataset_temporal_drift.png")


# ===================================================================
# 22. Dataset Comparison: Self-Reference Rate
# ===================================================================
def plot_dataset_self_reference(df_topk, output_dir):
    """Do some domains produce more self-referential decoding?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0].copy()
    if rank1.empty or "dataset_name" not in rank1.columns:
        return

    rank1["is_self_ref"] = rank1["token_id"] == rank1["ref_token_id"]
    ds_rates = rank1.groupby("dataset_name")["is_self_ref"].mean().sort_values()
    ds_rates.index = [_ds_label(x) for x in ds_rates.index]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ds_rates.plot.barh(ax=ax, color="coral", alpha=0.8)
    ax.set_xlabel("Self-Token Reference Rate")
    ax.set_title("Self-Token Reference Rate by Dataset")
    _save(fig, out, "22_dataset_self_reference.png")


# ===================================================================
# 23. Dataset Comparison: Prompt Anchoring
# ===================================================================
def plot_dataset_prompt_anchoring(df_topk, output_dir):
    """How strongly does each dataset anchor to the prompt?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0].copy()
    if rank1.empty or "dataset_name" not in rank1.columns:
        return

    rank1["is_prompt_ref"] = rank1["ref_token_position"] < rank1["context_length"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for ds in rank1["dataset_name"].unique():
        sub = rank1[rank1["dataset_name"] == ds]
        by_step = sub.groupby("decode_step")["is_prompt_ref"].mean()
        ax.plot(by_step.index, by_step.values * 100, label=_ds_label(ds), alpha=0.8)
    ax.set_xlabel("Decode Step"); ax.set_ylabel("% References from Prompt")
    ax.set_title("Prompt Anchoring Decay by Dataset")
    ax.legend(fontsize=8); ax.set_ylim(0, 105)
    _save(fig, out, "23_dataset_prompt_anchoring.png")


# ===================================================================
# 24. Similarity Entropy (information-theoretic measure)
# ===================================================================
def plot_similarity_entropy(df_agg, output_dir):
    """Compute an entropy-like measure from the similarity distribution.

    For each (step, layer), approximate the "attention entropy" using:
      H ≈ -mean_sim * log(mean_sim) - (1-mean_sim) * log(1-mean_sim)
    augmented by the spread (std_sim).

    High entropy = model is uncertain about which token to reference.
    Low entropy = sharp, confident retrieval.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return

    df = df_agg.copy()
    # Use std/mean ratio as a spread measure (coefficient of variation).
    df["cv"] = df["std_similarity"] / df["mean_similarity"].clip(lower=1e-8)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # CV by layer.
    by_layer = df.groupby("layer")["cv"].mean().sort_index()
    axes[0].plot(by_layer.index, by_layer.values, "g-", linewidth=2)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Coefficient of Variation")
    axes[0].set_title("Similarity Dispersion by Layer\n(higher = more uniform attention)")

    # CV heatmap.
    pivot = df.pivot_table(values="cv", index="layer", columns="decode_step", aggfunc="mean")
    sns.heatmap(pivot, cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("Similarity CV: Layer x Decode Step")
    axes[1].set_ylabel("Layer"); axes[1].set_xlabel("Decode Step")
    _save(fig, out, "24_similarity_entropy.png")


# ===================================================================
# 25. Layer Redundancy (adjacent layer similarity correlation)
# ===================================================================
def plot_layer_redundancy(df_agg, output_dir):
    """Adjacent-layer correlation of max_similarity patterns.

    If two adjacent layers are near-perfectly correlated, they might be
    redundant. Sharp drops in correlation indicate functional boundaries.
    This is related to layer pruning and early-exit theories.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return

    pivot = df_agg.pivot_table(
        values="max_similarity",
        index=["experiment_id", "decode_step"],
        columns="layer",
    )
    corr = pivot.corr()
    layers = sorted(corr.columns)
    adj_corrs = [corr.loc[layers[i], layers[i + 1]] for i in range(len(layers) - 1)]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    axes[0].plot(layers[:-1], adj_corrs, "b-", linewidth=2, marker=".", markersize=3)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Correlation with Layer+1")
    axes[0].set_title("Adjacent-Layer Correlation\n(drops indicate functional boundaries)")
    axes[0].axhline(0.95, color="red", ls="--", alpha=0.5, label="0.95 threshold")
    axes[0].legend(fontsize=8)

    # Skip-layer correlations (distance 1, 2, 4, 8, 16).
    skip_distances = [1, 2, 4, 8, 16]
    for d in skip_distances:
        skip_corrs = [corr.loc[layers[i], layers[i + d]]
                      for i in range(len(layers) - d)]
        axes[1].plot(layers[:len(skip_corrs)], skip_corrs,
                     label=f"skip={d}", alpha=0.8)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Correlation")
    axes[1].set_title("Skip-Layer Correlations\n(how fast does similarity decouple?)")
    axes[1].legend(fontsize=8)
    _save(fig, out, "25_layer_redundancy.png")


# ===================================================================
# 26. Attention Sink Analysis
# ===================================================================
def plot_attention_sink(df_topk, output_dir):
    """Do early tokens (positions 0-3) disproportionately attract references?

    The "attention sink" phenomenon: the first few tokens absorb outsized
    attention regardless of their content. Here we test whether activation
    similarity shows the same pattern.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rank1 = df_topk[df_topk["ref_rank"] == 0].copy()
    if rank1.empty:
        return

    # Fraction of rank-1 references pointing to positions 0-3.
    rank1["is_sink"] = rank1["ref_token_position"] <= 3

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # By layer.
    by_layer = rank1.groupby("layer")["is_sink"].mean().sort_index()
    axes[0].bar(by_layer.index, by_layer.values * 100, color="purple", alpha=0.7, width=1)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("% Refs to Positions 0-3")
    axes[0].set_title("Attention Sink Effect by Layer\n(% of rank-1 refs to first 4 tokens)")

    # Reference position histogram (zoomed into first 20 positions).
    ref_pos = rank1["ref_token_position"].dropna()
    first_20 = ref_pos[ref_pos < 20]
    axes[1].hist(first_20, bins=range(21), alpha=0.7, edgecolor="black", density=True)
    axes[1].set_xlabel("Reference Position"); axes[1].set_ylabel("Density")
    axes[1].set_title("Reference Position Distribution (first 20 tokens)")
    _save(fig, out, "26_attention_sink.png")


# ===================================================================
# 27. Context Length x Layer Interaction
# ===================================================================
def plot_context_layer_interaction(df_agg, output_dir):
    """How does context length modulate the layer-wise similarity profile?

    Tests whether longer contexts cause some layers to become more
    specialized or whether the overall profile shape is invariant.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df_agg.empty:
        return

    ctx_lengths = sorted(df_agg["context_length"].unique())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for cl in ctx_lengths:
        sub = df_agg[df_agg["context_length"] == cl]
        profile = sub.groupby("layer")["mean_similarity"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Mean Similarity")
    axes[0].set_title("Layer Profile by Context Length")
    axes[0].legend(fontsize=8)

    for cl in ctx_lengths:
        sub = df_agg[df_agg["context_length"] == cl]
        profile = sub.groupby("layer")["max_similarity"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Max Similarity")
    axes[1].set_title("Max Similarity Layer Profile by Context Length")
    axes[1].legend(fontsize=8)
    _save(fig, out, "27_context_layer_interaction.png")


# ===================================================================
# Master routine
# ===================================================================
def run_all_decode_analyses(
    results_dir: str = "./results_decode",
    figures_dir: Optional[str] = None,
):
    """Run the full decode analysis suite and save all figures."""
    if figures_dir is None:
        figures_dir = str(Path(results_dir) / "figures")

    df_topk = DecodeResultStore.load_topk(results_dir)
    df_agg = DecodeResultStore.load_aggregates(results_dir)

    if df_topk.empty and df_agg.empty:
        logger.error("No data found in %s. Run decode experiments first.", results_dir)
        return

    logger.info(
        "Loaded %d topk records, %d aggregate records. Generating figures…",
        len(df_topk), len(df_agg),
    )

    # --- Basic descriptive (1-8) ---
    plot_similarity_vs_distance(df_topk, figures_dir)
    plot_layer_decode_step_heatmap(df_agg, figures_dir)
    plot_max_similarity_distribution(df_agg, figures_dir)
    plot_threshold_fraction(df_agg, figures_dir)
    plot_temporal_drift(df_agg, figures_dir)
    plot_context_length_effect(df_agg, figures_dir)
    plot_reference_position_distribution(df_topk, figures_dir)
    plot_topk_similarity_decay(df_topk, figures_dir)

    # --- Structural / geometric (9-14) ---
    plot_layer_similarity_profile(df_agg, figures_dir)
    plot_layer_phase_transitions(df_agg, figures_dir)
    plot_inter_layer_correlation(df_agg, figures_dir)
    plot_similarity_concentration(df_agg, figures_dir)
    plot_mean_similarity_heatmap(df_agg, figures_dir)
    plot_std_similarity_heatmap(df_agg, figures_dir)

    # --- Reference provenance (15-19) ---
    plot_prompt_vs_decoded_reference(df_topk, figures_dir)
    plot_reference_source_by_context(df_topk, figures_dir)
    plot_self_token_reference(df_topk, figures_dir)
    plot_reference_distance_by_layer(df_topk, figures_dir)
    plot_topk_reference_spread(df_topk, figures_dir)

    # --- Dataset comparison (20-23) ---
    plot_dataset_layer_profiles(df_agg, figures_dir)
    plot_dataset_temporal_drift(df_agg, figures_dir)
    plot_dataset_self_reference(df_topk, figures_dir)
    plot_dataset_prompt_anchoring(df_topk, figures_dir)

    # --- Information-theoretic / structural theory (24-27) ---
    plot_similarity_entropy(df_agg, figures_dir)
    plot_layer_redundancy(df_agg, figures_dir)
    plot_attention_sink(df_topk, figures_dir)
    plot_context_layer_interaction(df_agg, figures_dir)

    logger.info("All %d figures saved to %s.", 27, figures_dir)
