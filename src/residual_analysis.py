"""
Visualization module for residual stream decomposition experiments.

Reads stored residual results and produces figures exploring which layers
do real computational work (large delta) vs skip (small delta), and how
layer contributions evolve during autoregressive decoding.

Analysis suite (10 plots):
  1. Delta norm by layer (prefill vs decode)
  2. Delta/residual ratio by layer (contribution profile)
  3. cos(delta, residual) by layer (alignment profile)
  4. cos(delta, embedding) by layer (embedding persistence)
  5. cos(delta[L], delta[L-1]) by layer (inter-layer delta correlation)
  6. Delta norm heatmap: layer x decode_step
  7. Delta/residual ratio heatmap: layer x decode_step
  8. Dataset comparison: delta profiles
  9. Context length effect on delta norms
  10. Layer contribution ranking (sorted delta/residual ratio)
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

from .residual_storage import ResidualResultStore

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
# 1. Delta Norm by Layer (prefill vs decode average)
# ===================================================================
def plot_delta_norm_by_layer(df, output_dir):
    """Line plot: delta_norm vs layer, separate lines for prefill and decode."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for phase in ["prefill", "decode"]:
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["delta_norm"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], linewidth=2, label=phase)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("||delta[L]|| = ||h[L] - h[L-1]||")
    ax.set_title("Delta Norm by Layer (Prefill vs Decode)")
    ax.legend(fontsize=10)
    _save(fig, out, "01_delta_norm_by_layer.png")


# ===================================================================
# 2. Delta/Residual Ratio by Layer (contribution profile)
# ===================================================================
def plot_delta_residual_ratio_by_layer(df, output_dir):
    """The 'contribution profile': which layers add the most relative to the stream."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for phase in ["prefill", "decode"]:
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["delta_residual_ratio"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], linewidth=2, label=phase)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("||delta[L]|| / ||h[L]||")
    ax.set_title("Delta/Residual Ratio by Layer (Layer Contribution Profile)")
    ax.legend(fontsize=10)
    _save(fig, out, "02_delta_residual_ratio_by_layer.png")


# ===================================================================
# 3. cos(delta, residual) by Layer — alignment profile
# ===================================================================
def plot_cos_delta_residual(df, output_dir):
    """How aligned is each layer's update with the residual stream?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for phase in ["prefill", "decode"]:
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["cos_delta_residual"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], linewidth=2, label=phase)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(delta[L], h[L])")
    ax.set_title("Delta-Residual Alignment by Layer")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    _save(fig, out, "03_cos_delta_residual.png")


# ===================================================================
# 4. cos(delta, embedding) by Layer — embedding persistence
# ===================================================================
def plot_cos_delta_embedding(df, output_dir):
    """How much does each layer's update still correlate with the initial embedding?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for phase in ["prefill", "decode"]:
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        profile = sub.groupby("layer")["cos_delta_embedding"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], linewidth=2, label=phase)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(delta[L], h[0])")
    ax.set_title("Delta-Embedding Alignment by Layer (Embedding Persistence)")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    _save(fig, out, "04_cos_delta_embedding.png")


# ===================================================================
# 5. cos(delta[L], delta[L-1]) by Layer — inter-layer delta correlation
# ===================================================================
def plot_cos_delta_prev_delta(df, output_dir):
    """Are consecutive layers doing similar things (high cos) or orthogonal work?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for phase in ["prefill", "decode"]:
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        # Filter out NaN values (layer 1 has no prev delta)
        sub = sub.dropna(subset=["cos_delta_prev_delta"])
        profile = sub.groupby("layer")["cos_delta_prev_delta"].agg(["mean", "std"]).reset_index()
        ax.plot(profile["layer"], profile["mean"], linewidth=2, label=phase)
        ax.fill_between(
            profile["layer"],
            profile["mean"] - profile["std"],
            profile["mean"] + profile["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(delta[L], delta[L-1])")
    ax.set_title("Inter-Layer Delta Correlation")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.legend(fontsize=10)
    _save(fig, out, "05_cos_delta_prev_delta.png")


# ===================================================================
# 6. Delta Norm Heatmap: layer x decode_step
# ===================================================================
def plot_delta_norm_heatmap(df, output_dir):
    """Heatmap showing delta_norm across layers and decode steps."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    decode = df[df["phase"] == "decode"]
    if decode.empty:
        return

    pivot = decode.pivot_table(
        values="delta_norm", index="layer", columns="step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_title("Delta Norm: Layer x Decode Step")
    ax.set_ylabel("Layer"); ax.set_xlabel("Decode Step")
    _save(fig, out, "06_delta_norm_heatmap.png")


# ===================================================================
# 7. Delta/Residual Ratio Heatmap: layer x decode_step
# ===================================================================
def plot_delta_ratio_heatmap(df, output_dir):
    """Heatmap showing delta/residual ratio across layers and decode steps."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    decode = df[df["phase"] == "decode"]
    if decode.empty:
        return

    pivot = decode.pivot_table(
        values="delta_residual_ratio", index="layer", columns="step", aggfunc="mean",
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(pivot.columns) * 0.15), max(6, len(pivot) * 0.3))
    )
    sns.heatmap(pivot, cmap="viridis", ax=ax)
    ax.set_title("Delta/Residual Ratio: Layer x Decode Step")
    ax.set_ylabel("Layer"); ax.set_xlabel("Decode Step")
    _save(fig, out, "07_delta_ratio_heatmap.png")


# ===================================================================
# 8. Dataset Comparison: Delta Profiles
# ===================================================================
def plot_dataset_delta_profiles(df, output_dir):
    """Do different text domains produce different layer-wise delta signatures?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty or "dataset_name" not in df.columns:
        return

    decode = df[df["phase"] == "decode"]
    if decode.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("delta_norm", "Delta Norm"),
        ("delta_residual_ratio", "Delta/Residual Ratio"),
        ("cos_delta_residual", "cos(delta, residual)"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for ds in decode["dataset_name"].unique():
            sub = decode[decode["dataset_name"] == ds]
            profile = sub.groupby("layer")[metric].mean().sort_index()
            ax.plot(profile.index, profile.values, label=_ds_label(ds), alpha=0.8)
        ax.set_xlabel("Layer"); ax.set_ylabel(title)
        ax.set_title(f"{title} by Layer — Dataset Comparison")
        ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "08_dataset_delta_profiles.png")


# ===================================================================
# 9. Context Length Effect on Delta Norms
# ===================================================================
def plot_context_length_effect(df, output_dir):
    """How does context length modulate the layer-wise delta norm profile?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    decode = df[df["phase"] == "decode"]
    if decode.empty:
        return

    ctx_lengths = sorted(decode["context_length"].unique())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Delta norm by context length.
    for cl in ctx_lengths:
        sub = decode[decode["context_length"] == cl]
        profile = sub.groupby("layer")["delta_norm"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Delta Norm")
    axes[0].set_title("Delta Norm Layer Profile by Context Length")
    axes[0].legend(fontsize=8)

    # Delta/residual ratio by context length.
    for cl in ctx_lengths:
        sub = decode[decode["context_length"] == cl]
        profile = sub.groupby("layer")["delta_residual_ratio"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Delta/Residual Ratio")
    axes[1].set_title("Contribution Profile by Context Length")
    axes[1].legend(fontsize=8)
    _save(fig, out, "09_context_length_effect.png")


# ===================================================================
# 10. Layer Contribution Ranking (sorted delta/residual ratio)
# ===================================================================
def plot_layer_contribution_ranking(df, output_dir):
    """Rank layers by their mean delta/residual ratio — which layers contribute most?"""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for ax, phase, title_phase in zip(axes, ["prefill", "decode"], ["Prefill", "Decode"]):
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        ranking = sub.groupby("layer")["delta_residual_ratio"].mean().sort_values(ascending=False)
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(ranking)))
        ax.barh(
            range(len(ranking)),
            ranking.values,
            color=colors,
            edgecolor="gray",
            linewidth=0.3,
        )
        ax.set_yticks(range(len(ranking)))
        ax.set_yticklabels([f"L{int(l)}" for l in ranking.index], fontsize=6)
        ax.set_xlabel("Mean Delta/Residual Ratio")
        ax.set_title(f"Layer Contribution Ranking — {title_phase}")
        ax.invert_yaxis()

    _save(fig, out, "10_layer_contribution_ranking.png")


# ===================================================================
# Master routine
# ===================================================================
def run_all_residual_analyses(
    results_dir: str = "./results_residual",
    figures_dir: Optional[str] = None,
):
    """Run the full residual analysis suite and save all figures."""
    if figures_dir is None:
        figures_dir = str(Path(results_dir) / "figures")

    df = ResidualResultStore.load_results(results_dir)

    if df.empty:
        logger.error("No data found in %s. Run residual experiments first.", results_dir)
        return

    logger.info(
        "Loaded %d residual records. Generating figures...",
        len(df),
    )

    # --- Core layer profiles (1-5) ---
    plot_delta_norm_by_layer(df, figures_dir)
    plot_delta_residual_ratio_by_layer(df, figures_dir)
    plot_cos_delta_residual(df, figures_dir)
    plot_cos_delta_embedding(df, figures_dir)
    plot_cos_delta_prev_delta(df, figures_dir)

    # --- Heatmaps (6-7) ---
    plot_delta_norm_heatmap(df, figures_dir)
    plot_delta_ratio_heatmap(df, figures_dir)

    # --- Comparative analyses (8-10) ---
    plot_dataset_delta_profiles(df, figures_dir)
    plot_context_length_effect(df, figures_dir)
    plot_layer_contribution_ranking(df, figures_dir)

    logger.info("All %d figures saved to %s.", 10, figures_dir)
