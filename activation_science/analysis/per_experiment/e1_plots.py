"""E1: Residual stream decomposition analysis — 10 plot functions.

Ported from ``src/residual_analysis.py``.  Analyses which layers do real
computational work (large delta) vs skip (small delta), and how layer
contributions evolve during autoregressive decoding.

Plots
-----
01. delta_norm_by_layer          — Line: delta_norm by layer (prefill vs decode)
02. delta_residual_ratio_by_layer — Line: contribution profile by layer
03. cos_delta_residual           — Line: delta-residual alignment by layer
04. cos_delta_embedding          — Line: embedding persistence by layer
05. cos_delta_prev_delta         — Line: inter-layer delta correlation
06. delta_norm_heatmap           — Heatmap: delta_norm (layer x step)
07. delta_ratio_heatmap          — Heatmap: delta_residual_ratio (layer x step)
08. dataset_delta_profiles       — 3-panel line by dataset
09. context_length_effect        — 2-panel by context_length
10. layer_contribution_ranking   — Horizontal bar: sorted contribution
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
    """Pick a representative subset of layers for line plots."""
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
    return ExperimentStore.load_table(results_dir, "residual")


# ===================================================================
# Plot 1: Delta Norm by Layer (prefill vs decode)
# ===================================================================
def plot_delta_norm_by_layer(df, figures_dir):
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
    _savefig(fig, figures_dir, "delta_norm_by_layer", 1)


# ===================================================================
# Plot 2: Delta/Residual Ratio by Layer
# ===================================================================
def plot_delta_residual_ratio_by_layer(df, figures_dir):
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
    _savefig(fig, figures_dir, "delta_residual_ratio_by_layer", 2)


# ===================================================================
# Plot 3: cos(delta, residual) by Layer
# ===================================================================
def plot_cos_delta_residual(df, figures_dir):
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
    _savefig(fig, figures_dir, "cos_delta_residual", 3)


# ===================================================================
# Plot 4: cos(delta, embedding) by Layer
# ===================================================================
def plot_cos_delta_embedding(df, figures_dir):
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
    _savefig(fig, figures_dir, "cos_delta_embedding", 4)


# ===================================================================
# Plot 5: cos(delta[L], delta[L-1]) by Layer
# ===================================================================
def plot_cos_delta_prev_delta(df, figures_dir):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for phase in ["prefill", "decode"]:
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
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
    _savefig(fig, figures_dir, "cos_delta_prev_delta", 5)


# ===================================================================
# Plot 6: Delta Norm Heatmap
# ===================================================================
def plot_delta_norm_heatmap(df, figures_dir):
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
    ax.set_ylabel("Layer")
    ax.set_xlabel("Decode Step")
    _savefig(fig, figures_dir, "delta_norm_heatmap", 6)


# ===================================================================
# Plot 7: Delta/Residual Ratio Heatmap
# ===================================================================
def plot_delta_ratio_heatmap(df, figures_dir):
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
    ax.set_ylabel("Layer")
    ax.set_xlabel("Decode Step")
    _savefig(fig, figures_dir, "delta_ratio_heatmap", 7)


# ===================================================================
# Plot 8: Dataset Comparison: Delta Profiles
# ===================================================================
def plot_dataset_delta_profiles(df, figures_dir):
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
        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Layer — Dataset Comparison")
        ax.legend(fontsize=7, ncol=2)
    _savefig(fig, figures_dir, "dataset_delta_profiles", 8)


# ===================================================================
# Plot 9: Context Length Effect
# ===================================================================
def plot_context_length_effect(df, figures_dir):
    if df.empty:
        return
    decode = df[df["phase"] == "decode"]
    if decode.empty or "context_length" not in decode.columns:
        return
    ctx_lengths = sorted(decode["context_length"].unique())
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for cl in ctx_lengths:
        sub = decode[decode["context_length"] == cl]
        profile = sub.groupby("layer")["delta_norm"].mean().sort_index()
        axes[0].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Delta Norm")
    axes[0].set_title("Delta Norm Layer Profile by Context Length")
    axes[0].legend(fontsize=8)
    for cl in ctx_lengths:
        sub = decode[decode["context_length"] == cl]
        profile = sub.groupby("layer")["delta_residual_ratio"].mean().sort_index()
        axes[1].plot(profile.index, profile.values, label=f"ctx={cl}", alpha=0.8)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Delta/Residual Ratio")
    axes[1].set_title("Contribution Profile by Context Length")
    axes[1].legend(fontsize=8)
    _savefig(fig, figures_dir, "context_length_effect", 9)


# ===================================================================
# Plot 10: Layer Contribution Ranking
# ===================================================================
def plot_layer_contribution_ranking(df, figures_dir):
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
    _savefig(fig, figures_dir, "layer_contribution_ranking", 10)


# ===================================================================
# Master function
# ===================================================================
def run_all_residual_analyses(results_dir: str, figures_dir: str):
    """Run all 10 residual decomposition analysis plots."""
    os.makedirs(figures_dir, exist_ok=True)
    logger.info("Loading residual data from %s", results_dir)

    df = _load_data(results_dir)
    if df.empty:
        logger.warning("No residual data found in %s. Skipping.", results_dir)
        return

    logger.info("Loaded %d residual records.", len(df))

    for name, fn, args in [
        ("delta_norm_by_layer", plot_delta_norm_by_layer, (df, figures_dir)),
        ("delta_residual_ratio_by_layer", plot_delta_residual_ratio_by_layer, (df, figures_dir)),
        ("cos_delta_residual", plot_cos_delta_residual, (df, figures_dir)),
        ("cos_delta_embedding", plot_cos_delta_embedding, (df, figures_dir)),
        ("cos_delta_prev_delta", plot_cos_delta_prev_delta, (df, figures_dir)),
        ("delta_norm_heatmap", plot_delta_norm_heatmap, (df, figures_dir)),
        ("delta_ratio_heatmap", plot_delta_ratio_heatmap, (df, figures_dir)),
        ("dataset_delta_profiles", plot_dataset_delta_profiles, (df, figures_dir)),
        ("context_length_effect", plot_context_length_effect, (df, figures_dir)),
        ("layer_contribution_ranking", plot_layer_contribution_ranking, (df, figures_dir)),
    ]:
        try:
            fn(*args)
        except Exception:
            logger.exception("Failed to generate %s", name)

    logger.info("All 10 residual plots complete. Figures saved to %s", figures_dir)
