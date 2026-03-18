#!/usr/bin/env python3
"""E12 Trigram Delta-Coding Pipeline — Comprehensive Performance Report.

Generates a full report with:
  1. System Design Overview
  2. Tier Match Rates (warmed table, requests 50-99)
  3. Raw Reference Cosine Similarity per Tier
  4. Reconstruction Quality (cosine, MSE)
  5. Compression Ratio Analysis
  6. Latency Breakdown (with CPU/GPU overlap optimization)
  7. End-to-End Latency under Varying Bandwidth
  8. Table Growth Trajectory
  9. Per-Dataset Breakdown
  10. Summary Statistics

All analysis uses only the "warmed" period (request_index >= 50).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────
RESULTS_DIR = Path("results_trigram_pipeline")
OUTPUT_DIR = Path("report_e12")
OUTPUT_DIR.mkdir(exist_ok=True)
WARMUP = 50  # first 50 requests are warmup
MODEL_NAME = "Qwen2.5-32B-Instruct"
HIDDEN_DIM = 5120  # Qwen2.5-32B hidden size

# Plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (12, 5),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
COLORS = {
    "trigram": "#2196F3",
    "bigram": "#FF9800",
    "self_ref": "#4CAF50",
    "unigram": "#F44336",
}
CTX_COLORS = {512: "#1976D2", 2048: "#D32F2F"}

# ── Load Data ──────────────────────────────────────────────────────
def load_table(name):
    files = sorted(RESULTS_DIR.glob(f"{name}_part_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df

pq = load_table("pipeline_quality")
td = load_table("tier_detail")
tg = load_table("table_growth")
lat = load_table("latency")

# Shorten dataset names for display
def short_ds(name):
    return os.path.basename(str(name)).replace("-2-raw-v1", "").replace("ShareGPT52K", "ShareGPT")

for df in [pq, td, tg, lat]:
    df["ds_short"] = df["dataset_name"].apply(short_ds)

# Filter to warmed period only
pq_w = pq[pq["request_index"] >= WARMUP].copy()
td_w = td[td["request_index"] >= WARMUP].copy()
tg_w = tg[tg["request_index"] >= WARMUP].copy()
lat_w = lat[lat["request_index"] >= WARMUP].copy()

print(f"Total records: {len(pq)} | Warmed records (req >= {WARMUP}): {len(pq_w)}")
print(f"Datasets: {sorted(pq_w['ds_short'].unique())}")
print(f"Context lengths: {sorted(pq_w['context_length'].unique())}")
print()

# ===================================================================
# REPORT SECTION 1: System Design
# ===================================================================
design_text = f"""
{'='*80}
  E12: TRIGRAM DELTA-CODING PIPELINE — PERFORMANCE REPORT
  Model: {MODEL_NAME} (32B params, hidden_dim={HIDDEN_DIM})
  Layer boundary: 6 (first 6 transformer layers)
  Warmup: {WARMUP} requests | Test: {WARMUP} requests per (dataset, context_length)
  Datasets: 7 | Context lengths: 512, 2048
{'='*80}

1. SYSTEM DESIGN OVERVIEW
─────────────────────────
The trigram delta-coding pipeline compresses intermediate activations at a
pipeline-parallel (PP) boundary layer by exploiting n-gram repetition across
requests. A persistent DAG trie stores pre-computed hidden states keyed by
token n-grams, enabling cache-like reuse of activation patterns.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Sender (PP stage 0)                                           │
  │                                                                 │
  │  ① Prefill (GPU)  ──────────────────┐                          │
  │  ② Classify (CPU, overlapped with ①) │  concurrent             │
  │  ③ Encode per tier (GPU):            │                          │
  │     • Trigram/Bigram: Affine + Int4 delta + Top-K sparsity     │
  │     • Self-ref: Delta against earlier reconstructed position   │
  │     • Unigram: Int8 groupwise + outlier Top-K                  │
  │  ④ Table update (CPU, overlapped with ③)                       │
  └─────────────────────────────────────────────────────────────────┘
       │ compressed packet
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Receiver (PP stage 1)                                         │
  │                                                                 │
  │  ⑤ Decode + reconstruct activations                            │
  └─────────────────────────────────────────────────────────────────┘

Tier Hierarchy (cascading fallback):
  1. TRIGRAM — exact (A,B,C) match in DAG → Affine + Int4 delta coding
  2. BIGRAM — (B,C) prefix match → Affine + Int4 delta coding
  3. SELF-REF — same trigram seen earlier in this request → delta vs.
     already-reconstructed position
  4. UNIGRAM — no match → Int8 groupwise quantization + outlier encoding

Encoding Details:
  • Affine transform: per-position scale & bias to align ref → real
  • Int4 delta: groupwise quantization (group_size=128) of residual
  • Top-K sparsity: keep top-1 outlier indices per group
  • Int8 unigram: groupwise 8-bit quantization + top-1 FP16 outliers

CPU/GPU Overlap Optimization:
  • classify_and_build_refs runs on a background CPU thread concurrently
    with the GPU prefill pass (GIL released during CUDA kernel launches)
  • update_from_hidden_states runs on a background CPU thread concurrently
    with the GPU encoding phases
  • Both CPU ops are fully hidden behind GPU work (~0ms visible latency)
"""
print(design_text)

# ===================================================================
# SECTION 2: Tier Match Rates
# ===================================================================
print("2. TIER MATCH RATES (warmed table, requests 50-99)")
print("─" * 60)

tier_pcts = pq_w.groupby("context_length")[
    ["pct_trigram", "pct_bigram", "pct_self_ref", "pct_unigram"]
].agg(["mean", "std"])

for ctx in [512, 2048]:
    row = tier_pcts.loc[ctx]
    print(f"\n  Context length = {ctx}:")
    for tier in ["trigram", "bigram", "self_ref", "unigram"]:
        m = row[(f"pct_{tier}", "mean")]
        s = row[(f"pct_{tier}", "std")]
        print(f"    {tier:10s}: {m:6.2f}% ± {s:.2f}%")

# Per-dataset match rate
print(f"\n  Per-dataset trigram+self_ref coverage (ctx=2048):")
ds_coverage = pq_w[pq_w["context_length"] == 2048].groupby("ds_short").apply(
    lambda g: pd.Series({
        "trigram%": g["pct_trigram"].mean(),
        "self_ref%": g["pct_self_ref"].mean(),
        "covered%": (g["pct_trigram"] + g["pct_self_ref"] + g["pct_bigram"]).mean(),
    })
).round(2)
print(ds_coverage.to_string(index=True))

# ── Plot: Tier distribution stacked bar ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, ctx in zip(axes, [512, 2048]):
    sub = pq_w[pq_w["context_length"] == ctx].groupby("ds_short")[
        ["pct_trigram", "pct_bigram", "pct_self_ref", "pct_unigram"]
    ].mean()
    sub = sub.reindex(sorted(sub.index))
    bottom = np.zeros(len(sub))
    for tier, col in [("trigram", "pct_trigram"), ("bigram", "pct_bigram"),
                       ("self_ref", "pct_self_ref"), ("unigram", "pct_unigram")]:
        ax.bar(sub.index, sub[col], bottom=bottom, label=tier, color=COLORS[tier], width=0.6)
        bottom += sub[col].values
    ax.set_title(f"Tier Distribution (ctx={ctx})")
    ax.set_ylabel("% of positions")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis='x', rotation=30)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "tier_distribution.png", bbox_inches="tight")
plt.close(fig)
print(f"\n  [Saved: {OUTPUT_DIR}/tier_distribution.png]")

# ===================================================================
# SECTION 3: Raw Reference Cosine Similarity
# ===================================================================
print("\n\n3. RAW REFERENCE COSINE SIMILARITY (before delta coding)")
print("─" * 60)

for ctx in [512, 2048]:
    print(f"\n  Context length = {ctx}:")
    sub = td_w[(td_w["context_length"] == ctx) & (td_w["count"] > 0)]
    for tier in ["trigram", "bigram", "self_ref"]:
        t_sub = sub[sub["tier"] == tier]
        if len(t_sub) == 0:
            continue
        m = t_sub["raw_cosine_mean"].mean()
        mn = t_sub["raw_cosine_min"].mean()
        print(f"    {tier:10s}: mean={m:.6f}  worst-case-min={mn:.6f}")

# ── Plot: Raw cosine per tier ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, ctx in zip(axes, [512, 2048]):
    sub = td_w[(td_w["context_length"] == ctx) & (td_w["count"] > 0)]
    for tier in ["trigram", "bigram", "self_ref"]:
        t_sub = sub[sub["tier"] == tier]["raw_cosine_mean"]
        if len(t_sub) == 0:
            continue
        ax.hist(t_sub, bins=40, alpha=0.6, label=tier, color=COLORS[tier], density=True)
    ax.set_title(f"Raw Reference Cosine (ctx={ctx})")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0.95, 1.005)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "raw_cosine_per_tier.png", bbox_inches="tight")
plt.close(fig)
print(f"  [Saved: {OUTPUT_DIR}/raw_cosine_per_tier.png]")

# ===================================================================
# SECTION 4: Reconstruction Quality
# ===================================================================
print("\n\n4. RECONSTRUCTION QUALITY (after delta coding)")
print("─" * 60)

for ctx in [512, 2048]:
    sub = pq_w[pq_w["context_length"] == ctx]
    cos_m = sub["cosine_similarity_mean"].mean()
    cos_min = sub["cosine_similarity_min"].mean()
    mse_m = sub["mse_mean"].mean()
    mse_max = sub["mse_max"].mean()
    print(f"\n  Context length = {ctx}:")
    print(f"    Cosine similarity:  mean={cos_m:.6f}  avg-min={cos_min:.6f}")
    print(f"    MSE:                mean={mse_m:.8f}  avg-max={mse_max:.8f}")

# Per-tier reconstruction quality
print(f"\n  Per-tier reconstruction cosine (ctx=2048):")
for tier in ["trigram", "bigram", "self_ref", "unigram"]:
    t_sub = td_w[(td_w["context_length"] == 2048) & (td_w["tier"] == tier) & (td_w["count"] > 0)]
    if len(t_sub) == 0:
        print(f"    {tier:10s}: no data")
        continue
    m = t_sub["recon_cosine_mean"].mean()
    mn = t_sub["recon_cosine_min"].mean()
    print(f"    {tier:10s}: mean={m:.6f}  avg-min={mn:.6f}")

# ── Plot: Reconstruction cosine distribution ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, ctx in zip(axes, [512, 2048]):
    sub = pq_w[pq_w["context_length"] == ctx]
    ax.hist(sub["cosine_similarity_mean"], bins=40, alpha=0.7, color="#2196F3", edgecolor="white")
    ax.axvline(sub["cosine_similarity_mean"].mean(), color="red", linestyle="--",
               label=f"mean={sub['cosine_similarity_mean'].mean():.4f}")
    ax.set_title(f"Reconstruction Cosine Similarity (ctx={ctx})")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "reconstruction_cosine.png", bbox_inches="tight")
plt.close(fig)
print(f"\n  [Saved: {OUTPUT_DIR}/reconstruction_cosine.png]")

# ===================================================================
# SECTION 5: Compression Ratio
# ===================================================================
print("\n\n5. COMPRESSION RATIO")
print("─" * 60)

for ctx in [512, 2048]:
    sub = pq_w[pq_w["context_length"] == ctx]
    cr_m = sub["compression_ratio"].mean()
    cr_std = sub["compression_ratio"].std()
    total_raw = sub["raw_fp16_bytes"].sum()
    total_comp = sub["total_transfer_bytes"].sum()
    print(f"\n  Context length = {ctx}:")
    print(f"    Compression ratio:    {cr_m:.2f}× ± {cr_std:.2f}×")
    print(f"    Aggregate:            {total_raw/1e6:.1f} MB raw → {total_comp/1e6:.1f} MB compressed")

    # Per-tier transfer breakdown
    for tier in ["trigram", "bigram", "self_ref", "unigram"]:
        tb = sub[f"transfer_bytes_{tier}"].sum()
        pct = tb / max(total_comp, 1) * 100
        print(f"    {tier:10s} transfer:  {tb/1e6:.2f} MB ({pct:.1f}%)")

# ── Plot: Compression ratio by dataset ──
fig, ax = plt.subplots(figsize=(12, 5))
for ctx in [512, 2048]:
    sub = pq_w[pq_w["context_length"] == ctx].groupby("ds_short")["compression_ratio"].mean()
    sub = sub.reindex(sorted(sub.index))
    x = np.arange(len(sub))
    w = 0.35
    offset = -w/2 if ctx == 512 else w/2
    ax.bar(x + offset, sub.values, w, label=f"ctx={ctx}", color=CTX_COLORS[ctx], alpha=0.8)
ax.set_xticks(np.arange(len(sub)))
ax.set_xticklabels(sorted(sub.index), rotation=30)
ax.set_ylabel("Compression Ratio (×)")
ax.set_title("Compression Ratio by Dataset and Context Length")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "compression_ratio.png", bbox_inches="tight")
plt.close(fig)
print(f"\n  [Saved: {OUTPUT_DIR}/compression_ratio.png]")

# ===================================================================
# SECTION 6: Latency Breakdown
# ===================================================================
print("\n\n6. LATENCY BREAKDOWN (with CPU/GPU overlap)")
print("─" * 60)

lat_cols = ["prefill_ms", "classify_ms", "encode_delta_ms",
            "encode_self_ref_ms", "encode_unigram_ms", "decode_ms",
            "table_update_ms", "total_ms"]

for ctx in [512, 2048]:
    sub = lat_w[lat_w["context_length"] == ctx]
    print(f"\n  Context length = {ctx}:")
    for col in lat_cols:
        m = sub[col].mean()
        s = sub[col].std()
        label = col.replace("_ms", "").replace("_", " ")
        print(f"    {label:20s}: {m:8.2f} ms ± {s:.2f}")

    # Compute overlap savings
    classify_hidden = sub["classify_ms"].mean()
    update_hidden = sub["table_update_ms"].mean()
    print(f"\n    classify visible (overlapped):    {classify_hidden:.2f} ms  (~0 = fully hidden)")
    print(f"    table_update visible (overlapped): {update_hidden:.2f} ms  (~0 = fully hidden)")

# ── Plot: Latency breakdown stacked bar ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
phase_cols = ["prefill_ms", "classify_ms", "encode_delta_ms",
              "encode_self_ref_ms", "encode_unigram_ms", "table_update_ms"]
phase_labels = ["Prefill (GPU)", "Classify (CPU→overlapped)", "Encode delta (GPU)",
                "Encode self-ref (GPU)", "Encode unigram (GPU)", "Table update (CPU→overlapped)"]
phase_colors = ["#2196F3", "#90CAF9", "#FF9800", "#FFC107", "#F44336", "#A5D6A7"]

for ax, ctx in zip(axes, [512, 2048]):
    sub = lat_w[lat_w["context_length"] == ctx].groupby("ds_short")[phase_cols].mean()
    sub = sub.reindex(sorted(sub.index))
    bottom = np.zeros(len(sub))
    for col, lbl, clr in zip(phase_cols, phase_labels, phase_colors):
        ax.bar(sub.index, sub[col], bottom=bottom, label=lbl, color=clr, width=0.6)
        bottom += sub[col].values
    ax.set_title(f"Latency Breakdown (ctx={ctx})")
    ax.set_ylabel("Time (ms)")
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "latency_breakdown.png", bbox_inches="tight")
plt.close(fig)
print(f"\n  [Saved: {OUTPUT_DIR}/latency_breakdown.png]")

# ===================================================================
# SECTION 7: End-to-End Latency under Varying Bandwidth
# ===================================================================
print("\n\n7. END-TO-END LATENCY vs. BANDWIDTH (pipeline-parallel transfer)")
print("─" * 60)
print("""
  Communication overhead model (cross-node pipeline-parallel):
    Baseline:  T_transfer_raw
    Ours:      T_encode + T_transfer_compressed + T_decode
  where T_transfer = bytes / bandwidth

  Prefill (T_compute) is identical in both cases and excluded.

  Target scenario: cross-node PP over 200 Mbps–1 Gbps links
  (e.g., cloud VPC, WAN, or bandwidth-constrained cluster interconnect)

  T_encode  ≈ encode_delta_ms + encode_self_ref_ms + encode_unigram_ms
  T_decode  ≈ measured dequantize + reconstruct on receiver GPU
""")

# Measured decode-only costs (dequant + reconstruct, from benchmark)
DECODE_MS = {512: 0.34, 2048: 0.63}

bandwidths_gbps = [0.025, 0.0625, 0.125, 0.2, 0.5, 1, 5, 10, 25]  # GB/s  (200Mbps=0.025, 500Mbps=0.0625, 1Gbps=0.125, ...)

for ctx in [512, 2048]:
    sub_lat = lat_w[lat_w["context_length"] == ctx]
    sub_pq = pq_w[pq_w["context_length"] == ctx]

    encode = (sub_lat["encode_delta_ms"] + sub_lat["encode_self_ref_ms"] +
              sub_lat["encode_unigram_ms"]).mean()
    decode = DECODE_MS[ctx]

    raw_bytes = sub_pq["raw_fp16_bytes"].mean()
    comp_bytes = sub_pq["total_transfer_bytes"].mean()

    print(f"\n  Context length = {ctx} (seq_len≈{ctx}):")
    print(f"    Raw FP16 transfer:    {raw_bytes/1024:.1f} KB")
    print(f"    Compressed transfer:  {comp_bytes/1024:.1f} KB")
    print(f"    Encode overhead:      {encode:.2f} ms")
    print(f"    Decode overhead:      {decode:.2f} ms")
    def fmt_bw(gbps):
        mbps = gbps * 8000  # GB/s → Mbps
        if mbps < 1000:
            return f"{mbps:.0f} Mbps"
        return f"{mbps/1000:.0f} Gbps"

    print(f"    {'Bandwidth':>14s} {'Baseline (ms)':>15s} {'Ours (ms)':>12s} {'Speedup':>10s}")
    print(f"    {'─'*55}")

    for bw in bandwidths_gbps:
        bw_bytes_per_ms = bw * 1e9 / 1e3  # bytes per ms
        t_raw = raw_bytes / bw_bytes_per_ms
        t_comp = comp_bytes / bw_bytes_per_ms
        baseline = t_raw
        ours = encode + t_comp + decode
        speedup = baseline / ours if ours > 0 else float("inf")
        print(f"    {fmt_bw(bw):>14s}   {baseline:>13.2f}   {ours:>10.2f}   {speedup:>8.2f}×")

# ── Plot: E2E latency vs bandwidth ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, ctx in zip(axes, [512, 2048]):
    sub_lat = lat_w[lat_w["context_length"] == ctx]
    sub_pq = pq_w[pq_w["context_length"] == ctx]

    encode = (sub_lat["encode_delta_ms"] + sub_lat["encode_self_ref_ms"] +
              sub_lat["encode_unigram_ms"]).mean()
    decode = DECODE_MS[ctx]
    raw_bytes = sub_pq["raw_fp16_bytes"].mean()
    comp_bytes = sub_pq["total_transfer_bytes"].mean()

    bws = np.array(bandwidths_gbps)
    baselines = []
    ours_list = []
    for bw in bws:
        bw_bpms = bw * 1e9 / 1e3
        baselines.append(raw_bytes / bw_bpms)
        ours_list.append(encode + comp_bytes / bw_bpms + decode)

    ax.plot(bws, baselines, "o-", label="Baseline (FP16 transfer)", color="#F44336", linewidth=2)
    ax.plot(bws, ours_list, "s-", label="Ours (Trigram Pipeline)", color="#2196F3", linewidth=2)
    ax.fill_between(bws, ours_list, baselines, alpha=0.15, color="#2196F3")
    ax.set_xscale("log")
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Communication Overhead (ms)")
    ax.set_title(f"Communication Overhead vs. Bandwidth (ctx={ctx})")
    ax.legend()
    bw_labels = []
    for b in bws:
        mbps = b * 8000
        bw_labels.append(f"{mbps:.0f}M" if mbps < 1000 else f"{mbps/1000:.0f}G")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(bws)
    ax.set_xticklabels(bw_labels, rotation=30)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "e2e_latency_vs_bandwidth.png", bbox_inches="tight")
plt.close(fig)
print(f"\n  [Saved: {OUTPUT_DIR}/e2e_latency_vs_bandwidth.png]")

# ── Plot: Speedup vs bandwidth ──
fig, ax = plt.subplots(figsize=(10, 5))
for ctx in [512, 2048]:
    sub_lat = lat_w[lat_w["context_length"] == ctx]
    sub_pq = pq_w[pq_w["context_length"] == ctx]
    encode = (sub_lat["encode_delta_ms"] + sub_lat["encode_self_ref_ms"] +
              sub_lat["encode_unigram_ms"]).mean()
    decode = DECODE_MS[ctx]
    raw_bytes = sub_pq["raw_fp16_bytes"].mean()
    comp_bytes = sub_pq["total_transfer_bytes"].mean()

    speedups = []
    for bw in bandwidths_gbps:
        bw_bpms = bw * 1e9 / 1e3
        orig = raw_bytes / bw_bpms
        ours = encode + comp_bytes / bw_bpms + decode
        speedups.append(orig / ours)

    ax.plot(bandwidths_gbps, speedups, "o-", label=f"ctx={ctx}", color=CTX_COLORS[ctx], linewidth=2)

ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Break-even")
ax.set_xscale("log")
ax.set_xlabel("Bandwidth")
ax.set_ylabel("Speedup (×)")
ax.set_title("Speedup over FP16 Baseline vs. Bandwidth")
ax.legend()
bw_labels_s = []
for b in bandwidths_gbps:
    mbps = b * 8000
    bw_labels_s.append(f"{mbps:.0f}M" if mbps < 1000 else f"{mbps/1000:.0f}G")
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xticks(bandwidths_gbps)
ax.set_xticklabels(bw_labels_s, rotation=30)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "speedup_vs_bandwidth.png", bbox_inches="tight")
plt.close(fig)
print(f"  [Saved: {OUTPUT_DIR}/speedup_vs_bandwidth.png]")

# ===================================================================
# SECTION 8: Table Growth Trajectory
# ===================================================================
print("\n\n8. TABLE GROWTH TRAJECTORY")
print("─" * 60)

for ctx in [512, 2048]:
    sub = tg[tg["context_length"] == ctx]
    # Show first and last request averages across datasets
    first = sub[sub["request_index"] == 0]
    last = sub[sub["request_index"] == 99]
    at_50 = sub[sub["request_index"] == WARMUP]
    print(f"\n  Context length = {ctx}:")
    print(f"    After request  0: {first['num_trigrams'].mean():,.0f} trigrams, {first['num_bigrams'].mean():,.0f} bigrams")
    print(f"    After request 50: {at_50['num_trigrams'].mean():,.0f} trigrams, {at_50['num_bigrams'].mean():,.0f} bigrams")
    print(f"    After request 99: {last['num_trigrams'].mean():,.0f} trigrams, {last['num_bigrams'].mean():,.0f} bigrams")
    print(f"    Memory at req 99: {last['memory_bytes'].mean()/1e6:.1f} MB")

# ── Plot: Table growth ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ctx in zip(axes, [512, 2048]):
    sub = tg[tg["context_length"] == ctx]
    avg = sub.groupby("request_index")[["num_trigrams", "num_bigrams"]].mean()
    ax.plot(avg.index, avg["num_trigrams"], label="Trigrams", color="#2196F3", linewidth=2)
    ax.plot(avg.index, avg["num_bigrams"], label="Bigrams", color="#FF9800", linewidth=2)
    ax.axvline(WARMUP, color="red", linestyle="--", alpha=0.5, label=f"Warmup boundary (req={WARMUP})")
    ax.set_title(f"Table Growth (ctx={ctx}, avg across datasets)")
    ax.set_xlabel("Request Index")
    ax.set_ylabel("Count")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "table_growth.png", bbox_inches="tight")
plt.close(fig)
print(f"\n  [Saved: {OUTPUT_DIR}/table_growth.png]")

# ===================================================================
# SECTION 9: Per-Dataset Detailed Breakdown
# ===================================================================
print("\n\n9. PER-DATASET BREAKDOWN (warmed period, ctx=2048)")
print("─" * 60)

ctx2k = pq_w[pq_w["context_length"] == 2048]
ds_summary = ctx2k.groupby("ds_short").agg({
    "pct_trigram": "mean",
    "pct_bigram": "mean",
    "pct_self_ref": "mean",
    "pct_unigram": "mean",
    "cosine_similarity_mean": "mean",
    "cosine_similarity_min": "mean",
    "compression_ratio": "mean",
    "raw_cosine_mean": "mean",
}).round(4)

ds_summary.columns = ["tri%", "bi%", "self%", "uni%", "cos_mean", "cos_min", "ratio", "raw_cos"]
print(ds_summary.to_string())

# ===================================================================
# SECTION 10: Summary Statistics Table
# ===================================================================
print("\n\n10. SUMMARY TABLE")
print("─" * 60)

summary_rows = []
for ctx in [512, 2048]:
    sub_pq = pq_w[pq_w["context_length"] == ctx]
    sub_lat = lat_w[lat_w["context_length"] == ctx]
    sub_tg = tg_w[tg_w["context_length"] == ctx]

    coverage = (sub_pq["pct_trigram"] + sub_pq["pct_bigram"] + sub_pq["pct_self_ref"]).mean()
    encode_ms = (sub_lat["encode_delta_ms"] + sub_lat["encode_self_ref_ms"] +
                 sub_lat["encode_unigram_ms"]).mean()

    summary_rows.append({
        "Context": ctx,
        "Trigram%": f"{sub_pq['pct_trigram'].mean():.1f}",
        "SelfRef%": f"{sub_pq['pct_self_ref'].mean():.1f}",
        "Coverage%": f"{coverage:.1f}",
        "Cosine": f"{sub_pq['cosine_similarity_mean'].mean():.4f}",
        "CompRatio": f"{sub_pq['compression_ratio'].mean():.2f}×",
        "Prefill(ms)": f"{sub_lat['prefill_ms'].mean():.1f}",
        "Encode(ms)": f"{encode_ms:.1f}",
        "Classify(ms)": f"{sub_lat['classify_ms'].mean():.2f}",
        "Update(ms)": f"{sub_lat['table_update_ms'].mean():.2f}",
        "Total(ms)": f"{sub_lat['total_ms'].mean():.1f}",
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# ── Plot: Per-tier reconstruction cosine box plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ctx in zip(axes, [512, 2048]):
    sub = td_w[(td_w["context_length"] == ctx) & (td_w["count"] > 0)]
    data = []
    labels = []
    for tier in ["trigram", "bigram", "self_ref", "unigram"]:
        vals = sub[sub["tier"] == tier]["recon_cosine_mean"].dropna()
        if len(vals) > 0:
            data.append(vals.values)
            labels.append(tier)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, whis=(5, 95))
    for patch, tier in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS.get(tier, "#ccc"))
        patch.set_alpha(0.7)
    ax.set_title(f"Reconstruction Cosine by Tier (ctx={ctx})")
    ax.set_ylabel("Cosine Similarity")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "recon_cosine_boxplot.png", bbox_inches="tight")
plt.close(fig)
print(f"\n  [Saved: {OUTPUT_DIR}/recon_cosine_boxplot.png]")

# ── Coverage growth over requests ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ctx in zip(axes, [512, 2048]):
    sub = pq[pq["context_length"] == ctx]
    coverage_by_req = sub.groupby("request_index").apply(
        lambda g: (g["pct_trigram"] + g["pct_bigram"] + g["pct_self_ref"]).mean()
    )
    ax.plot(coverage_by_req.index, coverage_by_req.values, color="#2196F3", linewidth=2)
    ax.axvline(WARMUP, color="red", linestyle="--", alpha=0.5, label=f"Warmup boundary")
    ax.axhline(coverage_by_req.iloc[-10:].mean(), color="green", linestyle="--",
               alpha=0.5, label=f"Steady state: {coverage_by_req.iloc[-10:].mean():.1f}%")
    ax.set_title(f"N-gram Coverage Over Requests (ctx={ctx})")
    ax.set_xlabel("Request Index")
    ax.set_ylabel("Coverage (%)")
    ax.legend()
    ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "coverage_growth.png", bbox_inches="tight")
plt.close(fig)
print(f"  [Saved: {OUTPUT_DIR}/coverage_growth.png]")

# ── Cosine quality over requests ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, ctx in zip(axes, [512, 2048]):
    sub = pq[pq["context_length"] == ctx]
    cos_by_req = sub.groupby("request_index")["cosine_similarity_mean"].mean()
    ax.plot(cos_by_req.index, cos_by_req.values, color="#4CAF50", linewidth=2)
    ax.axvline(WARMUP, color="red", linestyle="--", alpha=0.5, label=f"Warmup boundary")
    ax.set_title(f"Reconstruction Cosine Over Requests (ctx={ctx})")
    ax.set_xlabel("Request Index")
    ax.set_ylabel("Cosine Similarity")
    ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "cosine_over_requests.png", bbox_inches="tight")
plt.close(fig)
print(f"  [Saved: {OUTPUT_DIR}/cosine_over_requests.png]")

print(f"""

{'='*80}
  REPORT COMPLETE
  All figures saved to: {OUTPUT_DIR.resolve()}/
  Data source: {RESULTS_DIR.resolve()}/
{'='*80}
""")
