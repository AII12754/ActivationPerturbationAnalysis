#!/usr/bin/env python3
"""E12v2 analysis: generate summary tables and plots from parquet results.

Reads results_e12v2/{dataset}/*.parquet, generates:
  - report_e12v2/ with PNG plots
  - report_e12v2_cn.md summary report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("analyze_e12v2")

# ===================================================================
# Data loading
# ===================================================================

def load_all(results_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all parquet files grouped by dataset.

    Returns: {dataset_name: {"pipeline_quality": df, "tier_detail": df, ...}}
    """
    data = {}
    for ds_dir in sorted(results_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        ds_data = {}
        for pq_file in ds_dir.glob("*.parquet"):
            table_name = pq_file.stem
            ds_data[table_name] = pd.read_parquet(pq_file)
        if ds_data:
            data[ds_name] = ds_data
            logger.info("Loaded %s: %s", ds_name, list(ds_data.keys()))
    return data


# ===================================================================
# Plots
# ===================================================================

def plot_tier_distribution(data: Dict, out_dir: Path):
    """Stacked bar chart: tier distribution per dataset."""
    datasets = []
    pcts = {"trigram": [], "bigram": [], "self_ref": [], "unigram": []}

    for ds_name, tables in sorted(data.items()):
        df = tables.get("pipeline_quality")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty:
            continue
        datasets.append(ds_name)
        for tier in pcts:
            pcts[tier].append(test_df[f"pct_{tier}"].mean())

    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.6
    bottom = np.zeros(len(datasets))

    colors = {"trigram": "#2ecc71", "bigram": "#3498db", "self_ref": "#f39c12", "unigram": "#e74c3c"}
    for tier in ["trigram", "bigram", "self_ref", "unigram"]:
        vals = np.array(pcts[tier])
        ax.bar(x, vals, width, bottom=bottom, label=tier, color=colors[tier])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Tier Distribution per Dataset (Test Phase)")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out_dir / "tier_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved tier_distribution.png")


def plot_reconstruction_cosine(data: Dict, out_dir: Path):
    """Box plot: reconstruction cosine per tier across datasets."""
    records = []
    for ds_name, tables in sorted(data.items()):
        df = tables.get("tier_detail")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        for _, row in test_df.iterrows():
            if row["count"] > 0:
                records.append({
                    "dataset": ds_name,
                    "tier": row["tier"],
                    "recon_cosine": row["recon_cosine_mean"],
                })

    if not records:
        return

    rdf = pd.DataFrame(records)
    tiers = ["trigram", "bigram", "self_ref", "unigram"]
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"trigram": "#2ecc71", "bigram": "#3498db", "self_ref": "#f39c12", "unigram": "#e74c3c"}
    positions = []
    labels = []
    pos = 0
    for tier in tiers:
        subset = rdf[rdf["tier"] == tier]
        if subset.empty:
            continue
        vals = subset["recon_cosine"].values
        bp = ax.boxplot([vals], positions=[pos], widths=0.6, patch_artist=True)
        bp["boxes"][0].set_facecolor(colors.get(tier, "#999"))
        labels.append(tier)
        positions.append(pos)
        pos += 1

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reconstruction Cosine Similarity")
    ax.set_title("Reconstruction Quality by Tier")
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_cosine.png", dpi=150)
    plt.close(fig)
    logger.info("Saved reconstruction_cosine.png")


def plot_compression_ratio(data: Dict, out_dir: Path):
    """Bar chart: compression ratio per dataset."""
    datasets = []
    ratios = []

    for ds_name, tables in sorted(data.items()):
        df = tables.get("pipeline_quality")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty:
            continue
        datasets.append(ds_name)
        ratios.append(test_df["compression_ratio"].mean())

    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    ax.bar(x, ratios, color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Compression Ratio (×)")
    ax.set_title("Average Compression Ratio per Dataset")
    fig.tight_layout()
    fig.savefig(out_dir / "compression_ratio.png", dpi=150)
    plt.close(fig)
    logger.info("Saved compression_ratio.png")


def plot_latency_breakdown(data: Dict, out_dir: Path):
    """Stacked bar chart: latency breakdown per dataset."""
    datasets = []
    components = ["prefill_ms", "classify_ms", "encode_delta_ms",
                  "encode_self_ref_ms", "encode_unigram_ms", "decode_generate_ms",
                  "table_update_ms"]
    comp_data = {c: [] for c in components}

    for ds_name, tables in sorted(data.items()):
        df = tables.get("latency")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty:
            continue
        datasets.append(ds_name)
        for c in components:
            comp_data[c].append(test_df[c].mean() if c in test_df.columns else 0)

    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    width = 0.6
    bottom = np.zeros(len(datasets))

    cmap = plt.cm.Set2
    for idx, c in enumerate(components):
        vals = np.array(comp_data[c])
        ax.bar(x, vals, width, bottom=bottom, label=c.replace("_ms", ""),
               color=cmap(idx / len(components)))
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Breakdown per Dataset (Test Phase)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_breakdown.png", dpi=150)
    plt.close(fig)
    logger.info("Saved latency_breakdown.png")


def plot_table_growth(data: Dict, out_dir: Path):
    """Line plot: table growth (trigrams) over requests."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ds_name, tables in sorted(data.items()):
        df = tables.get("table_growth")
        if df is None:
            continue
        ax.plot(range(len(df)), df["num_trigrams"].values, label=ds_name, linewidth=1.5)

    ax.set_xlabel("Request Index (warmup + test)")
    ax.set_ylabel("Trigrams in Table")
    ax.set_title("Table Growth Over Requests")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "table_growth.png", dpi=150)
    plt.close(fig)
    logger.info("Saved table_growth.png")


def plot_coverage_growth(data: Dict, out_dir: Path):
    """Line plot: trigram coverage over test requests."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ds_name, tables in sorted(data.items()):
        df = tables.get("table_growth")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty:
            continue
        ax.plot(range(len(test_df)), test_df["trigram_coverage"].values,
                label=ds_name, linewidth=1.5)

    ax.set_xlabel("Test Request Index")
    ax.set_ylabel("Trigram Coverage")
    ax.set_title("Trigram Coverage Growth (Test Phase)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_growth.png", dpi=150)
    plt.close(fig)
    logger.info("Saved coverage_growth.png")


def plot_cosine_stability(data: Dict, out_dir: Path):
    """Line plot: cosine similarity over test requests."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ds_name, tables in sorted(data.items()):
        df = tables.get("pipeline_quality")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty:
            continue
        ax.plot(range(len(test_df)), test_df["cosine_similarity_mean"].values,
                label=ds_name, linewidth=1.5)

    ax.set_xlabel("Test Request Index")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Reconstruction Cosine Stability (Test Phase)")
    ax.legend()
    ax.set_ylim(0.8, 1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_stability.png", dpi=150)
    plt.close(fig)
    logger.info("Saved cosine_stability.png")


def plot_seq_length_distribution(data: Dict, out_dir: Path):
    """Histogram: sequence length distribution per dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for ds_name, tables in sorted(data.items()):
        df = tables.get("pipeline_quality")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty or "seq_len" not in test_df.columns:
            continue
        ax.hist(test_df["seq_len"].values, bins=30, alpha=0.5, label=ds_name)

    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Sequence Length Distribution per Dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "seq_length_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved seq_length_distribution.png")


def plot_bandwidth_comparison(data: Dict, out_dir: Path):
    """Bar chart: E2E latency vs effective bandwidth."""
    datasets = []
    total_ms = []
    transfer_mb = []

    for ds_name, tables in sorted(data.items()):
        df = tables.get("pipeline_quality")
        lat_df = tables.get("latency")
        if df is None or lat_df is None:
            continue
        test_pq = df[df["phase"] == "test"] if "phase" in df.columns else df
        test_lat = lat_df[lat_df["phase"] == "test"] if "phase" in lat_df.columns else lat_df
        if test_pq.empty or test_lat.empty:
            continue
        datasets.append(ds_name)
        total_ms.append(test_lat["total_ms"].mean())
        transfer_mb.append(test_pq["total_transfer_bytes"].mean() / 1e6)

    if not datasets:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(datasets))

    ax1.bar(x, total_ms, color="#3498db")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=30, ha="right")
    ax1.set_ylabel("Total Latency (ms)")
    ax1.set_title("E2E Latency per Dataset")

    ax2.bar(x, transfer_mb, color="#2ecc71")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=30, ha="right")
    ax2.set_ylabel("Transfer Size (MB)")
    ax2.set_title("Average Transfer Size per Dataset")

    fig.tight_layout()
    fig.savefig(out_dir / "bandwidth_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved bandwidth_comparison.png")


# ===================================================================
# Report generation
# ===================================================================

def generate_report(data: Dict, out_dir: Path, report_path: Path):
    """Generate markdown report."""
    lines = [
        "# E12v2 实验报告: 自然长度 Trigram 流水线 + 解码阶段",
        "",
        "## 1. 实验概述",
        "",
        "本实验（E12v2）使用 6 个数据集的原始文本长度运行 trigram delta-coding 流水线。",
        "相比 E12，E12v2 的改进：",
        "- 文本保持自然长度（无拼接/截断）",
        "- 独立的预热阶段和测试阶段",
        "- 包含 128 token 解码（生成）阶段",
        "- 每个数据集 100 个预热 + 100 个测试请求",
        "",
    ]

    # Summary table
    lines.append("## 2. 数据集与序列长度统计")
    lines.append("")
    lines.append("| 数据集 | 测试请求数 | 平均序列长度 | 中位序列长度 | 最短 | 最长 |")
    lines.append("|--------|-----------|-------------|-------------|------|------|")

    for ds_name, tables in sorted(data.items()):
        df = tables.get("pipeline_quality")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty or "seq_len" not in test_df.columns:
            continue
        sl = test_df["seq_len"]
        lines.append(f"| {ds_name} | {len(test_df)} | {sl.mean():.0f} | {sl.median():.0f} | {sl.min()} | {sl.max()} |")

    lines.append("")

    # Tier distribution
    lines.append("## 3. Tier 分布")
    lines.append("")
    lines.append("| 数据集 | Trigram% | Bigram% | Self-ref% | Unigram% |")
    lines.append("|--------|---------|---------|-----------|----------|")

    for ds_name, tables in sorted(data.items()):
        df = tables.get("pipeline_quality")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty:
            continue
        lines.append(
            f"| {ds_name} | {test_df['pct_trigram'].mean():.1f} | "
            f"{test_df['pct_bigram'].mean():.1f} | "
            f"{test_df['pct_self_ref'].mean():.1f} | "
            f"{test_df['pct_unigram'].mean():.1f} |"
        )

    lines.append("")
    lines.append("![Tier Distribution](report_e12v2/tier_distribution.png)")
    lines.append("")

    # Quality
    lines.append("## 4. 重建质量")
    lines.append("")
    lines.append("| 数据集 | Cosine Mean | Cosine Min | MSE Mean | 压缩比 |")
    lines.append("|--------|------------|------------|----------|--------|")

    for ds_name, tables in sorted(data.items()):
        df = tables.get("pipeline_quality")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty:
            continue
        lines.append(
            f"| {ds_name} | {test_df['cosine_similarity_mean'].mean():.4f} | "
            f"{test_df['cosine_similarity_min'].mean():.4f} | "
            f"{test_df['mse_mean'].mean():.6f} | "
            f"{test_df['compression_ratio'].mean():.2f}× |"
        )

    lines.append("")
    lines.append("![Reconstruction Cosine](report_e12v2/reconstruction_cosine.png)")
    lines.append("")
    lines.append("![Cosine Stability](report_e12v2/cosine_stability.png)")
    lines.append("")

    # Compression
    lines.append("## 5. 压缩效率")
    lines.append("")
    lines.append("![Compression Ratio](report_e12v2/compression_ratio.png)")
    lines.append("")

    # Latency
    lines.append("## 6. 延迟分析")
    lines.append("")
    lines.append("| 数据集 | Prefill(ms) | Classify(ms) | Encode(ms) | Decode(ms) | Total(ms) |")
    lines.append("|--------|------------|-------------|------------|------------|-----------|")

    for ds_name, tables in sorted(data.items()):
        df = tables.get("latency")
        if df is None:
            continue
        test_df = df[df["phase"] == "test"] if "phase" in df.columns else df
        if test_df.empty:
            continue
        encode_ms = test_df["encode_delta_ms"].mean() + test_df["encode_self_ref_ms"].mean() + test_df["encode_unigram_ms"].mean()
        lines.append(
            f"| {ds_name} | {test_df['prefill_ms'].mean():.1f} | "
            f"{test_df['classify_ms'].mean():.1f} | "
            f"{encode_ms:.1f} | "
            f"{test_df['decode_generate_ms'].mean():.1f} | "
            f"{test_df['total_ms'].mean():.1f} |"
        )

    lines.append("")
    lines.append("![Latency Breakdown](report_e12v2/latency_breakdown.png)")
    lines.append("")

    # Table growth
    lines.append("## 7. Table 增长")
    lines.append("")
    lines.append("![Table Growth](report_e12v2/table_growth.png)")
    lines.append("")
    lines.append("![Coverage Growth](report_e12v2/coverage_growth.png)")
    lines.append("")

    # Bandwidth
    lines.append("## 8. 带宽对比")
    lines.append("")
    lines.append("![Bandwidth Comparison](report_e12v2/bandwidth_comparison.png)")
    lines.append("")

    # Seq length
    lines.append("## 9. 序列长度分布")
    lines.append("")
    lines.append("![Seq Length Distribution](report_e12v2/seq_length_distribution.png)")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved to %s", report_path)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="E12v2 analysis and report generation")
    parser.add_argument("--results-dir", default="results_e12v2", help="Results directory")
    parser.add_argument("--output-dir", default="report_e12v2", help="Output directory for plots")
    parser.add_argument("--report", default="report_e12v2_cn.md", help="Report output path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(results_dir)
    if not data:
        logger.error("No data found in %s", results_dir)
        return

    # Generate all plots
    plot_tier_distribution(data, out_dir)
    plot_reconstruction_cosine(data, out_dir)
    plot_compression_ratio(data, out_dir)
    plot_latency_breakdown(data, out_dir)
    plot_table_growth(data, out_dir)
    plot_coverage_growth(data, out_dir)
    plot_cosine_stability(data, out_dir)
    plot_seq_length_distribution(data, out_dir)
    plot_bandwidth_comparison(data, out_dir)

    # Generate report
    generate_report(data, out_dir, Path(args.report))

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
