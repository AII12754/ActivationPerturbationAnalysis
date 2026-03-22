#!/usr/bin/env python3
"""Analyze trace compaction experiment results and generate report + plots.

Reads parquet files from results_trace_compaction/ and produces:
  - results_trace_compaction/report.md
  - results_trace_compaction/figures/*.png

Usage:
  python -m delta_coding_system.analyze_trace --input-dir results_trace_compaction
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("analyze_trace")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available — skipping plot generation")

ALL_SCHEMES = ["fp16", "int8", "int4", "delta-int8", "delta-int4"]
SCHEME_COLORS = {
    "fp16": "#95a5a6",
    "int8": "#e74c3c",
    "int4": "#f39c12",
    "delta-int8": "#2ecc71",
    "delta-int4": "#3498db",
}
SCHEME_LABELS = {
    "fp16": "FP16 (baseline)",
    "int8": "Pure INT8",
    "int4": "Pure INT4",
    "delta-int8": "Delta-INT8",
    "delta-int4": "Delta-INT4",
}
BANDWIDTHS_MBPS = [200, 500, 1000]


def _setup_chinese_font():
    """Try to set up Chinese font for matplotlib."""
    if not HAS_MPL:
        return
    for font_name in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def load_results(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all result parquet files."""
    results = {}
    for name in ["prefill_summary", "decode_summary", "raw_detail"]:
        path = input_dir / f"{name}.parquet"
        if path.exists():
            results[name] = pd.read_parquet(path)
            logger.info("Loaded %s: %d rows", name, len(results[name]))
        else:
            logger.warning("Missing: %s", path)
    return results


# ===================================================================
# Plot functions
# ===================================================================
def plot_compression_ratio(df_prefill: pd.DataFrame, fig_dir: Path):
    """Bar chart: compression ratio per event, grouped by scheme."""
    if not HAS_MPL:
        return

    events = df_prefill["event_id"].unique()
    schemes = [s for s in ALL_SCHEMES if s != "fp16" and s in df_prefill["scheme"].unique()]

    fig, ax = plt.subplots(figsize=(max(12, len(events) * 2), 6))
    x = np.arange(len(events))
    width = 0.8 / max(len(schemes), 1)

    for i, scheme in enumerate(schemes):
        ratios = []
        for ev in events:
            row = df_prefill[(df_prefill["event_id"] == ev) & (df_prefill["scheme"] == scheme)]
            ratios.append(row["compression_ratio"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, ratios, width,
                      label=SCHEME_LABELS.get(scheme, scheme),
                      color=SCHEME_COLORS.get(scheme, "#333"))
        for bar, r in zip(bars, ratios):
            if r > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{r:.1f}x", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width * (len(schemes) - 1) / 2)
    ax.set_xticklabels([e.split("_")[-1] for e in events], rotation=30)
    ax.set_xlabel("Compaction Event")
    ax.set_ylabel("Compression Ratio (x)")
    ax.set_title("Prefill Compression Ratio by Scheme")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(df_prefill[df_prefill["scheme"] != "fp16"]["compression_ratio"].max() * 1.3, 4))
    fig.tight_layout()
    fig.savefig(fig_dir / "compression_ratio.png", dpi=150)
    plt.close(fig)
    logger.info("Saved compression_ratio.png")


def plot_transfer_bytes(df_prefill: pd.DataFrame, fig_dir: Path):
    """Bar chart: total transfer bytes per event, grouped by scheme."""
    if not HAS_MPL:
        return

    events = df_prefill["event_id"].unique()
    schemes = [s for s in ALL_SCHEMES if s in df_prefill["scheme"].unique()]

    fig, ax = plt.subplots(figsize=(max(12, len(events) * 2), 6))
    x = np.arange(len(events))
    width = 0.8 / max(len(schemes), 1)

    for i, scheme in enumerate(schemes):
        bytes_vals = []
        for ev in events:
            row = df_prefill[(df_prefill["event_id"] == ev) & (df_prefill["scheme"] == scheme)]
            bytes_vals.append(row["total_bytes"].values[0] / 1e6 if len(row) > 0 else 0)
        ax.bar(x + i * width, bytes_vals, width,
               label=SCHEME_LABELS.get(scheme, scheme),
               color=SCHEME_COLORS.get(scheme, "#333"))

    ax.set_xticks(x + width * (len(schemes) - 1) / 2)
    ax.set_xticklabels([e.split("_")[-1] for e in events], rotation=30)
    ax.set_xlabel("Compaction Event")
    ax.set_ylabel("Transfer Size (MB)")
    ax.set_title("Prefill Transfer Size by Scheme")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "transfer_bytes.png", dpi=150)
    plt.close(fig)
    logger.info("Saved transfer_bytes.png")


def plot_latency(df_prefill: pd.DataFrame, df_decode: pd.DataFrame, fig_dir: Path):
    """Bar chart: end-to-end communication latency (prefill + decode) per bandwidth."""
    if not HAS_MPL:
        return

    events = df_prefill["event_id"].unique()
    schemes = [s for s in ALL_SCHEMES if s in df_prefill["scheme"].unique()]

    for bw in BANDWIDTHS_MBPS:
        col = f"latency_{bw}mbps_ms"
        if col not in df_prefill.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(12, len(events) * 2), 6))
        x = np.arange(len(events))
        width = 0.8 / max(len(schemes), 1)

        for i, scheme in enumerate(schemes):
            latencies = []
            for ev in events:
                lat = 0.0
                row_p = df_prefill[(df_prefill["event_id"] == ev) & (df_prefill["scheme"] == scheme)]
                if len(row_p) > 0 and col in row_p.columns:
                    lat += float(row_p[col].values[0])
                if df_decode is not None:
                    row_d = df_decode[(df_decode["event_id"] == ev) & (df_decode["scheme"] == scheme)]
                    if len(row_d) > 0 and col in row_d.columns:
                        lat += float(row_d[col].values[0])
                latencies.append(lat)
            ax.bar(x + i * width, latencies, width,
                   label=SCHEME_LABELS.get(scheme, scheme),
                   color=SCHEME_COLORS.get(scheme, "#333"))

        ax.set_xticks(x + width * (len(schemes) - 1) / 2)
        ax.set_xticklabels([e.split("_")[-1] for e in events], rotation=30)
        ax.set_xlabel("Compaction Event")
        ax.set_ylabel("E2E Communication Latency (ms)")
        ax.set_title(f"End-to-End Comm Latency @ {bw} Mbps (Prefill + Decode)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"latency_{bw}mbps.png", dpi=150)
        plt.close(fig)
        logger.info("Saved latency_%dmbps.png", bw)


def plot_delta_gain(df_prefill: pd.DataFrame, fig_dir: Path):
    """Bar chart: delta scheme byte reduction vs pure quantization."""
    if not HAS_MPL:
        return

    events = df_prefill["event_id"].unique()
    pairs = [("int8", "delta-int8"), ("int4", "delta-int4")]

    fig, axes = plt.subplots(1, len(pairs), figsize=(12, 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (pure, delta) in zip(axes, pairs):
        gains = []
        ev_labels = []
        for ev in events:
            row_pure = df_prefill[(df_prefill["event_id"] == ev) & (df_prefill["scheme"] == pure)]
            row_delta = df_prefill[(df_prefill["event_id"] == ev) & (df_prefill["scheme"] == delta)]
            if len(row_pure) > 0 and len(row_delta) > 0:
                pure_bytes = row_pure["total_bytes"].values[0]
                delta_bytes = row_delta["total_bytes"].values[0]
                gain_pct = (1 - delta_bytes / max(pure_bytes, 1)) * 100
                gains.append(gain_pct)
                ev_labels.append(ev.split("_")[-1])

        colors = ["#2ecc71" if g > 0 else "#e74c3c" for g in gains]
        bars = ax.bar(ev_labels, gains, color=colors, width=0.6)
        for bar, g in zip(bars, gains):
            y_offset = 1 if g >= 0 else -3
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    f"{g:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Compaction Event")
        ax.set_ylabel("Byte Reduction (%)")
        ax.set_title(f"Delta Gain: {SCHEME_LABELS[delta]} vs {SCHEME_LABELS[pure]}")
        ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "delta_gain.png", dpi=150)
    plt.close(fig)
    logger.info("Saved delta_gain.png")


# ===================================================================
# Table generation
# ===================================================================
def _short_event_id(ev: str) -> str:
    """Shorten event ID for table display."""
    return ev.split("_", 1)[-1] if "_" in ev else ev


def generate_compression_table(df: pd.DataFrame) -> str:
    """Generate markdown table of compression ratios."""
    events = df["event_id"].unique()
    schemes = [s for s in ALL_SCHEMES if s in df["scheme"].unique()]

    lines = ["| Event | " + " | ".join(SCHEME_LABELS.get(s, s) for s in schemes) + " |"]
    lines.append("|" + "---|" * (len(schemes) + 1))

    for ev in events:
        row_vals = [_short_event_id(ev)]
        for s in schemes:
            r = df[(df["event_id"] == ev) & (df["scheme"] == s)]
            if len(r) > 0:
                row_vals.append(f"{r['compression_ratio'].values[0]:.2f}x")
            else:
                row_vals.append("—")
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def generate_latency_table(df_prefill: pd.DataFrame, df_decode: pd.DataFrame, bw: int) -> str:
    """Generate markdown table of e2e latency for a given bandwidth."""
    col = f"latency_{bw}mbps_ms"
    events = df_prefill["event_id"].unique()
    schemes = [s for s in ALL_SCHEMES if s in df_prefill["scheme"].unique()]

    lines = [f"| Event | " + " | ".join(SCHEME_LABELS.get(s, s) for s in schemes) + " |"]
    lines.append("|" + "---|" * (len(schemes) + 1))

    for ev in events:
        row_vals = [_short_event_id(ev)]
        for s in schemes:
            lat = 0.0
            r_p = df_prefill[(df_prefill["event_id"] == ev) & (df_prefill["scheme"] == s)]
            if len(r_p) > 0 and col in r_p.columns:
                lat += float(r_p[col].values[0])
            if df_decode is not None:
                r_d = df_decode[(df_decode["event_id"] == ev) & (df_decode["scheme"] == s)]
                if len(r_d) > 0 and col in r_d.columns:
                    lat += float(r_d[col].values[0])
            row_vals.append(f"{lat:.2f} ms")
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def generate_transfer_table(df: pd.DataFrame) -> str:
    """Generate markdown table of transfer sizes in MB."""
    events = df["event_id"].unique()
    schemes = [s for s in ALL_SCHEMES if s in df["scheme"].unique()]

    lines = ["| Event | " + " | ".join(SCHEME_LABELS.get(s, s) for s in schemes) + " |"]
    lines.append("|" + "---|" * (len(schemes) + 1))

    for ev in events:
        row_vals = [_short_event_id(ev)]
        for s in schemes:
            r = df[(df["event_id"] == ev) & (df["scheme"] == s)]
            if len(r) > 0:
                mb = r["total_bytes"].values[0] / 1e6
                row_vals.append(f"{mb:.2f} MB")
            else:
                row_vals.append("—")
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


# ===================================================================
# Report generation
# ===================================================================
def generate_report(results: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate comprehensive markdown report with embedded figures."""
    df_prefill = results.get("prefill_summary")
    df_decode = results.get("decode_summary")

    if df_prefill is None:
        logger.error("No prefill_summary data found, cannot generate report.")
        return

    num_events = df_prefill["event_id"].nunique()
    available_schemes = [s for s in ALL_SCHEMES if s in df_prefill["scheme"].unique()]

    report_lines = [
        "# Trace Compaction: Compression Ratio & Communication Latency Report",
        "",
        "## 1. Experiment Overview",
        "",
        f"- **Total compaction events**: {num_events}",
        f"- **Schemes tested**: {', '.join(SCHEME_LABELS.get(s, s) for s in available_schemes)}",
        f"- **Bandwidths**: {', '.join(str(b) + ' Mbps' for b in BANDWIDTHS_MBPS)}",
        "",
        "### Scheme Definitions",
        "",
        "| Scheme | Description |",
        "|--------|-------------|",
        "| FP16 | Baseline: transmit full FP16 activations |",
        "| Pure INT8 | Group-wise INT8 quantization + top-k FP16 outliers |",
        "| Pure INT4 | Group-wise INT4 quantization + top-k FP16 outliers |",
        "| Delta-INT8 | Delta coding (affine + INT8 quantized delta) for referenced positions; "
        "pure INT8 for unigram; shared prefix cached (0 bytes) |",
        "| Delta-INT4 | Delta coding (affine + INT4 quantized delta) for referenced positions; "
        "pure INT4 for unigram; shared prefix cached (0 bytes) |",
        "",
        "---",
        "",
        "## 2. Prefill Phase: Compression Ratio",
        "",
        generate_compression_table(df_prefill),
        "",
        "![Compression Ratio](figures/compression_ratio.png)",
        "",
        "---",
        "",
        "## 3. Prefill Phase: Transfer Size",
        "",
        generate_transfer_table(df_prefill),
        "",
        "![Transfer Bytes](figures/transfer_bytes.png)",
        "",
        "---",
        "",
        "## 4. End-to-End Communication Latency",
        "",
    ]

    for bw in BANDWIDTHS_MBPS:
        report_lines.append(f"### @ {bw} Mbps")
        report_lines.append("")
        report_lines.append(generate_latency_table(df_prefill, df_decode, bw))
        report_lines.append("")
        report_lines.append(f"![Latency {bw}Mbps](figures/latency_{bw}mbps.png)")
        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## 5. Delta Gain (Delta vs Pure Quantization)",
        "",
        "Byte reduction percentage = (1 - delta_bytes / pure_bytes) × 100%",
        "",
        "![Delta Gain](figures/delta_gain.png)",
        "",
        "---",
        "",
    ])

    # Decode section
    if df_decode is not None and len(df_decode) > 0:
        report_lines.extend([
            "## 6. Decode Phase Summary",
            "",
            generate_compression_table(df_decode),
            "",
            "---",
            "",
        ])
    else:
        report_lines.extend([
            "## 6. Decode Phase Summary",
            "",
            "_No decode results available._",
            "",
            "---",
            "",
        ])

    # Summary statistics
    report_lines.extend([
        "## 7. Summary Statistics",
        "",
    ])

    for scheme in available_schemes:
        subset = df_prefill[df_prefill["scheme"] == scheme]
        if len(subset) > 0:
            avg_ratio = subset["compression_ratio"].mean()
            avg_bytes = subset["total_bytes"].mean()
            report_lines.append(
                f"- **{SCHEME_LABELS.get(scheme, scheme)}**: "
                f"avg ratio = {avg_ratio:.2f}x, avg transfer = {avg_bytes / 1e6:.2f} MB"
            )

    # Delta gain summary
    report_lines.append("")
    for pure, delta in [("int8", "delta-int8"), ("int4", "delta-int4")]:
        pure_df = df_prefill[df_prefill["scheme"] == pure]
        delta_df = df_prefill[df_prefill["scheme"] == delta]
        if len(pure_df) > 0 and len(delta_df) > 0:
            pure_avg = pure_df["total_bytes"].mean()
            delta_avg = delta_df["total_bytes"].mean()
            gain = (1 - delta_avg / max(pure_avg, 1)) * 100
            report_lines.append(
                f"- **{SCHEME_LABELS[delta]} vs {SCHEME_LABELS[pure]}**: "
                f"avg byte reduction = {gain:.1f}%"
            )

    report_lines.append("")

    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info("Report saved to %s", report_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze trace compaction experiment results")
    parser.add_argument("--input-dir", default="results_trace_compaction",
                        help="Directory containing result parquet files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("Input dir does not exist: %s", input_dir)
        sys.exit(1)

    _setup_chinese_font()

    results = load_results(input_dir)

    fig_dir = input_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_prefill = results.get("prefill_summary")
    df_decode = results.get("decode_summary")

    if df_prefill is not None:
        plot_compression_ratio(df_prefill, fig_dir)
        plot_transfer_bytes(df_prefill, fig_dir)
        plot_latency(df_prefill, df_decode, fig_dir)
        plot_delta_gain(df_prefill, fig_dir)

    generate_report(results, input_dir)
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
