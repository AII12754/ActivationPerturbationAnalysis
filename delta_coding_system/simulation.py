"""Simulation module for computing compression ratio and communication latency.

Provides analytical (no-GPU) calculation of transfer bytes for:
  - FP16 baseline
  - Pure INT8 quantization
  - Pure INT4 quantization

All values are derived from sequence length, hidden dimension, and
quantization parameters — no actual model forward pass needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class SchemeResult:
    """Result for a single compression scheme."""
    name: str
    total_bytes: int
    compression_ratio: float  # fp16_bytes / total_bytes
    latency_ms: Dict[int, float] = field(default_factory=dict)  # bandwidth_mbps -> ms


@dataclass
class SimulationResult:
    """Aggregated simulation results for all schemes."""
    seq_len: int
    hidden_dim: int
    fp16_bytes: int
    schemes: Dict[str, SchemeResult] = field(default_factory=dict)


def _network_ms(total_bytes: float, bandwidth_mbps: int) -> float:
    """Compute network transfer latency in milliseconds."""
    return float(total_bytes) * 8.0 / (float(bandwidth_mbps) * 1000.0)


def compute_scheme_bytes(
    seq_len: int,
    hidden_dim: int,
    bits: int,
    group_size: int = 128,
    top_k: int = 1,
) -> int:
    """Compute total transfer bytes for a pure quantization scheme.

    Per-position overhead breakdown:
      - quantized data: hidden_dim * (bits/8) bytes
      - scales: num_groups * 2 bytes (float16)
      - zero_points: num_groups * 2 bytes (float16)
      - topk_values: num_groups * top_k * 2 bytes (float16)
      - topk_indices: num_groups * top_k * 1 byte (uint8)
    """
    if bits == 16:
        return seq_len * hidden_dim * 2

    num_groups = hidden_dim // group_size

    if bits == 8:
        quant_bytes = hidden_dim * 1  # uint8
    elif bits == 4:
        quant_bytes = hidden_dim // 2  # 2 values per byte
    else:
        raise ValueError(f"Unsupported bit width: {bits}")

    # Per-group metadata
    scale_bytes = num_groups * 2        # float16 scales
    zero_bytes = num_groups * 2         # float16 zero points
    topk_val_bytes = num_groups * top_k * 2   # float16 outlier values
    topk_idx_bytes = num_groups * top_k * 1   # uint8 outlier indices

    per_position = quant_bytes + scale_bytes + zero_bytes + topk_val_bytes + topk_idx_bytes
    return seq_len * per_position


def simulate_compression(
    seq_len: int,
    hidden_dim: int,
    group_size: int = 128,
    top_k: int = 1,
    bandwidths: List[int] | None = None,
) -> SimulationResult:
    """Compute simulated compression ratio and latency for all pure schemes.

    Parameters
    ----------
    seq_len : number of tokens
    hidden_dim : model hidden dimension
    group_size : quantization group size
    top_k : number of FP16 outliers per group
    bandwidths : list of bandwidth values in Mbps

    Returns
    -------
    SimulationResult with FP16/INT8/INT4 scheme data.
    """
    if bandwidths is None:
        bandwidths = [200, 500, 1000]

    fp16_bytes = seq_len * hidden_dim * 2

    result = SimulationResult(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        fp16_bytes=fp16_bytes,
    )

    scheme_configs = [
        ("fp16", 16),
        ("int8", 8),
        ("int4", 4),
    ]

    for name, bits in scheme_configs:
        total_bytes = compute_scheme_bytes(seq_len, hidden_dim, bits, group_size, top_k)
        ratio = fp16_bytes / max(total_bytes, 1)
        latency = {bw: _network_ms(total_bytes, bw) for bw in bandwidths}
        result.schemes[name] = SchemeResult(
            name=name,
            total_bytes=total_bytes,
            compression_ratio=ratio,
            latency_ms=latency,
        )

    return result
