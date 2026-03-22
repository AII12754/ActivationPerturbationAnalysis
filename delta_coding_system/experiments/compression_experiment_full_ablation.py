#!/usr/bin/env python3
"""Unified compression experiment with full ablations and energy metrics.

This script reruns all explored strategies under one metric system and adds
component-level ablations so future experiments share the same analysis schema.

Recorded reference metrics:
  raw_ref_energy_explained = 1 - ||x - r||^2 / ||x||^2
  affine_energy_explained = 1 - ||x - (a r + b)||^2 / ||x||^2
  recon_energy_explained = 1 - ||x - x_hat||^2 / ||x||^2
  affine_gain = affine_energy_explained - raw_ref_energy_explained
  quantization_loss = affine_energy_explained - recon_energy_explained

Usage:
  python -m delta_coding_system.experiments.compression_experiment_full_ablation --gpu 0
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.codec import (
    apply_affine,
    compute_affine_params,
    compute_delta,
    groupwise_int4_dequantize_topk,
    groupwise_int4_quantize_topk,
    groupwise_int8_dequantize_topk,
    groupwise_int8_quantize_topk,
)
from delta_coding_system.experiments.compression_experiment import (
    groupwise_int2_dequantize_topk,
    groupwise_int2_quantize_topk,
    train_linear_predictor,
)
from delta_coding_system.table import NgramTable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("compress_full_ablation")

HIDDEN_DIM = 5120
GROUP_SIZE = 128

ALL_STRATEGY_NAMES = [
    "baseline",
    "sparse_010",
    "sparse_005",
    "int2_topk4",
    "int2_topk8",
    "adaptive_bw",
    "combined",
    "best_sparse",
    "linear_pred",
    "unigram_int4_k4",
    "unigram_int4_k8",
    "prev_int4_k1",
    "prev_int4_k2",
    "prev_int4_k4",
    "prev_int2_k4",
    "prev_int2_k8",
    "mean_pool",
    "ema_prev4_int4_k2",
    "prev2_blend_int4_k2",
    "prev_gs256_k2",
    "sparse_prev",
    "best_combined",
    "hybrid_sparse_prev",
    "delta_sparse_thr_20",
    "delta_sparse_thr_10",
    "delta_sparse_thr_05",
    "delta_top16_groups",
    "delta_top8_groups",
    "delta_top1024_ch",
    "delta_top512_ch",
    "delta_struct2of8",
    "delta_int2_top8",
    "ablate_delta_raw_ref",
    "ablate_delta_affine_only",
    "ablate_delta_affine_int4_k0",
    "ablate_delta_affine_int4_k1",
    "ablate_prev_raw_ref",
    "ablate_prev_affine_only",
    "ablate_prev_affine_int4_k0",
    "ablate_prev_affine_int4_k1",
    "ablate_prev_affine_int4_k2",
]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).item()


def _energy(tensor: torch.Tensor) -> float:
    return float(tensor.float().pow(2).sum().item())


def _error_energy(target: torch.Tensor, approx: torch.Tensor) -> float:
    return float((target.float() - approx.float()).pow(2).sum().item())


def _explained_energy(target: torch.Tensor, approx: torch.Tensor) -> float:
    target_energy = max(_energy(target), 1e-8)
    return 1.0 - _error_energy(target, approx) / target_energy


def _reference_metrics(real_h: torch.Tensor, ref_h: Optional[torch.Tensor]) -> Dict[str, float]:
    nan_metrics = {
        "raw_ref_cosine": float("nan"),
        "raw_ref_energy_explained": float("nan"),
        "affine_ref_cosine": float("nan"),
        "affine_energy_explained": float("nan"),
        "affine_gain": float("nan"),
        "raw_error_energy": float("nan"),
        "affine_error_energy": float("nan"),
    }
    if ref_h is None:
        return nan_metrics

    scale, bias = compute_affine_params(real_h, ref_h)
    affine_ref = apply_affine(ref_h, scale, bias)
    raw_explained = _explained_energy(real_h, ref_h)
    affine_explained = _explained_energy(real_h, affine_ref)
    return {
        "raw_ref_cosine": _cosine(real_h, ref_h),
        "raw_ref_energy_explained": raw_explained,
        "affine_ref_cosine": _cosine(real_h, affine_ref),
        "affine_energy_explained": affine_explained,
        "affine_gain": affine_explained - raw_explained,
        "raw_error_energy": _error_energy(real_h, ref_h),
        "affine_error_energy": _error_energy(real_h, affine_ref),
    }


def _quantize_no_outlier(
    tensor: torch.Tensor,
    group_size: int,
    bits: int,
) -> Tuple[torch.Tensor, int]:
    batch, hidden_dim = tensor.shape
    num_groups = hidden_dim // group_size
    grouped = tensor.float().reshape(batch, num_groups, group_size)
    g_min = grouped.min(dim=-1).values
    g_max = grouped.max(dim=-1).values

    if bits == 4:
        qmax = 15.0
        data_bytes = hidden_dim // 2
    elif bits == 2:
        qmax = 3.0
        data_bytes = hidden_dim // 4
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    scale = ((g_max - g_min) / qmax).unsqueeze(-1)
    zp = g_min.unsqueeze(-1)
    q = torch.clamp(torch.round((grouped - zp) / (scale + 1e-10)), 0, int(qmax))
    dequant = q * scale + zp
    recon = dequant.reshape(batch, hidden_dim).to(torch.float16)
    transfer = data_bytes + num_groups * 2 + num_groups * 2
    return recon, transfer


def _quantize_delta(
    delta: torch.Tensor,
    group_size: int,
    bits: int,
    top_k: int,
) -> Tuple[torch.Tensor, int]:
    if top_k == 0:
        return _quantize_no_outlier(delta, group_size, bits)
    if bits == 4:
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, delta.shape[-1])
        data_bytes = packed.nelement()
    elif bits == 2:
        packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(delta, group_size, top_k)
        dequant = groupwise_int2_dequantize_topk(packed, scales, zeros, tv, ti, group_size, delta.shape[-1])
        data_bytes = packed.nelement()
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    transfer = (
        data_bytes
        + scales.nelement() * 2
        + zeros.nelement() * 2
        + tv.nelement() * 2
        + ti.nelement() * 1
    )
    return dequant, transfer


def _encode_delta(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
    group_size: int,
    bits: int,
    top_k: int,
    include_ref_idx: bool,
    extra_param_bytes: int = 4,
) -> Dict[str, object]:
    ref_metrics = _reference_metrics(real_h, ref_h)
    scale, bias = compute_affine_params(real_h, ref_h)
    affine_ref = apply_affine(ref_h, scale, bias)
    delta = compute_delta(real_h, affine_ref)
    dequant, payload_bytes = _quantize_delta(delta, group_size, bits, top_k)
    recon = (affine_ref + dequant).to(torch.float16)
    transfer = payload_bytes + extra_param_bytes + (8 if include_ref_idx else 0)
    recon_explained = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": transfer,
        "metrics": {
            **ref_metrics,
            "recon_energy_explained": recon_explained,
            "residual_coding_gain": recon_explained - ref_metrics["affine_energy_explained"],
            "residual_coding_loss": 1.0 - recon_explained,
            "quantization_loss": ref_metrics["affine_energy_explained"] - recon_explained,
        },
    }


def _encode_direct_int4(
    real_h: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Dict[str, object]:
    dequant, payload_bytes = _quantize_delta(real_h, group_size, 4, top_k)
    recon_explained = _explained_energy(real_h, dequant)
    return {
        "recon": dequant,
        "bytes": payload_bytes,
        "metrics": {
            "raw_ref_cosine": float("nan"),
            "raw_ref_energy_explained": float("nan"),
            "affine_ref_cosine": float("nan"),
            "affine_energy_explained": float("nan"),
            "affine_gain": float("nan"),
            "raw_error_energy": float("nan"),
            "affine_error_energy": float("nan"),
            "recon_energy_explained": recon_explained,
            "residual_coding_gain": float("nan"),
            "residual_coding_loss": 1.0 - recon_explained,
            "quantization_loss": float("nan"),
        },
    }


def _encode_int8(real_h: torch.Tensor, group_size: int, top_k: int) -> Dict[str, object]:
    pkt = groupwise_int8_quantize_topk(real_h, group_size, top_k)
    recon = groupwise_int8_dequantize_topk(pkt)
    transfer = (
        pkt.quantized.nelement() * 1
        + pkt.scales.nelement() * 2
        + pkt.zero_points.nelement() * 2
        + pkt.topk_values.nelement() * 2
        + pkt.topk_indices.nelement() * 1
    )
    recon_explained = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": transfer,
        "metrics": {
            "raw_ref_cosine": float("nan"),
            "raw_ref_energy_explained": float("nan"),
            "affine_ref_cosine": float("nan"),
            "affine_energy_explained": float("nan"),
            "affine_gain": float("nan"),
            "raw_error_energy": float("nan"),
            "affine_error_energy": float("nan"),
            "recon_energy_explained": recon_explained,
            "residual_coding_gain": float("nan"),
            "residual_coding_loss": 1.0 - recon_explained,
            "quantization_loss": float("nan"),
        },
    }


def _encode_sparse_groups(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
    group_size: int,
    bits: int,
    top_k: int,
    threshold_ratio: Optional[float] = None,
    keep_groups: Optional[int] = None,
) -> Dict[str, object]:
    ref_metrics = _reference_metrics(real_h, ref_h)
    scale, bias = compute_affine_params(real_h, ref_h)
    affine_ref = apply_affine(ref_h, scale, bias)
    delta = compute_delta(real_h, affine_ref)

    batch, hidden_dim = delta.shape
    num_groups = hidden_dim // group_size
    grouped = delta.float().reshape(batch, num_groups, group_size)
    energy = grouped.pow(2).sum(dim=-1)
    if threshold_ratio is not None:
        max_energy = energy.max(dim=-1, keepdim=True).values
        mask = energy >= max_energy * threshold_ratio
    elif keep_groups is not None:
        keep = min(keep_groups, num_groups)
        indices = torch.topk(energy, k=keep, dim=-1).indices
        mask = torch.zeros_like(energy, dtype=torch.bool)
        mask.scatter_(1, indices, True)
    else:
        raise ValueError("Need threshold_ratio or keep_groups")

    bitmask_bytes = (num_groups + 7) // 8
    recon_grouped = torch.zeros_like(grouped)
    total_bytes = 0
    active_units = 0

    for batch_idx in range(batch):
        active = mask[batch_idx]
        active_count = int(active.sum().item())
        active_units += active_count
        if active_count == 0:
            total_bytes += bitmask_bytes
            continue

        active_delta = grouped[batch_idx, active].unsqueeze(0)
        dequant_active, payload_bytes = _quantize_delta(active_delta.reshape(1, -1), group_size, bits, top_k)
        recon_grouped[batch_idx, active] = dequant_active.float().reshape(1, active_count, group_size)
        total_bytes += bitmask_bytes + payload_bytes

    recon = (affine_ref + recon_grouped.reshape(batch, hidden_dim).to(torch.float16)).to(torch.float16)
    recon_explained = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": total_bytes + 2 + 2 + 8,
        "metrics": {
            **ref_metrics,
            "recon_energy_explained": recon_explained,
            "residual_coding_gain": recon_explained - ref_metrics["affine_energy_explained"],
            "residual_coding_loss": 1.0 - recon_explained,
            "quantization_loss": ref_metrics["affine_energy_explained"] - recon_explained,
        },
        "active_units": active_units,
    }


def _encode_top_channels(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
    keep_channels: int,
) -> Dict[str, object]:
    ref_metrics = _reference_metrics(real_h, ref_h)
    scale, bias = compute_affine_params(real_h, ref_h)
    affine_ref = apply_affine(ref_h, scale, bias)
    delta = compute_delta(real_h, affine_ref)
    keep = min(keep_channels, delta.shape[-1])
    abs_vals = delta.abs()
    idx = torch.topk(abs_vals, k=keep, dim=-1).indices
    vals = delta.gather(-1, idx)
    sparse_delta = torch.zeros_like(delta)
    sparse_delta.scatter_(1, idx, vals)
    recon = (affine_ref + sparse_delta).to(torch.float16)
    transfer = keep * 2 + keep * 2 + 2 + 2 + 8
    recon_explained = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": transfer,
        "metrics": {
            **ref_metrics,
            "recon_energy_explained": recon_explained,
            "residual_coding_gain": recon_explained - ref_metrics["affine_energy_explained"],
            "residual_coding_loss": 1.0 - recon_explained,
            "quantization_loss": ref_metrics["affine_energy_explained"] - recon_explained,
        },
        "active_units": keep,
    }


def _encode_structured_2of8(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
) -> Dict[str, object]:
    ref_metrics = _reference_metrics(real_h, ref_h)
    scale, bias = compute_affine_params(real_h, ref_h)
    affine_ref = apply_affine(ref_h, scale, bias)
    delta = compute_delta(real_h, affine_ref)
    batch, hidden_dim = delta.shape
    blocks = hidden_dim // 8
    grouped = delta.reshape(batch, blocks, 8)
    idx = torch.topk(grouped.abs(), k=2, dim=-1).indices
    vals = grouped.gather(-1, idx)
    sparse_delta = torch.zeros_like(grouped)
    sparse_delta.scatter_(-1, idx, vals)
    recon = (affine_ref + sparse_delta.reshape(batch, hidden_dim)).to(torch.float16)
    transfer = blocks * (2 * 2 + 2) + 2 + 2 + 8
    recon_explained = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": transfer,
        "metrics": {
            **ref_metrics,
            "recon_energy_explained": recon_explained,
            "residual_coding_gain": recon_explained - ref_metrics["affine_energy_explained"],
            "residual_coding_loss": 1.0 - recon_explained,
            "quantization_loss": ref_metrics["affine_energy_explained"] - recon_explained,
        },
        "active_units": blocks * 2,
    }


def _mean_pool_ref(h: torch.Tensor, pos: int, window: int = 4) -> Optional[torch.Tensor]:
    start = max(0, pos - window)
    if start == pos:
        return None
    return h[start:pos].mean(dim=0, keepdim=True).to(torch.float16)


def _ema_ref(history: List[torch.Tensor], decay: float = 0.7) -> Optional[torch.Tensor]:
    if not history:
        return None
    weights = torch.tensor(
        [decay ** (len(history) - 1 - idx) for idx in range(len(history))],
        dtype=torch.float32,
        device=history[0].device,
    )
    weights = weights / weights.sum()
    stacked = torch.cat(history, dim=0).float()
    return (stacked * weights.unsqueeze(-1)).sum(dim=0, keepdim=True).to(torch.float16)


def _blend_two_refs(
    real_h: torch.Tensor,
    ref1: torch.Tensor,
    ref2: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float], int]:
    feature = torch.stack([
        ref1.float().squeeze(0),
        ref2.float().squeeze(0),
        torch.ones_like(ref1.float().squeeze(0)),
    ], dim=1)
    target = real_h.float().squeeze(0)
    solution = torch.linalg.lstsq(feature, target).solution
    pred = (
        solution[0] * ref1.float()
        + solution[1] * ref2.float()
        + solution[2]
    ).to(torch.float16)
    raw_explained = _explained_energy(real_h, ref1)
    affine_explained = _explained_energy(real_h, pred)
    return pred, {
        "raw_ref_cosine": _cosine(real_h, ref1),
        "raw_ref_energy_explained": raw_explained,
        "affine_ref_cosine": _cosine(real_h, pred),
        "affine_energy_explained": affine_explained,
        "affine_gain": affine_explained - raw_explained,
        "raw_error_energy": _error_energy(real_h, ref1),
        "affine_error_energy": _error_energy(real_h, pred),
    }, 6


class StrategyRunner:
    def __init__(self, hidden_dim: int, group_size: int):
        self.hidden_dim = hidden_dim
        self.group_size = group_size

    def _baseline_delta(self, real_h: torch.Tensor, ref_h: torch.Tensor) -> Dict[str, object]:
        return _encode_delta(real_h, ref_h, self.group_size, 4, 1, True)

    def _baseline_unigram(self, real_h: torch.Tensor) -> Dict[str, object]:
        return _encode_int8(real_h, self.group_size, 1)

    def run_all(
        self,
        real_h: torch.Tensor,
        tier: str,
        table_ref: Optional[torch.Tensor],
        prev_h: Optional[torch.Tensor],
        prev2_h: Optional[torch.Tensor],
        mean_ref: Optional[torch.Tensor],
        ema_ref: Optional[torch.Tensor],
        pred_w: Optional[torch.Tensor],
        pred_b: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}

        if table_ref is not None:
            baseline = self._baseline_delta(real_h, table_ref)
            results["baseline"] = baseline
            results["sparse_010"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 4, 1, threshold_ratio=0.10)
            results["sparse_005"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 4, 1, threshold_ratio=0.05)
            results["int2_topk4"] = _encode_delta(real_h, table_ref, self.group_size, 2, 4, True)
            results["int2_topk8"] = _encode_delta(real_h, table_ref, self.group_size, 2, 8, True)
            results["adaptive_bw"] = _encode_delta(real_h, table_ref, self.group_size, 2 if tier == "trigram" else 4, 4 if tier == "trigram" else 1, True)
            results["combined"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 2 if tier == "trigram" else 4, 4 if tier == "trigram" else 1, threshold_ratio=0.10)
            results["best_sparse"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 4, 1, threshold_ratio=0.05)
            results["delta_sparse_thr_20"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 4, 1, threshold_ratio=0.20)
            results["delta_sparse_thr_10"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 4, 1, threshold_ratio=0.10)
            results["delta_sparse_thr_05"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 4, 1, threshold_ratio=0.05)
            results["delta_top16_groups"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 4, 1, keep_groups=16)
            results["delta_top8_groups"] = _encode_sparse_groups(real_h, table_ref, self.group_size, 4, 1, keep_groups=8)
            results["delta_top1024_ch"] = _encode_top_channels(real_h, table_ref, 1024)
            results["delta_top512_ch"] = _encode_top_channels(real_h, table_ref, 512)
            results["delta_struct2of8"] = _encode_structured_2of8(real_h, table_ref)
            results["delta_int2_top8"] = _encode_delta(real_h, table_ref, self.group_size, 2, 8, True)

            if pred_w is not None and pred_b is not None:
                pred_ref = (pred_w.unsqueeze(0) * table_ref + pred_b.unsqueeze(0)).to(torch.float16)
                ref_metrics = {
                    "raw_ref_cosine": _cosine(real_h, table_ref),
                    "raw_ref_energy_explained": _explained_energy(real_h, table_ref),
                    "affine_ref_cosine": _cosine(real_h, pred_ref),
                    "affine_energy_explained": _explained_energy(real_h, pred_ref),
                    "affine_gain": _explained_energy(real_h, pred_ref) - _explained_energy(real_h, table_ref),
                    "raw_error_energy": _error_energy(real_h, table_ref),
                    "affine_error_energy": _error_energy(real_h, pred_ref),
                }
                delta = compute_delta(real_h, pred_ref)
                dequant, payload = _quantize_delta(delta, self.group_size, 4, 1)
                recon = (pred_ref + dequant).to(torch.float16)
                results["linear_pred"] = {
                    "recon": recon,
                    "bytes": payload + 8,
                    "metrics": {
                        **ref_metrics,
                        "recon_energy_explained": _explained_energy(real_h, recon),
                        "residual_coding_gain": _explained_energy(real_h, recon) - ref_metrics["affine_energy_explained"],
                        "residual_coding_loss": 1.0 - _explained_energy(real_h, recon),
                        "quantization_loss": ref_metrics["affine_energy_explained"] - _explained_energy(real_h, recon),
                    },
                }
            else:
                results["linear_pred"] = baseline

            ref_metrics = _reference_metrics(real_h, table_ref)
            results["ablate_delta_raw_ref"] = {
                "recon": table_ref,
                "bytes": 8,
                "metrics": {
                    **ref_metrics,
                    "recon_energy_explained": ref_metrics["raw_ref_energy_explained"],
                    "residual_coding_gain": ref_metrics["raw_ref_energy_explained"] - ref_metrics["affine_energy_explained"],
                    "residual_coding_loss": 1.0 - ref_metrics["raw_ref_energy_explained"],
                    "quantization_loss": ref_metrics["affine_gain"],
                },
            }
            affine_ref = apply_affine(table_ref, *compute_affine_params(real_h, table_ref))
            results["ablate_delta_affine_only"] = {
                "recon": affine_ref,
                "bytes": 2 + 2 + 8,
                "metrics": {
                    **ref_metrics,
                    "recon_energy_explained": ref_metrics["affine_energy_explained"],
                    "residual_coding_gain": 0.0,
                    "residual_coding_loss": 1.0 - ref_metrics["affine_energy_explained"],
                    "quantization_loss": 0.0,
                },
            }
            results["ablate_delta_affine_int4_k0"] = _encode_delta(real_h, table_ref, self.group_size, 4, 0, True)
            results["ablate_delta_affine_int4_k1"] = baseline
            for name in ALL_STRATEGY_NAMES:
                if name not in results:
                    results[name] = baseline
            return results

        baseline = self._baseline_unigram(real_h)
        results["baseline"] = baseline
        results["unigram_int4_k4"] = _encode_direct_int4(real_h, self.group_size, 4)
        results["unigram_int4_k8"] = _encode_direct_int4(real_h, self.group_size, 8)

        if prev_h is not None:
            results["prev_int4_k1"] = _encode_delta(real_h, prev_h, self.group_size, 4, 1, False)
            results["prev_int4_k2"] = _encode_delta(real_h, prev_h, self.group_size, 4, 2, False)
            results["prev_int4_k4"] = _encode_delta(real_h, prev_h, self.group_size, 4, 4, False)
            results["prev_int2_k4"] = _encode_delta(real_h, prev_h, self.group_size, 2, 4, False)
            results["prev_int2_k8"] = _encode_delta(real_h, prev_h, self.group_size, 2, 8, False)
            results["sparse_prev"] = results["prev_int4_k1"]
            results["best_combined"] = results["prev_int4_k2"]
            results["hybrid_sparse_prev"] = results["prev_int4_k2"]
            results["prev_gs256_k2"] = _encode_delta(real_h, prev_h, 256, 4, 2, False)

            prev_metrics = _reference_metrics(real_h, prev_h)
            results["ablate_prev_raw_ref"] = {
                "recon": prev_h,
                "bytes": 0,
                "metrics": {
                    **prev_metrics,
                    "recon_energy_explained": prev_metrics["raw_ref_energy_explained"],
                    "residual_coding_gain": prev_metrics["raw_ref_energy_explained"] - prev_metrics["affine_energy_explained"],
                    "residual_coding_loss": 1.0 - prev_metrics["raw_ref_energy_explained"],
                    "quantization_loss": prev_metrics["affine_gain"],
                },
            }
            affine_prev = apply_affine(prev_h, *compute_affine_params(real_h, prev_h))
            results["ablate_prev_affine_only"] = {
                "recon": affine_prev,
                "bytes": 2 + 2,
                "metrics": {
                    **prev_metrics,
                    "recon_energy_explained": prev_metrics["affine_energy_explained"],
                    "residual_coding_gain": 0.0,
                    "residual_coding_loss": 1.0 - prev_metrics["affine_energy_explained"],
                    "quantization_loss": 0.0,
                },
            }
            results["ablate_prev_affine_int4_k0"] = _encode_delta(real_h, prev_h, self.group_size, 4, 0, False)
            results["ablate_prev_affine_int4_k1"] = results["prev_int4_k1"]
            results["ablate_prev_affine_int4_k2"] = results["prev_int4_k2"]
        else:
            for name in [
                "prev_int4_k1",
                "prev_int4_k2",
                "prev_int4_k4",
                "prev_int2_k4",
                "prev_int2_k8",
                "sparse_prev",
                "best_combined",
                "hybrid_sparse_prev",
                "prev_gs256_k2",
                "ablate_prev_raw_ref",
                "ablate_prev_affine_only",
                "ablate_prev_affine_int4_k0",
                "ablate_prev_affine_int4_k1",
                "ablate_prev_affine_int4_k2",
            ]:
                results[name] = baseline

        if mean_ref is not None:
            results["mean_pool"] = _encode_delta(real_h, mean_ref, self.group_size, 4, 2, False)
        else:
            results["mean_pool"] = baseline

        if ema_ref is not None:
            results["ema_prev4_int4_k2"] = _encode_delta(real_h, ema_ref, self.group_size, 4, 2, False)
        else:
            results["ema_prev4_int4_k2"] = baseline

        if prev_h is not None and prev2_h is not None:
            blend_ref, blend_metrics, blend_bytes = _blend_two_refs(real_h, prev_h, prev2_h)
            delta = compute_delta(real_h, blend_ref)
            dequant, payload_bytes = _quantize_delta(delta, self.group_size, 4, 2)
            recon = (blend_ref + dequant).to(torch.float16)
            results["prev2_blend_int4_k2"] = {
                "recon": recon,
                "bytes": payload_bytes + blend_bytes,
                "metrics": {
                    **blend_metrics,
                    "recon_energy_explained": _explained_energy(real_h, recon),
                    "residual_coding_gain": _explained_energy(real_h, recon) - blend_metrics["affine_energy_explained"],
                    "residual_coding_loss": 1.0 - _explained_energy(real_h, recon),
                    "quantization_loss": blend_metrics["affine_energy_explained"] - _explained_energy(real_h, recon),
                },
            }
        else:
            results["prev2_blend_int4_k2"] = baseline

        for name in ALL_STRATEGY_NAMES:
            if name not in results:
                results[name] = baseline
        return results


def run_experiment(args):
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    logger.info("Loading model on GPU %d...", args.gpu)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from delta_coding_system.run_experiment import load_dataset_texts

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    global HIDDEN_DIM
    HIDDEN_DIM = model.config.hidden_size
    logger.info("Model loaded. hidden_dim=%d", HIDDEN_DIM)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = StrategyRunner(HIDDEN_DIM, GROUP_SIZE)

    request_records = []
    position_records = []

    for ds_name in args.datasets:
        logger.info("%s", "=" * 60)
        logger.info("Dataset: %s", ds_name)
        texts = load_dataset_texts(ds_name)
        rng = random.Random(args.seed)
        rng.shuffle(texts)
        total_needed = args.warmup_requests + args.test_requests
        while len(texts) < total_needed:
            texts.extend(texts[: total_needed - len(texts)])

        table = NgramTable(device=device, dtype=torch.float16, max_entries=100000)

        logger.info("Warmup: %d requests...", args.warmup_requests)
        for warm_idx in range(args.warmup_requests):
            input_ids = tokenizer(
                texts[warm_idx],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_len,
            ).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            if token_ids.shape[0] >= 3:
                trigrams = token_ids.unfold(0, 3, 1)
                for tri_idx in range(trigrams.shape[0]):
                    tri = trigrams[tri_idx]
                    table._get_or_create_node(
                        tri[0].item(),
                        tri[1].item(),
                        h[tri_idx + 1].unsqueeze(0).to(torch.float16),
                    )
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[tri_idx + 2].unsqueeze(0).to(torch.float16)
            del out, h

        logger.info("Warmup done. Table: %d bi", table.stats["num_bigrams"])

        train_pairs = []
        logger.info("Test: %d requests...", args.test_requests)
        for req_idx in range(args.test_requests):
            text = texts[args.warmup_requests + req_idx]
            input_ids = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_len,
            ).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)

            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            seq_len = token_ids.shape[0]
            raw_bytes = seq_len * HIDDEN_DIM * 2

            tiers: List[str] = []
            refs: List[Optional[torch.Tensor]] = []
            for pos in range(seq_len):
                found = False
                if pos >= 2:
                    a, b, c = token_ids[pos - 2].item(), token_ids[pos - 1].item(), token_ids[pos].item()
                    tri_ref = table.get_trigram(a, b, c)
                    if tri_ref is not None:
                        tiers.append("trigram")
                        refs.append(tri_ref.to(torch.float16).to(device))
                        found = True
                if not found and pos >= 1:
                    b, c = token_ids[pos - 1].item(), token_ids[pos].item()
                    bi_ref = table.get_bigram(b, c)
                    if bi_ref is not None:
                        tiers.append("bigram")
                        refs.append(bi_ref.to(torch.float16).to(device))
                        found = True
                if not found:
                    tiers.append("unigram")
                    refs.append(None)

            if req_idx < 5:
                for pos, ref in enumerate(refs):
                    if ref is not None:
                        train_pairs.append((h[pos:pos + 1].to(torch.float16), ref))

            if req_idx == 5 and train_pairs:
                logger.info("  Training linear predictor with %d pairs...", len(train_pairs))
                pred_w, pred_b = train_linear_predictor(train_pairs, HIDDEN_DIM, device)
            elif req_idx < 5:
                pred_w, pred_b = None, None

            strategy_accum = defaultdict(lambda: {
                "bytes": 0,
                "cos_sum": 0.0,
                "cos_min": 1.0,
                "n": 0,
                "raw_ref_cosine_sum": 0.0,
                "raw_ref_cosine_n": 0,
                "raw_ref_energy_sum": 0.0,
                "raw_ref_energy_n": 0,
                "affine_ref_cosine_sum": 0.0,
                "affine_ref_cosine_n": 0,
                "affine_energy_sum": 0.0,
                "affine_energy_n": 0,
                "affine_gain_sum": 0.0,
                "affine_gain_n": 0,
                "recon_energy_sum": 0.0,
                "recon_energy_n": 0,
                "quantization_loss_sum": 0.0,
                "quantization_loss_n": 0,
                "residual_coding_gain_sum": 0.0,
                "residual_coding_gain_n": 0,
                "residual_coding_loss_sum": 0.0,
                "residual_coding_loss_n": 0,
            })

            for pos in range(seq_len):
                real = h[pos:pos + 1].to(torch.float16)
                table_ref = refs[pos]
                prev_h = h[pos - 1:pos].to(torch.float16) if pos > 0 else None
                prev2_h = h[pos - 2:pos - 1].to(torch.float16) if pos > 1 else None
                mean_ref = _mean_pool_ref(h, pos, 4)
                history = [h[idx:idx + 1].to(torch.float16) for idx in range(max(0, pos - 4), pos)]
                ema_ref = _ema_ref(history)
                results = runner.run_all(real, tiers[pos], table_ref, prev_h, prev2_h, mean_ref, ema_ref, pred_w, pred_b)

                for strategy_name, result in results.items():
                    recon = result["recon"]
                    transfer = int(result["bytes"])
                    metrics = result["metrics"]
                    cosine = _cosine(real, recon)
                    acc = strategy_accum[strategy_name]
                    acc["bytes"] += transfer
                    acc["cos_sum"] += cosine
                    acc["cos_min"] = min(acc["cos_min"], cosine)
                    acc["n"] += 1

                    metric_map = [
                        ("raw_ref_cosine", "raw_ref_cosine_sum", "raw_ref_cosine_n"),
                        ("raw_ref_energy_explained", "raw_ref_energy_sum", "raw_ref_energy_n"),
                        ("affine_ref_cosine", "affine_ref_cosine_sum", "affine_ref_cosine_n"),
                        ("affine_energy_explained", "affine_energy_sum", "affine_energy_n"),
                        ("affine_gain", "affine_gain_sum", "affine_gain_n"),
                        ("recon_energy_explained", "recon_energy_sum", "recon_energy_n"),
                        ("quantization_loss", "quantization_loss_sum", "quantization_loss_n"),
                        ("residual_coding_gain", "residual_coding_gain_sum", "residual_coding_gain_n"),
                        ("residual_coding_loss", "residual_coding_loss_sum", "residual_coding_loss_n"),
                    ]
                    for metric_name, sum_name, count_name in metric_map:
                        value = metrics[metric_name]
                        if value == value:
                            acc[sum_name] += value
                            acc[count_name] += 1

                    if req_idx < args.detail_requests:
                        position_records.append({
                            "dataset": ds_name,
                            "request_index": req_idx,
                            "position": pos,
                            "tier": tiers[pos],
                            "strategy": strategy_name,
                            "cosine": cosine,
                            "transfer_bytes": transfer,
                            **metrics,
                        })

            n_trigram = sum(1 for tier in tiers if tier == "trigram")
            n_bigram = sum(1 for tier in tiers if tier == "bigram")
            n_unigram = sum(1 for tier in tiers if tier == "unigram")
            for strategy_name, acc in strategy_accum.items():
                request_records.append({
                    "dataset": ds_name,
                    "request_index": req_idx,
                    "strategy": strategy_name,
                    "seq_len": seq_len,
                    "raw_fp16_bytes": raw_bytes,
                    "total_transfer_bytes": acc["bytes"],
                    "compression_ratio": raw_bytes / max(acc["bytes"], 1),
                    "cosine_mean": acc["cos_sum"] / max(acc["n"], 1),
                    "cosine_min": acc["cos_min"],
                    "raw_ref_cosine_mean": acc["raw_ref_cosine_sum"] / max(acc["raw_ref_cosine_n"], 1),
                    "raw_ref_energy_explained_mean": acc["raw_ref_energy_sum"] / max(acc["raw_ref_energy_n"], 1),
                    "affine_ref_cosine_mean": acc["affine_ref_cosine_sum"] / max(acc["affine_ref_cosine_n"], 1),
                    "affine_energy_explained_mean": acc["affine_energy_sum"] / max(acc["affine_energy_n"], 1),
                    "affine_gain_mean": acc["affine_gain_sum"] / max(acc["affine_gain_n"], 1),
                    "recon_energy_explained_mean": acc["recon_energy_sum"] / max(acc["recon_energy_n"], 1),
                    "quantization_loss_mean": acc["quantization_loss_sum"] / max(acc["quantization_loss_n"], 1),
                    "residual_coding_gain_mean": acc["residual_coding_gain_sum"] / max(acc["residual_coding_gain_n"], 1),
                    "residual_coding_loss_mean": acc["residual_coding_loss_sum"] / max(acc["residual_coding_loss_n"], 1),
                    "num_trigram": n_trigram,
                    "num_bigram": n_bigram,
                    "num_unigram": n_unigram,
                    "pct_trigram": n_trigram / max(seq_len, 1) * 100,
                    "pct_bigram": n_bigram / max(seq_len, 1) * 100,
                    "pct_unigram": n_unigram / max(seq_len, 1) * 100,
                })

            if token_ids.shape[0] >= 3:
                trigrams = token_ids.unfold(0, 3, 1)
                for tri_idx in range(trigrams.shape[0]):
                    tri = trigrams[tri_idx]
                    table._get_or_create_node(
                        tri[0].item(),
                        tri[1].item(),
                        h[tri_idx + 1].unsqueeze(0).to(torch.float16),
                    )
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[tri_idx + 2].unsqueeze(0).to(torch.float16)

            del out, h
            if (req_idx + 1) % 5 == 0:
                logger.info("  Test %d/%d complete", req_idx + 1, args.test_requests)

        gc.collect()
        torch.cuda.empty_cache()

    pq.write_table(pa.Table.from_pylist(request_records), str(output_dir / "strategies_full.parquet"))
    if position_records:
        pq.write_table(pa.Table.from_pylist(position_records), str(output_dir / "position_detail_full.parquet"))
    logger.info("Saved %d request records", len(request_records))

    import pandas as pd

    df = pd.DataFrame(request_records)
    summary = df.groupby("strategy").agg(
        cr=("compression_ratio", "mean"),
        cos=("cosine_mean", "mean"),
        raw_e=("raw_ref_energy_explained_mean", "mean"),
        affine_e=("affine_energy_explained_mean", "mean"),
        recon_e=("recon_energy_explained_mean", "mean"),
        residual_gain=("residual_coding_gain_mean", "mean"),
        residual_loss=("residual_coding_loss_mean", "mean"),
        affine_gain=("affine_gain_mean", "mean"),
        q_loss=("quantization_loss_mean", "mean"),
    ).sort_values("cr", ascending=False)

    print("\n" + "=" * 110)
    print("FULL COMPRESSION RERUN WITH ABLATIONS")
    print("=" * 110)
    print(f"\n{'Strategy':<28s} {'Ratio':>8s} {'Cos':>9s} {'RawE':>9s} {'AffineE':>9s} {'ReconE':>9s} {'ResGain':>9s} {'ResLoss':>9s}")
    print("-" * 110)
    for name, row in summary.iterrows():
        print(
            f"{name:<28s} {row['cr']:>7.3f}x {row['cos']:>9.6f} {row['raw_e']:>9.6f} "
            f"{row['affine_e']:>9.6f} {row['recon_e']:>9.6f} {row['residual_gain']:>9.6f} {row['residual_loss']:>9.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Unified compression rerun with ablations")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=30)
    parser.add_argument("--detail-requests", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_compression_full_ablation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext2", "sharegpt", "gsm8k", "cnn_dm", "alpaca", "triviaqa"],
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()