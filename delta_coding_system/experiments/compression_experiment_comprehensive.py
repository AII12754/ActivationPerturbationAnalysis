#!/usr/bin/env python3
"""Comprehensive compression evaluation with drift, ablations, and latency.

This script unifies four evaluation dimensions:
1. Compression ratio and packet entropy estimates.
2. Hidden-state reconstruction quality.
3. Downstream logit drift after replaying remaining layers.
4. Codec-side latency breakdown for decode-time strategies.

It deliberately extends the earlier strategy-drift script to cover:
- More delta-path variants and ablations.
- Delta without affine for decode-latency reduction.
- More unigram alternatives.
- Tier-level detail records for representative requests.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
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
)
from delta_coding_system.table import NgramTable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compress_comprehensive")

GROUP_SIZE = 128

DELTA_STRATEGIES = [
    "baseline_current",
    "delta_int2_k8_out8",
    "delta_int2_k8_out4",
    "delta_int2_k8_out8_entropy",
    "delta_noaffine_int4_k1",
    "delta_noaffine_int2_k8_out4",
    "delta_sparse_thr10",
    "delta_top1024_ch",
    "delta_struct2of8",
    "ablate_delta_raw_ref",
    "ablate_delta_affine_only",
]

UNIGRAM_STRATEGIES = [
    "baseline_current",
    "unigram_int4_k4",
    "prev_int4_k2",
    "prev_gs256_k2",
    "prev_int2_k8_out4",
    "zero_affine_int4_k2",
    "global_mean_int4_k2",
    "mean_pool_int4_k2",
    "ema_prev4_int4_k2",
    "prev2_blend_int4_k2",
    "ablate_prev_raw_ref",
    "ablate_prev_affine_only",
]

ALL_STRATEGIES = [
    "baseline_current",
    "delta_int2_k8_out8",
    "delta_int2_k8_out4",
    "delta_int2_k8_out8_entropy",
    "delta_noaffine_int4_k1",
    "delta_noaffine_int2_k8_out4",
    "delta_sparse_thr10",
    "delta_top1024_ch",
    "delta_struct2of8",
    "unigram_int4_k4",
    "prev_int4_k2",
    "prev_gs256_k2",
    "prev_int2_k8_out4",
    "zero_affine_int4_k2",
    "global_mean_int4_k2",
    "mean_pool_int4_k2",
    "ema_prev4_int4_k2",
    "prev2_blend_int4_k2",
    "ablate_delta_raw_ref",
    "ablate_delta_affine_only",
    "ablate_prev_raw_ref",
    "ablate_prev_affine_only",
]


def _serialize_tensor(tensor: Optional[torch.Tensor]) -> bytes:
    if tensor is None:
        return b""
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).item()


def _explained_energy(target: torch.Tensor, approx: torch.Tensor) -> float:
    denom = max(float(target.float().pow(2).sum().item()), 1e-8)
    err = float((target.float() - approx.float()).pow(2).sum().item())
    return 1.0 - err / denom


def _entropy_bytes(payload: bytes) -> Tuple[float, int, float]:
    if not payload:
        return 0.0, 0, 0.0
    counts = Counter(payload)
    total = len(payload)
    entropy_bpb = 0.0
    for count in counts.values():
        p = count / total
        entropy_bpb -= p * math.log2(p)
    ideal_bytes = total * entropy_bpb / 8.0
    zlib_bytes = len(zlib.compress(payload, level=9))
    return entropy_bpb, zlib_bytes, ideal_bytes


def _init_timing() -> Dict[str, float]:
    return {
        "affine_param_ms": 0.0,
        "affine_apply_ms": 0.0,
        "delta_ms": 0.0,
        "pack_ms": 0.0,
        "outlier_ms": 0.0,
        "decode_ms": 0.0,
        "reconstruct_ms": 0.0,
        "entropy_ms": 0.0,
        "codec_total_ms": 0.0,
    }


class _GpuTimeline:
    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = device.type == "cuda"
        self.events: List[torch.cuda.Event] = []

    def mark(self):
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append(event)
        return event

    def sync(self):
        if self.enabled:
            torch.cuda.synchronize(self.device)

    @staticmethod
    def elapsed(start, end) -> float:
        if start is None or end is None:
            return 0.0
        return float(start.elapsed_time(end))


def _quantize_outliers(values: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int, bytes]:
    if values.numel() == 0:
        return values.to(torch.float16), 0, b""
    if bits >= 16:
        qv = values.to(torch.float16)
        return qv, qv.nelement() * 2, _serialize_tensor(qv)

    batch, groups, k = values.shape
    vals = values.float()
    v_min = vals.min(dim=-1).values
    v_max = vals.max(dim=-1).values
    qmax = 255.0 if bits == 8 else 15.0
    scales = ((v_max - v_min) / qmax).to(torch.float16)
    zeros = v_min.to(torch.float16)
    q = torch.clamp(
        torch.round((vals - zeros.float().unsqueeze(-1)) / (scales.float().unsqueeze(-1) + 1e-10)),
        0,
        int(qmax),
    ).to(torch.uint8)
    if bits == 8:
        packed = q
        data_bytes = q.nelement()
        q_restore = packed.float()
    else:
        if k % 2 == 1:
            q = torch.cat([q, torch.zeros(batch, groups, 1, dtype=torch.uint8, device=q.device)], dim=-1)
        packed = (q[..., 0::2] << 4) | q[..., 1::2]
        data_bytes = packed.nelement()
        hi = (packed >> 4).to(torch.uint8)
        lo = (packed & 0x0F).to(torch.uint8)
        q_restore = torch.stack([hi, lo], dim=-1).reshape(batch, groups, -1)[..., :k].float()
    dequant = q_restore * scales.float().unsqueeze(-1) + zeros.float().unsqueeze(-1)
    payload = b"".join([_serialize_tensor(packed), _serialize_tensor(scales), _serialize_tensor(zeros)])
    return dequant.to(torch.float16), data_bytes + scales.nelement() * 2 + zeros.nelement() * 2, payload


def _decode_delta_with_custom_outliers(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    group_size: int,
    hidden_dim: int,
    bits: int,
) -> torch.Tensor:
    if bits == 4:
        return groupwise_int4_dequantize_topk(packed, scales, zeros, topk_values, topk_indices, group_size, hidden_dim)
    if bits == 2:
        return groupwise_int2_dequantize_topk(packed, scales, zeros, topk_values, topk_indices, group_size, hidden_dim)
    raise ValueError(bits)


def _reference_metrics(real_h: torch.Tensor, ref_h: Optional[torch.Tensor], affine_ref: Optional[torch.Tensor]) -> Dict[str, float]:
    if ref_h is None:
        return {
            "raw_ref_cosine": float("nan"),
            "raw_ref_energy_explained": float("nan"),
            "affine_ref_cosine": float("nan"),
            "affine_energy_explained": float("nan"),
            "affine_gain": float("nan"),
        }
    raw_e = _explained_energy(real_h, ref_h)
    if affine_ref is None:
        affine_ref = ref_h
    aff_e = _explained_energy(real_h, affine_ref)
    return {
        "raw_ref_cosine": _cosine(real_h, ref_h),
        "raw_ref_energy_explained": raw_e,
        "affine_ref_cosine": _cosine(real_h, affine_ref),
        "affine_energy_explained": aff_e,
        "affine_gain": aff_e - raw_e,
    }


def _encode_int8_unigram(real_h: torch.Tensor, top_k: int = 1, group_size: int = GROUP_SIZE) -> Dict[str, object]:
    timing = _init_timing()
    timeline = _GpuTimeline(real_h.device)
    timeline.sync()
    e0 = timeline.mark()
    pkt = groupwise_int8_quantize_topk(real_h, group_size, top_k)
    e1 = timeline.mark()
    recon = groupwise_int8_dequantize_topk(pkt)
    e2 = timeline.mark()
    timeline.sync()
    timing["pack_ms"] = timeline.elapsed(e0, e1)
    timing["decode_ms"] = timeline.elapsed(e1, e2)
    timing["codec_total_ms"] = timing["pack_ms"] + timing["decode_ms"]

    bytes_ = (
        pkt.quantized.nelement()
        + pkt.scales.nelement() * 2
        + pkt.zero_points.nelement() * 2
        + pkt.topk_values.nelement() * 2
        + pkt.topk_indices.nelement()
    )
    recon_energy = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": bytes_,
        "raw_bytes": bytes_,
        "zlib_bytes": bytes_,
        "ideal_entropy_bytes": float(bytes_),
        "entropy_bpb": float("nan"),
        "timing": timing,
        "metrics": {
            **_reference_metrics(real_h, None, None),
            "recon_energy_explained": recon_energy,
            "residual_coding_loss": 1.0 - recon_energy,
        },
    }


def _quantize_direct_int4(real_h: torch.Tensor, top_k: int, group_size: int = GROUP_SIZE) -> Dict[str, object]:
    timing = _init_timing()
    timeline = _GpuTimeline(real_h.device)
    timeline.sync()
    e0 = timeline.mark()
    # The Int4 helper zeroes out top-k values in-place, so quantize a clone.
    packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(real_h.clone(), group_size, top_k)
    e1 = timeline.mark()
    recon = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, real_h.shape[-1])
    e2 = timeline.mark()
    timeline.sync()
    timing["pack_ms"] = timeline.elapsed(e0, e1)
    timing["decode_ms"] = timeline.elapsed(e1, e2)
    timing["codec_total_ms"] = timing["pack_ms"] + timing["decode_ms"]
    payload = b"".join([
        _serialize_tensor(packed),
        _serialize_tensor(scales),
        _serialize_tensor(zeros),
        _serialize_tensor(tv),
        _serialize_tensor(ti),
    ])
    entropy_bpb, zlib_bytes, ideal_bytes = _entropy_bytes(payload)
    bytes_ = packed.nelement() + scales.nelement() * 2 + zeros.nelement() * 2 + tv.nelement() * 2 + ti.nelement()
    recon_energy = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": bytes_,
        "raw_bytes": bytes_,
        "zlib_bytes": zlib_bytes,
        "ideal_entropy_bytes": ideal_bytes,
        "entropy_bpb": entropy_bpb,
        "timing": timing,
        "metrics": {
            **_reference_metrics(real_h, None, None),
            "recon_energy_explained": recon_energy,
            "residual_coding_loss": 1.0 - recon_energy,
        },
    }


def _encode_delta_quantized(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
    bits: int,
    top_k: int,
    outlier_bits: int = 16,
    include_ref_idx: bool = True,
    group_size: int = GROUP_SIZE,
    entropy_override: bool = False,
    use_affine: bool = True,
) -> Dict[str, object]:
    timing = _init_timing()
    timeline = _GpuTimeline(real_h.device)
    timeline.sync()
    e0 = timeline.mark()
    if use_affine:
        scale, bias = compute_affine_params(real_h, ref_h)
    else:
        scale = torch.ones(real_h.shape[0], dtype=torch.float16, device=real_h.device)
        bias = torch.zeros(real_h.shape[0], dtype=torch.float16, device=real_h.device)
    e1 = timeline.mark()
    affine_ref = apply_affine(ref_h, scale, bias) if use_affine else ref_h
    e2 = timeline.mark()
    delta = compute_delta(real_h, affine_ref)
    e3 = timeline.mark()
    hidden_dim = delta.shape[-1]
    if bits == 4:
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)
    elif bits == 2:
        packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(delta, group_size, top_k)
    else:
        raise ValueError(bits)
    e4 = timeline.mark()
    q_outliers, outlier_bytes, outlier_payload = _quantize_outliers(tv, outlier_bits)
    e5 = timeline.mark()
    dequant = _decode_delta_with_custom_outliers(packed, scales, zeros, q_outliers, ti, group_size, hidden_dim, bits)
    e6 = timeline.mark()
    recon = (affine_ref + dequant).to(torch.float16)
    e7 = timeline.mark()
    timeline.sync()

    timing["affine_param_ms"] = timeline.elapsed(e0, e1)
    timing["affine_apply_ms"] = timeline.elapsed(e1, e2)
    timing["delta_ms"] = timeline.elapsed(e2, e3)
    timing["pack_ms"] = timeline.elapsed(e3, e4)
    timing["outlier_ms"] = timeline.elapsed(e4, e5)
    timing["decode_ms"] = timeline.elapsed(e5, e6)
    timing["reconstruct_ms"] = timeline.elapsed(e6, e7)

    t_cpu0 = time.perf_counter()
    payload = b"".join([
        _serialize_tensor(packed),
        _serialize_tensor(scales),
        _serialize_tensor(zeros),
        outlier_payload,
        _serialize_tensor(ti),
        _serialize_tensor(scale.to(torch.float16)) if use_affine else b"",
        _serialize_tensor(bias.to(torch.float16)) if use_affine else b"",
    ])
    entropy_bpb, zlib_bytes, ideal_bytes = _entropy_bytes(payload)
    timing["entropy_ms"] = (time.perf_counter() - t_cpu0) * 1000.0
    timing["codec_total_ms"] = sum(v for k, v in timing.items() if k != "codec_total_ms")

    raw_bytes = (
        packed.nelement()
        + scales.nelement() * 2
        + zeros.nelement() * 2
        + outlier_bytes
        + ti.nelement()
        + (4 if use_affine else 0)
        + (8 if include_ref_idx else 0)
    )
    effective_bytes = zlib_bytes if entropy_override else raw_bytes
    recon_energy = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": effective_bytes,
        "raw_bytes": raw_bytes,
        "zlib_bytes": zlib_bytes,
        "ideal_entropy_bytes": ideal_bytes,
        "entropy_bpb": entropy_bpb,
        "timing": timing,
        "metrics": {
            **_reference_metrics(real_h, ref_h, affine_ref),
            "recon_energy_explained": recon_energy,
            "residual_coding_loss": 1.0 - recon_energy,
        },
    }


def _encode_sparse_groups(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
    bits: int,
    top_k: int,
    threshold_ratio: float,
    group_size: int = GROUP_SIZE,
) -> Dict[str, object]:
    timing = _init_timing()
    timeline = _GpuTimeline(real_h.device)
    timeline.sync()
    e0 = timeline.mark()
    scale, bias = compute_affine_params(real_h, ref_h)
    e1 = timeline.mark()
    affine_ref = apply_affine(ref_h, scale, bias)
    e2 = timeline.mark()
    delta = compute_delta(real_h, affine_ref)
    e3 = timeline.mark()
    batch, hidden_dim = delta.shape
    num_groups = hidden_dim // group_size
    grouped = delta.float().reshape(batch, num_groups, group_size)
    energy = grouped.pow(2).sum(dim=-1)
    max_energy = energy.max(dim=-1, keepdim=True).values
    mask = energy >= max_energy * threshold_ratio
    recon_grouped = torch.zeros_like(grouped)
    total_bytes = 0
    payload_parts: List[bytes] = []
    bitmask_bytes = (num_groups + 7) // 8
    for batch_idx in range(batch):
        active = mask[batch_idx]
        active_count = int(active.sum().item())
        total_bytes += bitmask_bytes
        if active_count == 0:
            continue
        active_delta = grouped[batch_idx, active].reshape(1, -1)
        if bits == 4:
            packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(active_delta, group_size, top_k)
            dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, active_delta.shape[-1])
        elif bits == 2:
            packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(active_delta, group_size, top_k)
            dequant = groupwise_int2_dequantize_topk(packed, scales, zeros, tv, ti, group_size, active_delta.shape[-1])
        else:
            raise ValueError(bits)
        recon_grouped[batch_idx, active] = dequant.float().reshape(1, active_count, group_size)
        total_bytes += packed.nelement() + scales.nelement() * 2 + zeros.nelement() * 2 + tv.nelement() * 2 + ti.nelement()
        payload_parts.extend([
            _serialize_tensor(packed),
            _serialize_tensor(scales),
            _serialize_tensor(zeros),
            _serialize_tensor(tv),
            _serialize_tensor(ti),
        ])
    e4 = timeline.mark()
    recon = (affine_ref + recon_grouped.reshape(batch, hidden_dim).to(torch.float16)).to(torch.float16)
    e5 = timeline.mark()
    timeline.sync()
    timing["affine_param_ms"] = timeline.elapsed(e0, e1)
    timing["affine_apply_ms"] = timeline.elapsed(e1, e2)
    timing["delta_ms"] = timeline.elapsed(e2, e3)
    timing["pack_ms"] = timeline.elapsed(e3, e4)
    timing["reconstruct_ms"] = timeline.elapsed(e4, e5)
    t_cpu0 = time.perf_counter()
    payload = b"".join(payload_parts + [_serialize_tensor(scale.to(torch.float16)), _serialize_tensor(bias.to(torch.float16))])
    entropy_bpb, zlib_bytes, ideal_bytes = _entropy_bytes(payload)
    timing["entropy_ms"] = (time.perf_counter() - t_cpu0) * 1000.0
    timing["codec_total_ms"] = sum(v for k, v in timing.items() if k != "codec_total_ms")
    total_bytes += 4 + 8
    recon_energy = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": total_bytes,
        "raw_bytes": total_bytes,
        "zlib_bytes": zlib_bytes,
        "ideal_entropy_bytes": ideal_bytes,
        "entropy_bpb": entropy_bpb,
        "timing": timing,
        "metrics": {
            **_reference_metrics(real_h, ref_h, affine_ref),
            "recon_energy_explained": recon_energy,
            "residual_coding_loss": 1.0 - recon_energy,
        },
    }


def _encode_top_channels(real_h: torch.Tensor, ref_h: torch.Tensor, keep_channels: int) -> Dict[str, object]:
    timing = _init_timing()
    timeline = _GpuTimeline(real_h.device)
    timeline.sync()
    e0 = timeline.mark()
    scale, bias = compute_affine_params(real_h, ref_h)
    e1 = timeline.mark()
    affine_ref = apply_affine(ref_h, scale, bias)
    e2 = timeline.mark()
    delta = compute_delta(real_h, affine_ref)
    e3 = timeline.mark()
    keep = min(keep_channels, delta.shape[-1])
    abs_vals = delta.abs()
    idx = torch.topk(abs_vals, k=keep, dim=-1).indices
    vals = delta.gather(-1, idx)
    sparse_delta = torch.zeros_like(delta)
    sparse_delta.scatter_(1, idx, vals)
    e4 = timeline.mark()
    recon = (affine_ref + sparse_delta).to(torch.float16)
    e5 = timeline.mark()
    timeline.sync()
    timing["affine_param_ms"] = timeline.elapsed(e0, e1)
    timing["affine_apply_ms"] = timeline.elapsed(e1, e2)
    timing["delta_ms"] = timeline.elapsed(e2, e3)
    timing["pack_ms"] = timeline.elapsed(e3, e4)
    timing["reconstruct_ms"] = timeline.elapsed(e4, e5)
    payload = b"".join([
        _serialize_tensor(vals.to(torch.float16)),
        _serialize_tensor(idx.to(torch.int16)),
        _serialize_tensor(scale.to(torch.float16)),
        _serialize_tensor(bias.to(torch.float16)),
    ])
    t_cpu0 = time.perf_counter()
    entropy_bpb, zlib_bytes, ideal_bytes = _entropy_bytes(payload)
    timing["entropy_ms"] = (time.perf_counter() - t_cpu0) * 1000.0
    timing["codec_total_ms"] = sum(v for k, v in timing.items() if k != "codec_total_ms")
    transfer = keep * 2 + keep * 2 + 4 + 8
    recon_energy = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": transfer,
        "raw_bytes": transfer,
        "zlib_bytes": zlib_bytes,
        "ideal_entropy_bytes": ideal_bytes,
        "entropy_bpb": entropy_bpb,
        "timing": timing,
        "metrics": {
            **_reference_metrics(real_h, ref_h, affine_ref),
            "recon_energy_explained": recon_energy,
            "residual_coding_loss": 1.0 - recon_energy,
        },
    }


def _encode_structured_2of8(real_h: torch.Tensor, ref_h: torch.Tensor) -> Dict[str, object]:
    timing = _init_timing()
    timeline = _GpuTimeline(real_h.device)
    timeline.sync()
    e0 = timeline.mark()
    scale, bias = compute_affine_params(real_h, ref_h)
    e1 = timeline.mark()
    affine_ref = apply_affine(ref_h, scale, bias)
    e2 = timeline.mark()
    delta = compute_delta(real_h, affine_ref)
    e3 = timeline.mark()
    batch, hidden_dim = delta.shape
    blocks = hidden_dim // 8
    grouped = delta.reshape(batch, blocks, 8)
    idx = torch.topk(grouped.abs(), k=2, dim=-1).indices
    vals = grouped.gather(-1, idx)
    sparse_delta = torch.zeros_like(grouped)
    sparse_delta.scatter_(-1, idx, vals)
    e4 = timeline.mark()
    recon = (affine_ref + sparse_delta.reshape(batch, hidden_dim)).to(torch.float16)
    e5 = timeline.mark()
    timeline.sync()
    timing["affine_param_ms"] = timeline.elapsed(e0, e1)
    timing["affine_apply_ms"] = timeline.elapsed(e1, e2)
    timing["delta_ms"] = timeline.elapsed(e2, e3)
    timing["pack_ms"] = timeline.elapsed(e3, e4)
    timing["reconstruct_ms"] = timeline.elapsed(e4, e5)
    payload = b"".join([
        _serialize_tensor(vals.to(torch.float16)),
        _serialize_tensor(idx.to(torch.uint8)),
        _serialize_tensor(scale.to(torch.float16)),
        _serialize_tensor(bias.to(torch.float16)),
    ])
    t_cpu0 = time.perf_counter()
    entropy_bpb, zlib_bytes, ideal_bytes = _entropy_bytes(payload)
    timing["entropy_ms"] = (time.perf_counter() - t_cpu0) * 1000.0
    timing["codec_total_ms"] = sum(v for k, v in timing.items() if k != "codec_total_ms")
    transfer = blocks * (2 * 2 + 2) + 4 + 8
    recon_energy = _explained_energy(real_h, recon)
    return {
        "recon": recon,
        "bytes": transfer,
        "raw_bytes": transfer,
        "zlib_bytes": zlib_bytes,
        "ideal_entropy_bytes": ideal_bytes,
        "entropy_bpb": entropy_bpb,
        "timing": timing,
        "metrics": {
            **_reference_metrics(real_h, ref_h, affine_ref),
            "recon_energy_explained": recon_energy,
            "residual_coding_loss": 1.0 - recon_energy,
        },
    }


def _encode_raw_reference(real_h: torch.Tensor, ref_h: torch.Tensor, include_ref_idx: bool) -> Dict[str, object]:
    recon = ref_h.to(torch.float16)
    metrics = _reference_metrics(real_h, ref_h, ref_h)
    timing = _init_timing()
    timing["codec_total_ms"] = 0.0
    recon_energy = _explained_energy(real_h, recon)
    transfer = 8 if include_ref_idx else 0
    return {
        "recon": recon,
        "bytes": transfer,
        "raw_bytes": transfer,
        "zlib_bytes": transfer,
        "ideal_entropy_bytes": float(transfer),
        "entropy_bpb": 0.0,
        "timing": timing,
        "metrics": {
            **metrics,
            "recon_energy_explained": recon_energy,
            "residual_coding_loss": 1.0 - recon_energy,
        },
    }


def _encode_affine_only(real_h: torch.Tensor, ref_h: torch.Tensor, include_ref_idx: bool) -> Dict[str, object]:
    timing = _init_timing()
    timeline = _GpuTimeline(real_h.device)
    timeline.sync()
    e0 = timeline.mark()
    scale, bias = compute_affine_params(real_h, ref_h)
    e1 = timeline.mark()
    recon = apply_affine(ref_h, scale, bias).to(torch.float16)
    e2 = timeline.mark()
    timeline.sync()
    timing["affine_param_ms"] = timeline.elapsed(e0, e1)
    timing["affine_apply_ms"] = timeline.elapsed(e1, e2)
    timing["codec_total_ms"] = timing["affine_param_ms"] + timing["affine_apply_ms"]
    metrics = _reference_metrics(real_h, ref_h, recon)
    recon_energy = _explained_energy(real_h, recon)
    transfer = 4 + (8 if include_ref_idx else 0)
    return {
        "recon": recon,
        "bytes": transfer,
        "raw_bytes": transfer,
        "zlib_bytes": transfer,
        "ideal_entropy_bytes": float(transfer),
        "entropy_bpb": 0.0,
        "timing": timing,
        "metrics": {
            **metrics,
            "recon_energy_explained": recon_energy,
            "residual_coding_loss": 1.0 - recon_energy,
        },
    }


def _mean_pool_ref(history: List[torch.Tensor], window: int = 4) -> Optional[torch.Tensor]:
    if not history:
        return None
    selected = history[-window:]
    return torch.cat(selected, dim=0).mean(dim=0, keepdim=True).to(torch.float16)


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


def _blend_two_refs(real_h: torch.Tensor, ref1: torch.Tensor, ref2: torch.Tensor) -> torch.Tensor:
    feature = torch.stack([
        ref1.float().squeeze(0),
        ref2.float().squeeze(0),
        torch.ones_like(ref1.float().squeeze(0)),
    ], dim=1)
    target = real_h.float().squeeze(0)
    solution = torch.linalg.lstsq(feature, target).solution
    pred = solution[0] * ref1.float() + solution[1] * ref2.float() + solution[2]
    return pred.to(torch.float16)


def _run_remaining_layers(model, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    cache_position = torch.arange(seq_len, device=hidden_states.device)
    causal_mask = model.model._update_causal_mask(attention_mask, hidden_states, cache_position, None, False)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    hidden = hidden_states
    start_layer = getattr(_run_remaining_layers, "start_layer", 0)
    for layer in model.model.layers[start_layer:]:
        hidden = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


class StrategyRunner:
    def run_all(
        self,
        real_h: torch.Tensor,
        tier: str,
        table_ref: Optional[torch.Tensor],
        prev_h: Optional[torch.Tensor],
        prev2_h: Optional[torch.Tensor],
        history: List[torch.Tensor],
        global_mean_ref: torch.Tensor,
    ) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        zero_ref = torch.zeros_like(real_h)

        if table_ref is not None:
            results["baseline_current"] = _encode_delta_quantized(real_h, table_ref, 4, 1, outlier_bits=16, include_ref_idx=True, use_affine=True)
            results["delta_int2_k8_out8"] = _encode_delta_quantized(real_h, table_ref, 2, 8, outlier_bits=8, include_ref_idx=True, use_affine=True)
            results["delta_int2_k8_out4"] = _encode_delta_quantized(real_h, table_ref, 2, 8, outlier_bits=4, include_ref_idx=True, use_affine=True)
            results["delta_int2_k8_out8_entropy"] = _encode_delta_quantized(real_h, table_ref, 2, 8, outlier_bits=8, include_ref_idx=True, use_affine=True, entropy_override=True)
            results["delta_noaffine_int4_k1"] = _encode_delta_quantized(real_h, table_ref, 4, 1, outlier_bits=16, include_ref_idx=True, use_affine=False)
            results["delta_noaffine_int2_k8_out4"] = _encode_delta_quantized(real_h, table_ref, 2, 8, outlier_bits=4, include_ref_idx=True, use_affine=False)
            results["delta_sparse_thr10"] = _encode_sparse_groups(real_h, table_ref, 4, 1, threshold_ratio=0.10)
            results["delta_top1024_ch"] = _encode_top_channels(real_h, table_ref, 1024)
            results["delta_struct2of8"] = _encode_structured_2of8(real_h, table_ref)
            results["ablate_delta_raw_ref"] = _encode_raw_reference(real_h, table_ref, include_ref_idx=True)
            results["ablate_delta_affine_only"] = _encode_affine_only(real_h, table_ref, include_ref_idx=True)
            baseline = results["baseline_current"]
            for name in ALL_STRATEGIES:
                if name not in results:
                    results[name] = baseline
            return results

        results["baseline_current"] = _encode_int8_unigram(real_h, top_k=1)
        results["unigram_int4_k4"] = _quantize_direct_int4(real_h, top_k=4)

        if prev_h is not None:
            results["prev_int4_k2"] = _encode_delta_quantized(real_h, prev_h, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=True)
            results["prev_gs256_k2"] = _encode_delta_quantized(real_h, prev_h, 4, 2, outlier_bits=16, include_ref_idx=False, group_size=256, use_affine=True)
            results["prev_int2_k8_out4"] = _encode_delta_quantized(real_h, prev_h, 2, 8, outlier_bits=4, include_ref_idx=False, use_affine=True)
            results["ablate_prev_raw_ref"] = _encode_raw_reference(real_h, prev_h, include_ref_idx=False)
            results["ablate_prev_affine_only"] = _encode_affine_only(real_h, prev_h, include_ref_idx=False)
        else:
            for name in ["prev_int4_k2", "prev_gs256_k2", "prev_int2_k8_out4", "ablate_prev_raw_ref", "ablate_prev_affine_only"]:
                results[name] = results["baseline_current"]

        results["zero_affine_int4_k2"] = _encode_delta_quantized(real_h, zero_ref, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=True)
        results["global_mean_int4_k2"] = _encode_delta_quantized(real_h, global_mean_ref, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=True)

        mean_ref = _mean_pool_ref(history)
        if mean_ref is not None:
            results["mean_pool_int4_k2"] = _encode_delta_quantized(real_h, mean_ref, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=True)
        else:
            results["mean_pool_int4_k2"] = results["baseline_current"]

        ema_ref = _ema_ref(history)
        if ema_ref is not None:
            results["ema_prev4_int4_k2"] = _encode_delta_quantized(real_h, ema_ref, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=True)
        else:
            results["ema_prev4_int4_k2"] = results["baseline_current"]

        if prev_h is not None and prev2_h is not None:
            blend_ref = _blend_two_refs(real_h, prev_h, prev2_h)
            results["prev2_blend_int4_k2"] = _encode_delta_quantized(real_h, blend_ref, 4, 2, outlier_bits=16, include_ref_idx=False, use_affine=False)
        else:
            results["prev2_blend_int4_k2"] = results["baseline_current"]

        baseline = results["baseline_current"]
        for name in ALL_STRATEGIES:
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
    _run_remaining_layers.start_layer = args.layer_boundary

    runner = StrategyRunner()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_records = []
    drift_records = []
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
        mean_sum = None
        mean_count = 0

        logger.info("Warmup: %d requests...", args.warmup_requests)
        for warm_idx in range(args.warmup_requests):
            input_ids = tokenizer(texts[warm_idx], return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            if mean_sum is None:
                mean_sum = h.sum(dim=0, keepdim=True).float()
            else:
                mean_sum += h.sum(dim=0, keepdim=True).float()
            mean_count += h.shape[0]
            if token_ids.shape[0] >= 3:
                trigrams = token_ids.unfold(0, 3, 1)
                for tri_idx in range(trigrams.shape[0]):
                    tri = trigrams[tri_idx]
                    table._get_or_create_node(tri[0].item(), tri[1].item(), h[tri_idx + 1].unsqueeze(0).to(torch.float16))
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[tri_idx + 2].unsqueeze(0).to(torch.float16)
            del out, h

        global_mean_ref = (mean_sum / max(mean_count, 1)).to(torch.float16)

        logger.info("Test: %d requests...", args.test_requests)
        for req_idx in range(args.test_requests):
            text = texts[args.warmup_requests + req_idx]
            input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            original_logits = out.logits
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            seq_len = token_ids.shape[0]
            raw_bytes = seq_len * h.shape[-1] * 2

            tiers: List[str] = []
            refs: List[Optional[torch.Tensor]] = []
            for pos in range(seq_len):
                ref = None
                tier = "unigram"
                if pos >= 2:
                    tri_ref = table.get_trigram(token_ids[pos - 2].item(), token_ids[pos - 1].item(), token_ids[pos].item())
                    if tri_ref is not None:
                        ref = tri_ref.to(torch.float16).to(device)
                        tier = "trigram"
                if ref is None and pos >= 1:
                    bi_ref = table.get_bigram(token_ids[pos - 1].item(), token_ids[pos].item())
                    if bi_ref is not None:
                        ref = bi_ref.to(torch.float16).to(device)
                        tier = "bigram"
                tiers.append(tier)
                refs.append(ref)

            strategy_accum = defaultdict(lambda: {
                "bytes": 0,
                "raw_bytes": 0,
                "zlib_bytes": 0,
                "ideal_bytes": 0.0,
                "entropy_bpb_sum": 0.0,
                "cos_sum": 0.0,
                "cos_min": 1.0,
                "recon_e_sum": 0.0,
                "count": 0,
                "timing_sum": _init_timing(),
                "delta_count": 0,
                "unigram_count": 0,
            })
            recon_sequences = defaultdict(list)
            history: List[torch.Tensor] = []

            for pos in range(seq_len):
                real = h[pos:pos + 1].to(torch.float16)
                prev_h = h[pos - 1:pos].to(torch.float16) if pos > 0 else None
                prev2_h = h[pos - 2:pos - 1].to(torch.float16) if pos > 1 else None
                results = runner.run_all(real, tiers[pos], refs[pos], prev_h, prev2_h, history, global_mean_ref.to(device))
                for strategy_name, result in results.items():
                    recon = result["recon"]
                    recon_sequences[strategy_name].append(recon)
                    metrics = result["metrics"]
                    cos = _cosine(real, recon)
                    acc = strategy_accum[strategy_name]
                    acc["bytes"] += result["bytes"]
                    acc["raw_bytes"] += result["raw_bytes"]
                    acc["zlib_bytes"] += result["zlib_bytes"]
                    acc["ideal_bytes"] += result["ideal_entropy_bytes"]
                    if result["entropy_bpb"] == result["entropy_bpb"]:
                        acc["entropy_bpb_sum"] += result["entropy_bpb"]
                    acc["cos_sum"] += cos
                    acc["cos_min"] = min(acc["cos_min"], cos)
                    acc["recon_e_sum"] += metrics["recon_energy_explained"]
                    acc["count"] += 1
                    if tiers[pos] == "unigram":
                        acc["unigram_count"] += 1
                    else:
                        acc["delta_count"] += 1
                    for key, value in result["timing"].items():
                        acc["timing_sum"][key] += value

                    if req_idx < args.detail_requests:
                        position_records.append({
                            "dataset": ds_name,
                            "request_index": req_idx,
                            "position": pos,
                            "tier": tiers[pos],
                            "strategy": strategy_name,
                            "seq_len": seq_len,
                            "transfer_bytes": result["bytes"],
                            "raw_packet_bytes": result["raw_bytes"],
                            "zlib_packet_bytes": result["zlib_bytes"],
                            "ideal_entropy_bytes": result["ideal_entropy_bytes"],
                            "cosine": cos,
                            "raw_ref_cosine": metrics["raw_ref_cosine"],
                            "affine_ref_cosine": metrics["affine_ref_cosine"],
                            "raw_ref_energy_explained": metrics["raw_ref_energy_explained"],
                            "affine_energy_explained": metrics["affine_energy_explained"],
                            "affine_gain": metrics["affine_gain"],
                            "recon_energy_explained": metrics["recon_energy_explained"],
                            "residual_coding_loss": metrics["residual_coding_loss"],
                            **{f"timing_{k}": v for k, v in result["timing"].items()},
                        })

                history.append(real)

            for strategy_name, acc in strategy_accum.items():
                if req_idx < args.drift_requests:
                    recon_hidden = torch.cat(recon_sequences[strategy_name], dim=0).unsqueeze(0)
                    with torch.no_grad():
                        recon_logits = _run_remaining_layers(model, recon_hidden, attention_mask)
                    logit_cos = F.cosine_similarity(original_logits.float(), recon_logits.float(), dim=-1).squeeze(0)
                    orig_top1 = original_logits.argmax(dim=-1).squeeze(0)
                    recon_top1 = recon_logits.argmax(dim=-1).squeeze(0)
                    mismatch = orig_top1 != recon_top1
                    first_drift = int(torch.where(mismatch)[0][0].item()) if mismatch.any() else seq_len
                    below = logit_cos < 0.999
                    first_logit_cos_below_0999 = int(torch.where(below)[0][0].item()) if below.any() else seq_len
                    kl = F.kl_div(
                        F.log_softmax(recon_logits.float(), dim=-1),
                        F.softmax(original_logits.float(), dim=-1),
                        reduction="none",
                    ).sum(dim=-1).squeeze(0)
                    drift_records.append({
                        "dataset": ds_name,
                        "request_index": req_idx,
                        "strategy": strategy_name,
                        "seq_len": seq_len,
                        "top1_match_rate": float((~mismatch).float().mean().item()),
                        "first_top1_drift_pos": first_drift,
                        "first_logit_cos_below_0_999": first_logit_cos_below_0999,
                        "logit_cosine_mean": float(logit_cos.mean().item()),
                        "logit_cosine_min": float(logit_cos.min().item()),
                        "kl_mean": float(kl.mean().item()),
                        "kl_max": float(kl.max().item()),
                    })

                count = max(acc["count"], 1)
                request_records.append({
                    "dataset": ds_name,
                    "request_index": req_idx,
                    "strategy": strategy_name,
                    "seq_len": seq_len,
                    "raw_fp16_bytes": raw_bytes,
                    "total_transfer_bytes": acc["bytes"],
                    "total_raw_packet_bytes": acc["raw_bytes"],
                    "total_zlib_packet_bytes": acc["zlib_bytes"],
                    "total_ideal_entropy_bytes": acc["ideal_bytes"],
                    "compression_ratio": raw_bytes / max(acc["bytes"], 1),
                    "compression_ratio_raw_packet": raw_bytes / max(acc["raw_bytes"], 1),
                    "compression_ratio_zlib_packet": raw_bytes / max(acc["zlib_bytes"], 1),
                    "compression_ratio_ideal_entropy": raw_bytes / max(acc["ideal_bytes"], 1e-8),
                    "cosine_mean": acc["cos_sum"] / count,
                    "cosine_min": acc["cos_min"],
                    "recon_energy_explained_mean": acc["recon_e_sum"] / count,
                    "residual_coding_loss_mean": 1.0 - acc["recon_e_sum"] / count,
                    "packet_entropy_bits_per_byte_mean": acc["entropy_bpb_sum"] / count,
                    "num_delta_positions": acc["delta_count"],
                    "num_unigram_positions": acc["unigram_count"],
                    **{f"timing_{k}_sum": v for k, v in acc["timing_sum"].items()},
                    **{f"timing_{k}_mean": v / count for k, v in acc["timing_sum"].items()},
                })

            if token_ids.shape[0] >= 3:
                trigrams = token_ids.unfold(0, 3, 1)
                for tri_idx in range(trigrams.shape[0]):
                    tri = trigrams[tri_idx]
                    table._get_or_create_node(tri[0].item(), tri[1].item(), h[tri_idx + 1].unsqueeze(0).to(torch.float16))
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[tri_idx + 2].unsqueeze(0).to(torch.float16)

            del out, h
            if (req_idx + 1) % 5 == 0:
                logger.info("  Test %d/%d complete", req_idx + 1, args.test_requests)

        gc.collect()
        torch.cuda.empty_cache()

    pq.write_table(pa.Table.from_pylist(request_records), str(output_dir / "comprehensive_request_summary.parquet"))
    if drift_records:
        pq.write_table(pa.Table.from_pylist(drift_records), str(output_dir / "comprehensive_drift.parquet"))
    if position_records:
        pq.write_table(pa.Table.from_pylist(position_records), str(output_dir / "comprehensive_position_detail.parquet"))
    logger.info(
        "Saved %d request records, %d drift records, %d position records",
        len(request_records),
        len(drift_records),
        len(position_records),
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive compression evaluation")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=20)
    parser.add_argument("--test-requests", type=int, default=20)
    parser.add_argument("--drift-requests", type=int, default=3)
    parser.add_argument("--detail-requests", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_comprehensive_compression")
    parser.add_argument("--datasets", nargs="+", default=["wikitext2", "sharegpt", "gsm8k", "cnn_dm", "alpaca", "triviaqa"])
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()