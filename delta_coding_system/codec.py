"""Delta-coding encode/decode primitives for pipeline-parallel activation transfer.

Implements the encode/decode pipeline:
  Sender:  reference lookup -> affine transform -> delta -> Int4 quantize + top-k outliers
  Receiver: dequantize + outlier overlay -> affine reference -> reconstruct

For unigram (no-reference) positions:
  Int8 group quantization + top-k FP16 outliers

All operations are GPU-native.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import zlib

import torch


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class DeltaPacket:
    """Packed delta-coded activation for inter-stage transfer."""
    quantized_data: torch.Tensor    # (batch, hidden_dim // 2) uint8
    scales: torch.Tensor            # (batch, num_groups) float16
    zero_points: torch.Tensor       # (batch, num_groups) float16
    topk_values: torch.Tensor       # (batch, num_groups, k) float16
    topk_indices: torch.Tensor      # (batch, num_groups, k) uint8
    affine_scale: torch.Tensor      # (batch,) float16
    affine_bias: torch.Tensor       # (batch,) float16
    ref_indices: torch.Tensor       # (batch,) int64
    group_size: int
    top_k: int


@dataclass
class Int8OutlierPacket:
    """Packed Int8-quantized activation with top-k fp16 outlier extraction."""
    quantized: torch.Tensor             # (batch, hidden_dim) uint8
    scales: torch.Tensor                # (batch, num_groups) float16
    zero_points: torch.Tensor           # (batch, num_groups) float16
    topk_values: torch.Tensor           # (batch, num_groups, top_k) float16
    topk_indices: torch.Tensor          # (batch, num_groups, top_k) uint8
    group_size: int
    top_k: int


def serialize_tensor(tensor: Optional[torch.Tensor]) -> bytes:
    """Serialize a tensor into a contiguous raw byte payload on CPU."""
    if tensor is None or tensor.numel() == 0:
        return b""
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def entropy_coded_num_bytes(tensors: Iterable[Optional[torch.Tensor]], level: int = 1) -> int:
    """Return compressed payload size in bytes for a sequence of tensors.

    This provides a real lossless entropy-coding proxy for transfer accounting.
    It is intentionally CPU-side and should be used only for payload-size estimation.
    """
    payload = b"".join(serialize_tensor(tensor) for tensor in tensors)
    if not payload:
        return 0
    return len(zlib.compress(payload, level=level))


# ===================================================================
# Core Operations
# ===================================================================
def compute_affine_params(
    new_acts: torch.Tensor,
    ref_acts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute affine transform parameters: scale and bias.

    ``scale = dot(new, ref) / dot(ref, ref)``
    ``bias = mean(new - scale * ref)``
    """
    new_f = new_acts.float()
    ref_f = ref_acts.float()
    dot_nr = (new_f * ref_f).sum(dim=-1)
    dot_rr = (ref_f * ref_f).sum(dim=-1)
    scale = dot_nr / (dot_rr + 1e-8)
    bias = (new_f - scale.unsqueeze(-1) * ref_f).mean(dim=-1)
    return scale, bias


def apply_affine(
    ref_acts: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply affine transform: ``scale * ref + bias``."""
    return scale.unsqueeze(-1) * ref_acts + bias.unsqueeze(-1)


def compute_delta(
    new_acts: torch.Tensor,
    ref_transformed: torch.Tensor,
) -> torch.Tensor:
    """Compute delta: ``new - ref_transformed``."""
    return new_acts - ref_transformed


def groupwise_int4_quantize_topk(
    delta: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise Int4 quantization with top-k fp16 outlier extraction."""
    batch, hidden_dim = delta.shape
    num_groups = hidden_dim // group_size

    grouped = delta.reshape(batch, num_groups, group_size)

    abs_grouped = grouped.abs()
    topk_vals_abs, topk_idx = torch.topk(abs_grouped, top_k, dim=-1)
    topk_values = grouped.gather(-1, topk_idx).to(torch.float16)
    topk_indices = topk_idx.to(torch.uint8)

    # Zero out top-k positions in-place (topk_values already saved above)
    grouped.scatter_(-1, topk_idx, 0.0)

    g_min = grouped.min(dim=-1).values
    g_max = grouped.max(dim=-1).values
    scales = ((g_max - g_min) / 15.0).to(torch.float16)
    zero_points = g_min.to(torch.float16)

    scales_f = scales.float().unsqueeze(-1)
    zeros_f = zero_points.float().unsqueeze(-1)
    q = torch.clamp(
        torch.round((grouped - zeros_f) / (scales_f + 1e-10)),
        0, 15,
    ).to(torch.uint8)

    q_flat = q.reshape(batch, hidden_dim)
    even = q_flat[:, 0::2]
    odd = q_flat[:, 1::2]
    packed = (even << 4) | odd

    return packed, scales, zero_points, topk_values, topk_indices


def groupwise_int4_dequantize_topk(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    group_size: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Dequantize Int4 groups and overlay top-k fp16 outliers."""
    batch = packed.shape[0]
    num_groups = hidden_dim // group_size

    even = (packed >> 4).to(torch.uint8)
    odd = (packed & 0x0F).to(torch.uint8)

    # Interleave even/odd via stack+reshape to avoid zeros allocation
    q_flat = torch.stack([even, odd], dim=-1).reshape(batch, hidden_dim)

    q_grouped = q_flat.reshape(batch, num_groups, group_size)

    scales_f = scales.float().unsqueeze(-1)
    zeros_f = zero_points.float().unsqueeze(-1)
    dequant = q_grouped.float() * scales_f + zeros_f

    topk_idx_long = topk_indices.long()
    dequant.scatter_(-1, topk_idx_long, topk_values.float())

    return dequant.reshape(batch, hidden_dim).to(torch.float16)


# ===================================================================
# Fused quantize-dequantize (skip bit-packing round-trip)
# ===================================================================

def fused_int4_quantize_dequantize(
    tensor: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, int]:
    """Fused Int4 quantize→dequantize without materializing packed representation.

    For the encode path we only need the *reconstructed* output (to measure
    quantization loss) and the transfer size.  Skipping the uint8 pack/unpack
    round-trip eliminates ~8 CUDA kernel launches.

    Returns ``(reconstructed_fp16, transfer_bytes)``.
    """
    batch, hidden_dim = tensor.shape
    num_groups = hidden_dim // group_size

    # Work in float32 for precision
    grouped = tensor.float().reshape(batch, num_groups, group_size)

    # Extract top-k outlier values (kept as-is in fp16)
    abs_grouped = grouped.abs()
    _, topk_idx = torch.topk(abs_grouped, top_k, dim=-1)
    topk_values = grouped.gather(-1, topk_idx)

    # Zero out outlier positions before computing group statistics
    grouped.scatter_(-1, topk_idx, 0.0)

    g_min = grouped.min(dim=-1, keepdim=True).values
    g_max = grouped.max(dim=-1, keepdim=True).values
    scale = (g_max - g_min) / 15.0

    # Quantize→dequantize in float (no int4 packing)
    q = torch.clamp(torch.round((grouped - g_min) / (scale + 1e-10)), 0, 15)
    dequant = q * scale + g_min

    # Overlay top-k outliers
    dequant.scatter_(-1, topk_idx, topk_values)

    # Analytical transfer size:
    #   packed int4: batch * hidden_dim / 2 (uint8)
    #   scales:       batch * num_groups * 2 (fp16)
    #   zero_points:  batch * num_groups * 2 (fp16)
    #   topk_values:  batch * num_groups * top_k * 2 (fp16)
    #   topk_indices: batch * num_groups * top_k * 1 (uint8)
    transfer = batch * (
        hidden_dim // 2
        + num_groups * 4
        + num_groups * top_k * 3
    )

    return dequant.reshape(batch, hidden_dim).to(torch.float16), transfer


def fused_int4_delta_encode(
    real_batch: torch.Tensor,
    ref_batch: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, int]:
    """Fused no-affine delta encode: delta→int4 quantize→dequantize→reconstruct.

    Combines ``compute_delta`` + ``fused_int4_quantize_dequantize`` +
    ``(ref + dequant)`` into a single function with fewer intermediate tensors.

    Returns ``(reconstructed_fp16, transfer_bytes)``.
    """
    batch, hidden_dim = real_batch.shape
    num_groups = hidden_dim // group_size

    # Delta in float32
    delta_grouped = (real_batch.float() - ref_batch.float()).reshape(
        batch, num_groups, group_size
    )

    # Top-k outlier extraction
    abs_delta = delta_grouped.abs()
    _, topk_idx = torch.topk(abs_delta, top_k, dim=-1)
    topk_values = delta_grouped.gather(-1, topk_idx)

    delta_grouped.scatter_(-1, topk_idx, 0.0)

    g_min = delta_grouped.min(dim=-1, keepdim=True).values
    g_max = delta_grouped.max(dim=-1, keepdim=True).values
    scale = (g_max - g_min) / 15.0

    q = torch.clamp(torch.round((delta_grouped - g_min) / (scale + 1e-10)), 0, 15)
    dequant_delta = q * scale + g_min
    dequant_delta.scatter_(-1, topk_idx, topk_values)

    # Reconstruct = ref + dequant_delta
    ref_grouped = ref_batch.float().reshape(batch, num_groups, group_size)
    recon = (ref_grouped + dequant_delta).reshape(batch, hidden_dim).to(torch.float16)

    transfer = batch * (
        hidden_dim // 2
        + num_groups * 4
        + num_groups * top_k * 3
    )
    return recon, transfer


def fused_int4_affine_delta_encode(
    real_batch: torch.Tensor,
    ref_batch: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, int]:
    """Fused affine delta encode: affine→delta→int4 quantize→dequantize→reconstruct.

    Returns ``(reconstructed_fp16, transfer_bytes)``.
    """
    batch, hidden_dim = real_batch.shape
    num_groups = hidden_dim // group_size

    # Affine parameters
    new_f = real_batch.float()
    ref_f = ref_batch.float()
    dot_nr = (new_f * ref_f).sum(dim=-1, keepdim=True)
    dot_rr = (ref_f * ref_f).sum(dim=-1, keepdim=True)
    aff_scale = dot_nr / (dot_rr + 1e-8)
    aff_bias = (new_f - aff_scale * ref_f).mean(dim=-1, keepdim=True)

    # Delta from affine-transformed reference
    ref_t = aff_scale * ref_f + aff_bias
    delta_grouped = (new_f - ref_t).reshape(batch, num_groups, group_size)

    # Top-k outlier extraction
    abs_delta = delta_grouped.abs()
    _, topk_idx = torch.topk(abs_delta, top_k, dim=-1)
    topk_values = delta_grouped.gather(-1, topk_idx)

    delta_grouped.scatter_(-1, topk_idx, 0.0)

    g_min = delta_grouped.min(dim=-1, keepdim=True).values
    g_max = delta_grouped.max(dim=-1, keepdim=True).values
    scale = (g_max - g_min) / 15.0

    q = torch.clamp(torch.round((delta_grouped - g_min) / (scale + 1e-10)), 0, 15)
    dequant_delta = q * scale + g_min
    dequant_delta.scatter_(-1, topk_idx, topk_values)

    # Reconstruct = affine(ref) + dequant_delta
    ref_t_grouped = ref_t.reshape(batch, num_groups, group_size)
    recon = (ref_t_grouped + dequant_delta).reshape(batch, hidden_dim).to(torch.float16)

    # Transfer: packed + scales + zeros + topk + affine params
    transfer = batch * (
        hidden_dim // 2
        + num_groups * 4
        + num_groups * top_k * 3
        + 4  # affine_scale(2) + affine_bias(2)
    )
    return recon, transfer


def fused_int8_quantize_dequantize(
    tensor: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, int]:
    """Fused Int8 quantize→dequantize without materializing the Int8OutlierPacket.

    Returns ``(reconstructed_fp16, transfer_bytes)``.
    """
    batch, hidden_dim = tensor.shape
    effective_group_size = min(group_size, hidden_dim)
    effective_top_k = min(top_k, effective_group_size)
    num_groups = hidden_dim // effective_group_size

    grouped = tensor.float().reshape(batch, num_groups, effective_group_size)

    abs_vals = grouped.abs()
    _, tk_idx = abs_vals.topk(effective_top_k, dim=-1)
    tk_vals = grouped.gather(-1, tk_idx)

    grouped.scatter_(-1, tk_idx, 0.0)

    g_min = grouped.min(dim=-1, keepdim=True).values
    g_max = grouped.max(dim=-1, keepdim=True).values
    scale = (g_max - g_min) / 255.0

    q = torch.clamp(torch.round((grouped - g_min) / (scale + 1e-10)), 0, 255)
    dequant = q * scale + g_min

    dequant.scatter_(-1, tk_idx, tk_vals)

    # Transfer: quantized(uint8) + scales(fp16) + zeros(fp16) + topk_values(fp16) + topk_indices(uint8)
    transfer = batch * (
        hidden_dim  # uint8 quantized
        + num_groups * 4  # scales + zero_points (fp16 each)
        + num_groups * effective_top_k * 3  # topk: values(fp16) + indices(uint8)
    )
    return dequant.reshape(batch, hidden_dim).to(torch.float16), transfer


def groupwise_int2_quantize_topk(
    delta: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise Int2 quantization with top-k fp16 outlier extraction."""
    batch, hidden_dim = delta.shape
    num_groups = hidden_dim // group_size
    grouped = delta.reshape(batch, num_groups, group_size)

    abs_grouped = grouped.abs()
    _, topk_idx = torch.topk(abs_grouped, top_k, dim=-1)
    topk_values = grouped.gather(-1, topk_idx).to(torch.float16)
    topk_indices = topk_idx.to(torch.uint8)

    grouped.scatter_(-1, topk_idx, 0.0)

    g_min = grouped.min(dim=-1).values
    g_max = grouped.max(dim=-1).values
    scales = ((g_max - g_min) / 3.0).to(torch.float16)
    zero_points = g_min.to(torch.float16)

    scales_f = scales.float().unsqueeze(-1)
    zeros_f = zero_points.float().unsqueeze(-1)
    q = torch.clamp(
        torch.round((grouped - zeros_f) / (scales_f + 1e-10)),
        0, 3,
    ).to(torch.uint8)

    q_flat = q.reshape(batch, hidden_dim)
    packed = (
        (q_flat[:, 0::4] << 6)
        | (q_flat[:, 1::4] << 4)
        | (q_flat[:, 2::4] << 2)
        | q_flat[:, 3::4]
    )

    return packed, scales, zero_points, topk_values, topk_indices


def groupwise_int2_dequantize_topk(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    group_size: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Dequantize Int2 groups and overlay top-k fp16 outliers."""
    batch = packed.shape[0]
    num_groups = hidden_dim // group_size

    v0 = (packed >> 6) & 0x03
    v1 = (packed >> 4) & 0x03
    v2 = (packed >> 2) & 0x03
    v3 = packed & 0x03
    q_flat = torch.stack([v0, v1, v2, v3], dim=-1).reshape(batch, hidden_dim).to(torch.uint8)

    q_grouped = q_flat.reshape(batch, num_groups, group_size)
    scales_f = scales.float().unsqueeze(-1)
    zeros_f = zero_points.float().unsqueeze(-1)
    dequant = q_grouped.float() * scales_f + zeros_f

    topk_idx_long = topk_indices.long()
    dequant.scatter_(-1, topk_idx_long, topk_values.float())

    return dequant.reshape(batch, hidden_dim).to(torch.float16)


def groupwise_int8_quantize_topk(
    tensor: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Int8OutlierPacket:
    """Group-wise Int8 quantization with top-k fp16 outlier extraction."""
    batch, hidden_dim = tensor.shape
    effective_group_size = min(group_size, hidden_dim)
    if effective_group_size <= 0:
        raise ValueError("group_size must be positive")
    if hidden_dim % effective_group_size != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by effective group size {effective_group_size}"
        )
    effective_top_k = min(top_k, effective_group_size)
    num_groups = hidden_dim // effective_group_size

    grouped = tensor.float().reshape(batch, num_groups, effective_group_size)

    abs_vals = grouped.abs()
    _, tk_idx = abs_vals.topk(effective_top_k, dim=-1)
    tk_vals = torch.gather(grouped, -1, tk_idx)

    # Zero out top-k positions in-place (tk_vals already saved above)
    grouped.scatter_(-1, tk_idx, 0.0)

    g_min = grouped.min(dim=-1).values
    g_max = grouped.max(dim=-1).values
    scales = ((g_max - g_min) / 255.0).to(torch.float16)
    zero_points = g_min.to(torch.float16)

    scales_f = scales.float().unsqueeze(-1)
    zeros_f = zero_points.float().unsqueeze(-1)
    q = torch.clamp(
        torch.round((grouped - zeros_f) / (scales_f + 1e-10)),
        0, 255,
    ).to(torch.uint8)

    quantized = q.reshape(batch, hidden_dim)

    return Int8OutlierPacket(
        quantized=quantized,
        scales=scales,
        zero_points=zero_points,
        topk_values=tk_vals.to(torch.float16),
        topk_indices=tk_idx.to(torch.uint8),
        group_size=effective_group_size,
        top_k=effective_top_k,
    )


def groupwise_int8_dequantize_topk(packet: Int8OutlierPacket) -> torch.Tensor:
    """Dequantize Int8 + overlay top-k outliers."""
    hidden_dim = packet.quantized.shape[1]
    batch = packet.quantized.shape[0]
    num_groups = hidden_dim // packet.group_size

    q_grouped = packet.quantized.reshape(batch, num_groups, packet.group_size)
    scales_f = packet.scales.float().unsqueeze(-1)
    zeros_f = packet.zero_points.float().unsqueeze(-1)
    dequant = q_grouped.float() * scales_f + zeros_f

    tk_idx = packet.topk_indices.long()
    tk_vals = packet.topk_values.float()
    dequant.scatter_(-1, tk_idx, tk_vals)

    return dequant.reshape(batch, hidden_dim).to(torch.float16)


def reconstruct_activation(
    dequant_delta: torch.Tensor,
    ref_acts: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct activation: ``affine(ref) + dequantized_delta``."""
    ref_transformed = apply_affine(ref_acts, scale, bias)
    return ref_transformed + dequant_delta


def compute_transfer_size(packet: DeltaPacket) -> int:
    """Compute total transfer size in bytes for a DeltaPacket."""
    total = 0
    total += packet.quantized_data.nelement() * packet.quantized_data.element_size()
    total += packet.scales.nelement() * packet.scales.element_size()
    total += packet.zero_points.nelement() * packet.zero_points.element_size()
    total += packet.topk_values.nelement() * packet.topk_values.element_size()
    total += packet.topk_indices.nelement() * packet.topk_indices.element_size()
    total += packet.affine_scale.nelement() * packet.affine_scale.element_size()
    total += packet.affine_bias.nelement() * packet.affine_bias.element_size()
    total += packet.ref_indices.nelement() * packet.ref_indices.element_size()
    return total


def compute_transfer_size_int8_outlier(packet: Int8OutlierPacket) -> int:
    """Compute total transfer size in bytes for an Int8OutlierPacket."""
    total = 0
    total += packet.quantized.nelement() * packet.quantized.element_size()
    total += packet.scales.nelement() * packet.scales.element_size()
    total += packet.zero_points.nelement() * packet.zero_points.element_size()
    total += packet.topk_values.nelement() * packet.topk_values.element_size()
    total += packet.topk_indices.nelement() * packet.topk_indices.element_size()
    return total


# ===================================================================
# High-level encode/decode helper
# ===================================================================
def encode_decode_single(
    real_h: torch.Tensor,      # (1, hidden_dim)
    ref_h: torch.Tensor,       # (1, hidden_dim) or None
    tier: str,
    group_size: int,
    top_k: int,
    int8_group_size: int,
    int8_outlier_top_k: int,
    hidden_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Encode and decode a single position. Returns (reconstructed, transfer_bytes)."""

    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
        # Affine + Int4 delta
        scale, bias = compute_affine_params(real_h, ref_h)
        ref_t = apply_affine(ref_h, scale, bias)
        delta = compute_delta(real_h, ref_t)
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, hidden_dim)
        recon = reconstruct_activation(dequant, ref_h, scale, bias).to(torch.float16)

        pkt = DeltaPacket(
            quantized_data=packed, scales=scales, zero_points=zeros,
            topk_values=tv, topk_indices=ti,
            affine_scale=scale.to(torch.float16),
            affine_bias=bias.to(torch.float16),
            ref_indices=torch.zeros(1, dtype=torch.long, device=device),
            group_size=group_size, top_k=top_k,
        )
        transfer_bytes = compute_transfer_size(pkt)
        return recon, transfer_bytes
    else:
        # Int8 + outlier (unigram)
        int8_pkt = groupwise_int8_quantize_topk(real_h, int8_group_size, int8_outlier_top_k)
        recon = groupwise_int8_dequantize_topk(int8_pkt)
        transfer_bytes = compute_transfer_size_int8_outlier(int8_pkt)
        return recon, transfer_bytes
