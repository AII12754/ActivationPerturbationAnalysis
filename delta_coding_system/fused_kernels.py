"""Fused CUDA kernels for quantize→dequantize operations.

Provides three optimization tiers:
  1. Triton kernel    — best performance, requires ``triton`` package
  2. torch.compile    — good performance, requires PyTorch ≥ 2.0
  3. Eager PyTorch    — baseline (fused functions from codec.py)

All tiers produce numerically identical results.

Usage::

    from delta_coding_system.fused_kernels import (
        fused_int4_qdq,         # quantize→dequantize (no reference)
        fused_int4_delta_qdq,   # delta + quantize→dequantize + reconstruct
        fused_int8_qdq,         # int8 quantize→dequantize (no reference)
    )
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

_TRITON_AVAILABLE = False
_TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ===================================================================
# Triton kernels
# ===================================================================

if _TRITON_AVAILABLE:

    @triton.jit
    def _int4_qdq_kernel(
        input_ptr,
        output_ptr,
        batch,
        hidden_dim: tl.constexpr,
        group_size: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_GROUP: tl.constexpr,
    ):
        """Fused Int4 quantize→dequantize kernel.

        Each program instance handles one (batch_idx, group_idx) pair.
        Loads the group, finds top-k outliers, quantizes the rest to
        4-bit with min/max scaling, dequantizes back, overlays outliers.
        """
        pid = tl.program_id(0)
        num_groups = hidden_dim // group_size
        batch_idx = pid // num_groups
        group_idx = pid % num_groups

        base_offset = batch_idx * hidden_dim + group_idx * group_size
        offsets = tl.arange(0, BLOCK_GROUP)
        mask = offsets < group_size

        # Load group into SRAM
        vals = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)

        # Find top-k outlier positions via iterative argmax.
        # For small k (1-4), this is cheaper than a full sort.
        outlier_vals = tl.zeros([top_k], dtype=tl.float32)
        outlier_mask = tl.zeros([BLOCK_GROUP], dtype=tl.int32)

        for _k in range(top_k):
            # Mask already-found outliers
            masked_abs = tl.where(outlier_mask == 0, abs_vals, -1.0)
            max_idx = tl.argmax(masked_abs, axis=0)
            outlier_mask = tl.where(offsets == max_idx, 1, outlier_mask)

        # Compute min/max excluding outliers
        non_outlier_vals = tl.where(outlier_mask == 0, vals, 0.0)
        g_min = tl.min(non_outlier_vals, axis=0)
        g_max = tl.max(non_outlier_vals, axis=0)
        scale = (g_max - g_min) / 15.0

        # Quantize → dequantize
        q = tl.where(
            outlier_mask == 0,
            tl.math.round((non_outlier_vals - g_min) / (scale + 1e-10)),
            0.0,
        )
        q = tl.maximum(tl.minimum(q, 15.0), 0.0)
        dequant = q * scale + g_min

        # Overlay outliers (keep original values)
        result = tl.where(outlier_mask != 0, vals, dequant)

        tl.store(output_ptr + base_offset + offsets, result, mask=mask)


    @triton.jit
    def _int8_qdq_kernel(
        input_ptr,
        output_ptr,
        batch,
        hidden_dim: tl.constexpr,
        group_size: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_GROUP: tl.constexpr,
    ):
        """Fused Int8 quantize→dequantize kernel."""
        pid = tl.program_id(0)
        num_groups = hidden_dim // group_size
        batch_idx = pid // num_groups
        group_idx = pid % num_groups

        base_offset = batch_idx * hidden_dim + group_idx * group_size
        offsets = tl.arange(0, BLOCK_GROUP)
        mask = offsets < group_size

        vals = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)

        outlier_mask = tl.zeros([BLOCK_GROUP], dtype=tl.int32)
        for _k in range(top_k):
            masked_abs = tl.where(outlier_mask == 0, abs_vals, -1.0)
            max_idx = tl.argmax(masked_abs, axis=0)
            outlier_mask = tl.where(offsets == max_idx, 1, outlier_mask)

        non_outlier_vals = tl.where(outlier_mask == 0, vals, 0.0)
        g_min = tl.min(non_outlier_vals, axis=0)
        g_max = tl.max(non_outlier_vals, axis=0)
        scale = (g_max - g_min) / 255.0

        q = tl.where(
            outlier_mask == 0,
            tl.math.round((non_outlier_vals - g_min) / (scale + 1e-10)),
            0.0,
        )
        q = tl.maximum(tl.minimum(q, 255.0), 0.0)
        dequant = q * scale + g_min

        result = tl.where(outlier_mask != 0, vals, dequant)
        tl.store(output_ptr + base_offset + offsets, result, mask=mask)


    @triton.jit
    def _delta_int4_qdq_kernel(
        real_ptr,
        ref_ptr,
        output_ptr,
        batch,
        hidden_dim: tl.constexpr,
        group_size: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_GROUP: tl.constexpr,
    ):
        """Fused delta→Int4 quantize→dequantize→reconstruct kernel.

        Reads real and ref, computes delta, quantizes/dequantizes delta,
        reconstructs ``ref + dequant_delta``.
        """
        pid = tl.program_id(0)
        num_groups = hidden_dim // group_size
        batch_idx = pid // num_groups
        group_idx = pid % num_groups

        base_offset = batch_idx * hidden_dim + group_idx * group_size
        offsets = tl.arange(0, BLOCK_GROUP)
        mask = offsets < group_size

        real_vals = tl.load(real_ptr + base_offset + offsets, mask=mask, other=0.0)
        ref_vals = tl.load(ref_ptr + base_offset + offsets, mask=mask, other=0.0)
        delta = real_vals - ref_vals
        abs_delta = tl.abs(delta)

        outlier_mask = tl.zeros([BLOCK_GROUP], dtype=tl.int32)
        for _k in range(top_k):
            masked_abs = tl.where(outlier_mask == 0, abs_delta, -1.0)
            max_idx = tl.argmax(masked_abs, axis=0)
            outlier_mask = tl.where(offsets == max_idx, 1, outlier_mask)

        non_outlier = tl.where(outlier_mask == 0, delta, 0.0)
        g_min = tl.min(non_outlier, axis=0)
        g_max = tl.max(non_outlier, axis=0)
        scale = (g_max - g_min) / 15.0

        q = tl.where(
            outlier_mask == 0,
            tl.math.round((non_outlier - g_min) / (scale + 1e-10)),
            0.0,
        )
        q = tl.maximum(tl.minimum(q, 15.0), 0.0)
        dequant_delta = q * scale + g_min
        dequant_delta = tl.where(outlier_mask != 0, delta, dequant_delta)

        result = ref_vals + dequant_delta
        tl.store(output_ptr + base_offset + offsets, result, mask=mask)


# ===================================================================
# Python wrappers
# ===================================================================

def _triton_int4_qdq(tensor: torch.Tensor, group_size: int, top_k: int) -> torch.Tensor:
    batch, hidden_dim = tensor.shape
    num_groups = hidden_dim // group_size
    inp = tensor.float().contiguous()
    out = torch.empty_like(inp)
    grid = (batch * num_groups,)
    # BLOCK_GROUP must be power-of-2 ≥ group_size for Triton
    block_group = max(32, 1 << (group_size - 1).bit_length())
    _int4_qdq_kernel[grid](
        inp, out, batch, hidden_dim, group_size, top_k, block_group,
    )
    return out.to(torch.float16)


def _triton_int8_qdq(tensor: torch.Tensor, group_size: int, top_k: int) -> torch.Tensor:
    batch, hidden_dim = tensor.shape
    effective_group_size = min(group_size, hidden_dim)
    effective_top_k = min(top_k, effective_group_size)
    num_groups = hidden_dim // effective_group_size
    inp = tensor.float().contiguous()
    out = torch.empty_like(inp)
    grid = (batch * num_groups,)
    block_group = max(32, 1 << (effective_group_size - 1).bit_length())
    _int8_qdq_kernel[grid](
        inp, out, batch, hidden_dim, effective_group_size, effective_top_k, block_group,
    )
    return out.to(torch.float16)


def _triton_delta_int4_qdq(
    real: torch.Tensor, ref: torch.Tensor, group_size: int, top_k: int,
) -> torch.Tensor:
    batch, hidden_dim = real.shape
    num_groups = hidden_dim // group_size
    real_f = real.float().contiguous()
    ref_f = ref.float().contiguous()
    out = torch.empty_like(real_f)
    grid = (batch * num_groups,)
    block_group = max(32, 1 << (group_size - 1).bit_length())
    _delta_int4_qdq_kernel[grid](
        real_f, ref_f, out, batch, hidden_dim, group_size, top_k, block_group,
    )
    return out.to(torch.float16)


# ===================================================================
# torch.compile wrappers (auto-fuse via Triton backend)
# ===================================================================

def _eager_int4_qdq(tensor: torch.Tensor, group_size: int, top_k: int) -> torch.Tensor:
    """Eager PyTorch fused Int4 quantize→dequantize (no packing)."""
    from delta_coding_system.codec import fused_int4_quantize_dequantize
    recon, _ = fused_int4_quantize_dequantize(tensor, group_size, top_k)
    return recon


def _eager_int8_qdq(tensor: torch.Tensor, group_size: int, top_k: int) -> torch.Tensor:
    from delta_coding_system.codec import fused_int8_quantize_dequantize
    recon, _ = fused_int8_quantize_dequantize(tensor, group_size, top_k)
    return recon


def _eager_delta_int4_qdq(
    real: torch.Tensor, ref: torch.Tensor, group_size: int, top_k: int,
) -> torch.Tensor:
    from delta_coding_system.codec import fused_int4_delta_encode
    recon, _ = fused_int4_delta_encode(real, ref, group_size, top_k)
    return recon


if _TORCH_COMPILE_AVAILABLE:
    _compiled_int4_qdq = torch.compile(_eager_int4_qdq, mode="max-autotune")
    _compiled_int8_qdq = torch.compile(_eager_int8_qdq, mode="max-autotune")
    _compiled_delta_int4_qdq = torch.compile(_eager_delta_int4_qdq, mode="max-autotune")
else:
    _compiled_int4_qdq = _eager_int4_qdq
    _compiled_int8_qdq = _eager_int8_qdq
    _compiled_delta_int4_qdq = _eager_delta_int4_qdq


# ===================================================================
# Transfer size helpers (analytical, no tensor materialization)
# ===================================================================

def int4_transfer_bytes(batch: int, hidden_dim: int, group_size: int, top_k: int) -> int:
    """Analytical Int4 transfer size without materializing the packet."""
    num_groups = hidden_dim // group_size
    return batch * (
        hidden_dim // 2              # packed uint8
        + num_groups * 4             # scales + zero_points (fp16 each)
        + num_groups * top_k * 3     # topk_values(fp16) + topk_indices(uint8)
    )


def int8_transfer_bytes(batch: int, hidden_dim: int, group_size: int, top_k: int) -> int:
    """Analytical Int8 transfer size without materializing the packet."""
    effective_gs = min(group_size, hidden_dim)
    effective_k = min(top_k, effective_gs)
    num_groups = hidden_dim // effective_gs
    return batch * (
        hidden_dim                   # quantized uint8
        + num_groups * 4             # scales + zero_points (fp16 each)
        + num_groups * effective_k * 3
    )


# ===================================================================
# Public API — auto-selects best available backend
# ===================================================================

class _BackendSelector:
    """Selects the best available kernel backend at first call."""

    def __init__(self):
        self._backend: str | None = None

    def _select(self) -> str:
        if self._backend is not None:
            return self._backend
        if _TRITON_AVAILABLE:
            self._backend = "triton"
            logger.info("fused_kernels: using Triton backend")
        elif _TORCH_COMPILE_AVAILABLE:
            self._backend = "compile"
            logger.info("fused_kernels: using torch.compile backend")
        else:
            self._backend = "eager"
            logger.info("fused_kernels: using eager PyTorch backend")
        return self._backend

    @property
    def backend(self) -> str:
        return self._select()


_selector = _BackendSelector()


def fused_int4_qdq(
    tensor: torch.Tensor, group_size: int, top_k: int,
) -> Tuple[torch.Tensor, int]:
    """Fused Int4 quantize→dequantize. Returns (recon_fp16, transfer_bytes)."""
    b = _selector.backend
    batch, hidden_dim = tensor.shape
    transfer = int4_transfer_bytes(batch, hidden_dim, group_size, top_k)
    if b == "triton":
        return _triton_int4_qdq(tensor, group_size, top_k), transfer
    if b == "compile":
        return _compiled_int4_qdq(tensor, group_size, top_k), transfer
    return _eager_int4_qdq(tensor, group_size, top_k), transfer


def fused_int4_delta_qdq(
    real: torch.Tensor,
    ref: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, int]:
    """Fused delta→Int4 quantize→dequantize→reconstruct. Returns (recon_fp16, transfer_bytes)."""
    b = _selector.backend
    batch, hidden_dim = real.shape
    transfer = int4_transfer_bytes(batch, hidden_dim, group_size, top_k)
    if b == "triton":
        return _triton_delta_int4_qdq(real, ref, group_size, top_k), transfer
    if b == "compile":
        return _compiled_delta_int4_qdq(real, ref, group_size, top_k), transfer
    return _eager_delta_int4_qdq(real, ref, group_size, top_k), transfer


def fused_int8_qdq(
    tensor: torch.Tensor, group_size: int, top_k: int,
) -> Tuple[torch.Tensor, int]:
    """Fused Int8 quantize→dequantize. Returns (recon_fp16, transfer_bytes)."""
    b = _selector.backend
    batch, hidden_dim = tensor.shape
    transfer = int8_transfer_bytes(batch, hidden_dim, group_size, top_k)
    if b == "triton":
        return _triton_int8_qdq(tensor, group_size, top_k), transfer
    if b == "compile":
        return _compiled_int8_qdq(tensor, group_size, top_k), transfer
    return _eager_int8_qdq(tensor, group_size, top_k), transfer
