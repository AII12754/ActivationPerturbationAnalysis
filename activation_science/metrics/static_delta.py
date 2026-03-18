"""Static token delta-coding for prefill activation compression.

Pre-computes activations for each token without context (single-token input)
and stores them as a static embedding table. At prefill time, delta-codes
real activations against these static references, quantizes, and transfers
the compressed delta.

All operations are GPU-native with CUDA event-based timing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .delta_coding import (
    DeltaPacket,
    TimingResult,
    apply_affine,
    compute_affine_params,
    compute_delta,
    gpu_timed,
    groupwise_int4_dequantize_topk,
    groupwise_int4_quantize_topk,
    reconstruct_activation,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Static Token Table
# ===================================================================
class StaticTokenTable:
    """Pre-computed activation table for individual tokens (no context).

    For each token ID, stores the hidden-state vector produced by running
    the model on that single token in isolation at a given layer.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.table: Dict[int, torch.Tensor] = {}
        self.dense_table: Optional[torch.Tensor] = None
        self._max_token_id: int = 0

    @torch.inference_mode()
    def build_from_model(
        self,
        model,
        token_ids: torch.Tensor,
        layer_idx: int,
        batch_size: int = 256,
    ):
        """Run model on single-token inputs in batches and store activations.

        Parameters
        ----------
        model : HuggingFace causal LM
        token_ids : torch.Tensor
            1-D tensor of unique token IDs to build entries for.
        layer_idx : int
            Which layer's hidden states to capture.
        batch_size : int
            Number of tokens to process per forward pass.
        """
        device = next(model.parameters()).device
        unique_ids = token_ids.unique()

        # Skip tokens already built
        new_ids = [tid.item() for tid in unique_ids if tid.item() not in self.table]
        if not new_ids:
            self._build_dense_table()
            return

        logger.info(
            "Building static table: %d new tokens (layer %d, batch_size %d).",
            len(new_ids), layer_idx, batch_size,
        )

        for start in range(0, len(new_ids), batch_size):
            batch_ids = new_ids[start:start + batch_size]
            # Each token is a single-token input: (batch, 1)
            input_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device).unsqueeze(1)

            outputs = model(
                input_ids=input_tensor,
                output_hidden_states=True,
                use_cache=False,
            )

            # hidden_states is a tuple of (num_layers+1) tensors, each (batch, 1, hidden_dim)
            hidden = outputs.hidden_states[layer_idx]  # (batch, 1, hidden_dim)
            hidden = hidden.squeeze(1).to(self.dtype).detach()  # (batch, hidden_dim)

            for i, tid in enumerate(batch_ids):
                self.table[tid] = hidden[i]

        self._build_dense_table()

    def _build_dense_table(self):
        """Build a dense (max_token_id+1, hidden_dim) tensor for fast gather."""
        if not self.table:
            return

        self._max_token_id = max(self.table.keys())
        sample = next(iter(self.table.values()))
        hidden_dim = sample.shape[0]

        self.dense_table = torch.zeros(
            self._max_token_id + 1, hidden_dim,
            device=self.device, dtype=self.dtype,
        )
        for tid, vec in self.table.items():
            self.dense_table[tid] = vec.to(self.device)

    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up static activations by token ID — O(1) gather.

        Parameters
        ----------
        token_ids : torch.Tensor
            Shape ``(seq_len,)`` or ``(batch,)`` of token IDs.

        Returns
        -------
        torch.Tensor of shape ``(seq_len, hidden_dim)``.
        """
        return self.dense_table[token_ids.long()]


# ===================================================================
# Delta Magnitude & Affine Stats
# ===================================================================
def compute_delta_magnitude_stats(delta: torch.Tensor) -> Dict[str, float]:
    """Compute magnitude statistics of a delta tensor.

    Parameters
    ----------
    delta : torch.Tensor
        Shape ``(batch, hidden_dim)``.

    Returns
    -------
    Dict with l2_norm_mean, l2_norm_max, max_abs_mean, max_abs_max, mean_abs_mean.
    """
    delta_f = delta.float()
    l2_norms = delta_f.norm(dim=-1)  # (batch,)
    max_abs = delta_f.abs().max(dim=-1).values  # (batch,)
    mean_abs = delta_f.abs().mean(dim=-1)  # (batch,)

    return {
        "delta_l2_norm_mean": l2_norms.mean().item(),
        "delta_l2_norm_max": l2_norms.max().item(),
        "delta_max_abs_mean": max_abs.mean().item(),
        "delta_max_abs_max": max_abs.max().item(),
        "delta_mean_abs_mean": mean_abs.mean().item(),
    }


def compute_affine_param_stats(
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> Dict[str, float]:
    """Compute statistics of affine parameters.

    Parameters
    ----------
    scale : torch.Tensor — shape ``(batch,)``
    bias : torch.Tensor — shape ``(batch,)``

    Returns
    -------
    Dict with affine_scale_mean/std, affine_bias_mean/std.
    """
    s = scale.float()
    b = bias.float()
    return {
        "affine_scale_mean": s.mean().item(),
        "affine_scale_std": s.std().item() if s.numel() > 1 else 0.0,
        "affine_bias_mean": b.mean().item(),
        "affine_bias_std": b.std().item() if b.numel() > 1 else 0.0,
    }


# ===================================================================
# Timed Encode Pipeline (Static)
# ===================================================================
def run_timed_static_encode_pipeline(
    real_acts: torch.Tensor,
    token_ids: torch.Tensor,
    static_table: StaticTokenTable,
    group_size: int,
    top_k: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Tuple[DeltaPacket, Dict[str, TimingResult]]:
    """Run the static delta encode pipeline with per-operation GPU timing.

    Treats seq_len as the batch dimension — all existing (batch, hidden_dim)
    functions work directly.

    Parameters
    ----------
    real_acts : torch.Tensor
        Shape ``(seq_len, hidden_dim)`` — real prefill activations.
    token_ids : torch.Tensor
        Shape ``(seq_len,)`` — token IDs for static lookup.
    static_table : StaticTokenTable
    group_size : int
    top_k : int
    num_warmup : int
    num_runs : int

    Returns
    -------
    (DeltaPacket, Dict[str, TimingResult])
    """
    timings: Dict[str, TimingResult] = {}

    # 1. Static lookup
    static_acts, timings["static_lookup"] = gpu_timed(
        static_table.lookup, token_ids,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 2. Affine parameter computation
    (scale, bias), timings["affine_compute"] = gpu_timed(
        compute_affine_params, real_acts, static_acts,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 3. Delta computation (affine transform + subtract)
    def _delta_fn():
        ref_t = apply_affine(static_acts, scale, bias)
        return compute_delta(real_acts, ref_t)

    delta, timings["delta_compute"] = gpu_timed(
        _delta_fn,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 4. Quantize + top-k
    quant_result, timings["quantize_topk"] = gpu_timed(
        groupwise_int4_quantize_topk, delta, group_size, top_k,
        num_warmup=num_warmup, num_runs=num_runs,
    )
    packed, scales, zeros, topk_vals, topk_idx = quant_result

    # 5. Pack into DeltaPacket (token_ids stored in ref_indices)
    def _pack_fn():
        return DeltaPacket(
            quantized_data=packed,
            scales=scales,
            zero_points=zeros,
            topk_values=topk_vals,
            topk_indices=topk_idx,
            affine_scale=scale.to(torch.float16),
            affine_bias=bias.to(torch.float16),
            ref_indices=token_ids.long(),
            group_size=group_size,
            top_k=top_k,
        )

    packet, timings["pack"] = gpu_timed(
        _pack_fn,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 6. Total encode (end-to-end)
    def _full_encode():
        sa = static_table.lookup(token_ids)
        s, b = compute_affine_params(real_acts, sa)
        rt = apply_affine(sa, s, b)
        d = compute_delta(real_acts, rt)
        pk, sc, zp, tv, ti = groupwise_int4_quantize_topk(d, group_size, top_k)
        return DeltaPacket(
            quantized_data=pk, scales=sc, zero_points=zp,
            topk_values=tv, topk_indices=ti,
            affine_scale=s.to(torch.float16), affine_bias=b.to(torch.float16),
            ref_indices=token_ids.long(), group_size=group_size, top_k=top_k,
        )

    _, timings["total_encode"] = gpu_timed(
        _full_encode,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    return packet, timings


# ===================================================================
# Timed Decode Pipeline (Static)
# ===================================================================
def run_timed_static_decode_pipeline(
    packet: DeltaPacket,
    static_table: StaticTokenTable,
    hidden_dim: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Tuple[torch.Tensor, Dict[str, TimingResult]]:
    """Run the static delta decode pipeline with per-operation GPU timing.

    Parameters
    ----------
    packet : DeltaPacket
        ref_indices field contains token IDs for static table lookup.
    static_table : StaticTokenTable
    hidden_dim : int
    num_warmup : int
    num_runs : int

    Returns
    -------
    (reconstructed, Dict[str, TimingResult])
    """
    timings: Dict[str, TimingResult] = {}

    # 1. Static lookup (receiver side)
    static_acts, timings["static_lookup"] = gpu_timed(
        static_table.lookup, packet.ref_indices,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 2. Dequantize + top-k overlay
    dequant, timings["dequantize_topk"] = gpu_timed(
        groupwise_int4_dequantize_topk,
        packet.quantized_data, packet.scales, packet.zero_points,
        packet.topk_values, packet.topk_indices,
        packet.group_size, hidden_dim,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 3. Reconstruction
    scale_f = packet.affine_scale.float()
    bias_f = packet.affine_bias.float()

    reconstructed, timings["reconstruction"] = gpu_timed(
        reconstruct_activation, dequant, static_acts, scale_f, bias_f,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 4. Total decode (end-to-end)
    def _full_decode():
        sa = static_table.lookup(packet.ref_indices)
        dq = groupwise_int4_dequantize_topk(
            packet.quantized_data, packet.scales, packet.zero_points,
            packet.topk_values, packet.topk_indices,
            packet.group_size, hidden_dim,
        )
        return reconstruct_activation(dq, sa, scale_f, bias_f)

    _, timings["total_decode"] = gpu_timed(
        _full_decode,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    return reconstructed, timings
