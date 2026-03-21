#!/usr/bin/env python3
"""Comprehensive compression improvement experiments.

Tests multiple strategies to improve compression ratio beyond the baseline 2.52×:
  1. Baseline (current: Int4 delta + Int8 unigram)
  2. Sparse group bitmask — skip near-zero delta groups
  3. Int2 delta quantization — for high-quality trigram matches
  4. Unigram → Int4 — downgrade unigram from Int8 to Int4
  5. Previous-token reference — use adjacent hidden as ref for unigram
  6. Adaptive bit-width — Int2 for trigram, Int4 for rest
  7. Sparse + prev-token combined
  8. All-combined best strategy
  9. Small linear predictor replacing affine

Usage:
  python -m delta_coding_system.compression_experiment --gpu 0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.table import NgramTable
from delta_coding_system.codec import (
    compute_affine_params,
    apply_affine,
    compute_delta,
    groupwise_int4_quantize_topk,
    groupwise_int4_dequantize_topk,
    groupwise_int8_quantize_topk,
    groupwise_int8_dequantize_topk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("compress_exp")

HIDDEN_DIM = 5120
GROUP_SIZE = 128
NUM_GROUPS = HIDDEN_DIM // GROUP_SIZE  # 40

# ===================================================================
# Quantization primitives for new strategies
# ===================================================================

def groupwise_int2_quantize_topk(
    delta: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise Int2 quantization (4 levels: 0,1,2,3) with top-k outlier extraction."""
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

    # Pack 4 values per byte (2 bits each)
    q_flat = q.reshape(batch, hidden_dim)
    # Pack: val0 in bits [7:6], val1 in [5:4], val2 in [3:2], val3 in [1:0]
    packed = (q_flat[:, 0::4] << 6) | (q_flat[:, 1::4] << 4) | (q_flat[:, 2::4] << 2) | q_flat[:, 3::4]

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
    """Dequantize Int2 groups and overlay top-k outliers."""
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


def sparse_group_encode(
    delta: torch.Tensor,
    group_size: int,
    top_k: int,
    threshold_ratio: float = 0.1,
    quant_bits: int = 4,
) -> Tuple[torch.Tensor, int]:
    """Sparse group encoding: skip groups with small energy.

    Returns (reconstructed, transfer_bytes).
    """
    batch, hidden_dim = delta.shape
    num_groups = hidden_dim // group_size
    grouped = delta.float().reshape(batch, num_groups, group_size)

    # Compute per-group energy (L2 norm)
    group_energy = (grouped ** 2).sum(dim=-1).sqrt()  # (batch, num_groups)
    max_energy = group_energy.max(dim=-1, keepdim=True).values
    # Threshold: skip groups with < threshold_ratio of max energy
    group_mask = (group_energy > max_energy * threshold_ratio)  # (batch, num_groups)

    # Count active groups per batch item
    num_active = group_mask.sum(dim=-1)  # (batch,)

    # Reconstruct with only active groups
    recon_grouped = torch.zeros_like(grouped)

    total_bytes = 0
    for b in range(batch):
        mask_b = group_mask[b]  # (num_groups,)
        n_active = mask_b.sum().item()

        # Bitmask: ceil(num_groups / 8) bytes
        bitmask_bytes = (num_groups + 7) // 8  # 5 bytes for 40 groups

        if n_active > 0:
            active_groups = grouped[b, mask_b].unsqueeze(0)  # (1, n_active, group_size)

            # Top-k extraction per active group
            abs_vals = active_groups.abs()
            _, tk_idx = abs_vals.topk(top_k, dim=-1)
            tk_vals = active_groups.gather(-1, tk_idx)
            active_groups_zeroed = active_groups.clone()
            active_groups_zeroed.scatter_(-1, tk_idx, 0.0)

            g_min = active_groups_zeroed.min(dim=-1).values
            g_max = active_groups_zeroed.max(dim=-1).values

            if quant_bits == 4:
                max_val = 15
            elif quant_bits == 2:
                max_val = 3
            else:
                max_val = 255

            scale = ((g_max - g_min) / max_val).unsqueeze(-1)
            zp = g_min.unsqueeze(-1)
            q = torch.clamp(torch.round((active_groups_zeroed - zp) / (scale + 1e-10)), 0, max_val)

            # Dequantize
            dequant_active = q * scale + zp
            dequant_active.scatter_(-1, tk_idx, tk_vals)

            recon_grouped[b, mask_b] = dequant_active.squeeze(0)

            # Compute bytes for active groups
            if quant_bits == 4:
                data_bytes = n_active * group_size // 2  # Int4 packed
            elif quant_bits == 2:
                data_bytes = n_active * group_size // 4  # Int2 packed
            else:
                data_bytes = n_active * group_size

            scale_bytes = n_active * 2  # FP16
            zp_bytes = n_active * 2     # FP16
            tk_val_bytes = n_active * top_k * 2  # FP16
            tk_idx_bytes = n_active * top_k * 1  # uint8

            total_bytes += bitmask_bytes + data_bytes + scale_bytes + zp_bytes + tk_val_bytes + tk_idx_bytes

    recon = recon_grouped.reshape(batch, hidden_dim).to(torch.float16)
    return recon, total_bytes


def train_linear_predictor(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    hidden_dim: int,
    device: torch.device,
    n_components: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Train a small linear predictor: new ≈ W @ ref + b.

    Uses low-rank factorization: W = A @ B where A is (H, r) and B is (r, H).
    Returns (AB_matrix, bias) for inference.
    """
    if len(pairs) < 10:
        # Not enough data; fall back to identity
        return torch.eye(hidden_dim, dtype=torch.float16, device=device), \
               torch.zeros(hidden_dim, dtype=torch.float16, device=device)

    # Stack pairs
    refs = torch.cat([p[1] for p in pairs], dim=0).float()  # (N, H)
    news = torch.cat([p[0] for p in pairs], dim=0).float()  # (N, H)

    # Channel-wise linear regression: for each channel i, new_i = w_i * ref_i + b_i
    # This is much more efficient than full matrix and captures per-channel scaling
    ref_mean = refs.mean(dim=0)
    new_mean = news.mean(dim=0)
    ref_centered = refs - ref_mean
    new_centered = news - new_mean

    # Per-channel slope: w_i = sum(ref_i * new_i) / sum(ref_i^2)
    numerator = (ref_centered * new_centered).sum(dim=0)
    denominator = (ref_centered ** 2).sum(dim=0) + 1e-8
    w = numerator / denominator  # (H,)
    b = new_mean - w * ref_mean  # (H,)

    return w.to(torch.float16).to(device), b.to(torch.float16).to(device)


# ===================================================================
# Strategy functions
# ===================================================================

def strategy_baseline(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
) -> Tuple[torch.Tensor, int, str]:
    """Baseline: Int4 delta + Int8 unigram. Returns (recon, bytes, label)."""
    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
        scale, bias = compute_affine_params(real_h, ref_h)
        ref_t = apply_affine(ref_h, scale, bias)
        delta = compute_delta(real_h, ref_t)
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, GROUP_SIZE, 1)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, GROUP_SIZE, HIDDEN_DIM)
        recon = (ref_t + dequant).to(torch.float16)

        transfer = (packed.nelement() * 1  # uint8
                    + scales.nelement() * 2  # fp16
                    + zeros.nelement() * 2
                    + tv.nelement() * 2
                    + ti.nelement() * 1
                    + 2 + 2 + 8)  # affine_scale, affine_bias, ref_idx
        return recon, transfer, "baseline_delta"
    else:
        pkt = groupwise_int8_quantize_topk(real_h, GROUP_SIZE, 1)
        recon = groupwise_int8_dequantize_topk(pkt)
        transfer = (pkt.quantized.nelement() * 1
                    + pkt.scales.nelement() * 2
                    + pkt.zero_points.nelement() * 2
                    + pkt.topk_values.nelement() * 2
                    + pkt.topk_indices.nelement() * 1)
        return recon, transfer, "baseline_int8"


def strategy_sparse_group(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
    threshold: float = 0.1,
) -> Tuple[torch.Tensor, int, str]:
    """Sparse group bitmask: skip near-zero delta groups."""
    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
        scale, bias = compute_affine_params(real_h, ref_h)
        ref_t = apply_affine(ref_h, scale, bias)
        delta = compute_delta(real_h, ref_t)
        recon_delta, data_bytes = sparse_group_encode(delta, GROUP_SIZE, 1, threshold, quant_bits=4)
        recon = (ref_t + recon_delta).to(torch.float16)
        transfer = data_bytes + 2 + 2 + 8  # affine_scale, affine_bias, ref_idx
        return recon, transfer, "sparse_delta"
    else:
        return strategy_baseline(real_h, ref_h, tier)


def strategy_int2_delta(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
    top_k: int = 4,
) -> Tuple[torch.Tensor, int, str]:
    """Int2 (4 levels) for delta-coded positions, higher top_k to compensate."""
    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
        scale, bias = compute_affine_params(real_h, ref_h)
        ref_t = apply_affine(ref_h, scale, bias)
        delta = compute_delta(real_h, ref_t)
        packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(delta, GROUP_SIZE, top_k)
        dequant = groupwise_int2_dequantize_topk(packed, scales, zeros, tv, ti, GROUP_SIZE, HIDDEN_DIM)
        recon = (ref_t + dequant).to(torch.float16)

        transfer = (packed.nelement() * 1  # uint8
                    + scales.nelement() * 2
                    + zeros.nelement() * 2
                    + tv.nelement() * 2
                    + ti.nelement() * 1
                    + 2 + 2 + 8)
        return recon, transfer, "int2_delta"
    else:
        return strategy_baseline(real_h, ref_h, tier)


def strategy_unigram_int4(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
    top_k_uni: int = 4,
) -> Tuple[torch.Tensor, int, str]:
    """Unigram positions use Int4+higher top_k instead of Int8."""
    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
        return strategy_baseline(real_h, ref_h, tier)
    else:
        # Int4 with higher top_k for unigram
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(real_h, GROUP_SIZE, top_k_uni)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, GROUP_SIZE, HIDDEN_DIM)
        transfer = (packed.nelement() * 1
                    + scales.nelement() * 2
                    + zeros.nelement() * 2
                    + tv.nelement() * 2
                    + ti.nelement() * 1)
        return dequant, transfer, "unigram_int4"


def strategy_prev_token_ref(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
    prev_h: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, str]:
    """Use previous token's hidden as reference for unigram positions."""
    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
        return strategy_baseline(real_h, ref_h, tier)
    elif prev_h is not None:
        # Use prev token hidden as reference
        scale, bias = compute_affine_params(real_h, prev_h)
        ref_t = apply_affine(prev_h, scale, bias)
        delta = compute_delta(real_h, ref_t)
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, GROUP_SIZE, 1)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, GROUP_SIZE, HIDDEN_DIM)
        recon = (ref_t + dequant).to(torch.float16)
        transfer = (packed.nelement() * 1
                    + scales.nelement() * 2
                    + zeros.nelement() * 2
                    + tv.nelement() * 2
                    + ti.nelement() * 1
                    + 2 + 2)  # affine_scale, affine_bias (no ref_idx needed, always prev)
        return recon, transfer, "prev_token_ref"
    else:
        return strategy_baseline(real_h, ref_h, tier)


def strategy_adaptive_bitwidth(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
) -> Tuple[torch.Tensor, int, str]:
    """Adaptive: Int2 for trigram (best match), Int4 for bigram/self_ref."""
    if tier == "trigram" and ref_h is not None:
        return strategy_int2_delta(real_h, ref_h, tier, top_k=4)
    elif tier in ("bigram", "self_ref") and ref_h is not None:
        return strategy_baseline(real_h, ref_h, tier)
    else:
        return strategy_baseline(real_h, ref_h, tier)


def strategy_sparse_prevtoken(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
    prev_h: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, str]:
    """Combined: sparse group for delta + prev-token ref for unigram."""
    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
        return strategy_sparse_group(real_h, ref_h, tier)
    elif prev_h is not None:
        return strategy_prev_token_ref(real_h, ref_h, tier, prev_h)
    else:
        return strategy_baseline(real_h, ref_h, tier)


def strategy_combined_best(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
    prev_h: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, str]:
    """Combined best: sparse+Int2 for trigram, sparse+Int4 for bigram, prev-token for unigram."""
    if tier == "trigram" and ref_h is not None:
        # Sparse + Int2, top_k=4
        scale, bias = compute_affine_params(real_h, ref_h)
        ref_t = apply_affine(ref_h, scale, bias)
        delta = compute_delta(real_h, ref_t)
        recon_delta, data_bytes = sparse_group_encode(delta, GROUP_SIZE, 4, 0.1, quant_bits=2)
        recon = (ref_t + recon_delta).to(torch.float16)
        transfer = data_bytes + 2 + 2 + 8
        return recon, transfer, "combined_tri"
    elif tier in ("bigram", "self_ref") and ref_h is not None:
        # Sparse + Int4
        return strategy_sparse_group(real_h, ref_h, tier)
    elif prev_h is not None:
        # Prev-token reference + Int4
        return strategy_prev_token_ref(real_h, ref_h, tier, prev_h)
    else:
        # Fallback: Int4 with top_k=4
        return strategy_unigram_int4(real_h, ref_h, tier, top_k_uni=4)


def strategy_linear_predictor(
    real_h: torch.Tensor,
    ref_h: Optional[torch.Tensor],
    tier: str,
    predictor_w: Optional[torch.Tensor] = None,
    predictor_b: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int, str]:
    """Small learned channel-wise linear predictor instead of per-sample affine."""
    if tier in ("trigram", "bigram", "self_ref") and ref_h is not None and predictor_w is not None:
        # Use channel-wise linear: pred = w * ref + b
        pred = (predictor_w.unsqueeze(0) * ref_h + predictor_b.unsqueeze(0)).to(torch.float16)
        delta = compute_delta(real_h, pred)
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, GROUP_SIZE, 1)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, GROUP_SIZE, HIDDEN_DIM)
        recon = (pred + dequant).to(torch.float16)

        # No affine_scale/bias needed (predictor is shared), just ref_idx + quantized data
        transfer = (packed.nelement() * 1
                    + scales.nelement() * 2
                    + zeros.nelement() * 2
                    + tv.nelement() * 2
                    + ti.nelement() * 1
                    + 8)  # ref_idx only, no per-sample affine
        return recon, transfer, "linear_pred"
    else:
        return strategy_baseline(real_h, ref_h, tier)


# ===================================================================
# Main experiment
# ===================================================================

def run_experiment(args):
    """Run the full compression experiment."""
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # Load model
    logger.info("Loading model on GPU %d...", args.gpu)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map={"": device}, trust_remote_code=True,
    )
    model.eval()
    hidden_dim = model.config.hidden_size
    global HIDDEN_DIM, NUM_GROUPS
    HIDDEN_DIM = hidden_dim
    NUM_GROUPS = hidden_dim // GROUP_SIZE
    logger.info("Model loaded. hidden_dim=%d", hidden_dim)

    # GPU warmup
    wids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    _ = model(wids, output_hidden_states=True, use_cache=False)
    del wids, _
    torch.cuda.synchronize()

    # Load datasets
    from delta_coding_system.run_experiment import load_dataset_texts
    datasets_to_test = args.datasets
    logger.info("Datasets: %s", datasets_to_test)

    # Create table for classification
    table = NgramTable(
        device=device,
        dtype=torch.float16,
        max_entries=100000,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    for ds_name in datasets_to_test:
        logger.info("=" * 60)
        logger.info("Dataset: %s", ds_name)
        logger.info("=" * 60)

        texts = load_dataset_texts(ds_name)
        rng = random.Random(args.seed)
        rng.shuffle(texts)

        warmup_n = args.warmup_requests
        test_n = args.test_requests
        total_needed = warmup_n + test_n
        while len(texts) < total_needed:
            texts.extend(texts[:total_needed - len(texts)])

        warmup_texts = texts[:warmup_n]
        test_texts = texts[warmup_n:warmup_n + test_n]

        # Reset table
        table = NgramTable(device=device, dtype=torch.float16, max_entries=100000)

        # ---- Warmup: build table ----
        logger.info("Warmup: %d requests...", warmup_n)
        for i, text in enumerate(warmup_texts):
            input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=args.max_seq_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary]  # (1, seq_len, H)
            h = h.squeeze(0)  # (seq_len, H)
            token_ids = input_ids.squeeze(0)  # (seq_len,)

            # Build references and update table
            seq_len = token_ids.shape[0]
            if seq_len >= 3:
                trigrams = token_ids.unfold(0, 3, 1)  # (seq_len-2, 3)
                for t_idx in range(trigrams.shape[0]):
                    tri = trigrams[t_idx]
                    table._get_or_create_node(
                        tri[0].item(), tri[1].item(),
                        h[t_idx + 1].unsqueeze(0).to(torch.float16),
                    )
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[t_idx + 2].unsqueeze(0).to(torch.float16)

            del out, h
            if (i + 1) % 10 == 0:
                logger.info("  warmup %d/%d  table: %d tri, %d bi",
                            i + 1, warmup_n, table.stats["num_trigrams"], table.stats["num_bigrams"])

        logger.info("Warmup done. Table: %d tri, %d bi",
                    table.stats["num_trigrams"], table.stats["num_bigrams"])

        # ---- Collect training pairs for linear predictor ----
        train_pairs = []

        # ---- Test ----
        logger.info("Test: %d requests...", test_n)
        for req_idx, text in enumerate(test_texts):
            input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=args.max_seq_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary].squeeze(0)  # (seq_len, H)
            token_ids = input_ids.squeeze(0)
            seq_len = token_ids.shape[0]

            # Classify each position
            tiers = []
            refs = []
            for pos in range(seq_len):
                if pos >= 2:
                    a, b, c = token_ids[pos - 2].item(), token_ids[pos - 1].item(), token_ids[pos].item()
                    tri_ref = table.get_trigram(a, b, c)
                    if tri_ref is not None:
                        tiers.append("trigram")
                        refs.append(tri_ref.to(torch.float16).to(device))
                        continue
                if pos >= 1:
                    b, c = token_ids[pos - 1].item(), token_ids[pos].item()
                    bi_ref = table.get_bigram(b, c)
                    if bi_ref is not None:
                        tiers.append("bigram")
                        refs.append(bi_ref.to(torch.float16).to(device))
                        continue
                tiers.append("unigram")
                refs.append(None)

            # Collect training pairs for linear predictor (first 5 requests)
            if req_idx < 5:
                for pos in range(seq_len):
                    if refs[pos] is not None:
                        train_pairs.append((h[pos:pos+1].to(torch.float16), refs[pos]))

            # Train linear predictor after collecting enough data
            if req_idx == 5 and train_pairs:
                logger.info("  Training linear predictor with %d pairs...", len(train_pairs))
                pred_w, pred_b = train_linear_predictor(train_pairs, hidden_dim, device)
            elif req_idx < 5:
                pred_w, pred_b = None, None

            # Per-position results for all strategies
            raw_bytes = seq_len * hidden_dim * 2  # FP16

            strategy_results = defaultdict(lambda: {"bytes": 0, "cos_sum": 0.0, "cos_min": 1.0, "count": 0})

            strategies = {
                "baseline": lambda rh, refh, t, ph: strategy_baseline(rh, refh, t),
                "sparse_010": lambda rh, refh, t, ph: strategy_sparse_group(rh, refh, t, 0.10),
                "sparse_005": lambda rh, refh, t, ph: strategy_sparse_group(rh, refh, t, 0.05),
                "int2_topk4": lambda rh, refh, t, ph: strategy_int2_delta(rh, refh, t, top_k=4),
                "int2_topk8": lambda rh, refh, t, ph: strategy_int2_delta(rh, refh, t, top_k=8),
                "unigram_int4_k4": lambda rh, refh, t, ph: strategy_unigram_int4(rh, refh, t, top_k_uni=4),
                "unigram_int4_k8": lambda rh, refh, t, ph: strategy_unigram_int4(rh, refh, t, top_k_uni=8),
                "prev_token": lambda rh, refh, t, ph: strategy_prev_token_ref(rh, refh, t, ph),
                "adaptive_bw": lambda rh, refh, t, ph: strategy_adaptive_bitwidth(rh, refh, t),
                "sparse_prev": lambda rh, refh, t, ph: strategy_sparse_prevtoken(rh, refh, t, ph),
                "combined": lambda rh, refh, t, ph: strategy_combined_best(rh, refh, t, ph),
            }
            if pred_w is not None:
                strategies["linear_pred"] = lambda rh, refh, t, ph: strategy_linear_predictor(
                    rh, refh, t, pred_w, pred_b)

            for pos in range(seq_len):
                real = h[pos:pos+1].to(torch.float16)
                ref = refs[pos]
                tier = tiers[pos]
                prev = h[pos-1:pos].to(torch.float16) if pos > 0 else None

                for s_name, s_fn in strategies.items():
                    recon, transfer, label = s_fn(real, ref, tier, prev)
                    cos = F.cosine_similarity(real.float(), recon.float(), dim=-1).item()
                    sr = strategy_results[s_name]
                    sr["bytes"] += transfer
                    sr["cos_sum"] += cos
                    sr["cos_min"] = min(sr["cos_min"], cos)
                    sr["count"] += 1

            # Record per-request results
            tier_counts = {"trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0}
            for t in tiers:
                tier_counts[t] = tier_counts.get(t, 0) + 1

            for s_name, sr in strategy_results.items():
                record = {
                    "dataset": ds_name,
                    "request_index": req_idx,
                    "strategy": s_name,
                    "seq_len": seq_len,
                    "raw_fp16_bytes": raw_bytes,
                    "total_transfer_bytes": sr["bytes"],
                    "compression_ratio": raw_bytes / max(sr["bytes"], 1),
                    "cosine_mean": sr["cos_sum"] / max(sr["count"], 1),
                    "cosine_min": sr["cos_min"],
                    "num_trigram": tier_counts.get("trigram", 0),
                    "num_bigram": tier_counts.get("bigram", 0),
                    "num_self_ref": tier_counts.get("self_ref", 0),
                    "num_unigram": tier_counts.get("unigram", 0),
                    "pct_trigram": tier_counts.get("trigram", 0) / max(seq_len, 1) * 100,
                    "pct_bigram": tier_counts.get("bigram", 0) / max(seq_len, 1) * 100,
                    "pct_unigram": tier_counts.get("unigram", 0) / max(seq_len, 1) * 100,
                }
                all_records.append(record)

            # Update table with new data
            if seq_len >= 3:
                trigrams = token_ids.unfold(0, 3, 1)
                for t_idx in range(trigrams.shape[0]):
                    tri = trigrams[t_idx]
                    table._get_or_create_node(
                        tri[0].item(), tri[1].item(),
                        h[t_idx + 1].unsqueeze(0).to(torch.float16),
                    )
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[t_idx + 2].unsqueeze(0).to(torch.float16)

            del out, h

            if (req_idx + 1) % 5 == 0:
                # Print progress
                base_cr = 0
                for r in all_records[-len(strategies):]:
                    if r["strategy"] == "baseline":
                        base_cr = r["compression_ratio"]
                        break
                logger.info("  Test %d/%d  baseline_ratio=%.2fx  seq_len=%d",
                            req_idx + 1, test_n, base_cr, seq_len)

        gc.collect()
        torch.cuda.empty_cache()

    # Save all records
    out_path = output_dir / "compression_strategies.parquet"
    tbl = pa.Table.from_pylist(all_records)
    pq.write_table(tbl, str(out_path))
    logger.info("Saved %d records to %s", len(all_records), out_path)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPRESSION STRATEGY COMPARISON")
    print("=" * 80)

    import pandas as pd
    df = pd.DataFrame(all_records)
    summary = df.groupby("strategy").agg({
        "compression_ratio": "mean",
        "cosine_mean": "mean",
        "cosine_min": "mean",
    }).sort_values("compression_ratio", ascending=False)

    print(f"\n{'Strategy':<25s} {'Compression':>12s} {'Cosine Mean':>12s} {'Cosine Min':>12s}")
    print("-" * 65)
    for name, row in summary.iterrows():
        print(f"{name:<25s} {row['compression_ratio']:>10.3f}x {row['cosine_mean']:>12.6f} {row['cosine_min']:>12.6f}")

    # Per-dataset breakdown
    print("\n\nPer-Dataset Breakdown:")
    for ds in df["dataset"].unique():
        print(f"\n--- {ds} ---")
        ds_df = df[df["dataset"] == ds]
        ds_summary = ds_df.groupby("strategy").agg({
            "compression_ratio": "mean",
            "cosine_mean": "mean",
        }).sort_values("compression_ratio", ascending=False)
        for name, row in ds_summary.iterrows():
            print(f"  {name:<25s} {row['compression_ratio']:>8.3f}x  cos={row['cosine_mean']:.6f}")

    logger.info("Done!")
    return df


def main():
    parser = argparse.ArgumentParser(description="Compression improvement experiments")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_compression_exp")
    parser.add_argument("--datasets", nargs="+",
                        default=["wikitext2", "sharegpt", "gsm8k", "cnn_dm", "alpaca", "triviaqa"])
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
