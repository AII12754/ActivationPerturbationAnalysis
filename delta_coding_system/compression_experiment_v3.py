#!/usr/bin/env python3
"""Round 3 compression experiments.

Focus areas:
1. Measure raw cosine similarity between references and target activations.
2. Estimate how much error energy affine alignment removes.
3. Explore additional sparse delta transmission schemes.
4. Try a few additional reference constructions beyond plain prev-token.

Usage:
  python -m delta_coding_system.compression_experiment_v3 --gpu 0
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from delta_coding_system.compression_experiment import (
    groupwise_int2_dequantize_topk,
    groupwise_int2_quantize_topk,
)
from delta_coding_system.table import NgramTable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("compress_v3")

HIDDEN_DIM = 5120
GROUP_SIZE = 128


def _tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).item()


def _error_energy(target: torch.Tensor, approx: torch.Tensor) -> float:
    diff = (target.float() - approx.float()).pow(2).sum().item()
    return float(diff)


def _norm_energy(tensor: torch.Tensor) -> float:
    return float(tensor.float().pow(2).sum().item())


def _reference_metrics(real_h: torch.Tensor, ref_h: Optional[torch.Tensor]) -> Dict[str, float]:
    if ref_h is None:
        return {
            "raw_ref_cosine": float("nan"),
            "affine_ref_cosine": float("nan"),
            "raw_error_energy": float("nan"),
            "affine_error_energy": float("nan"),
            "affine_energy_reduction": float("nan"),
            "raw_rel_l2": float("nan"),
            "affine_rel_l2": float("nan"),
        }

    scale, bias = compute_affine_params(real_h, ref_h)
    ref_affine = apply_affine(ref_h, scale, bias)
    real_energy = max(_norm_energy(real_h), 1e-8)
    raw_error = _error_energy(real_h, ref_h)
    affine_error = _error_energy(real_h, ref_affine)
    return {
        "raw_ref_cosine": _tensor_cosine(real_h, ref_h),
        "affine_ref_cosine": _tensor_cosine(real_h, ref_affine),
        "raw_error_energy": raw_error,
        "affine_error_energy": affine_error,
        "affine_energy_reduction": 1.0 - affine_error / max(raw_error, 1e-8),
        "raw_rel_l2": (raw_error / real_energy) ** 0.5,
        "affine_rel_l2": (affine_error / real_energy) ** 0.5,
    }


def _quantize_delta(
    delta: torch.Tensor,
    group_size: int,
    top_k: int,
    bits: int,
) -> Tuple[torch.Tensor, int]:
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


def _delta_encode_decode(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
    group_size: int,
    top_k: int,
    bits: int = 4,
    include_ref_idx: bool = True,
    extra_param_bytes: int = 4,
) -> Tuple[torch.Tensor, int, Dict[str, float]]:
    metrics = _reference_metrics(real_h, ref_h)
    scale, bias = compute_affine_params(real_h, ref_h)
    ref_t = apply_affine(ref_h, scale, bias)
    delta = compute_delta(real_h, ref_t)
    dequant, payload_bytes = _quantize_delta(delta, group_size, top_k, bits)
    recon = (ref_t + dequant).to(torch.float16)
    transfer = payload_bytes + extra_param_bytes + (8 if include_ref_idx else 0)
    return recon, transfer, metrics


def _int8_encode_decode(real_h: torch.Tensor, group_size: int, top_k: int) -> Tuple[torch.Tensor, int]:
    pkt = groupwise_int8_quantize_topk(real_h, group_size, top_k)
    recon = groupwise_int8_dequantize_topk(pkt)
    transfer = (
        pkt.quantized.nelement() * 1
        + pkt.scales.nelement() * 2
        + pkt.zero_points.nelement() * 2
        + pkt.topk_values.nelement() * 2
        + pkt.topk_indices.nelement() * 1
    )
    return recon, transfer


def _sparse_group_threshold_encode(
    delta: torch.Tensor,
    group_size: int,
    top_k: int,
    threshold_ratio: float,
    bits: int,
) -> Tuple[torch.Tensor, int, int]:
    batch, hidden_dim = delta.shape
    num_groups = hidden_dim // group_size
    grouped = delta.float().reshape(batch, num_groups, group_size)
    energy = grouped.pow(2).sum(dim=-1)
    max_energy = energy.max(dim=-1, keepdim=True).values
    mask = energy >= max_energy * threshold_ratio
    return _encode_masked_groups(grouped, mask, top_k, bits)


def _sparse_top_groups_encode(
    delta: torch.Tensor,
    group_size: int,
    top_k: int,
    keep_groups: int,
    bits: int,
) -> Tuple[torch.Tensor, int, int]:
    batch, hidden_dim = delta.shape
    num_groups = hidden_dim // group_size
    grouped = delta.float().reshape(batch, num_groups, group_size)
    energy = grouped.pow(2).sum(dim=-1)
    actual_keep = min(keep_groups, num_groups)
    idx = torch.topk(energy, k=actual_keep, dim=-1).indices
    mask = torch.zeros_like(energy, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return _encode_masked_groups(grouped, mask, top_k, bits)


def _encode_masked_groups(
    grouped: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
    bits: int,
) -> Tuple[torch.Tensor, int, int]:
    batch, num_groups, group_size = grouped.shape
    hidden_dim = num_groups * group_size
    recon = torch.zeros_like(grouped)
    total_bytes = 0
    total_active = 0
    bitmask_bytes = (num_groups + 7) // 8

    for batch_idx in range(batch):
        active_mask = mask[batch_idx]
        active_count = int(active_mask.sum().item())
        total_active += active_count
        if active_count == 0:
            total_bytes += bitmask_bytes
            continue

        active = grouped[batch_idx, active_mask].unsqueeze(0)
        abs_vals = active.abs()
        _, tk_idx = abs_vals.topk(top_k, dim=-1)
        tk_vals = active.gather(-1, tk_idx)
        work = active.clone()
        work.scatter_(-1, tk_idx, 0.0)

        if bits == 4:
            qmax = 15.0
            data_bytes = active_count * group_size // 2
        elif bits == 2:
            qmax = 3.0
            data_bytes = active_count * group_size // 4
        else:
            raise ValueError(f"Unsupported bits: {bits}")

        g_min = work.min(dim=-1).values
        g_max = work.max(dim=-1).values
        scale = ((g_max - g_min) / qmax).unsqueeze(-1)
        zp = g_min.unsqueeze(-1)
        q = torch.clamp(torch.round((work - zp) / (scale + 1e-10)), 0, int(qmax))
        dequant = q * scale + zp
        dequant.scatter_(-1, tk_idx, tk_vals)
        recon[batch_idx, active_mask] = dequant.squeeze(0)

        total_bytes += (
            bitmask_bytes
            + data_bytes
            + active_count * 2
            + active_count * 2
            + active_count * top_k * 2
            + active_count * top_k * 1
        )

    return recon.reshape(batch, hidden_dim).to(torch.float16), total_bytes, total_active


def _sparse_top_channels_encode(
    delta: torch.Tensor,
    keep_ratio: float,
) -> Tuple[torch.Tensor, int, int]:
    batch, hidden_dim = delta.shape
    keep = max(1, int(hidden_dim * keep_ratio))
    abs_vals = delta.abs()
    idx = torch.topk(abs_vals, k=keep, dim=-1).indices
    vals = delta.gather(-1, idx)
    recon = torch.zeros_like(delta)
    recon.scatter_(1, idx, vals)
    transfer = batch * (keep * 2 + keep * 2 + 2)
    return recon.to(torch.float16), transfer, keep


def _structured_2of8_encode(delta: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    batch, hidden_dim = delta.shape
    blocks = hidden_dim // 8
    grouped = delta.reshape(batch, blocks, 8)
    abs_vals = grouped.abs()
    idx = torch.topk(abs_vals, k=2, dim=-1).indices
    vals = grouped.gather(-1, idx)
    recon = torch.zeros_like(grouped)
    recon.scatter_(-1, idx, vals)
    transfer = batch * blocks * (2 * 2 + 2)
    kept = blocks * 2
    return recon.reshape(batch, hidden_dim).to(torch.float16), transfer, kept


def _ema_reference(history: List[torch.Tensor], decay: float = 0.7) -> Optional[torch.Tensor]:
    if not history:
        return None
    weights = []
    for idx in range(len(history)):
        age = len(history) - 1 - idx
        weights.append(decay ** age)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=history[0].device)
    weight_tensor = weight_tensor / weight_tensor.sum()
    stacked = torch.cat(history, dim=0).float()
    ref = (stacked * weight_tensor.unsqueeze(-1)).sum(dim=0, keepdim=True)
    return ref.to(torch.float16)


def _solve_two_ref_blend(
    real_h: torch.Tensor,
    ref1: torch.Tensor,
    ref2: torch.Tensor,
) -> Tuple[torch.Tensor, int, Dict[str, float]]:
    features = torch.stack([
        ref1.float().squeeze(0),
        ref2.float().squeeze(0),
        torch.ones_like(ref1.float().squeeze(0)),
    ], dim=1)
    target = real_h.float().squeeze(0)
    solution = torch.linalg.lstsq(features, target).solution
    pred = (
        solution[0] * ref1.float()
        + solution[1] * ref2.float()
        + solution[2]
    ).to(torch.float16)
    metrics = {
        "raw_ref_cosine": _tensor_cosine(real_h, ref1),
        "affine_ref_cosine": _tensor_cosine(real_h, pred),
        "raw_error_energy": _error_energy(real_h, ref1),
        "affine_error_energy": _error_energy(real_h, pred),
        "affine_energy_reduction": 1.0 - _error_energy(real_h, pred) / max(_error_energy(real_h, ref1), 1e-8),
        "raw_rel_l2": (_error_energy(real_h, ref1) / max(_norm_energy(real_h), 1e-8)) ** 0.5,
        "affine_rel_l2": (_error_energy(real_h, pred) / max(_norm_energy(real_h), 1e-8)) ** 0.5,
    }
    return pred, 6, metrics


class StrategyRunner:
    def __init__(self, hidden_dim: int, group_size: int):
        self.hidden_dim = hidden_dim
        self.group_size = group_size

    def run_all(
        self,
        real_h: torch.Tensor,
        tier: str,
        table_ref: Optional[torch.Tensor],
        prev_h: Optional[torch.Tensor],
        prev2_h: Optional[torch.Tensor],
        ema_ref: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        baseline_metrics = _reference_metrics(real_h, table_ref)

        if table_ref is not None:
            recon, transfer, metrics = _delta_encode_decode(real_h, table_ref, self.group_size, 1, 4)
            results["baseline"] = {
                "recon": recon,
                "bytes": transfer,
                "metrics": metrics,
                "active_units": self.hidden_dim // self.group_size,
            }

            recon, transfer, metrics = _delta_encode_decode(real_h, table_ref, self.group_size, 1, 4)
            results["prev_int4_k2"] = {
                "recon": recon,
                "bytes": transfer,
                "metrics": metrics,
                "active_units": self.hidden_dim // self.group_size,
            }

            ref_t = apply_affine(table_ref, *compute_affine_params(real_h, table_ref))
            delta = compute_delta(real_h, ref_t)

            sparse_specs = {
                "delta_sparse_thr_20": lambda: _sparse_group_threshold_encode(delta, self.group_size, 1, 0.20, 4),
                "delta_sparse_thr_10": lambda: _sparse_group_threshold_encode(delta, self.group_size, 1, 0.10, 4),
                "delta_sparse_thr_05": lambda: _sparse_group_threshold_encode(delta, self.group_size, 1, 0.05, 4),
                "delta_top16_groups": lambda: _sparse_top_groups_encode(delta, self.group_size, 1, 16, 4),
                "delta_top8_groups": lambda: _sparse_top_groups_encode(delta, self.group_size, 1, 8, 4),
                "delta_top1024_ch": lambda: _sparse_top_channels_encode(delta, 1024 / self.hidden_dim),
                "delta_top512_ch": lambda: _sparse_top_channels_encode(delta, 512 / self.hidden_dim),
                "delta_struct2of8": lambda: _structured_2of8_encode(delta),
                "delta_int2_top8": lambda: (*_quantize_delta(delta, self.group_size, 8, 2), self.hidden_dim),
            }

            for name, fn in sparse_specs.items():
                sparse_recon, sparse_bytes, active_units = fn()
                recon = (ref_t + sparse_recon).to(torch.float16)
                results[name] = {
                    "recon": recon,
                    "bytes": sparse_bytes + 2 + 2 + 8,
                    "metrics": baseline_metrics,
                    "active_units": active_units,
                }

            if prev_h is not None:
                recon, transfer, metrics = _delta_encode_decode(real_h, prev_h, self.group_size, 2, 4, include_ref_idx=False)
                results["hybrid_sparse_prev"] = {
                    "recon": recon,
                    "bytes": transfer,
                    "metrics": metrics,
                    "active_units": self.hidden_dim // self.group_size,
                }
            else:
                results["hybrid_sparse_prev"] = results["baseline"]

            return results

        baseline_recon, baseline_bytes = _int8_encode_decode(real_h, self.group_size, 1)
        results["baseline"] = {
            "recon": baseline_recon,
            "bytes": baseline_bytes,
            "metrics": {k: float("nan") for k in _reference_metrics(real_h, None)},
            "active_units": self.hidden_dim,
        }

        if prev_h is not None:
            recon, transfer, metrics = _delta_encode_decode(real_h, prev_h, self.group_size, 2, 4, include_ref_idx=False)
            results["prev_int4_k2"] = {
                "recon": recon,
                "bytes": transfer,
                "metrics": metrics,
                "active_units": self.hidden_dim // self.group_size,
            }
        else:
            results["prev_int4_k2"] = results["baseline"]

        if ema_ref is not None:
            recon, transfer, metrics = _delta_encode_decode(real_h, ema_ref, self.group_size, 2, 4, include_ref_idx=False)
            results["ema_prev4_int4_k2"] = {
                "recon": recon,
                "bytes": transfer,
                "metrics": metrics,
                "active_units": self.hidden_dim // self.group_size,
            }
        else:
            results["ema_prev4_int4_k2"] = results["baseline"]

        if prev_h is not None and prev2_h is not None:
            blend_ref, blend_bytes, blend_metrics = _solve_two_ref_blend(real_h, prev_h, prev2_h)
            delta = compute_delta(real_h, blend_ref)
            dequant, payload_bytes = _quantize_delta(delta, self.group_size, 2, 4)
            recon = (blend_ref + dequant).to(torch.float16)
            results["prev2_blend_int4_k2"] = {
                "recon": recon,
                "bytes": payload_bytes + blend_bytes,
                "metrics": blend_metrics,
                "active_units": self.hidden_dim // self.group_size,
            }
        else:
            results["prev2_blend_int4_k2"] = results["baseline"]

        if prev_h is not None:
            recon, transfer, metrics = _delta_encode_decode(real_h, prev_h, 256, 2, 4, include_ref_idx=False)
            results["prev_gs256_k2"] = {
                "recon": recon,
                "bytes": transfer,
                "metrics": metrics,
                "active_units": self.hidden_dim // 256,
            }
        else:
            results["prev_gs256_k2"] = results["baseline"]

        if prev_h is not None:
            results["hybrid_sparse_prev"] = results["prev_int4_k2"]
        else:
            results["hybrid_sparse_prev"] = results["baseline"]

        passthrough = [
            "delta_sparse_thr_20",
            "delta_sparse_thr_10",
            "delta_sparse_thr_05",
            "delta_top16_groups",
            "delta_top8_groups",
            "delta_top1024_ch",
            "delta_top512_ch",
            "delta_struct2of8",
            "delta_int2_top8",
        ]
        for name in passthrough:
            results[name] = results["baseline"]

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

    runner = StrategyRunner(HIDDEN_DIM, GROUP_SIZE)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_records = []
    position_records = []

    for ds_name in args.datasets:
        logger.info("%s", "=" * 60)
        logger.info("Dataset: %s", ds_name)

        texts = load_dataset_texts(ds_name)
        rng = random.Random(args.seed)
        rng.shuffle(texts)

        warmup_n = args.warmup_requests
        test_n = args.test_requests
        total_needed = warmup_n + test_n
        while len(texts) < total_needed:
            texts.extend(texts[: total_needed - len(texts)])

        table = NgramTable(device=device, dtype=torch.float16, max_entries=100000)

        logger.info("Warmup: %d requests...", warmup_n)
        for warm_idx in range(warmup_n):
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
        logger.info("Test: %d requests...", test_n)

        for req_idx in range(test_n):
            text = texts[warmup_n + req_idx]
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
                    a = token_ids[pos - 2].item()
                    b = token_ids[pos - 1].item()
                    c = token_ids[pos].item()
                    tri_ref = table.get_trigram(a, b, c)
                    if tri_ref is not None:
                        tiers.append("trigram")
                        refs.append(tri_ref.to(torch.float16).to(device))
                        found = True
                if not found and pos >= 1:
                    b = token_ids[pos - 1].item()
                    c = token_ids[pos].item()
                    bi_ref = table.get_bigram(b, c)
                    if bi_ref is not None:
                        tiers.append("bigram")
                        refs.append(bi_ref.to(torch.float16).to(device))
                        found = True
                if not found:
                    tiers.append("unigram")
                    refs.append(None)

            strategy_accum = defaultdict(lambda: {
                "bytes": 0,
                "cos_sum": 0.0,
                "cos_min": 1.0,
                "n": 0,
                "raw_ref_cos_sum": 0.0,
                "raw_ref_cos_n": 0,
                "affine_ref_cos_sum": 0.0,
                "affine_ref_cos_n": 0,
                "energy_red_sum": 0.0,
                "energy_red_n": 0,
                "raw_rel_l2_sum": 0.0,
                "raw_rel_l2_n": 0,
                "affine_rel_l2_sum": 0.0,
                "affine_rel_l2_n": 0,
                "active_units_sum": 0.0,
            })

            for pos in range(seq_len):
                real = h[pos:pos + 1].to(torch.float16)
                table_ref = refs[pos]
                prev_h = h[pos - 1:pos].to(torch.float16) if pos > 0 else None
                prev2_h = h[pos - 2:pos - 1].to(torch.float16) if pos > 1 else None
                start = max(0, pos - 4)
                history = [h[idx:idx + 1].to(torch.float16) for idx in range(start, pos)]
                ema_ref = _ema_reference(history)

                results = runner.run_all(real, tiers[pos], table_ref, prev_h, prev2_h, ema_ref)

                for strategy_name, result in results.items():
                    recon = result["recon"]
                    transfer = int(result["bytes"])
                    metrics = result["metrics"]
                    active_units = int(result["active_units"])

                    cosine = _tensor_cosine(real, recon)
                    acc = strategy_accum[strategy_name]
                    acc["bytes"] += transfer
                    acc["cos_sum"] += cosine
                    acc["cos_min"] = min(acc["cos_min"], cosine)
                    acc["n"] += 1
                    acc["active_units_sum"] += active_units

                    for metric_name, sum_key, count_key in [
                        ("raw_ref_cosine", "raw_ref_cos_sum", "raw_ref_cos_n"),
                        ("affine_ref_cosine", "affine_ref_cos_sum", "affine_ref_cos_n"),
                        ("affine_energy_reduction", "energy_red_sum", "energy_red_n"),
                        ("raw_rel_l2", "raw_rel_l2_sum", "raw_rel_l2_n"),
                        ("affine_rel_l2", "affine_rel_l2_sum", "affine_rel_l2_n"),
                    ]:
                        value = metrics[metric_name]
                        if value == value:
                            acc[sum_key] += value
                            acc[count_key] += 1

                    if req_idx < args.detail_requests:
                        position_records.append({
                            "dataset": ds_name,
                            "request_index": req_idx,
                            "position": pos,
                            "tier": tiers[pos],
                            "strategy": strategy_name,
                            "cosine": cosine,
                            "transfer_bytes": transfer,
                            "raw_ref_cosine": metrics["raw_ref_cosine"],
                            "affine_ref_cosine": metrics["affine_ref_cosine"],
                            "affine_energy_reduction": metrics["affine_energy_reduction"],
                            "raw_rel_l2": metrics["raw_rel_l2"],
                            "affine_rel_l2": metrics["affine_rel_l2"],
                            "active_units": active_units,
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
                    "raw_ref_cosine_mean": acc["raw_ref_cos_sum"] / max(acc["raw_ref_cos_n"], 1),
                    "affine_ref_cosine_mean": acc["affine_ref_cos_sum"] / max(acc["affine_ref_cos_n"], 1),
                    "affine_energy_reduction_mean": acc["energy_red_sum"] / max(acc["energy_red_n"], 1),
                    "raw_rel_l2_mean": acc["raw_rel_l2_sum"] / max(acc["raw_rel_l2_n"], 1),
                    "affine_rel_l2_mean": acc["affine_rel_l2_sum"] / max(acc["affine_rel_l2_n"], 1),
                    "active_units_mean": acc["active_units_sum"] / max(acc["n"], 1),
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
                logger.info("  Test %d/%d complete", req_idx + 1, test_n)

        gc.collect()
        torch.cuda.empty_cache()

    pq.write_table(pa.Table.from_pylist(request_records), str(output_dir / "strategies_v3.parquet"))
    if position_records:
        pq.write_table(pa.Table.from_pylist(position_records), str(output_dir / "position_detail_v3.parquet"))
    logger.info("Saved %d request records", len(request_records))

    import pandas as pd

    df = pd.DataFrame(request_records)
    summary = df.groupby("strategy").agg(
        cr=("compression_ratio", "mean"),
        cos=("cosine_mean", "mean"),
        raw_cos=("raw_ref_cosine_mean", "mean"),
        affine_cos=("affine_ref_cosine_mean", "mean"),
        energy_red=("affine_energy_reduction_mean", "mean"),
        active=("active_units_mean", "mean"),
    ).sort_values("cr", ascending=False)

    print("\n" + "=" * 90)
    print("ROUND 3: RAW REFERENCE / SPARSE DELTA EXPERIMENTS")
    print("=" * 90)
    print(f"\n{'Strategy':<22s} {'Ratio':>8s} {'Cos':>9s} {'RawCos':>9s} {'AffCos':>9s} {'EnergyRed':>10s} {'Active':>8s}")
    print("-" * 90)
    for name, row in summary.iterrows():
        print(
            f"{name:<22s} {row['cr']:>7.3f}x {row['cos']:>9.6f} {row['raw_cos']:>9.6f} "
            f"{row['affine_cos']:>9.6f} {row['energy_red']:>10.4f} {row['active']:>8.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Round 3 compression experiments")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=30)
    parser.add_argument("--detail-requests", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_compression_exp_v3")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext2", "sharegpt", "gsm8k", "cnn_dm", "alpaca", "triviaqa"],
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()