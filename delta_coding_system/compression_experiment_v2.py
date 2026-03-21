#!/usr/bin/env python3
"""Round 2: Refined compression experiments focusing on prev-token variants.

Tests variations of the prev-token reference strategy:
  1. prev_token_int4_k1 — baseline prev-token (Int4, top_k=1)
  2. prev_token_int4_k2 — prev-token with top_k=2
  3. prev_token_int4_k4 — prev-token with top_k=4
  4. prev_token_int2_k4 — prev-token with Int2, top_k=4
  5. prev_token_int2_k8 — prev-token with Int2, top_k=8
  6. mean_pool_ref — running mean of last 4 tokens as reference
  7. best_combined — Int4 delta + prev-token Int4 k2
  8. best_combined_sparse — same + sparse groups for delta
  9. prev_token_adaptive — Int4 k2 for prev-token + Int4 k1 for delta

Usage:
  python delta_coding_system/compression_experiment_v2.py --gpu 0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
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
from delta_coding_system.compression_experiment import (
    groupwise_int2_quantize_topk,
    groupwise_int2_dequantize_topk,
    sparse_group_encode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("compress_v2")

HIDDEN_DIM = 5120
GROUP_SIZE = 128
NUM_GROUPS = HIDDEN_DIM // GROUP_SIZE


def _delta_encode_decode(real_h, ref_h, group_size, top_k, bits=4):
    """Encode delta with affine + quantization. Returns (recon, transfer_bytes)."""
    scale, bias = compute_affine_params(real_h, ref_h)
    ref_t = apply_affine(ref_h, scale, bias)
    delta = compute_delta(real_h, ref_t)

    if bits == 4:
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, real_h.shape[-1])
        data_bytes = packed.nelement() * 1
    elif bits == 2:
        packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(delta, group_size, top_k)
        dequant = groupwise_int2_dequantize_topk(packed, scales, zeros, tv, ti, group_size, real_h.shape[-1])
        data_bytes = packed.nelement() * 1
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    recon = (ref_t + dequant).to(torch.float16)
    transfer = (data_bytes
                + scales.nelement() * 2
                + zeros.nelement() * 2
                + tv.nelement() * 2
                + ti.nelement() * 1
                + 2 + 2 + 8)  # affine_scale, bias, ref_idx
    return recon, transfer


def _prevtoken_encode_decode(real_h, prev_h, group_size, top_k, bits=4):
    """Encode using prev token as reference. No ref_idx needed."""
    scale, bias = compute_affine_params(real_h, prev_h)
    ref_t = apply_affine(prev_h, scale, bias)
    delta = compute_delta(real_h, ref_t)

    if bits == 4:
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)
        dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, real_h.shape[-1])
        data_bytes = packed.nelement() * 1
    elif bits == 2:
        packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(delta, group_size, top_k)
        dequant = groupwise_int2_dequantize_topk(packed, scales, zeros, tv, ti, group_size, real_h.shape[-1])
        data_bytes = packed.nelement() * 1
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    recon = (ref_t + dequant).to(torch.float16)
    transfer = (data_bytes
                + scales.nelement() * 2
                + zeros.nelement() * 2
                + tv.nelement() * 2
                + ti.nelement() * 1
                + 2 + 2)  # affine_scale, bias only (implicit prev ref)
    return recon, transfer


def _int8_encode_decode(real_h, group_size, top_k):
    """Int8 unigram encoding."""
    pkt = groupwise_int8_quantize_topk(real_h, group_size, top_k)
    recon = groupwise_int8_dequantize_topk(pkt)
    transfer = (pkt.quantized.nelement() * 1
                + pkt.scales.nelement() * 2
                + pkt.zero_points.nelement() * 2
                + pkt.topk_values.nelement() * 2
                + pkt.topk_indices.nelement() * 1)
    return recon, transfer


def _mean_pool_ref(h_buffer, pos, window=4):
    """Get mean-pooled reference from last `window` positions."""
    start = max(0, pos - window)
    if start == pos:
        return None
    ref = h_buffer[start:pos].mean(dim=0, keepdim=True).to(torch.float16)
    return ref


class StrategyRunner:
    """Runs all strategy variants for a single position."""

    def __init__(self, hidden_dim, group_size):
        self.hidden_dim = hidden_dim
        self.group_size = group_size

    def run_all(self, real_h, ref_h, tier, prev_h, mean_ref):
        """Returns dict of {strategy_name: (recon, transfer_bytes)}."""
        results = {}
        gs = self.group_size
        hd = self.hidden_dim

        # 1. Baseline: Int4 delta + Int8 unigram
        if ref_h is not None:
            results["baseline"] = _delta_encode_decode(real_h, ref_h, gs, 1, 4)
        else:
            results["baseline"] = _int8_encode_decode(real_h, gs, 1)

        # 2-5. prev_token variants (only for unigram; delta positions use delta)
        if ref_h is not None:
            # For delta positions: all strategies use the same delta encoding
            for name in ["prev_int4_k1", "prev_int4_k2", "prev_int4_k4",
                         "prev_int2_k4", "prev_int2_k8"]:
                results[name] = results["baseline"]  # same for delta positions
        elif prev_h is not None:
            results["prev_int4_k1"] = _prevtoken_encode_decode(real_h, prev_h, gs, 1, 4)
            results["prev_int4_k2"] = _prevtoken_encode_decode(real_h, prev_h, gs, 2, 4)
            results["prev_int4_k4"] = _prevtoken_encode_decode(real_h, prev_h, gs, 4, 4)
            results["prev_int2_k4"] = _prevtoken_encode_decode(real_h, prev_h, gs, 4, 2)
            results["prev_int2_k8"] = _prevtoken_encode_decode(real_h, prev_h, gs, 8, 2)
        else:
            for name in ["prev_int4_k1", "prev_int4_k2", "prev_int4_k4",
                         "prev_int2_k4", "prev_int2_k8"]:
                results[name] = results["baseline"]

        # 6. mean_pool_ref: use mean of last 4 tokens
        if ref_h is not None:
            results["mean_pool"] = results["baseline"]
        elif mean_ref is not None:
            results["mean_pool"] = _prevtoken_encode_decode(real_h, mean_ref, gs, 2, 4)
        else:
            results["mean_pool"] = results["baseline"]

        # 7. best_combined: Int4 k1 delta + prev Int4 k2 unigram
        if ref_h is not None:
            results["best_combined"] = _delta_encode_decode(real_h, ref_h, gs, 1, 4)
        elif prev_h is not None:
            results["best_combined"] = _prevtoken_encode_decode(real_h, prev_h, gs, 2, 4)
        else:
            results["best_combined"] = results["baseline"]

        # 8. best_combined_sparse: sparse delta + prev Int4 k2 unigram
        if ref_h is not None:
            scale, bias = compute_affine_params(real_h, ref_h)
            ref_t = apply_affine(ref_h, scale, bias)
            delta = compute_delta(real_h, ref_t)
            recon_delta, data_bytes = sparse_group_encode(delta, gs, 1, 0.05, quant_bits=4)
            recon = (ref_t + recon_delta).to(torch.float16)
            transfer = data_bytes + 2 + 2 + 8
            results["best_sparse"] = (recon, transfer)
        elif prev_h is not None:
            results["best_sparse"] = _prevtoken_encode_decode(real_h, prev_h, gs, 2, 4)
        else:
            results["best_sparse"] = results["baseline"]

        # 9. Ultra-aggressive: Int2 k8 delta + prev Int2 k8 unigram
        if ref_h is not None:
            results["ultra_compress"] = _delta_encode_decode(real_h, ref_h, gs, 8, 2)
        elif prev_h is not None:
            results["ultra_compress"] = _prevtoken_encode_decode(real_h, prev_h, gs, 8, 2)
        else:
            packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(real_h, gs, 8)
            dequant = groupwise_int2_dequantize_topk(packed, scales, zeros, tv, ti, gs, hd)
            transfer = (packed.nelement() * 1 + scales.nelement() * 2 + zeros.nelement() * 2
                        + tv.nelement() * 2 + ti.nelement() * 1)
            results["ultra_compress"] = (dequant, transfer)

        # 10. prev_token with larger group_size (256) to reduce metadata
        if ref_h is not None:
            results["prev_gs256"] = _delta_encode_decode(real_h, ref_h, gs, 1, 4)
        elif prev_h is not None:
            gs256 = 256
            results["prev_gs256"] = _prevtoken_encode_decode(real_h, prev_h, gs256, 2, 4)
        else:
            results["prev_gs256"] = results["baseline"]

        return results


def run_experiment(args):
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    logger.info("Loading model on GPU %d...", args.gpu)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map={"": device}, trust_remote_code=True,
    )
    model.eval()
    global HIDDEN_DIM, NUM_GROUPS
    HIDDEN_DIM = model.config.hidden_size
    NUM_GROUPS = HIDDEN_DIM // GROUP_SIZE
    logger.info("Model loaded. hidden_dim=%d", HIDDEN_DIM)

    wids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    _ = model(wids, output_hidden_states=True, use_cache=False)
    del wids, _
    torch.cuda.synchronize()

    from delta_coding_system.run_experiment import load_dataset_texts

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = StrategyRunner(HIDDEN_DIM, GROUP_SIZE)
    all_records = []
    # Per-position detail records for quality analysis
    position_records = []

    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", ds_name)

        texts = load_dataset_texts(ds_name)
        rng = random.Random(args.seed)
        rng.shuffle(texts)
        warmup_n = args.warmup_requests
        test_n = args.test_requests
        total_needed = warmup_n + test_n
        while len(texts) < total_needed:
            texts.extend(texts[:total_needed - len(texts)])

        table = NgramTable(device=device, dtype=torch.float16, max_entries=100000)

        # Warmup
        logger.info("Warmup: %d requests...", warmup_n)
        for i in range(warmup_n):
            input_ids = tokenizer(texts[i], return_tensors="pt", truncation=True,
                                  max_length=args.max_seq_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            seq_len = token_ids.shape[0]
            if seq_len >= 3:
                trigrams = token_ids.unfold(0, 3, 1)
                for t_idx in range(trigrams.shape[0]):
                    tri = trigrams[t_idx]
                    table._get_or_create_node(tri[0].item(), tri[1].item(),
                                              h[t_idx+1].unsqueeze(0).to(torch.float16))
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[t_idx+2].unsqueeze(0).to(torch.float16)
            del out, h
            if (i+1) % 10 == 0:
                logger.info("  warmup %d/%d", i+1, warmup_n)

        logger.info("Warmup done. Table: %d bi", table.stats["num_bigrams"])

        # Test
        logger.info("Test: %d requests...", test_n)
        for req_idx in range(test_n):
            text = texts[warmup_n + req_idx]
            input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=args.max_seq_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            seq_len = token_ids.shape[0]
            raw_bytes = seq_len * HIDDEN_DIM * 2

            # Classify
            tiers = []
            refs = []
            for pos in range(seq_len):
                found = False
                if pos >= 2:
                    a, b, c = token_ids[pos-2].item(), token_ids[pos-1].item(), token_ids[pos].item()
                    tri_ref = table.get_trigram(a, b, c)
                    if tri_ref is not None:
                        tiers.append("trigram")
                        refs.append(tri_ref.to(torch.float16).to(device))
                        found = True
                if not found and pos >= 1:
                    b, c = token_ids[pos-1].item(), token_ids[pos].item()
                    bi_ref = table.get_bigram(b, c)
                    if bi_ref is not None:
                        tiers.append("bigram")
                        refs.append(bi_ref.to(torch.float16).to(device))
                        found = True
                if not found:
                    tiers.append("unigram")
                    refs.append(None)

            # Run all strategies per position
            strategy_accum = defaultdict(lambda: {"bytes": 0, "cos_sum": 0.0, "cos_min": 1.0, "n": 0})

            for pos in range(seq_len):
                real = h[pos:pos+1].to(torch.float16)
                ref = refs[pos]
                tier = tiers[pos]
                prev = h[pos-1:pos].to(torch.float16) if pos > 0 else None
                mean_ref = _mean_pool_ref(h, pos, 4)

                results = runner.run_all(real, ref, tier, prev, mean_ref)

                for s_name, (recon, transfer) in results.items():
                    cos = F.cosine_similarity(real.float(), recon.float(), dim=-1).item()
                    sa = strategy_accum[s_name]
                    sa["bytes"] += transfer
                    sa["cos_sum"] += cos
                    sa["cos_min"] = min(sa["cos_min"], cos)
                    sa["n"] += 1

                    # Per-position detail for first 3 requests per dataset
                    if req_idx < 3:
                        position_records.append({
                            "dataset": ds_name,
                            "request_index": req_idx,
                            "position": pos,
                            "tier": tier,
                            "strategy": s_name,
                            "cosine": cos,
                            "transfer_bytes": transfer,
                        })

            # Per-request summary
            n_trigram = sum(1 for t in tiers if t == "trigram")
            n_bigram = sum(1 for t in tiers if t == "bigram")
            n_unigram = sum(1 for t in tiers if t == "unigram")

            for s_name, sa in strategy_accum.items():
                all_records.append({
                    "dataset": ds_name,
                    "request_index": req_idx,
                    "strategy": s_name,
                    "seq_len": seq_len,
                    "raw_fp16_bytes": raw_bytes,
                    "total_transfer_bytes": sa["bytes"],
                    "compression_ratio": raw_bytes / max(sa["bytes"], 1),
                    "cosine_mean": sa["cos_sum"] / max(sa["n"], 1),
                    "cosine_min": sa["cos_min"],
                    "num_trigram": n_trigram,
                    "num_bigram": n_bigram,
                    "num_unigram": n_unigram,
                    "pct_trigram": n_trigram / max(seq_len, 1) * 100,
                    "pct_bigram": n_bigram / max(seq_len, 1) * 100,
                    "pct_unigram": n_unigram / max(seq_len, 1) * 100,
                })

            # Update table
            if seq_len >= 3:
                trigrams = token_ids.unfold(0, 3, 1)
                for t_idx in range(trigrams.shape[0]):
                    tri = trigrams[t_idx]
                    table._get_or_create_node(tri[0].item(), tri[1].item(),
                                              h[t_idx+1].unsqueeze(0).to(torch.float16))
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[t_idx+2].unsqueeze(0).to(torch.float16)

            del out, h
            if (req_idx + 1) % 5 == 0:
                base_cr = strategy_accum["baseline"]["bytes"]
                if base_cr > 0:
                    base_cr = raw_bytes / base_cr
                logger.info("  Test %d/%d baseline=%.2fx seq=%d", req_idx+1, test_n, base_cr, seq_len)

        gc.collect()
        torch.cuda.empty_cache()

    # Save
    pq.write_table(pa.Table.from_pylist(all_records), str(output_dir / "strategies_v2.parquet"))
    if position_records:
        pq.write_table(pa.Table.from_pylist(position_records), str(output_dir / "position_detail.parquet"))
    logger.info("Saved %d records", len(all_records))

    # Print summary
    import pandas as pd
    df = pd.DataFrame(all_records)
    print("\n" + "=" * 80)
    print("ROUND 2: REFINED COMPRESSION STRATEGIES")
    print("=" * 80)

    summary = df.groupby("strategy").agg(
        cr=("compression_ratio", "mean"),
        cos=("cosine_mean", "mean"),
        cos_min=("cosine_min", "mean"),
    ).sort_values("cr", ascending=False)

    baseline_cr = summary.loc["baseline", "cr"]
    print(f"\n{'Strategy':<25s} {'Ratio':>8s} {'vs Base':>8s} {'Cos Mean':>10s} {'Cos Min':>10s}")
    print("-" * 65)
    for name, row in summary.iterrows():
        improve = (row["cr"] / baseline_cr - 1) * 100
        print(f"{name:<25s} {row['cr']:>6.3f}x {improve:>+7.1f}% {row['cos']:>10.6f} {row['cos_min']:>10.6f}")

    print("\n\nPer-Dataset:")
    for ds in df["dataset"].unique():
        print(f"\n--- {ds} ---")
        ds_sum = df[df["dataset"]==ds].groupby("strategy").agg(
            cr=("compression_ratio", "mean"),
            cos=("cosine_mean", "mean"),
        ).sort_values("cr", ascending=False)
        for name, row in ds_sum.iterrows():
            print(f"  {name:<25s} {row['cr']:>8.3f}x  cos={row['cos']:.6f}")

    # Position-level quality analysis
    if position_records:
        pdf = pd.DataFrame(position_records)
        print("\n\nPer-Tier Quality (Position-level):")
        for tier in ["trigram", "bigram", "unigram"]:
            tdf = pdf[pdf["tier"] == tier]
            if len(tdf) == 0:
                continue
            print(f"\n  --- {tier} ---")
            tsummary = tdf.groupby("strategy").agg(
                cos_mean=("cosine", "mean"),
                cos_min=("cosine", "min"),
                bytes_mean=("transfer_bytes", "mean"),
            ).sort_values("bytes_mean")
            for name, row in tsummary.iterrows():
                print(f"    {name:<25s} bytes={row['bytes_mean']:>7.0f}  cos={row['cos_mean']:.6f}  cos_min={row['cos_min']:.6f}")

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_compression_exp_v2")
    parser.add_argument("--datasets", nargs="+",
                        default=["wikitext2", "sharegpt", "gsm8k", "cnn_dm", "alpaca", "triviaqa"])
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
