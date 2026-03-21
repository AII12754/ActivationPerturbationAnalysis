#!/usr/bin/env python3
"""Focused delta-path ablation for bitwidth, outliers, and entropy coding.

Goals:
1. Isolate the effect of bitwidth on delta-coded positions.
2. Isolate the marginal value of top-k outliers at each bitwidth.
3. Estimate whether entropy coding is worth pursuing.

Outputs:
  - per-request summary parquet
  - per-position detail parquet
  - entropy summary parquet
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import random
import sys
import zlib
from collections import Counter, defaultdict
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
logger = logging.getLogger("delta_quant_entropy")

HIDDEN_DIM = 5120
GROUP_SIZE = 128

DELTA_CONFIGS = [
    ("delta_ref_only", "ref_only", None, None),
    ("delta_affine_only", "affine_only", None, None),
    ("delta_int8_k0", "quantized", 8, 0),
    ("delta_int8_k1", "quantized", 8, 1),
    ("delta_int8_k2", "quantized", 8, 2),
    ("delta_int8_k4", "quantized", 8, 4),
    ("delta_int4_k0", "quantized", 4, 0),
    ("delta_int4_k1", "quantized", 4, 1),
    ("delta_int4_k2", "quantized", 4, 2),
    ("delta_int4_k4", "quantized", 4, 4),
    ("delta_int4_k8", "quantized", 4, 8),
    ("delta_int2_k0", "quantized", 2, 0),
    ("delta_int2_k2", "quantized", 2, 2),
    ("delta_int2_k4", "quantized", 2, 4),
    ("delta_int2_k8", "quantized", 2, 8),
    ("delta_int2_k16", "quantized", 2, 16),
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


def _serialize_tensor(tensor: Optional[torch.Tensor]) -> bytes:
    if tensor is None:
        return b""
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def _byte_entropy(payload: bytes) -> Tuple[float, float]:
    if not payload:
        return 0.0, 0.0
    counts = Counter(payload)
    total = len(payload)
    entropy_bits_per_byte = 0.0
    for count in counts.values():
        prob = count / total
        entropy_bits_per_byte -= prob * math.log2(prob)
    ideal_bytes = total * entropy_bits_per_byte / 8.0
    return entropy_bits_per_byte, ideal_bytes


def _quantize_no_outlier(
    tensor: torch.Tensor,
    group_size: int,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bytes]:
    batch, hidden_dim = tensor.shape
    num_groups = hidden_dim // group_size
    grouped = tensor.float().reshape(batch, num_groups, group_size)
    g_min = grouped.min(dim=-1).values
    g_max = grouped.max(dim=-1).values

    if bits == 8:
        qmax = 255.0
        data_bytes = hidden_dim
    elif bits == 4:
        qmax = 15.0
        data_bytes = hidden_dim // 2
    elif bits == 2:
        qmax = 3.0
        data_bytes = hidden_dim // 4
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    scales = ((g_max - g_min) / qmax).to(torch.float16)
    zeros = g_min.to(torch.float16)
    scale_f = scales.float().unsqueeze(-1)
    zero_f = zeros.float().unsqueeze(-1)
    q = torch.clamp(torch.round((grouped - zero_f) / (scale_f + 1e-10)), 0, int(qmax)).to(torch.uint8)

    if bits == 8:
        quantized = q.reshape(batch, hidden_dim)
    elif bits == 4:
        q_flat = q.reshape(batch, hidden_dim)
        quantized = (q_flat[:, 0::2] << 4) | q_flat[:, 1::2]
    else:
        q_flat = q.reshape(batch, hidden_dim)
        quantized = (q_flat[:, 0::4] << 6) | (q_flat[:, 1::4] << 4) | (q_flat[:, 2::4] << 2) | q_flat[:, 3::4]

    dequant = q.float() * scale_f + zero_f
    recon = dequant.reshape(batch, hidden_dim).to(torch.float16)
    payload = b"".join([
        _serialize_tensor(quantized),
        _serialize_tensor(scales),
        _serialize_tensor(zeros),
    ])
    total_bytes = data_bytes + scales.nelement() * 2 + zeros.nelement() * 2
    return recon, scales, zeros, total_bytes, payload


def _encode_delta_variant(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
    mode: str,
    bits: Optional[int],
    top_k: Optional[int],
) -> Dict[str, object]:
    raw_ref = ref_h
    scale, bias = compute_affine_params(real_h, ref_h)
    affine_ref = apply_affine(ref_h, scale, bias)
    ref_metrics = {
        "raw_ref_cosine": _cosine(real_h, raw_ref),
        "raw_ref_energy_explained": _explained_energy(real_h, raw_ref),
        "affine_ref_cosine": _cosine(real_h, affine_ref),
        "affine_energy_explained": _explained_energy(real_h, affine_ref),
        "affine_gain": _explained_energy(real_h, affine_ref) - _explained_energy(real_h, raw_ref),
    }

    if mode == "ref_only":
        recon = raw_ref
        packet_bytes = 8
        packet_payload = b"\x00" * 8
    elif mode == "affine_only":
        recon = affine_ref
        packet_bytes = 12
        packet_payload = b"".join([
            b"\x00" * 8,
            _serialize_tensor(scale.to(torch.float16)),
            _serialize_tensor(bias.to(torch.float16)),
        ])
    else:
        delta = compute_delta(real_h, affine_ref)
        if bits == 8 and top_k and top_k > 0:
            pkt = groupwise_int8_quantize_topk(delta, GROUP_SIZE, top_k)
            dequant = groupwise_int8_dequantize_topk(pkt)
            payload_parts = [
                _serialize_tensor(pkt.quantized),
                _serialize_tensor(pkt.scales),
                _serialize_tensor(pkt.zero_points),
                _serialize_tensor(pkt.topk_values),
                _serialize_tensor(pkt.topk_indices),
            ]
            packet_bytes = (
                pkt.quantized.nelement()
                + pkt.scales.nelement() * 2
                + pkt.zero_points.nelement() * 2
                + pkt.topk_values.nelement() * 2
                + pkt.topk_indices.nelement()
                + 2 + 2 + 8
            )
        elif bits == 4 and top_k and top_k > 0:
            packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, GROUP_SIZE, top_k)
            dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, GROUP_SIZE, delta.shape[-1])
            payload_parts = [
                _serialize_tensor(packed),
                _serialize_tensor(scales),
                _serialize_tensor(zeros),
                _serialize_tensor(tv),
                _serialize_tensor(ti),
            ]
            packet_bytes = packed.nelement() + scales.nelement() * 2 + zeros.nelement() * 2 + tv.nelement() * 2 + ti.nelement() + 2 + 2 + 8
        elif bits == 2 and top_k and top_k > 0:
            packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(delta, GROUP_SIZE, top_k)
            dequant = groupwise_int2_dequantize_topk(packed, scales, zeros, tv, ti, GROUP_SIZE, delta.shape[-1])
            payload_parts = [
                _serialize_tensor(packed),
                _serialize_tensor(scales),
                _serialize_tensor(zeros),
                _serialize_tensor(tv),
                _serialize_tensor(ti),
            ]
            packet_bytes = packed.nelement() + scales.nelement() * 2 + zeros.nelement() * 2 + tv.nelement() * 2 + ti.nelement() + 2 + 2 + 8
        else:
            dequant, scales, zeros, quant_bytes, payload = _quantize_no_outlier(delta, GROUP_SIZE, bits)
            payload_parts = [payload]
            packet_bytes = quant_bytes + 2 + 2 + 8
        recon = (affine_ref + dequant).to(torch.float16)
        packet_payload = b"".join(payload_parts + [_serialize_tensor(scale.to(torch.float16)), _serialize_tensor(bias.to(torch.float16))])

    recon_e = _explained_energy(real_h, recon)
    entropy_bpb, ideal_entropy_bytes = _byte_entropy(packet_payload)
    zlib_bytes = len(zlib.compress(packet_payload, level=9)) if packet_payload else 0

    return {
        "recon": recon,
        "transfer_bytes": packet_bytes,
        "packet_entropy_bits_per_byte": entropy_bpb,
        "packet_entropy_ideal_bytes": ideal_entropy_bytes,
        "packet_zlib_bytes": zlib_bytes,
        "metrics": {
            **ref_metrics,
            "recon_energy_explained": recon_e,
            "residual_coding_loss": 1.0 - recon_e,
        },
    }


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
            input_ids = tokenizer(texts[warm_idx], return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            if token_ids.shape[0] >= 3:
                trigrams = token_ids.unfold(0, 3, 1)
                for tri_idx in range(trigrams.shape[0]):
                    tri = trigrams[tri_idx]
                    table._get_or_create_node(tri[0].item(), tri[1].item(), h[tri_idx + 1].unsqueeze(0).to(torch.float16))
                    node = table._get_node(tri[0].item(), tri[1].item())
                    if node is not None:
                        node.suffixes[tri[2].item()] = h[tri_idx + 2].unsqueeze(0).to(torch.float16)
            del out, h

        logger.info("Test: %d requests...", args.test_requests)
        for req_idx in range(args.test_requests):
            text = texts[args.warmup_requests + req_idx]
            input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            seq_len = token_ids.shape[0]

            strategy_accum = defaultdict(lambda: {
                "bytes": 0,
                "zlib_bytes": 0,
                "ideal_bytes": 0.0,
                "entropy_bpb_sum": 0.0,
                "cos_sum": 0.0,
                "cos_min": 1.0,
                "recon_e_sum": 0.0,
                "raw_e_sum": 0.0,
                "aff_e_sum": 0.0,
                "aff_gain_sum": 0.0,
                "count": 0,
            })
            delta_positions = 0

            for pos in range(seq_len):
                ref = None
                if pos >= 2:
                    tri_ref = table.get_trigram(token_ids[pos - 2].item(), token_ids[pos - 1].item(), token_ids[pos].item())
                    if tri_ref is not None:
                        ref = tri_ref.to(torch.float16).to(device)
                if ref is None and pos >= 1:
                    bi_ref = table.get_bigram(token_ids[pos - 1].item(), token_ids[pos].item())
                    if bi_ref is not None:
                        ref = bi_ref.to(torch.float16).to(device)
                if ref is None:
                    continue

                delta_positions += 1
                real = h[pos:pos + 1].to(torch.float16)
                for strategy_name, mode, bits, top_k in DELTA_CONFIGS:
                    result = _encode_delta_variant(real, ref, mode, bits, top_k)
                    recon = result["recon"]
                    metrics = result["metrics"]
                    cos = _cosine(real, recon)
                    acc = strategy_accum[strategy_name]
                    acc["bytes"] += result["transfer_bytes"]
                    acc["zlib_bytes"] += result["packet_zlib_bytes"]
                    acc["ideal_bytes"] += result["packet_entropy_ideal_bytes"]
                    acc["entropy_bpb_sum"] += result["packet_entropy_bits_per_byte"]
                    acc["cos_sum"] += cos
                    acc["cos_min"] = min(acc["cos_min"], cos)
                    acc["recon_e_sum"] += metrics["recon_energy_explained"]
                    acc["raw_e_sum"] += metrics["raw_ref_energy_explained"]
                    acc["aff_e_sum"] += metrics["affine_energy_explained"]
                    acc["aff_gain_sum"] += metrics["affine_gain"]
                    acc["count"] += 1

                    if req_idx < args.detail_requests:
                        position_records.append({
                            "dataset": ds_name,
                            "request_index": req_idx,
                            "position": pos,
                            "strategy": strategy_name,
                            "cosine": cos,
                            "transfer_bytes": result["transfer_bytes"],
                            "packet_zlib_bytes": result["packet_zlib_bytes"],
                            "packet_entropy_ideal_bytes": result["packet_entropy_ideal_bytes"],
                            "packet_entropy_bits_per_byte": result["packet_entropy_bits_per_byte"],
                            **metrics,
                        })

            raw_delta_bytes = delta_positions * HIDDEN_DIM * 2
            for strategy_name, acc in strategy_accum.items():
                if acc["count"] == 0:
                    continue
                request_records.append({
                    "dataset": ds_name,
                    "request_index": req_idx,
                    "strategy": strategy_name,
                    "delta_positions": delta_positions,
                    "raw_fp16_bytes": raw_delta_bytes,
                    "total_transfer_bytes": acc["bytes"],
                    "total_zlib_bytes": acc["zlib_bytes"],
                    "total_entropy_ideal_bytes": acc["ideal_bytes"],
                    "compression_ratio": raw_delta_bytes / max(acc["bytes"], 1),
                    "compression_ratio_zlib": raw_delta_bytes / max(acc["zlib_bytes"], 1),
                    "compression_ratio_entropy_ideal": raw_delta_bytes / max(acc["ideal_bytes"], 1e-8),
                    "cosine_mean": acc["cos_sum"] / acc["count"],
                    "cosine_min": acc["cos_min"],
                    "raw_ref_energy_explained_mean": acc["raw_e_sum"] / acc["count"],
                    "affine_energy_explained_mean": acc["aff_e_sum"] / acc["count"],
                    "affine_gain_mean": acc["aff_gain_sum"] / acc["count"],
                    "recon_energy_explained_mean": acc["recon_e_sum"] / acc["count"],
                    "residual_coding_loss_mean": 1.0 - (acc["recon_e_sum"] / acc["count"]),
                    "packet_entropy_bits_per_byte_mean": acc["entropy_bpb_sum"] / acc["count"],
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

    pq.write_table(pa.Table.from_pylist(request_records), str(output_dir / "delta_quant_entropy.parquet"))
    if position_records:
        pq.write_table(pa.Table.from_pylist(position_records), str(output_dir / "delta_quant_entropy_detail.parquet"))
    logger.info("Saved %d request records", len(request_records))

    import pandas as pd
    df = pd.DataFrame(request_records)
    summary = df.groupby("strategy").agg(
        cr=("compression_ratio", "mean"),
        cr_zlib=("compression_ratio_zlib", "mean"),
        cr_ideal=("compression_ratio_entropy_ideal", "mean"),
        cos=("cosine_mean", "mean"),
        recon_e=("recon_energy_explained_mean", "mean"),
        loss=("residual_coding_loss_mean", "mean"),
        entropy_bpb=("packet_entropy_bits_per_byte_mean", "mean"),
    ).sort_values("cr", ascending=False)

    print("\n" + "=" * 110)
    print("DELTA QUANTIZATION / OUTLIER / ENTROPY ABLATION")
    print("=" * 110)
    print(f"\n{'Strategy':<18s} {'CR':>8s} {'CR-zlib':>9s} {'CR-ideal':>9s} {'Cos':>9s} {'ReconE':>9s} {'Loss':>9s} {'EntBpb':>8s}")
    print("-" * 110)
    for name, row in summary.iterrows():
        print(
            f"{name:<18s} {row['cr']:>7.3f}x {row['cr_zlib']:>8.3f}x {row['cr_ideal']:>8.3f}x "
            f"{row['cos']:>9.6f} {row['recon_e']:>9.6f} {row['loss']:>9.6f} {row['entropy_bpb']:>8.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Delta quantization and entropy ablation")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=30)
    parser.add_argument("--test-requests", type=int, default=30)
    parser.add_argument("--detail-requests", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_delta_quant_entropy")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext2", "sharegpt", "gsm8k", "cnn_dm", "alpaca", "triviaqa"],
    )
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()