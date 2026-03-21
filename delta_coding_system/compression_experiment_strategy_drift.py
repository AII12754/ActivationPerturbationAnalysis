#!/usr/bin/env python3
"""Strategy selection with downstream output drift.

Focus:
1. Low-precision outliers, especially for Int2.
2. Int2 with entropy-coded byte estimates.
3. Unigram alternatives: zero-reference affine and warmup mean activation.
4. Real downstream drift after running remaining transformer layers.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("strategy_drift")

GROUP_SIZE = 128


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


def _quantize_outliers(values: torch.Tensor, bits: int) -> Tuple[torch.Tensor, int, bytes]:
    if bits >= 16:
        return values.to(torch.float16), values.nelement() * 2, _serialize_tensor(values.to(torch.float16))

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
    else:
        if k % 2 == 1:
            q = torch.cat([q, torch.zeros(batch, groups, 1, dtype=torch.uint8, device=q.device)], dim=-1)
        packed = (q[..., 0::2] << 4) | q[..., 1::2]
        data_bytes = packed.nelement()
    dequant = packed if bits == 8 else packed
    if bits == 8:
        q_restore = packed.float()
    else:
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


def _encode_delta(
    real_h: torch.Tensor,
    ref_h: torch.Tensor,
    bits: int,
    top_k: int,
    outlier_bits: int = 16,
    include_ref_idx: bool = True,
    group_size: int = GROUP_SIZE,
    entropy_override: bool = False,
) -> Dict[str, object]:
    scale, bias = compute_affine_params(real_h, ref_h)
    affine_ref = apply_affine(ref_h, scale, bias)
    delta = compute_delta(real_h, affine_ref)
    hidden_dim = delta.shape[-1]

    if bits == 4:
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)
    elif bits == 2:
        packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(delta, group_size, top_k)
    else:
        raise ValueError(bits)

    q_outliers, outlier_bytes, outlier_payload = _quantize_outliers(tv, outlier_bits)
    dequant = _decode_delta_with_custom_outliers(packed, scales, zeros, q_outliers, ti, group_size, hidden_dim, bits)
    recon = (affine_ref + dequant).to(torch.float16)

    payload = b"".join([
        _serialize_tensor(packed),
        _serialize_tensor(scales),
        _serialize_tensor(zeros),
        outlier_payload,
        _serialize_tensor(ti),
        _serialize_tensor(scale.to(torch.float16)),
        _serialize_tensor(bias.to(torch.float16)),
    ])
    entropy_bpb, zlib_bytes, ideal_bytes = _entropy_bytes(payload)
    raw_bytes = packed.nelement() + scales.nelement() * 2 + zeros.nelement() * 2 + outlier_bytes + ti.nelement() + 2 + 2 + (8 if include_ref_idx else 0)
    effective_bytes = zlib_bytes if entropy_override else raw_bytes
    return {
        "recon": recon,
        "bytes": effective_bytes,
        "raw_bytes": raw_bytes,
        "zlib_bytes": zlib_bytes,
        "ideal_entropy_bytes": ideal_bytes,
        "entropy_bpb": entropy_bpb,
        "metrics": {
            "raw_ref_cosine": _cosine(real_h, ref_h),
            "raw_ref_energy_explained": _explained_energy(real_h, ref_h),
            "affine_ref_cosine": _cosine(real_h, affine_ref),
            "affine_energy_explained": _explained_energy(real_h, affine_ref),
            "affine_gain": _explained_energy(real_h, affine_ref) - _explained_energy(real_h, ref_h),
            "recon_energy_explained": _explained_energy(real_h, recon),
            "residual_coding_loss": 1.0 - _explained_energy(real_h, recon),
        },
    }


def _encode_int8_unigram(real_h: torch.Tensor) -> Dict[str, object]:
    pkt = groupwise_int8_quantize_topk(real_h, GROUP_SIZE, 1)
    recon = groupwise_int8_dequantize_topk(pkt)
    bytes_ = pkt.quantized.nelement() + pkt.scales.nelement() * 2 + pkt.zero_points.nelement() * 2 + pkt.topk_values.nelement() * 2 + pkt.topk_indices.nelement()
    return {
        "recon": recon,
        "bytes": bytes_,
        "raw_bytes": bytes_,
        "zlib_bytes": bytes_,
        "ideal_entropy_bytes": float(bytes_),
        "entropy_bpb": float('nan'),
        "metrics": {
            "raw_ref_cosine": float('nan'),
            "raw_ref_energy_explained": float('nan'),
            "affine_ref_cosine": float('nan'),
            "affine_energy_explained": float('nan'),
            "affine_gain": float('nan'),
            "recon_energy_explained": _explained_energy(real_h, recon),
            "residual_coding_loss": 1.0 - _explained_energy(real_h, recon),
        },
    }


def _run_remaining_layers(model, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    cache_position = torch.arange(seq_len, device=hidden_states.device)
    causal_mask = model.model._update_causal_mask(attention_mask, hidden_states, cache_position, None, False)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    hidden = hidden_states
    start_layer = getattr(_run_remaining_layers, 'start_layer', 0)
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
    def __init__(self, layer_boundary: int):
        self.layer_boundary = layer_boundary

    def run_all(
        self,
        real_h: torch.Tensor,
        tier: str,
        table_ref: Optional[torch.Tensor],
        prev_h: Optional[torch.Tensor],
        global_mean_ref: torch.Tensor,
    ) -> Dict[str, Dict[str, object]]:
        zero_ref = torch.zeros_like(real_h)
        results: Dict[str, Dict[str, object]] = {}

        if table_ref is not None:
            results['baseline'] = _encode_delta(real_h, table_ref, 4, 1, outlier_bits=16, include_ref_idx=True)
            results['delta_int2_k8_prev_int4_k2'] = _encode_delta(real_h, table_ref, 2, 8, outlier_bits=16, include_ref_idx=True)
            results['delta_int2_k8_out8_prev_int4_k2'] = _encode_delta(real_h, table_ref, 2, 8, outlier_bits=8, include_ref_idx=True)
            results['delta_int2_k8_out4_prev_int4_k2'] = _encode_delta(real_h, table_ref, 2, 8, outlier_bits=4, include_ref_idx=True)
            results['delta_int2_k8_out8_entropy_prev_int4_k2'] = _encode_delta(real_h, table_ref, 2, 8, outlier_bits=8, include_ref_idx=True, entropy_override=True)
            passthrough = results['baseline']
            for name in [
                'prev_int4_k2', 'prev_gs256_k2', 'prev_int2_k8', 'prev_int2_k8_out8', 'prev_int2_k8_out4',
                'global_mean_int4_k2', 'zero_affine_int4_k2'
            ]:
                results[name] = passthrough
            return results

        results['baseline'] = _encode_int8_unigram(real_h)
        if prev_h is not None:
            results['prev_int4_k2'] = _encode_delta(real_h, prev_h, 4, 2, outlier_bits=16, include_ref_idx=False)
            results['prev_gs256_k2'] = _encode_delta(real_h, prev_h, 4, 2, outlier_bits=16, include_ref_idx=False, group_size=256)
            results['prev_int2_k8'] = _encode_delta(real_h, prev_h, 2, 8, outlier_bits=16, include_ref_idx=False)
            results['prev_int2_k8_out8'] = _encode_delta(real_h, prev_h, 2, 8, outlier_bits=8, include_ref_idx=False)
            results['prev_int2_k8_out4'] = _encode_delta(real_h, prev_h, 2, 8, outlier_bits=4, include_ref_idx=False)
        else:
            for name in ['prev_int4_k2', 'prev_gs256_k2', 'prev_int2_k8', 'prev_int2_k8_out8', 'prev_int2_k8_out4']:
                results[name] = results['baseline']

        results['global_mean_int4_k2'] = _encode_delta(real_h, global_mean_ref, 4, 2, outlier_bits=16, include_ref_idx=False)
        results['zero_affine_int4_k2'] = _encode_delta(real_h, zero_ref, 4, 2, outlier_bits=16, include_ref_idx=False)
        passthrough = results['baseline']
        for name in ['delta_int2_k8_prev_int4_k2', 'delta_int2_k8_out8_prev_int4_k2', 'delta_int2_k8_out4_prev_int4_k2', 'delta_int2_k8_out8_entropy_prev_int4_k2']:
            results[name] = passthrough
        return results


def run_experiment(args):
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    logger.info('Loading model on GPU %d...', args.gpu)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from delta_coding_system.run_experiment import load_dataset_texts

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map={'': device}, trust_remote_code=True)
    model.eval()
    _run_remaining_layers.start_layer = args.layer_boundary

    runner = StrategyRunner(args.layer_boundary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_records = []
    drift_records = []

    for ds_name in args.datasets:
        logger.info('%s', '=' * 60)
        logger.info('Dataset: %s', ds_name)
        texts = load_dataset_texts(ds_name)
        rng = random.Random(args.seed)
        rng.shuffle(texts)
        total_needed = args.warmup_requests + args.test_requests
        while len(texts) < total_needed:
            texts.extend(texts[: total_needed - len(texts)])

        table = NgramTable(device=device, dtype=torch.float16, max_entries=100000)
        mean_sum = None
        mean_count = 0
        logger.info('Warmup: %d requests...', args.warmup_requests)
        for warm_idx in range(args.warmup_requests):
            input_ids = tokenizer(texts[warm_idx], return_tensors='pt', truncation=True, max_length=args.max_seq_len).input_ids.to(device)
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

        logger.info('Test: %d requests...', args.test_requests)
        for req_idx in range(args.test_requests):
            text = texts[args.warmup_requests + req_idx]
            input_ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=args.max_seq_len).input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True, use_cache=False)
            original_logits = out.logits
            h = out.hidden_states[args.layer_boundary].squeeze(0)
            token_ids = input_ids.squeeze(0)
            seq_len = token_ids.shape[0]
            raw_bytes = seq_len * h.shape[-1] * 2

            tiers = []
            refs = []
            for pos in range(seq_len):
                ref = None
                tier = 'unigram'
                if pos >= 2:
                    tri_ref = table.get_trigram(token_ids[pos - 2].item(), token_ids[pos - 1].item(), token_ids[pos].item())
                    if tri_ref is not None:
                        ref = tri_ref.to(torch.float16).to(device)
                        tier = 'trigram'
                if ref is None and pos >= 1:
                    bi_ref = table.get_bigram(token_ids[pos - 1].item(), token_ids[pos].item())
                    if bi_ref is not None:
                        ref = bi_ref.to(torch.float16).to(device)
                        tier = 'bigram'
                tiers.append(tier)
                refs.append(ref)

            strategy_accum = defaultdict(lambda: {
                'bytes': 0, 'raw_bytes': 0, 'zlib_bytes': 0, 'ideal_bytes': 0.0, 'entropy_bpb_sum': 0.0,
                'cos_sum': 0.0, 'cos_min': 1.0, 'recon_e_sum': 0.0, 'count': 0,
            })
            recon_sequences = defaultdict(list)

            for pos in range(seq_len):
                real = h[pos:pos + 1].to(torch.float16)
                prev_h = h[pos - 1:pos].to(torch.float16) if pos > 0 else None
                results = runner.run_all(real, tiers[pos], refs[pos], prev_h, global_mean_ref.to(device))
                for strategy_name, result in results.items():
                    recon = result['recon']
                    recon_sequences[strategy_name].append(recon)
                    metrics = result['metrics']
                    cos = _cosine(real, recon)
                    acc = strategy_accum[strategy_name]
                    acc['bytes'] += result['bytes']
                    acc['raw_bytes'] += result['raw_bytes']
                    acc['zlib_bytes'] += result['zlib_bytes']
                    acc['ideal_bytes'] += result['ideal_entropy_bytes']
                    if result['entropy_bpb'] == result['entropy_bpb']:
                        acc['entropy_bpb_sum'] += result['entropy_bpb']
                    acc['cos_sum'] += cos
                    acc['cos_min'] = min(acc['cos_min'], cos)
                    acc['recon_e_sum'] += metrics['recon_energy_explained']
                    acc['count'] += 1

            for strategy_name, acc in strategy_accum.items():
                if req_idx < args.drift_requests:
                    recon_hidden = torch.cat(recon_sequences[strategy_name], dim=0).unsqueeze(0)
                    with torch.no_grad():
                        recon_logits = _run_remaining_layers(model, recon_hidden, attention_mask)
                    logit_cos = F.cosine_similarity(original_logits.float(), recon_logits.float(), dim=-1).squeeze(0)
                    orig_top1 = original_logits.argmax(dim=-1).squeeze(0)
                    recon_top1 = recon_logits.argmax(dim=-1).squeeze(0)
                    mismatch = (orig_top1 != recon_top1)
                    first_drift = int(torch.where(mismatch)[0][0].item()) if mismatch.any() else seq_len
                    first_logit_cos_below_0999 = int(torch.where(logit_cos < 0.999)[0][0].item()) if (logit_cos < 0.999).any() else seq_len
                    kl = F.kl_div(
                        F.log_softmax(recon_logits.float(), dim=-1),
                        F.softmax(original_logits.float(), dim=-1),
                        reduction='none',
                    ).sum(dim=-1).squeeze(0)
                    drift_records.append({
                        'dataset': ds_name,
                        'request_index': req_idx,
                        'strategy': strategy_name,
                        'seq_len': seq_len,
                        'top1_match_rate': float((~mismatch).float().mean().item()),
                        'first_top1_drift_pos': first_drift,
                        'first_logit_cos_below_0_999': first_logit_cos_below_0999,
                        'logit_cosine_mean': float(logit_cos.mean().item()),
                        'logit_cosine_min': float(logit_cos.min().item()),
                        'kl_mean': float(kl.mean().item()),
                        'kl_max': float(kl.max().item()),
                    })

                request_records.append({
                    'dataset': ds_name,
                    'request_index': req_idx,
                    'strategy': strategy_name,
                    'seq_len': seq_len,
                    'raw_fp16_bytes': raw_bytes,
                    'total_transfer_bytes': acc['bytes'],
                    'total_raw_packet_bytes': acc['raw_bytes'],
                    'total_zlib_packet_bytes': acc['zlib_bytes'],
                    'total_ideal_entropy_bytes': acc['ideal_bytes'],
                    'compression_ratio': raw_bytes / max(acc['bytes'], 1),
                    'compression_ratio_raw_packet': raw_bytes / max(acc['raw_bytes'], 1),
                    'compression_ratio_zlib_packet': raw_bytes / max(acc['zlib_bytes'], 1),
                    'compression_ratio_ideal_entropy': raw_bytes / max(acc['ideal_bytes'], 1e-8),
                    'cosine_mean': acc['cos_sum'] / max(acc['count'], 1),
                    'cosine_min': acc['cos_min'],
                    'recon_energy_explained_mean': acc['recon_e_sum'] / max(acc['count'], 1),
                    'residual_coding_loss_mean': 1.0 - acc['recon_e_sum'] / max(acc['count'], 1),
                    'packet_entropy_bits_per_byte_mean': acc['entropy_bpb_sum'] / max(acc['count'], 1),
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
                logger.info('  Test %d/%d complete', req_idx + 1, args.test_requests)

        gc.collect()
        torch.cuda.empty_cache()

    pq.write_table(pa.Table.from_pylist(request_records), str(output_dir / 'strategy_drift_summary.parquet'))
    if drift_records:
        pq.write_table(pa.Table.from_pylist(drift_records), str(output_dir / 'strategy_drift_logits.parquet'))
    logger.info('Saved %d request records and %d drift records', len(request_records), len(drift_records))


def main():
    parser = argparse.ArgumentParser(description='Strategy drift experiment')
    parser.add_argument('--model', default='/root/share/models/Qwen2.5-32B-Instruct')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--warmup-requests', type=int, default=20)
    parser.add_argument('--test-requests', type=int, default=20)
    parser.add_argument('--drift-requests', type=int, default=3)
    parser.add_argument('--max-seq-len', type=int, default=384)
    parser.add_argument('--layer-boundary', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='results_strategy_drift')
    parser.add_argument('--datasets', nargs='+', default=['wikitext2', 'sharegpt', 'gsm8k', 'cnn_dm', 'alpaca', 'triviaqa'])
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()