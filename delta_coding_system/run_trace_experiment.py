#!/usr/bin/env python3
"""Run delta-coding trace compaction experiments.

For each compaction event in compaction_prompt_lite.json:
  1. Process prompt_before to build NgramTable and cache hidden states
  2. Process prompt_after to get actual hidden states
  3. Compute transfer bytes for 5 schemes:
     - FP16 baseline, pure INT8, pure INT4 (simulated)
     - delta-INT8, delta-INT4 (actual delta encoding)
  4. Compute communication latency at multiple bandwidths

Each event is processed independently — no cross-event table sharing.

Usage:
  python -m delta_coding_system.run_trace_experiment --gpu 0
  python -m delta_coding_system.run_trace_experiment --gpu 0 --decode-tokens 32
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from delta_coding_system.codec import (
    DeltaPacket,
    compute_affine_params,
    apply_affine,
    compute_delta,
    compute_transfer_size,
    compute_transfer_size_int8_outlier,
    groupwise_int4_dequantize_topk,
    groupwise_int4_quantize_topk,
    groupwise_int8_dequantize_topk,
    groupwise_int8_quantize_topk,
    reconstruct_activation,
)
from delta_coding_system.pipeline import OverlappedPipeline
from delta_coding_system.simulation import (
    SchemeResult,
    SimulationResult,
    _network_ms,
    simulate_compression,
)
from delta_coding_system.table import NgramTable
from delta_coding_system.trace_loader import TraceEvent, load_trace_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("trace_experiment")

# Default trace path (relative to project root)
DEFAULT_TRACE_PATH = str(
    PROJECT_ROOT.parent / "trace" / "openclaw-memory" / "prompt_dataset" / "compaction_prompt_lite.json"
)

BANDWIDTHS_MBPS = [200, 500, 1000]
ALL_SCHEMES = ["fp16", "int8", "int4", "delta-int8", "delta-int4"]


# ===================================================================
# Helpers
# ===================================================================
def find_shared_prefix_len(ids_before: List[int], ids_after: List[int]) -> int:
    """Find the length of the shared token prefix between two sequences."""
    min_len = min(len(ids_before), len(ids_after))
    shared = 0
    for i in range(min_len):
        if ids_before[i] == ids_after[i]:
            shared += 1
        else:
            break
    return shared


def _save_records(records: List[Dict[str, Any]], path: Path) -> None:
    """Save a list of dicts as a Parquet file."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(records)
    pq.write_table(tbl, str(path))
    logger.info("Saved %d records to %s", len(records), path)


# ===================================================================
# Per-event experiment
# ===================================================================
@torch.inference_mode()
def run_single_event(
    event: TraceEvent,
    model,
    tokenizer,
    device: torch.device,
    layer_boundary: int,
    group_size: int,
    top_k: int,
    decode_tokens: int,
    max_seq_len: int,
    bandwidths: List[int],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Run experiment for a single trace compaction event.

    Returns (prefill_records, decode_records, detail_records).
    """
    logger.info("=" * 60)
    logger.info("Processing event: %s", event.id)
    logger.info("=" * 60)

    hidden_dim = model.config.hidden_size
    from activation_science.core.extraction import prefill as do_prefill, decode_step, select_next_token

    # ── Phase A: Process prompt_before ──
    logger.info("[Phase A] Processing prompt_before (tokens_before=%d)...", event.tokens_before)

    ids_before = tokenizer.encode(event.prompt_before, add_special_tokens=False)
    if len(ids_before) > max_seq_len:
        ids_before = ids_before[:max_seq_len]
    input_before = torch.tensor([ids_before], dtype=torch.long, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch_before = do_prefill(model, input_before, use_cache=False)
    torch.cuda.synchronize()
    prefill_before_ms = (time.perf_counter() - t0) * 1000.0

    layer_idx = min(layer_boundary, batch_before.num_layers)
    hidden_before = batch_before.hidden_states[layer_idx].squeeze(0).to(torch.float16)
    del batch_before
    torch.cuda.empty_cache()

    logger.info("  prompt_before: %d tokens, prefill %.1f ms", len(ids_before), prefill_before_ms)

    # Build NgramTable from prompt_before
    table = NgramTable(device=device, dtype=torch.float16, max_entries=100000)
    table.update_from_hidden_states(ids_before, hidden_before)
    table_stats = table.stats
    logger.info("  NgramTable built: %d trigrams, %d bigrams",
                table_stats["num_trigrams"], table_stats["num_bigrams"])

    # ── Phase B: Process prompt_after ──
    logger.info("[Phase B] Processing prompt_after...")

    ids_after = tokenizer.encode(event.prompt_after, add_special_tokens=False)
    if len(ids_after) > max_seq_len:
        ids_after = ids_after[:max_seq_len]
    input_after = torch.tensor([ids_after], dtype=torch.long, device=device)

    shared_prefix_len = find_shared_prefix_len(ids_before, ids_after)
    changed_len = len(ids_after) - shared_prefix_len
    logger.info("  prompt_after: %d tokens, shared_prefix: %d, changed: %d",
                len(ids_after), shared_prefix_len, changed_len)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch_after = do_prefill(model, input_after, use_cache=True)
    torch.cuda.synchronize()
    prefill_after_ms = (time.perf_counter() - t0) * 1000.0

    hidden_after = batch_after.hidden_states[layer_idx].squeeze(0).to(torch.float16)
    past_kv = batch_after.past_key_values
    next_tok = select_next_token(batch_after.last_logits, do_sample=False)
    del batch_after

    logger.info("  prefill_after: %.1f ms", prefill_after_ms)

    seq_len_after = len(ids_after)

    # ── Phase C: Compute transfer bytes for all 5 schemes (prefill) ──
    prefill_records: List[Dict] = []
    detail_records: List[Dict] = []

    # C.1 — Simulated schemes (fp16, int8, int4)
    sim_result = simulate_compression(
        seq_len=seq_len_after, hidden_dim=hidden_dim,
        group_size=group_size, top_k=top_k, bandwidths=bandwidths,
    )
    for scheme_name in ["fp16", "int8", "int4"]:
        sr = sim_result.schemes[scheme_name]
        rec: Dict[str, Any] = {
            "event_id": event.id,
            "scheme": scheme_name,
            "phase": "prefill",
            "seq_len": seq_len_after,
            "shared_prefix_len": shared_prefix_len,
            "changed_len": changed_len,
            "total_bytes": sr.total_bytes,
            "fp16_bytes": sim_result.fp16_bytes,
            "compression_ratio": sr.compression_ratio,
            "prefill_fwd_ms": prefill_after_ms,
        }
        for bw in bandwidths:
            rec[f"latency_{bw}mbps_ms"] = sr.latency_ms[bw]
        prefill_records.append(rec)

    # C.2 — Delta schemes: actual encoding on changed positions
    tiers, ref_acts, self_ref_sources, first_occ_map = table.classify_and_build_refs(
        ids_after, hidden_dim,
    )

    for bits, scheme_name in [(8, "delta-int8"), (4, "delta-int4")]:
        total_transfer = 0
        tier_counts = {"trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0}
        recon_cosines: List[float] = []
        reconstructed = torch.zeros_like(hidden_after)

        for pos in range(seq_len_after):
            real_h = hidden_after[pos].unsqueeze(0)  # (1, hidden_dim)

            if pos < shared_prefix_len:
                # Shared prefix: transfer = 0, receiver already has cached activations
                if pos < len(hidden_before):
                    reconstructed[pos] = hidden_before[pos]
                else:
                    reconstructed[pos] = hidden_after[pos]
                detail_records.append({
                    "event_id": event.id, "scheme": scheme_name,
                    "position": pos, "tier": "cached", "transfer_bytes": 0,
                })
                continue

            tier = tiers[pos]
            ref_h = ref_acts[pos].unsqueeze(0)
            has_ref = tier in ("trigram", "bigram") and ref_h.abs().sum().item() > 0
            tier_counts[tier] += 1
            xfer = 0

            if tier == "self_ref":
                # Use reconstructed position as reference
                src_pos = self_ref_sources[pos]
                if src_pos is not None and src_pos < pos:
                    ref_h = reconstructed[src_pos].unsqueeze(0)
                    has_ref = True
                else:
                    has_ref = False

            if has_ref:
                # Delta encoding with affine transform
                scale, bias = compute_affine_params(real_h, ref_h)
                ref_t = apply_affine(ref_h, scale, bias)
                delta = compute_delta(real_h, ref_t)

                if bits == 4:
                    packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(
                        delta, group_size, top_k,
                    )
                    dequant = groupwise_int4_dequantize_topk(
                        packed, scales, zeros, tv, ti, group_size, hidden_dim,
                    )
                    pkt = DeltaPacket(
                        quantized_data=packed, scales=scales, zero_points=zeros,
                        topk_values=tv, topk_indices=ti,
                        affine_scale=scale.to(torch.float16),
                        affine_bias=bias.to(torch.float16),
                        ref_indices=torch.zeros(1, dtype=torch.long, device=device),
                        group_size=group_size, top_k=top_k,
                    )
                    xfer = compute_transfer_size(pkt)
                else:  # bits == 8
                    int8_pkt = groupwise_int8_quantize_topk(delta, group_size, top_k)
                    dequant = groupwise_int8_dequantize_topk(int8_pkt)
                    # INT8 delta transfer: int8 packet + affine (4B) + ref_idx (8B)
                    xfer = compute_transfer_size_int8_outlier(int8_pkt) + 4 + 8

                recon = reconstruct_activation(dequant, ref_h, scale, bias).to(torch.float16)
                reconstructed[pos] = recon.squeeze(0)
            else:
                # Unigram fallback: pure quantization without delta
                if bits == 4:
                    packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(
                        real_h.clone(), group_size, top_k,
                    )
                    recon_h = groupwise_int4_dequantize_topk(
                        packed, scales, zeros, tv, ti, group_size, hidden_dim,
                    )
                    xfer = (packed.nelement() * packed.element_size()
                            + scales.nelement() * scales.element_size()
                            + zeros.nelement() * zeros.element_size()
                            + tv.nelement() * tv.element_size()
                            + ti.nelement() * ti.element_size())
                else:  # bits == 8
                    int8_pkt = groupwise_int8_quantize_topk(real_h, group_size, top_k)
                    recon_h = groupwise_int8_dequantize_topk(int8_pkt)
                    xfer = compute_transfer_size_int8_outlier(int8_pkt)

                reconstructed[pos] = recon_h.squeeze(0).to(torch.float16)

            total_transfer += xfer
            cos = F.cosine_similarity(
                real_h.float(), reconstructed[pos].unsqueeze(0).float(), dim=-1,
            ).item()
            recon_cosines.append(cos)

            detail_records.append({
                "event_id": event.id, "scheme": scheme_name,
                "position": pos, "tier": tier, "transfer_bytes": xfer,
            })

        fp16_bytes = seq_len_after * hidden_dim * 2
        ratio = fp16_bytes / max(total_transfer, 1)

        rec = {
            "event_id": event.id,
            "scheme": scheme_name,
            "phase": "prefill",
            "seq_len": seq_len_after,
            "shared_prefix_len": shared_prefix_len,
            "changed_len": changed_len,
            "total_bytes": total_transfer,
            "fp16_bytes": fp16_bytes,
            "compression_ratio": ratio,
            "prefill_fwd_ms": prefill_after_ms,
            "num_trigram": tier_counts["trigram"],
            "num_bigram": tier_counts["bigram"],
            "num_self_ref": tier_counts["self_ref"],
            "num_unigram": tier_counts["unigram"],
            "recon_cosine_mean": sum(recon_cosines) / len(recon_cosines) if recon_cosines else 0.0,
            "recon_cosine_min": min(recon_cosines) if recon_cosines else 0.0,
        }
        for bw in bandwidths:
            rec[f"latency_{bw}mbps_ms"] = _network_ms(total_transfer, bw)
        prefill_records.append(rec)

        logger.info("  %s: transfer=%d bytes, ratio=%.2fx, tiers=%s",
                     scheme_name, total_transfer, ratio, tier_counts)

    # ── Phase D: Decode test ──
    decode_records: List[Dict] = []

    if decode_tokens > 0:
        logger.info("[Phase D] Decode %d tokens...", decode_tokens)

        pipeline = OverlappedPipeline(
            model=model,
            tokenizer=tokenizer,
            layer_boundary=layer_boundary,
            table_dtype=torch.float16,
            max_table_entries=100000,
            group_size=group_size,
            top_k=top_k,
            int8_group_size=group_size,
            int8_outlier_top_k=top_k,
            decode_tokens=decode_tokens,
            max_seq_len=max_seq_len,
            device=device,
            domain_aware=False,
            delta_strategy="delta_noaffine_int4_k1",
            unigram_strategy="unigram_int4_k4",
        )
        # Inject the table built from prompt_before
        pipeline.table = table

        local_prompt_refs = pipeline._build_local_prompt_refs(ids_after, hidden_after)

        decode_result = pipeline.process_decode(
            past_kv=past_kv,
            next_tok=next_tok,
            input_ids=ids_after,
            prefill_hidden=hidden_after,
            local_prompt_refs=local_prompt_refs,
            phase="test",
        )
        pipeline.shutdown()

        # Record decode results for all schemes
        num_groups = hidden_dim // group_size
        for scheme_name in ALL_SCHEMES:
            if scheme_name in ("fp16", "int8", "int4"):
                # Simulated decode transfer
                sim_decode = simulate_compression(
                    seq_len=decode_result.decode_tokens,
                    hidden_dim=hidden_dim,
                    group_size=group_size,
                    top_k=top_k,
                    bandwidths=bandwidths,
                )
                sr = sim_decode.schemes[scheme_name]
                rec = {
                    "event_id": event.id,
                    "scheme": scheme_name,
                    "phase": "decode",
                    "decode_tokens": decode_result.decode_tokens,
                    "total_bytes": sr.total_bytes,
                    "fp16_bytes": sim_decode.fp16_bytes,
                    "compression_ratio": sr.compression_ratio,
                }
                for bw in bandwidths:
                    rec[f"latency_{bw}mbps_ms"] = sr.latency_ms[bw]
                decode_records.append(rec)

            elif scheme_name == "delta-int4":
                # Actual delta-int4 from pipeline decode result
                rec = {
                    "event_id": event.id,
                    "scheme": "delta-int4",
                    "phase": "decode",
                    "decode_tokens": decode_result.decode_tokens,
                    "total_bytes": decode_result.total_transfer_bytes,
                    "fp16_bytes": decode_result.raw_fp16_bytes,
                    "compression_ratio": decode_result.compression_ratio,
                    "recon_cosine_mean": decode_result.recon_cosine_mean,
                    "recon_cosine_min": decode_result.recon_cosine_min,
                    "num_trigram": decode_result.num_trigram,
                    "num_bigram": decode_result.num_bigram,
                    "num_self_ref": decode_result.num_self_ref,
                    "num_unigram": decode_result.num_unigram,
                }
                for bw in bandwidths:
                    rec[f"latency_{bw}mbps_ms"] = _network_ms(
                        decode_result.total_transfer_bytes, bw,
                    )
                decode_records.append(rec)

            elif scheme_name == "delta-int8":
                # Derive delta-int8 from same tier distribution as delta-int4
                n_delta = (decode_result.num_trigram + decode_result.num_bigram
                           + decode_result.num_self_ref)
                n_unigram = decode_result.num_unigram

                # INT8 delta per position
                int8_meta_per_pos = (num_groups * 2 + num_groups * 2
                                     + num_groups * top_k * 2 + num_groups * top_k)
                int8_delta_bytes = n_delta * (hidden_dim + int8_meta_per_pos + 4 + 8)
                int8_uni_bytes = n_unigram * (hidden_dim + int8_meta_per_pos)
                total_delta_int8 = int8_delta_bytes + int8_uni_bytes
                fp16_bytes_d = decode_result.decode_tokens * hidden_dim * 2

                rec = {
                    "event_id": event.id,
                    "scheme": "delta-int8",
                    "phase": "decode",
                    "decode_tokens": decode_result.decode_tokens,
                    "total_bytes": total_delta_int8,
                    "fp16_bytes": fp16_bytes_d,
                    "compression_ratio": fp16_bytes_d / max(total_delta_int8, 1),
                    "num_trigram": decode_result.num_trigram,
                    "num_bigram": decode_result.num_bigram,
                    "num_self_ref": decode_result.num_self_ref,
                    "num_unigram": decode_result.num_unigram,
                }
                for bw in bandwidths:
                    rec[f"latency_{bw}mbps_ms"] = _network_ms(total_delta_int8, bw)
                decode_records.append(rec)

        logger.info("  Decode: %d tokens, delta-int4 ratio=%.2fx",
                     decode_result.decode_tokens, decode_result.compression_ratio)

    # Cleanup
    del hidden_before, hidden_after, past_kv, next_tok, table
    gc.collect()
    torch.cuda.empty_cache()

    return prefill_records, decode_records, detail_records


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Trace compaction delta-coding experiment")
    parser.add_argument("--model", default="/root/share/models/Qwen2.5-32B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer-boundary", type=int, default=6)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE_PATH)
    parser.add_argument("--output-dir", default="results_trace_compaction")
    parser.add_argument("--bandwidths", nargs="+", type=int, default=BANDWIDTHS_MBPS)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # Load trace events
    events = load_trace_events(args.trace_path)
    logger.info("Loaded %d trace events", len(events))

    # Load model
    logger.info("Loading model %s on %s...", args.model, device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map={"": device}, trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded. hidden_size=%d", model.config.hidden_size)

    # GPU warmup
    wids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    _ = model(wids, output_hidden_states=True, use_cache=False)
    del wids, _
    torch.cuda.synchronize()

    # Run experiments — each event independently
    all_prefill: List[Dict] = []
    all_decode: List[Dict] = []
    all_detail: List[Dict] = []

    for event in events:
        prefill_recs, decode_recs, detail_recs = run_single_event(
            event=event,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layer_boundary=args.layer_boundary,
            group_size=args.group_size,
            top_k=args.top_k,
            decode_tokens=args.decode_tokens,
            max_seq_len=args.max_seq_len,
            bandwidths=args.bandwidths,
        )
        all_prefill.extend(prefill_recs)
        all_decode.extend(decode_recs)
        all_detail.extend(detail_recs)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_records(all_prefill, out_dir / "prefill_summary.parquet")
    _save_records(all_decode, out_dir / "decode_summary.parquet")
    _save_records(all_detail, out_dir / "raw_detail.parquet")

    # Save config
    config = {
        "model": args.model,
        "layer_boundary": args.layer_boundary,
        "decode_tokens": args.decode_tokens,
        "group_size": args.group_size,
        "top_k": args.top_k,
        "max_seq_len": args.max_seq_len,
        "trace_path": args.trace_path,
        "bandwidths": args.bandwidths,
        "num_events": len(events),
        "schemes": ALL_SCHEMES,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("All results saved to %s", out_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
