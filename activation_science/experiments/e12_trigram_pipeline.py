"""E12: Integrated trigram delta-coding pipeline experiment.

End-to-end simulation of a pre-built trigram table with online updates,
cascading fallback (trigram → bigram → self-ref → unigram), and
appropriate encoding per tier.

Each ``run()`` call processes multiple prompts in sequence, maintaining the
NgramTable across requests to simulate realistic table growth.  The sweep
is over (dataset, context_length) only.

Output tables:
  1. pipeline_quality — per-request aggregate quality & compression
  2. tier_detail — per-tier breakdown within each request
  3. table_growth — table size trajectory after each request
  4. latency — per-request timing breakdown
"""

from __future__ import annotations

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill
from ..core.datasets import PromptGenerator
from ..metrics.delta_coding import (
    compute_affine_params,
    apply_affine,
    compute_delta,
    groupwise_int4_quantize_topk,
    groupwise_int4_dequantize_topk,
    reconstruct_activation,
    compute_reconstruction_quality,
    compute_transfer_size,
    DeltaPacket,
    groupwise_int8_quantize_topk,
    groupwise_int8_dequantize_topk,
    compute_transfer_size_int8_outlier,
)
from ..metrics.ngram_table import NgramTable
from .base import BaseExperiment

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Table schemas
# -------------------------------------------------------------------
PIPELINE_QUALITY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "context_length",
    "request_index",
    "seq_len", "hidden_dim", "layer_boundary",
    "group_size", "top_k", "int8_group_size", "int8_outlier_top_k",
    # Tier counts
    "num_trigram", "num_bigram", "num_self_ref", "num_unigram",
    "pct_trigram", "pct_bigram", "pct_self_ref", "pct_unigram",
    # Raw reference quality (before delta coding)
    "raw_cosine_mean", "raw_cosine_min",
    # Overall reconstruction quality
    "cosine_similarity_mean", "cosine_similarity_min",
    "mse_mean", "mse_max",
    # Transfer
    "total_transfer_bytes", "raw_fp16_bytes", "compression_ratio",
    "transfer_bytes_trigram", "transfer_bytes_bigram",
    "transfer_bytes_self_ref", "transfer_bytes_unigram",
]

TIER_DETAIL_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "context_length",
    "request_index", "layer_boundary",
    "tier", "count",
    "raw_cosine_mean", "raw_cosine_min",
    "recon_cosine_mean", "recon_cosine_min",
    "mse_mean", "mse_max",
    "transfer_bytes",
]

TABLE_GROWTH_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "context_length",
    "request_index",
    "num_trigrams", "num_bigrams", "memory_bytes",
    "trigram_coverage", "bigram_coverage",
    "new_trigrams_added", "new_bigrams_added", "update_time_ms",
    "evicted_count",
]

LATENCY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "context_length",
    "request_index",
    "prefill_ms", "classify_ms",
    "encode_delta_ms", "encode_self_ref_ms", "encode_unigram_ms",
    "decode_ms", "table_update_ms", "total_ms",
]


class TrigramPipelineExperiment(BaseExperiment):
    """E12: Integrated trigram delta-coding pipeline."""

    experiment_id = "e12"
    experiment_name = "trigram_pipeline"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {
            "pipeline_quality": PIPELINE_QUALITY_COLUMNS,
            "tier_detail": TIER_DETAIL_COLUMNS,
            "table_growth": TABLE_GROWTH_COLUMNS,
            "latency": LATENCY_COLUMNS,
        }

    @classmethod
    def default_config_section(cls) -> str:
        return "trigram_pipeline"

    @classmethod
    def build_sweep_jobs(cls, config: Dict[str, Any]) -> List[ExperimentJob]:
        sweep = config["sweep"]
        context_lengths = sweep.get("context_lengths", [512, 2048])
        dataset_specs = resolve_dataset_list(config)

        jobs: List[ExperimentJob] = []
        for ds_spec, ctx_len in product(dataset_specs, context_lengths):
            params = {
                "dataset_name": ds_spec["name"],
                "dataset_config": ds_spec["config"],
                "dataset_split": ds_spec["split"],
                "context_length": ctx_len,
                "prompt_index": 0,  # placeholder — run() handles multi-prompt internally
            }
            job_id = make_experiment_id(params)
            jobs.append(ExperimentJob(job_id=job_id, params=params))

        logger.info("Built %d trigram-pipeline sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run(
        self,
        model,
        tokenizer,
        prompt_text: str,
        config: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        tp_cfg = config.get("trigram_pipeline", {})

        layer_boundary = tp_cfg.get("layer_boundary", 4)
        group_size = tp_cfg.get("group_size", 128)
        top_k = tp_cfg.get("top_k", 4)
        int8_group_size = tp_cfg.get("int8_group_size", 128)
        int8_outlier_top_k = tp_cfg.get("residual_top_k", 1)  # outlier top-k for Int8 unigram
        act_dtype = resolve_dtype(tp_cfg.get("activation_dtype", "float16"))
        ngram_batch_size = tp_cfg.get("ngram_batch_size", 128)
        table_mode = tp_cfg.get("table_mode", "cold")
        table_update_mode = tp_cfg.get("table_update_mode", "trigram_forward")
        sweep_cfg = config.get("sweep", {})
        num_prompts = sweep_cfg.get("num_prompts_per_length", 10)
        context_length = config.get("context_length", kwargs.get("context_length", 2048))

        device = next(model.parameters()).device

        pipeline_quality_records: List[Dict[str, Any]] = []
        tier_detail_records: List[Dict[str, Any]] = []
        table_growth_records: List[Dict[str, Any]] = []
        latency_records: List[Dict[str, Any]] = []

        # Create NgramTable — persists across all prompts (DAG trie storage)
        table_dtype_str = tp_cfg.get("table_dtype", "float16")
        table_dtype = resolve_dtype(table_dtype_str)
        max_table_entries = tp_cfg.get("max_table_entries", 0)
        ngram_table = NgramTable(device=device, dtype=table_dtype, max_entries=max_table_entries)

        # Get hidden_dim from model config (needed before first prefill for classify)
        hidden_dim = model.config.hidden_size

        # Single-thread executor for overlapping CPU table ops with GPU work
        executor = ThreadPoolExecutor(max_workers=1)

        # Generate multiple prompts
        prompt_cfg = config.get("prompts", {})
        ds_name = config.get("_dataset_name", prompt_cfg.get("dataset_name", "wikitext"))
        ds_config = config.get("_dataset_config", prompt_cfg.get("dataset_config", "default"))
        ds_split = config.get("_dataset_split", prompt_cfg.get("dataset_split", "train"))

        # Use smaller passages (15K chars ≈ 4K tokens) so every request gets
        # a unique prompt even for small datasets.  num_candidates must exceed
        # num_prompts to avoid index-wrapping / prompt recycling.
        prompt_gen = PromptGenerator(
            tokenizer=tokenizer,
            dataset_name=ds_name,
            dataset_config=ds_config,
            dataset_split=ds_split,
            num_candidates=max(num_prompts + 10, 120),
            passage_target_chars=8_000,
        )

        # GPU warmup: run a dummy forward pass to trigger CUDA lazy init,
        # JIT compilation, and memory pool allocation before timing begins.
        _warmup_ids = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
        _ = model(_warmup_ids, output_hidden_states=True, use_cache=False)
        del _warmup_ids, _
        torch.cuda.synchronize()

        for req_idx in range(num_prompts):
            t_total_start = time.perf_counter()

            # Get prompt text
            p_text = prompt_gen.generate(context_length, index=req_idx)

            # ===========================================================
            # Phase 1 + 2: Prefill (GPU) overlapped with classify (CPU)
            # ===========================================================
            input_ids = tokenizer.encode(p_text, add_special_tokens=False)
            input_ids = input_ids[:context_length]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            seq_len = len(input_ids)

            # Launch classify on background CPU thread (only needs token_ids + DAG dict)
            # GIL is released during CUDA kernel launches, so this runs freely.
            classify_future = executor.submit(
                ngram_table.classify_and_build_refs,
                input_ids, hidden_dim,
            )

            # Phase 1: Prefill on GPU (main thread) — runs concurrently with classify
            torch.cuda.synchronize()
            t_prefill_start = time.perf_counter()
            batch = prefill(model, input_tensor, use_cache=False)
            torch.cuda.synchronize()
            layer_idx = min(layer_boundary, batch.num_layers)
            real_acts = batch.hidden_states[layer_idx].squeeze(0).to(act_dtype)  # (seq_len, hidden_dim)

            del batch

            t_prefill_ms = (time.perf_counter() - t_prefill_start) * 1000.0

            # Warm start: seed table from first prompt
            if req_idx == 0 and table_mode == "warm":
                if table_update_mode == "sequence_extract":
                    ngram_table.update_from_hidden_states(input_ids, real_acts)
                else:
                    ngram_table.update_from_request(model, input_ids, layer_idx, ngram_batch_size)

            # Phase 2: Collect classify results (should already be done, hidden by prefill)
            t_classify_start = time.perf_counter()
            tiers, ref_acts, self_ref_sources, first_occ_map = classify_future.result()
            t_classify_ms = (time.perf_counter() - t_classify_start) * 1000.0

            # Partition indices by tier
            trigram_indices = [i for i, t in enumerate(tiers) if t == "trigram"]
            bigram_indices = [i for i, t in enumerate(tiers) if t == "bigram"]
            self_ref_indices = [i for i, t in enumerate(tiers) if t == "self_ref"]
            unigram_indices = [i for i, t in enumerate(tiers) if t == "unigram"]

            reconstructed = torch.zeros_like(real_acts)
            transfer_bytes_by_tier: Dict[str, int] = {
                "trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0,
            }

            # ===========================================================
            # Phase 6 (early launch): Update table in background CPU thread
            # Overlaps with encoding phases 3a-3c (GPU-only).
            # update_future.result() is collected after encoding completes,
            # before the next iteration's classify — no race on the DAG dict.
            # ===========================================================
            update_future = None
            if table_update_mode == "sequence_extract":
                update_future = executor.submit(
                    ngram_table.update_from_hidden_states,
                    input_ids, real_acts,
                )

            # ===========================================================
            # Phase 3a: Encode TRIGRAM + BIGRAM positions (batched)
            # ===========================================================
            torch.cuda.synchronize()
            t_encode_delta_start = time.perf_counter()

            delta_indices = trigram_indices + bigram_indices
            if delta_indices:
                idx_t = torch.tensor(delta_indices, dtype=torch.long, device=device)
                real_batch = real_acts[idx_t]      # (N, hidden_dim)
                ref_batch = ref_acts[idx_t]        # (N, hidden_dim)

                scale, bias = compute_affine_params(real_batch, ref_batch)
                ref_t = apply_affine(ref_batch, scale, bias)
                delta = compute_delta(real_batch, ref_t)
                packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)

                # Decode
                dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, hidden_dim)
                recon_batch = reconstruct_activation(dequant, ref_batch, scale, bias).to(act_dtype)
                reconstructed[idx_t] = recon_batch

                # Transfer size
                # Build a DeltaPacket per position for size calculation
                n_delta = len(delta_indices)
                packet = DeltaPacket(
                    quantized_data=packed, scales=scales, zero_points=zeros,
                    topk_values=tv, topk_indices=ti,
                    affine_scale=scale.to(torch.float16),
                    affine_bias=bias.to(torch.float16),
                    ref_indices=torch.zeros(n_delta, dtype=torch.long, device=device),
                    group_size=group_size, top_k=top_k,
                )
                total_delta_bytes = compute_transfer_size(packet)

                # Split between trigram and bigram
                n_tri = len(trigram_indices)
                n_bi = len(bigram_indices)
                if n_delta > 0:
                    per_pos = total_delta_bytes / n_delta
                    transfer_bytes_by_tier["trigram"] = int(per_pos * n_tri)
                    transfer_bytes_by_tier["bigram"] = int(per_pos * n_bi)

            torch.cuda.synchronize()
            t_encode_delta_ms = (time.perf_counter() - t_encode_delta_start) * 1000.0

            # ===========================================================
            # Phase 3b: Encode UNIGRAM positions (batched Int8 + outliers)
            # Moved BEFORE self-ref so all first-occurrence sources have
            # reconstructions available for batched self-ref.
            # ===========================================================
            t_encode_unigram_start = time.perf_counter()

            if unigram_indices:
                idx_u = torch.tensor(unigram_indices, dtype=torch.long, device=device)
                real_uni = real_acts[idx_u]

                int8_pkt = groupwise_int8_quantize_topk(real_uni, int8_group_size, int8_outlier_top_k)
                recon_uni = groupwise_int8_dequantize_topk(int8_pkt)
                transfer_bytes_by_tier["unigram"] = compute_transfer_size_int8_outlier(int8_pkt)

                reconstructed[idx_u] = recon_uni

            torch.cuda.synchronize()
            t_encode_unigram_ms = (time.perf_counter() - t_encode_unigram_start) * 1000.0

            # ===========================================================
            # Phase 3c: Encode SELF_REF positions (BATCHED)
            # All self-ref sources are first-occurrence positions with tier
            # trigram/bigram/unigram — all already reconstructed above.
            # ===========================================================
            t_encode_self_ref_start = time.perf_counter()

            if self_ref_indices:
                sorted_self_ref = sorted(self_ref_indices)
                idx_sr = torch.tensor(sorted_self_ref, dtype=torch.long, device=device)
                real_sr = real_acts[idx_sr]  # (N, hidden_dim)

                # Gather source reconstructions (all sources already filled)
                source_positions = [self_ref_sources[pos] for pos in sorted_self_ref]
                src_t = torch.tensor(source_positions, dtype=torch.long, device=device)
                ref_sr = reconstructed[src_t]  # (N, hidden_dim)

                # Store refs for raw cosine measurement
                ref_acts[idx_sr] = ref_sr

                # Batched affine + Int4 delta encoding
                scale, bias = compute_affine_params(real_sr, ref_sr)
                ref_t = apply_affine(ref_sr, scale, bias)
                delta = compute_delta(real_sr, ref_t)
                packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)

                dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, hidden_dim)
                recon_sr = reconstruct_activation(dequant, ref_sr, scale, bias).to(act_dtype)
                reconstructed[idx_sr] = recon_sr

                n_sr = len(sorted_self_ref)
                pkt = DeltaPacket(
                    quantized_data=packed, scales=scales, zero_points=zeros,
                    topk_values=tv, topk_indices=ti,
                    affine_scale=scale.to(torch.float16),
                    affine_bias=bias.to(torch.float16),
                    ref_indices=torch.zeros(n_sr, dtype=torch.long, device=device),
                    group_size=group_size, top_k=top_k,
                )
                transfer_bytes_by_tier["self_ref"] = compute_transfer_size(pkt)

            torch.cuda.synchronize()
            t_encode_self_ref_ms = (time.perf_counter() - t_encode_self_ref_start) * 1000.0

            # ===========================================================
            # Phase 4: Decode timing (already done inline above)
            # ===========================================================
            t_decode_ms = 0.0  # decode is interleaved with encode above

            # ===========================================================
            # Phase 5: Measure quality per tier + aggregate
            # ===========================================================
            base_meta = {
                "dataset_name": ds_name,
                "context_length": context_length,
                "request_index": req_idx,
                "layer_boundary": layer_boundary,
            }

            # -- Per-tier detail --
            for tier_name, tier_indices in [
                ("trigram", trigram_indices),
                ("bigram", bigram_indices),
                ("self_ref", self_ref_indices),
                ("unigram", unigram_indices),
            ]:
                if not tier_indices:
                    tier_detail_records.append({
                        **base_meta,
                        "tier": tier_name, "count": 0,
                        "raw_cosine_mean": 0.0, "raw_cosine_min": 0.0,
                        "recon_cosine_mean": 0.0, "recon_cosine_min": 0.0,
                        "mse_mean": 0.0, "mse_max": 0.0,
                        "transfer_bytes": 0,
                    })
                    continue

                idx_t = torch.tensor(tier_indices, dtype=torch.long, device=device)
                real_tier = real_acts[idx_t]
                ref_tier = ref_acts[idx_t]
                recon_tier = reconstructed[idx_t]

                # Raw cosine (ref vs real) — self_ref refs are now populated
                if tier_name == "unigram":
                    raw_cos = torch.zeros(len(tier_indices), device=device)
                else:
                    raw_cos = F.cosine_similarity(real_tier.float(), ref_tier.float(), dim=-1)

                # Reconstruction cosine
                recon_cos = F.cosine_similarity(real_tier.float(), recon_tier.float(), dim=-1)
                mse = ((real_tier.float() - recon_tier.float()) ** 2).mean(dim=-1)

                tier_detail_records.append({
                    **base_meta,
                    "tier": tier_name,
                    "count": len(tier_indices),
                    "raw_cosine_mean": raw_cos.mean().item(),
                    "raw_cosine_min": raw_cos.min().item() if raw_cos.numel() > 0 else 0.0,
                    "recon_cosine_mean": recon_cos.mean().item(),
                    "recon_cosine_min": recon_cos.min().item(),
                    "mse_mean": mse.mean().item(),
                    "mse_max": mse.max().item(),
                    "transfer_bytes": transfer_bytes_by_tier[tier_name],
                })

            # -- Aggregate quality --
            # Raw reference cosine (non-unigram positions only)
            non_uni_indices = trigram_indices + bigram_indices + sorted(self_ref_indices)
            if non_uni_indices:
                idx_nu = torch.tensor(non_uni_indices, dtype=torch.long, device=device)
                raw_cos_all = F.cosine_similarity(
                    real_acts[idx_nu].float(), ref_acts[idx_nu].float(), dim=-1,
                )
                raw_cosine_mean = raw_cos_all.mean().item()
                raw_cosine_min = raw_cos_all.min().item()
            else:
                raw_cosine_mean = 0.0
                raw_cosine_min = 0.0

            overall_cos = F.cosine_similarity(real_acts.float(), reconstructed.float(), dim=-1)
            overall_mse = ((real_acts.float() - reconstructed.float()) ** 2).mean(dim=-1)
            total_transfer = sum(transfer_bytes_by_tier.values())
            raw_fp16 = seq_len * hidden_dim * 2  # fp16 = 2 bytes

            pipeline_quality_records.append({
                **base_meta,
                "seq_len": seq_len,
                "hidden_dim": hidden_dim,
                "group_size": group_size,
                "top_k": top_k,
                "int8_group_size": int8_group_size,
                "int8_outlier_top_k": int8_outlier_top_k,
                "num_trigram": len(trigram_indices),
                "num_bigram": len(bigram_indices),
                "num_self_ref": len(self_ref_indices),
                "num_unigram": len(unigram_indices),
                "pct_trigram": len(trigram_indices) / max(seq_len, 1) * 100,
                "pct_bigram": len(bigram_indices) / max(seq_len, 1) * 100,
                "pct_self_ref": len(self_ref_indices) / max(seq_len, 1) * 100,
                "pct_unigram": len(unigram_indices) / max(seq_len, 1) * 100,
                "raw_cosine_mean": raw_cosine_mean,
                "raw_cosine_min": raw_cosine_min,
                "cosine_similarity_mean": overall_cos.mean().item(),
                "cosine_similarity_min": overall_cos.min().item(),
                "mse_mean": overall_mse.mean().item(),
                "mse_max": overall_mse.max().item(),
                "total_transfer_bytes": total_transfer,
                "raw_fp16_bytes": raw_fp16,
                "compression_ratio": raw_fp16 / max(total_transfer, 1),
                "transfer_bytes_trigram": transfer_bytes_by_tier["trigram"],
                "transfer_bytes_bigram": transfer_bytes_by_tier["bigram"],
                "transfer_bytes_self_ref": transfer_bytes_by_tier["self_ref"],
                "transfer_bytes_unigram": transfer_bytes_by_tier["unigram"],
            })

            # ===========================================================
            # Phase 6: Collect table update results
            # (launched before Phase 3 for sequence_extract mode)
            # ===========================================================
            t_update_start = time.perf_counter()

            if update_future is not None:
                new_tri, new_bi, update_ms = update_future.result()
            elif table_update_mode != "sequence_extract":
                new_tri, new_bi, update_ms = ngram_table.update_from_request(
                    model, input_ids, layer_idx, ngram_batch_size,
                )
            else:
                new_tri, new_bi, update_ms = 0, 0, 0.0

            t_update_ms = (time.perf_counter() - t_update_start) * 1000.0

            # Coverage: what fraction of this request's trigrams/bigrams are now in table
            total_trigrams_in_request = max(seq_len - 2, 0)
            total_bigrams_in_request = max(seq_len - 1, 0)
            tri_hits = len(trigram_indices) + len([i for i in self_ref_indices])
            bi_hits = len(bigram_indices)

            tbl_stats = ngram_table.stats
            table_growth_records.append({
                **base_meta,
                "num_trigrams": tbl_stats["num_trigrams"],
                "num_bigrams": tbl_stats["num_bigrams"],
                "memory_bytes": tbl_stats["memory_bytes"],
                "trigram_coverage": tri_hits / max(total_trigrams_in_request, 1),
                "bigram_coverage": bi_hits / max(total_bigrams_in_request, 1),
                "new_trigrams_added": new_tri,
                "new_bigrams_added": new_bi,
                "update_time_ms": update_ms,
                "evicted_count": ngram_table._last_evicted,
            })

            # -- Latency record --
            t_total_ms = (time.perf_counter() - t_total_start) * 1000.0

            latency_records.append({
                **base_meta,
                "prefill_ms": t_prefill_ms,
                "classify_ms": t_classify_ms,
                "encode_delta_ms": t_encode_delta_ms,
                "encode_self_ref_ms": t_encode_self_ref_ms,
                "encode_unigram_ms": t_encode_unigram_ms,
                "decode_ms": t_decode_ms,
                "table_update_ms": t_update_ms,
                "total_ms": t_total_ms,
            })

            logger.info(
                "E12 req=%d/%d  seq=%d  tri=%d  bi=%d  self=%d  uni=%d  "
                "cos=%.4f  ratio=%.2f×  table=%d/%d",
                req_idx + 1, num_prompts, seq_len,
                len(trigram_indices), len(bigram_indices),
                len(self_ref_indices), len(unigram_indices),
                overall_cos.mean().item(),
                raw_fp16 / max(total_transfer, 1),
                tbl_stats["num_trigrams"], tbl_stats["num_bigrams"],
            )

            # Cleanup between requests — avoid gc.collect()/empty_cache() per
            # iteration as they introduce unpredictable stalls (10–200 ms).
            # The CUDA allocator reuses freed blocks without explicit cache flush.
            del real_acts, reconstructed, ref_acts

        executor.shutdown(wait=False)

        # Cleanup table GPU tensors before returning
        del ngram_table
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return {
            "pipeline_quality": pipeline_quality_records,
            "tier_detail": tier_detail_records,
            "table_growth": table_growth_records,
            "latency": latency_records,
        }
