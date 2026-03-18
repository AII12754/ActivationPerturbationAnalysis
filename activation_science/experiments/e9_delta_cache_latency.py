"""E9: Delta-cache latency experiment.

Measures the precise GPU latency overhead of every operation in the
delta-coding pipeline (reference lookup, affine transform, Int4 quantize
with top-k outliers, dequantize, reconstruct) using real model activations
in a batched decode serving scenario.

Supports three lookup strategies:
- exact: Full cosine-similarity search over entire history.
- early_exit: Chunked reverse-recency search with early termination.
- lsh: SimHash-based approximate lookup with compact-gather.

Motivated by E0 reuse feasibility results showing viable activation reuse
at layers 1-8 and 57-64 in a 3-stage pipeline-parallel serving setup.
"""

from __future__ import annotations

import gc
import logging
import os
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill, decode_step, select_next_token
from ..metrics.delta_coding import (
    LookupStats,
    PerRequestHistoryBuffer,
    EarlyExitHistoryBuffer,
    LSHHistoryBuffer,
    RecentWindowHistoryBuffer,
    compute_reconstruction_quality,
    compute_transfer_size,
    create_history_buffer,
    run_baseline_memcpy,
    run_timed_decode_pipeline,
    run_timed_encode_pipeline,
    run_timed_encode_pipeline_with_stats,
)
from .base import BaseExperiment

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Table schemas
# -------------------------------------------------------------------
LATENCY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "batch_size", "group_size", "top_k", "history_len", "layer_boundary", "hidden_dim",
    "operation", "mean_ms", "std_ms", "min_ms", "max_ms",
    "transfer_bytes", "baseline_transfer_bytes", "compression_ratio",
    "lookup_strategy", "similarity_threshold", "chunk_size",
    "tokens_searched_mean", "tokens_searched_max", "early_exit_rate",
    "num_hash_tables", "num_planes_per_table", "bucket_hit_rate", "num_candidates_mean",
    "window_size", "max_hamming",
]

QUALITY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "batch_size", "group_size", "top_k", "history_len", "layer_boundary", "hidden_dim",
    "cosine_similarity_mean", "cosine_similarity_min", "mse_mean", "mse_max",
    "max_abs_error_mean", "max_abs_error_max",
    "lookup_strategy", "similarity_threshold", "chunk_size",
    "num_hash_tables", "num_planes_per_table", "ref_match_rate",
    "window_size", "max_hamming",
]


class DeltaCacheLatencyExperiment(BaseExperiment):
    """E9: Delta-cache latency profiling experiment."""

    experiment_id = "e9"
    experiment_name = "delta_cache_latency"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"latency": LATENCY_COLUMNS, "quality": QUALITY_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "delta_cache"

    @classmethod
    def build_sweep_jobs(cls, config: Dict[str, Any]) -> List[ExperimentJob]:
        """Build sweep jobs over (dataset, context_length, prompt_index).

        Inner sweep (batch_size, group_size, top_k, history_len) happens
        inside ``run()`` to avoid redundant forward passes.
        """
        sweep = config["sweep"]
        context_lengths = sweep.get("context_lengths", [2048])
        num_prompts = sweep.get("num_prompts_per_length", 2)
        dataset_specs = resolve_dataset_list(config)

        jobs: List[ExperimentJob] = []
        for ds_spec, ctx_len, prompt_idx in product(
            dataset_specs, context_lengths, range(num_prompts),
        ):
            params = {
                "dataset_name": ds_spec["name"],
                "dataset_config": ds_spec["config"],
                "dataset_split": ds_spec["split"],
                "context_length": ctx_len,
                "prompt_index": prompt_idx,
            }
            job_id = make_experiment_id(params)
            jobs.append(ExperimentJob(job_id=job_id, params=params))

        logger.info("Built %d delta-cache sweep jobs (%d datasets).", len(jobs), len(dataset_specs))
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
        dc_cfg = config.get("delta_cache", {})

        num_decode = dc_cfg.get("num_decode_tokens", 32)
        batch_sizes = dc_cfg.get("batch_sizes", [1, 4, 8, 16, 32, 64])
        group_sizes = dc_cfg.get("group_sizes", [128, 256])
        top_ks = dc_cfg.get("top_ks", [1, 2, 4])
        history_lens = dc_cfg.get("history_lens", [512, 1024, 2048])
        layer_boundaries = dc_cfg.get("layer_boundaries", [4, 60])
        num_warmup = dc_cfg.get("num_warmup", 3)
        num_timing_runs = dc_cfg.get("num_timing_runs", 10)
        act_dtype = resolve_dtype(dc_cfg.get("activation_dtype", "float16"))

        # Strategy configs
        early_exit_cfg = dc_cfg.get("early_exit", {})
        lsh_cfg = dc_cfg.get("lsh", {})
        recent_window_cfg = dc_cfg.get("recent_window", {})
        strategy_layer_filter = dc_cfg.get("strategy_layer_filter", {
            "early_exit": [4], "recent_window": [4], "lsh": [60], "exact": layer_boundaries,
        })

        strategy_configs: List[Tuple[str, dict]] = [("exact", {})]
        for thresh, chunk_sz in product(
            early_exit_cfg.get("thresholds", [0.95]),
            early_exit_cfg.get("chunk_sizes", [64]),
        ):
            strategy_configs.append((
                "early_exit",
                {"similarity_threshold": thresh, "chunk_size": chunk_sz},
            ))
        for ws in recent_window_cfg.get("window_sizes", [256]):
            strategy_configs.append((
                "recent_window",
                {"window_size": ws},
            ))
        for n_tables, n_planes, max_h in product(
            lsh_cfg.get("num_tables", [8]),
            lsh_cfg.get("num_planes", [12]),
            lsh_cfg.get("max_hamming", [0]),
        ):
            strategy_configs.append((
                "lsh",
                {"num_tables": n_tables, "num_planes": n_planes, "max_hamming": max_h},
            ))

        max_batch = max(batch_sizes)
        max_history = max(history_lens)

        device = next(model.parameters()).device
        context_length = config.get("context_length", kwargs.get("context_length", 2048))

        latency_records: List[Dict[str, Any]] = []
        quality_records: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------
        # Phase 1: Collect activations from max_batch independent sequences
        # ---------------------------------------------------------------
        # prefill_activations[lb][seq_idx] = (context_length, hidden_dim)
        # decode_activations[lb][seq_idx] = (num_decode, hidden_dim)
        prefill_activations: Dict[int, List[torch.Tensor]] = {lb: [] for lb in layer_boundaries}
        decode_activations: Dict[int, List[torch.Tensor]] = {lb: [] for lb in layer_boundaries}
        hidden_dim = None

        logger.info("Phase 1: Collecting activations from %d sequences.", max_batch)

        for seq_idx in range(max_batch):
            # Use different offsets into the prompt for diversity
            offset = seq_idx * 50
            text = prompt_text[offset:] if offset < len(prompt_text) else prompt_text
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            input_ids = input_ids[:context_length]
            if len(input_ids) < 2:
                input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[:context_length]

            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Prefill
            batch = prefill(model, input_tensor, use_cache=True)
            all_hidden = batch.hidden_states
            num_layers = batch.num_layers

            if hidden_dim is None:
                hidden_dim = all_hidden[0].shape[2]

            for lb in layer_boundaries:
                layer_idx = min(lb, num_layers)
                h = all_hidden[layer_idx].squeeze(0).to(act_dtype)  # (seq_len, hidden_dim)
                prefill_activations[lb].append(h.detach())

            past_key_values = batch.past_key_values
            logits = batch.last_logits

            # Decode
            seq_decode_acts: Dict[int, List[torch.Tensor]] = {lb: [] for lb in layer_boundaries}
            for step in range(num_decode):
                next_token = select_next_token(logits, do_sample=False)
                batch = decode_step(model, next_token, past_key_values)
                for lb in layer_boundaries:
                    layer_idx = min(lb, batch.num_layers)
                    h = batch.hidden_states[layer_idx].squeeze(0).squeeze(0).to(act_dtype)
                    seq_decode_acts[lb].append(h.detach())
                past_key_values = batch.past_key_values
                logits = batch.last_logits
                del batch

            for lb in layer_boundaries:
                decode_activations[lb].append(torch.stack(seq_decode_acts[lb], dim=0))

            # Free KV-cache
            del past_key_values, logits
            gc.collect()
            torch.cuda.empty_cache()

            if (seq_idx + 1) % 8 == 0:
                logger.info("  Collected %d / %d sequences.", seq_idx + 1, max_batch)

        logger.info("Phase 1 complete. hidden_dim=%d", hidden_dim)

        # ---------------------------------------------------------------
        # Phase 2: Run timed delta-coding for each config combination
        # ---------------------------------------------------------------
        logger.info("Phase 2: Timing delta-coding pipeline.")

        for lb in layer_boundaries:
            for batch_size in batch_sizes:
                if batch_size > len(prefill_activations[lb]):
                    continue

                for history_len in history_lens:
                    # Select new_acts from decode step 0
                    new_acts = torch.stack(
                        [decode_activations[lb][i][0] for i in range(batch_size)],
                        dim=0,
                    )  # (batch_size, hidden_dim)

                    # Compute exact references ONCE for ref_match_rate comparison
                    exact_buf = PerRequestHistoryBuffer(
                        num_requests=batch_size,
                        max_history_len=history_len,
                        hidden_dim=hidden_dim,
                        device=device,
                        dtype=act_dtype,
                    )
                    for i in range(batch_size):
                        pa = prefill_activations[lb][i]
                        exact_buf.add_batch_prefill(i, pa[:history_len])
                    exact_ref_idx, _ = exact_buf.find_best_references(new_acts)

                    # Baseline memcpy (once per batch_size)
                    new_acts_fp16 = new_acts.to(torch.float16)
                    baseline_timing = run_baseline_memcpy(
                        new_acts_fp16, num_warmup=num_warmup, num_runs=num_timing_runs,
                    )
                    baseline_bytes = new_acts_fp16.nelement() * new_acts_fp16.element_size()

                    for strategy, strat_kwargs in strategy_configs:
                        # Filter strategies by layer
                        allowed_layers = strategy_layer_filter.get(strategy, layer_boundaries)
                        if lb not in allowed_layers:
                            continue

                        # Build history buffer for this strategy
                        hist_buf = create_history_buffer(
                            strategy, batch_size, history_len, hidden_dim,
                            device, act_dtype, **strat_kwargs,
                        )
                        for i in range(batch_size):
                            pa = prefill_activations[lb][i]
                            hist_buf.add_batch_prefill(i, pa[:history_len])

                        # Strategy metadata for records
                        strat_meta = {
                            "lookup_strategy": strategy,
                            "similarity_threshold": strat_kwargs.get("similarity_threshold"),
                            "chunk_size": strat_kwargs.get("chunk_size"),
                            "num_hash_tables": strat_kwargs.get("num_tables"),
                            "num_planes_per_table": strat_kwargs.get("num_planes"),
                            "window_size": strat_kwargs.get("window_size"),
                            "max_hamming": strat_kwargs.get("max_hamming"),
                        }

                        for group_size in group_sizes:
                            if hidden_dim % group_size != 0:
                                logger.warning(
                                    "Skipping group_size=%d (hidden_dim=%d not divisible).",
                                    group_size, hidden_dim,
                                )
                                continue

                            for top_k in top_ks:
                                if top_k >= group_size:
                                    continue

                                logger.debug(
                                    "  lb=%d bs=%d hist=%d gs=%d k=%d strat=%s",
                                    lb, batch_size, history_len, group_size, top_k, strategy,
                                )

                                # Encode pipeline with stats
                                packet, encode_timings, lookup_stats = run_timed_encode_pipeline_with_stats(
                                    new_acts, hist_buf, group_size, top_k,
                                    num_warmup=num_warmup, num_runs=num_timing_runs,
                                )

                                # Decode pipeline
                                reconstructed, decode_timings = run_timed_decode_pipeline(
                                    packet, hist_buf, hidden_dim,
                                    num_warmup=num_warmup, num_runs=num_timing_runs,
                                )

                                # Transfer size and compression ratio
                                transfer_bytes = compute_transfer_size(packet)
                                compression_ratio = baseline_bytes / max(transfer_bytes, 1)

                                # Reconstruction quality
                                quality = compute_reconstruction_quality(new_acts, reconstructed)

                                # Reference match rate vs exact
                                ref_match_rate = (packet.ref_indices == exact_ref_idx).float().mean().item()

                                # Lookup stats fields
                                ls_fields = {}
                                if lookup_stats is not None:
                                    ls_fields = {
                                        "tokens_searched_mean": lookup_stats.tokens_searched_mean,
                                        "tokens_searched_max": lookup_stats.tokens_searched_max,
                                        "early_exit_rate": lookup_stats.early_exit_rate,
                                        "bucket_hit_rate": lookup_stats.bucket_hit_rate,
                                        "num_candidates_mean": lookup_stats.num_candidates_mean,
                                    }

                                # Base metadata for this config
                                base_meta = {
                                    "batch_size": batch_size,
                                    "group_size": group_size,
                                    "top_k": top_k,
                                    "history_len": history_len,
                                    "layer_boundary": lb,
                                    "hidden_dim": hidden_dim,
                                }
                                base_meta.update(strat_meta)

                                # Emit latency records — one per operation
                                all_timings = {}
                                all_timings.update(encode_timings)
                                all_timings.update(decode_timings)
                                all_timings["baseline_memcpy"] = baseline_timing

                                for op_name, timing in all_timings.items():
                                    rec = dict(base_meta)
                                    rec["operation"] = op_name
                                    rec["mean_ms"] = timing.mean_ms
                                    rec["std_ms"] = timing.std_ms
                                    rec["min_ms"] = timing.min_ms
                                    rec["max_ms"] = timing.max_ms
                                    rec["transfer_bytes"] = transfer_bytes
                                    rec["baseline_transfer_bytes"] = baseline_bytes
                                    rec["compression_ratio"] = compression_ratio
                                    rec.update(ls_fields)
                                    latency_records.append(rec)

                                # Emit quality record
                                q_rec = dict(base_meta)
                                q_rec.update(quality)
                                q_rec["ref_match_rate"] = ref_match_rate
                                q_rec.update(ls_fields)
                                quality_records.append(q_rec)

                        # Free strategy buffer
                        del hist_buf
                        torch.cuda.empty_cache()

                    # Free exact buffer
                    del exact_buf
                    torch.cuda.empty_cache()

        # Clean up phase-1 tensors
        del prefill_activations, decode_activations
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            "E9 complete: %d latency records, %d quality records.",
            len(latency_records), len(quality_records),
        )

        return {"latency": latency_records, "quality": quality_records}
