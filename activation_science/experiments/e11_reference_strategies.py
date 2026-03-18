"""E11: Reference strategy comparison experiment.

Tests 9 reference strategies (+ 1 hybrid) for delta-coding prefill
activations. All strategies are evaluated on the same prefill activations
for direct comparability.

Strategies:
  1. Static (baseline) — single-token, no context
  2. N-gram bigram — 2-token input, last-position hidden state
  3. N-gram trigram — 3-token input, last-position hidden state
  4. Cluster mean — per-token mean across all occurrences (oracle)
  5. Sequential — previous position's real activation (semi-deployable)
  6. Sliding window — mean of W preceding real activations (semi)
  7. Global mean — mean of all real activations (oracle)
  8. EMA — exponential moving average of preceding activations (semi)
  9. Low-rank SVD — rank-r truncated SVD of full activation matrix (oracle)
  10. Hybrid — static + sequential with threshold switch
"""

from __future__ import annotations

import gc
import logging
import time
from itertools import product
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill
from ..metrics.delta_coding import gpu_timed
from ..metrics.static_delta import StaticTokenTable
from ..metrics.reference_strategies import (
    build_static_references,
    build_ngram_references,
    build_cluster_mean_references,
    build_sequential_references,
    build_sliding_window_references,
    build_global_mean_references,
    build_ema_references,
    build_lowrank_references,
    build_hybrid_references,
    evaluate_strategy,
)
from .base import BaseExperiment

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Table schemas
# -------------------------------------------------------------------
QUALITY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer_boundary", "hidden_dim", "group_size", "top_k",
    "strategy", "strategy_param",
    "num_unique_refs",
    "raw_cosine_mean", "raw_cosine_min", "raw_cosine_max", "raw_cosine_std",
    "raw_mse_mean", "raw_mse_max",
    "cosine_similarity_mean", "cosine_similarity_min",
    "mse_mean", "mse_max", "max_abs_error_mean", "max_abs_error_max",
    "delta_l2_norm_mean", "delta_l2_norm_max",
    "delta_max_abs_mean", "delta_max_abs_max",
    "affine_scale_mean", "affine_scale_std",
    "affine_bias_mean", "affine_bias_std",
    "transfer_bytes", "compression_ratio",
    "ref_build_time_ms",
]

POSITION_QUALITY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer_boundary", "hidden_dim", "group_size", "top_k",
    "strategy", "strategy_param",
    "bin_start", "bin_end", "num_positions",
    "raw_cosine_mean", "raw_cosine_min",
    "raw_mse_mean", "raw_mse_max",
    "cosine_similarity_mean", "cosine_similarity_min",
    "mse_mean", "mse_max", "max_abs_error_mean", "max_abs_error_max",
]

LATENCY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer_boundary", "hidden_dim", "group_size", "top_k",
    "strategy", "strategy_param",
    "ref_build_mean_ms", "ref_build_std_ms", "ref_build_min_ms", "ref_build_max_ms",
]


class ReferenceStrategyExperiment(BaseExperiment):
    """E11: Reference strategy comparison for prefill compression."""

    experiment_id = "e11"
    experiment_name = "reference_strategies"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {
            "quality": QUALITY_COLUMNS,
            "position_quality": POSITION_QUALITY_COLUMNS,
            "latency": LATENCY_COLUMNS,
        }

    @classmethod
    def default_config_section(cls) -> str:
        return "reference_strategies"

    @classmethod
    def build_sweep_jobs(cls, config: Dict[str, Any]) -> List[ExperimentJob]:
        sweep = config["sweep"]
        context_lengths = sweep.get("context_lengths", [512, 2048])
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

        logger.info("Built %d reference-strategy sweep jobs (%d datasets).", len(jobs), len(dataset_specs))
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
        rs_cfg = config.get("reference_strategies", {})

        layer_boundaries = rs_cfg.get("layer_boundaries", [1, 2, 3, 4, 5, 6, 7, 8])
        group_size = rs_cfg.get("group_size", 128)
        top_k = rs_cfg.get("top_k", 4)
        num_warmup = rs_cfg.get("num_warmup", 3)
        num_timing_runs = rs_cfg.get("num_timing_runs", 10)
        act_dtype = resolve_dtype(rs_cfg.get("activation_dtype", "float16"))
        static_batch_size = rs_cfg.get("static_table_batch_size", 256)
        position_bins = rs_cfg.get("position_bins", [[0, 64], [64, 256], [256, 512], [512, 2048]])
        strategies_cfg = rs_cfg.get("strategies", {})

        device = next(model.parameters()).device
        context_length = config.get("context_length", kwargs.get("context_length", 2048))

        quality_records: List[Dict[str, Any]] = []
        position_quality_records: List[Dict[str, Any]] = []
        latency_records: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------
        # Phase 1: Prefill — tokenize and extract real activations
        # ---------------------------------------------------------------
        logger.info("E11 Phase 1: Prefill — extracting real activations.")

        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = input_ids[:context_length]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        token_ids_1d = input_tensor.squeeze(0)  # (seq_len,)
        seq_len = token_ids_1d.shape[0]

        batch = prefill(model, input_tensor, use_cache=False)
        all_hidden = batch.hidden_states
        num_layers = batch.num_layers

        real_acts_by_layer: Dict[int, torch.Tensor] = {}
        hidden_dim = None
        for lb in layer_boundaries:
            layer_idx = min(lb, num_layers)
            h = all_hidden[layer_idx].squeeze(0).to(act_dtype)  # (seq_len, hidden_dim)
            real_acts_by_layer[lb] = h.detach()
            if hidden_dim is None:
                hidden_dim = h.shape[1]

        del batch, all_hidden
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            "E11 Phase 1 complete. seq_len=%d, hidden_dim=%d, unique_tokens=%d",
            seq_len, hidden_dim, token_ids_1d.unique().shape[0],
        )

        # ---------------------------------------------------------------
        # Phase 2 & 3: Build references + Evaluate per layer
        # ---------------------------------------------------------------
        logger.info("E11 Phase 2+3: Build references and evaluate strategies.")

        unique_token_ids = token_ids_1d.unique()

        for lb in layer_boundaries:
            layer_idx = min(lb, num_layers)
            real_acts = real_acts_by_layer[lb]

            # Build static table for this layer (reused by multiple strategies)
            static_table = StaticTokenTable(device=device, dtype=act_dtype)
            static_table.build_from_model(model, unique_token_ids, layer_idx, batch_size=static_batch_size)

            # Collect (strategy_name, strategy_param, ref_acts, info, build_time_ms)
            strategy_results: List[tuple] = []

            # --- 1. Static ---
            if strategies_cfg.get("static", True):
                ref, info, timing = self._timed_build(
                    build_static_references, token_ids_1d, static_table,
                    num_warmup=num_warmup, num_runs=num_timing_runs,
                )
                strategy_results.append(("static", "", ref, info, timing))

            # --- 2 & 3. N-gram ---
            ngram_cfg = strategies_cfg.get("ngram", {})
            if ngram_cfg if isinstance(ngram_cfg, bool) else ngram_cfg.get("enabled", False):
                n_values = ngram_cfg.get("n_values", [2, 3]) if isinstance(ngram_cfg, dict) else [2, 3]
                ngram_batch = ngram_cfg.get("batch_size", 128) if isinstance(ngram_cfg, dict) else 128
                for n_val in n_values:
                    ref, info, timing = self._timed_build(
                        build_ngram_references, model, token_ids_1d, layer_idx,
                        n=n_val, batch_size=ngram_batch, dtype=act_dtype, static_table=static_table,
                        num_warmup=num_warmup, num_runs=num_timing_runs,
                    )
                    label = {2: "bigram", 3: "trigram"}.get(n_val, f"{n_val}gram")
                    strategy_results.append((f"ngram_{label}", str(n_val), ref, info, timing))

            # --- 4. Cluster mean ---
            if strategies_cfg.get("cluster_mean", True):
                ref, info, timing = self._timed_build(
                    build_cluster_mean_references, real_acts, token_ids_1d,
                    num_warmup=num_warmup, num_runs=num_timing_runs,
                )
                strategy_results.append(("cluster_mean", "", ref, info, timing))

            # --- 5. Sequential ---
            if strategies_cfg.get("sequential", True):
                ref, info, timing = self._timed_build(
                    build_sequential_references, real_acts, static_table, token_ids_1d,
                    num_warmup=num_warmup, num_runs=num_timing_runs,
                )
                strategy_results.append(("sequential", "", ref, info, timing))

            # --- 6. Sliding window ---
            sw_cfg = strategies_cfg.get("sliding_window", {})
            if sw_cfg if isinstance(sw_cfg, bool) else sw_cfg.get("enabled", False):
                window_sizes = sw_cfg.get("window_sizes", [4, 16]) if isinstance(sw_cfg, dict) else [4, 16]
                for ws in window_sizes:
                    ref, info, timing = self._timed_build(
                        build_sliding_window_references, real_acts, ws, static_table, token_ids_1d,
                        num_warmup=num_warmup, num_runs=num_timing_runs,
                    )
                    strategy_results.append((f"sliding_window_w{ws}", str(ws), ref, info, timing))

            # --- 7. Global mean ---
            if strategies_cfg.get("global_mean", True):
                ref, info, timing = self._timed_build(
                    build_global_mean_references, real_acts,
                    num_warmup=num_warmup, num_runs=num_timing_runs,
                )
                strategy_results.append(("global_mean", "", ref, info, timing))

            # --- 8. EMA ---
            ema_cfg = strategies_cfg.get("ema", {})
            if ema_cfg if isinstance(ema_cfg, bool) else ema_cfg.get("enabled", False):
                alphas = ema_cfg.get("alphas", [0.3]) if isinstance(ema_cfg, dict) else [0.3]
                for alpha in alphas:
                    ref, info, timing = self._timed_build(
                        build_ema_references, real_acts, alpha, static_table, token_ids_1d,
                        num_warmup=num_warmup, num_runs=num_timing_runs,
                    )
                    strategy_results.append((f"ema_a{alpha}", str(alpha), ref, info, timing))

            # --- 9. Low-rank SVD ---
            lr_cfg = strategies_cfg.get("lowrank", {})
            if lr_cfg if isinstance(lr_cfg, bool) else lr_cfg.get("enabled", False):
                ranks = lr_cfg.get("ranks", [32]) if isinstance(lr_cfg, dict) else [32]
                for rank in ranks:
                    ref, info, timing = self._timed_build(
                        build_lowrank_references, real_acts, rank=rank,
                        num_warmup=num_warmup, num_runs=num_timing_runs,
                    )
                    strategy_results.append((f"lowrank_r{rank}", str(rank), ref, info, timing))

            # --- 10. Hybrid ---
            hybrid_cfg = strategies_cfg.get("hybrid", {})
            if hybrid_cfg if isinstance(hybrid_cfg, bool) else hybrid_cfg.get("enabled", False):
                threshold = hybrid_cfg.get("cosine_threshold", 0.95) if isinstance(hybrid_cfg, dict) else 0.95
                ref, info, timing = self._timed_build(
                    build_hybrid_references, real_acts, token_ids_1d, static_table, threshold=threshold,
                    num_warmup=num_warmup, num_runs=num_timing_runs,
                )
                strategy_results.append((f"hybrid_t{threshold}", str(threshold), ref, info, timing))

            # ---------------------------------------------------------------
            # Phase 3: Evaluate all strategies for this layer
            # ---------------------------------------------------------------
            base_meta = {
                "layer_boundary": lb,
                "hidden_dim": hidden_dim,
                "group_size": group_size,
                "top_k": top_k,
            }

            for strategy_name, strategy_param, ref_acts, info, build_timing in strategy_results:
                logger.debug("  Evaluating lb=%d strategy=%s", lb, strategy_name)

                metrics = evaluate_strategy(real_acts, ref_acts, group_size, top_k)

                # --- Quality record ---
                q_rec = dict(base_meta)
                q_rec["strategy"] = strategy_name
                q_rec["strategy_param"] = strategy_param
                q_rec["num_unique_refs"] = info.get("num_unique_refs", 0)
                q_rec["ref_build_time_ms"] = build_timing.mean_ms
                q_rec.update(metrics)
                quality_records.append(q_rec)

                # --- Latency record ---
                l_rec = dict(base_meta)
                l_rec["strategy"] = strategy_name
                l_rec["strategy_param"] = strategy_param
                l_rec["ref_build_mean_ms"] = build_timing.mean_ms
                l_rec["ref_build_std_ms"] = build_timing.std_ms
                l_rec["ref_build_min_ms"] = build_timing.min_ms
                l_rec["ref_build_max_ms"] = build_timing.max_ms
                latency_records.append(l_rec)

                # --- Position-binned quality ---
                # Compute raw cosine for binning
                raw_cos = F.cosine_similarity(real_acts.float(), ref_acts.float(), dim=-1)
                raw_diff = real_acts.float() - ref_acts.float()
                raw_mse_per_pos = (raw_diff ** 2).mean(dim=-1)

                # Reconstruct for per-bin quality (reuse evaluate_strategy logic)
                from ..metrics.delta_coding import (
                    compute_affine_params, apply_affine, compute_delta,
                    groupwise_int4_quantize_topk, groupwise_int4_dequantize_topk,
                    reconstruct_activation, compute_reconstruction_quality,
                )
                scale, bias = compute_affine_params(real_acts, ref_acts)
                ref_t = apply_affine(ref_acts, scale, bias)
                delta = compute_delta(real_acts, ref_t)
                packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(delta, group_size, top_k)
                dequant = groupwise_int4_dequantize_topk(packed, scales, zeros, tv, ti, group_size, hidden_dim)
                reconstructed = reconstruct_activation(dequant, ref_acts, scale.float(), bias.float())

                for bin_start, bin_end in position_bins:
                    if bin_start >= seq_len:
                        continue
                    actual_end = min(bin_end, seq_len)
                    num_positions = actual_end - bin_start
                    if num_positions <= 0:
                        continue

                    bin_quality = compute_reconstruction_quality(
                        real_acts[bin_start:actual_end],
                        reconstructed[bin_start:actual_end],
                    )
                    bin_raw_cos = raw_cos[bin_start:actual_end]
                    bin_raw_mse = raw_mse_per_pos[bin_start:actual_end]

                    pq_rec = dict(base_meta)
                    pq_rec["strategy"] = strategy_name
                    pq_rec["strategy_param"] = strategy_param
                    pq_rec["bin_start"] = bin_start
                    pq_rec["bin_end"] = actual_end
                    pq_rec["num_positions"] = num_positions
                    pq_rec["raw_cosine_mean"] = bin_raw_cos.mean().item()
                    pq_rec["raw_cosine_min"] = bin_raw_cos.min().item()
                    pq_rec["raw_mse_mean"] = bin_raw_mse.mean().item()
                    pq_rec["raw_mse_max"] = bin_raw_mse.max().item()
                    pq_rec.update(bin_quality)
                    position_quality_records.append(pq_rec)

                del ref_acts, reconstructed, delta
                torch.cuda.empty_cache()

            del static_table
            torch.cuda.empty_cache()

        # Cleanup
        del real_acts_by_layer
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            "E11 complete: %d quality, %d position_quality, %d latency records.",
            len(quality_records), len(position_quality_records), len(latency_records),
        )

        return {
            "quality": quality_records,
            "position_quality": position_quality_records,
            "latency": latency_records,
        }

    @staticmethod
    def _timed_build(fn, *args, num_warmup: int = 3, num_runs: int = 10, **kwargs):
        """Time a reference builder function using CUDA events.

        Returns (ref_acts, info_dict, TimingResult).
        N-gram builders require model forward passes that are expensive,
        so only 1 warmup and 1 run for those — timing is dominated by
        the model forward pass anyway.
        """
        from ..metrics.delta_coding import TimingResult

        # Detect if this is an n-gram build (has model as first arg)
        # N-gram builds are expensive — only run once
        is_ngram = hasattr(args[0], 'parameters') if args else False

        if is_ngram:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ref_acts, info = fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            timing = TimingResult(
                mean_ms=elapsed, std_ms=0.0,
                min_ms=elapsed, max_ms=elapsed, num_runs=1,
            )
            return ref_acts, info, timing

        # Standard timed build
        result, timing = gpu_timed(fn, *args, num_warmup=num_warmup, num_runs=num_runs, **kwargs)
        ref_acts, info = result
        return ref_acts, info, timing
