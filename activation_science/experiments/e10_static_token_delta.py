"""E10: Static token delta-coding experiment.

Tests whether pre-computed single-token activations (no context) can serve
as effective references for delta-coding prefill activations. At shallow
layers, token activations depend weakly on context, so a static embedding
table should yield high-quality deltas with minimal overhead.

Core hypothesis: context dependency grows with position and layer depth.
Layers 1-3 should be near-perfect, layer 4 the target sweet spot,
layers 5-8 degrading.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill
from ..metrics.delta_coding import (
    apply_affine,
    compute_affine_params,
    compute_delta,
    compute_reconstruction_quality,
    compute_transfer_size,
)
from ..metrics.static_delta import (
    StaticTokenTable,
    compute_affine_param_stats,
    compute_delta_magnitude_stats,
    run_timed_static_decode_pipeline,
    run_timed_static_encode_pipeline,
)
from .base import BaseExperiment

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Table schemas
# -------------------------------------------------------------------
LATENCY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer_boundary", "hidden_dim", "group_size", "top_k",
    "operation", "mean_ms", "std_ms", "min_ms", "max_ms",
    "transfer_bytes", "baseline_transfer_bytes", "compression_ratio",
    "bytes_per_token",
]

QUALITY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer_boundary", "hidden_dim", "group_size", "top_k",
    "raw_cosine_mean", "raw_cosine_min", "raw_cosine_max",
    "raw_mse_mean", "raw_mse_max",
    "cosine_similarity_mean", "cosine_similarity_min",
    "mse_mean", "mse_max", "max_abs_error_mean", "max_abs_error_max",
    "delta_l2_norm_mean", "delta_l2_norm_max",
    "delta_max_abs_mean", "delta_max_abs_max",
    "affine_scale_mean", "affine_scale_std",
    "affine_bias_mean", "affine_bias_std",
]

POSITION_QUALITY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer_boundary", "hidden_dim", "group_size", "top_k",
    "bin_start", "bin_end", "num_positions",
    "raw_cosine_mean", "raw_cosine_min",
    "raw_mse_mean", "raw_mse_max",
    "cosine_similarity_mean", "cosine_similarity_min",
    "mse_mean", "mse_max", "max_abs_error_mean", "max_abs_error_max",
]

BANDWIDTH_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer_boundary", "hidden_dim", "group_size", "top_k",
    "bandwidth_mbps", "raw_transfer_ms", "delta_transfer_ms",
    "speedup", "compression_ratio",
]


class StaticTokenDeltaExperiment(BaseExperiment):
    """E10: Static token delta-coding for prefill compression."""

    experiment_id = "e10"
    experiment_name = "static_token_delta"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {
            "latency": LATENCY_COLUMNS,
            "quality": QUALITY_COLUMNS,
            "position_quality": POSITION_QUALITY_COLUMNS,
            "bandwidth": BANDWIDTH_COLUMNS,
        }

    @classmethod
    def default_config_section(cls) -> str:
        return "static_delta"

    @classmethod
    def build_sweep_jobs(cls, config: Dict[str, Any]) -> List[ExperimentJob]:
        """Build sweep jobs over (dataset, context_length, prompt_index)."""
        sweep = config["sweep"]
        context_lengths = sweep.get("context_lengths", [256, 512, 1024, 2048])
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

        logger.info("Built %d static-delta sweep jobs (%d datasets).", len(jobs), len(dataset_specs))
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
        sd_cfg = config.get("static_delta", {})

        layer_boundaries = sd_cfg.get("layer_boundaries", [1, 2, 3, 4, 5, 6, 7, 8])
        group_sizes = sd_cfg.get("group_sizes", [128, 256])
        top_ks = sd_cfg.get("top_ks", [1, 2, 4])
        num_warmup = sd_cfg.get("num_warmup", 3)
        num_timing_runs = sd_cfg.get("num_timing_runs", 10)
        act_dtype = resolve_dtype(sd_cfg.get("activation_dtype", "float16"))
        static_batch_size = sd_cfg.get("static_table_batch_size", 256)
        position_bins = sd_cfg.get("position_bins", [[0, 64], [64, 256], [256, 512], [512, 2048]])
        bandwidths_mbps = sd_cfg.get("bandwidths_mbps", [200, 500, 1000])

        device = next(model.parameters()).device
        context_length = config.get("context_length", kwargs.get("context_length", 2048))

        latency_records: List[Dict[str, Any]] = []
        quality_records: List[Dict[str, Any]] = []
        position_quality_records: List[Dict[str, Any]] = []
        bandwidth_records: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------
        # Phase 1: Prefill — tokenize and extract real activations
        # ---------------------------------------------------------------
        logger.info("Phase 1: Prefill — extracting real activations.")

        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = input_ids[:context_length]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        token_ids_1d = input_tensor.squeeze(0)  # (seq_len,)
        seq_len = token_ids_1d.shape[0]

        batch = prefill(model, input_tensor, use_cache=False)
        all_hidden = batch.hidden_states
        num_layers = batch.num_layers

        # Extract per-layer activations: (seq_len, hidden_dim)
        real_acts_by_layer: Dict[int, torch.Tensor] = {}
        hidden_dim = None
        for lb in layer_boundaries:
            layer_idx = min(lb, num_layers)
            h = all_hidden[layer_idx].squeeze(0).to(act_dtype)  # (seq_len, hidden_dim)
            real_acts_by_layer[lb] = h.detach()
            if hidden_dim is None:
                hidden_dim = h.shape[1]

        # Free prefill outputs
        del batch, all_hidden
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            "Phase 1 complete. seq_len=%d, hidden_dim=%d, unique_tokens=%d",
            seq_len, hidden_dim, token_ids_1d.unique().shape[0],
        )

        # ---------------------------------------------------------------
        # Phase 2: Build static tables — one per layer
        # ---------------------------------------------------------------
        logger.info("Phase 2: Building static token tables.")

        unique_token_ids = token_ids_1d.unique()
        static_tables: Dict[int, StaticTokenTable] = {}

        for lb in layer_boundaries:
            layer_idx = min(lb, num_layers)
            table = StaticTokenTable(device=device, dtype=act_dtype)
            table.build_from_model(model, unique_token_ids, layer_idx, batch_size=static_batch_size)
            static_tables[lb] = table

        logger.info("Phase 2 complete. %d tables built.", len(static_tables))

        # ---------------------------------------------------------------
        # Phase 3: Sweep — encode/decode, quality, bandwidth
        # ---------------------------------------------------------------
        logger.info("Phase 3: Sweep over layers, group_sizes, top_ks.")

        baseline_bytes = seq_len * hidden_dim * 2  # fp16

        for lb in layer_boundaries:
            real_acts = real_acts_by_layer[lb]  # (seq_len, hidden_dim)
            static_table = static_tables[lb]

            # --- Raw cosine: real vs static (before affine), computed once per layer ---
            static_acts_all = static_table.lookup(token_ids_1d)
            raw_cos = F.cosine_similarity(real_acts.float(), static_acts_all.float(), dim=-1)  # (seq_len,)
            raw_diff = real_acts.float() - static_acts_all.float()
            raw_mse_per_pos = (raw_diff ** 2).mean(dim=-1)  # (seq_len,)
            raw_cosine_stats = {
                "raw_cosine_mean": raw_cos.mean().item(),
                "raw_cosine_min": raw_cos.min().item(),
                "raw_cosine_max": raw_cos.max().item(),
                "raw_mse_mean": raw_mse_per_pos.mean().item(),
                "raw_mse_max": raw_mse_per_pos.max().item(),
            }

            # Pre-compute per-bin raw cosine stats for position_quality
            raw_cosine_by_bin = {}
            for bin_start, bin_end in position_bins:
                if bin_start >= seq_len:
                    continue
                actual_end = min(bin_end, seq_len)
                if actual_end - bin_start <= 0:
                    continue
                bin_cos = raw_cos[bin_start:actual_end]
                bin_raw_mse = raw_mse_per_pos[bin_start:actual_end]
                raw_cosine_by_bin[(bin_start, bin_end)] = {
                    "raw_cosine_mean": bin_cos.mean().item(),
                    "raw_cosine_min": bin_cos.min().item(),
                    "raw_mse_mean": bin_raw_mse.mean().item(),
                    "raw_mse_max": bin_raw_mse.max().item(),
                }

            del static_acts_all, raw_diff

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
                        "  lb=%d gs=%d k=%d seq_len=%d",
                        lb, group_size, top_k, seq_len,
                    )

                    # --- Encode pipeline ---
                    packet, encode_timings = run_timed_static_encode_pipeline(
                        real_acts, token_ids_1d, static_table,
                        group_size, top_k,
                        num_warmup=num_warmup, num_runs=num_timing_runs,
                    )

                    # --- Decode pipeline ---
                    reconstructed, decode_timings = run_timed_static_decode_pipeline(
                        packet, static_table, hidden_dim,
                        num_warmup=num_warmup, num_runs=num_timing_runs,
                    )

                    # --- Transfer size ---
                    transfer_bytes = compute_transfer_size(packet)
                    compression_ratio = baseline_bytes / max(transfer_bytes, 1)
                    bytes_per_token = transfer_bytes / max(seq_len, 1)

                    # --- Quality metrics (global) ---
                    quality = compute_reconstruction_quality(real_acts, reconstructed)

                    # --- Delta magnitude & affine stats ---
                    # Re-derive delta and affine params for stats
                    static_acts_lookup = static_table.lookup(token_ids_1d)
                    scale, bias = compute_affine_params(real_acts, static_acts_lookup)
                    ref_t = apply_affine(static_acts_lookup, scale, bias)
                    delta = compute_delta(real_acts, ref_t)

                    delta_stats = compute_delta_magnitude_stats(delta)
                    affine_stats = compute_affine_param_stats(scale, bias)

                    # Base metadata
                    base_meta = {
                        "layer_boundary": lb,
                        "hidden_dim": hidden_dim,
                        "group_size": group_size,
                        "top_k": top_k,
                    }

                    # --- Latency records ---
                    all_timings = {}
                    all_timings.update(encode_timings)
                    all_timings.update({f"decode_{k}": v for k, v in decode_timings.items()})

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
                        rec["bytes_per_token"] = bytes_per_token
                        latency_records.append(rec)

                    # --- Quality record ---
                    q_rec = dict(base_meta)
                    q_rec.update(raw_cosine_stats)
                    q_rec.update(quality)
                    q_rec.update(delta_stats)
                    q_rec.update(affine_stats)
                    quality_records.append(q_rec)

                    # --- Position-binned quality ---
                    for bin_start, bin_end in position_bins:
                        if bin_start >= seq_len:
                            continue
                        actual_end = min(bin_end, seq_len)
                        num_positions = actual_end - bin_start

                        if num_positions <= 0:
                            continue

                        bin_orig = real_acts[bin_start:actual_end]
                        bin_recon = reconstructed[bin_start:actual_end]
                        bin_quality = compute_reconstruction_quality(bin_orig, bin_recon)

                        pq_rec = dict(base_meta)
                        pq_rec["bin_start"] = bin_start
                        pq_rec["bin_end"] = actual_end
                        pq_rec["num_positions"] = num_positions
                        bin_key = (bin_start, bin_end)
                        if bin_key in raw_cosine_by_bin:
                            pq_rec.update(raw_cosine_by_bin[bin_key])
                        pq_rec.update(bin_quality)
                        position_quality_records.append(pq_rec)

                    # --- Bandwidth analysis ---
                    for bw_mbps in bandwidths_mbps:
                        bw_bytes_per_ms = (bw_mbps * 1e6) / (8 * 1000)  # bytes/ms
                        raw_transfer_ms = baseline_bytes / bw_bytes_per_ms
                        delta_transfer_ms = transfer_bytes / bw_bytes_per_ms
                        speedup = raw_transfer_ms / max(delta_transfer_ms, 1e-9)

                        bw_rec = dict(base_meta)
                        bw_rec["bandwidth_mbps"] = bw_mbps
                        bw_rec["raw_transfer_ms"] = raw_transfer_ms
                        bw_rec["delta_transfer_ms"] = delta_transfer_ms
                        bw_rec["speedup"] = speedup
                        bw_rec["compression_ratio"] = compression_ratio
                        bandwidth_records.append(bw_rec)

                    del packet, reconstructed, delta
                    torch.cuda.empty_cache()

        # Cleanup
        del real_acts_by_layer, static_tables
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(
            "E10 complete: %d latency, %d quality, %d position_quality, %d bandwidth records.",
            len(latency_records), len(quality_records),
            len(position_quality_records), len(bandwidth_records),
        )

        return {
            "latency": latency_records,
            "quality": quality_records,
            "position_quality": position_quality_records,
            "bandwidth": bandwidth_records,
        }
