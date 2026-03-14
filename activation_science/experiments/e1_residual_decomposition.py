"""E1: Residual stream decomposition experiment.

Measures which layers do real work (large delta) vs skip (small delta),
and how the deltas relate to the residual stream and embeddings.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List

import torch

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill, decode_step, select_next_token
from ..metrics.dynamics import compute_residual_update
from .base import BaseExperiment

logger = logging.getLogger(__name__)

RESIDUAL_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id",
    "context_length", "num_decode_tokens",
    "phase", "step", "layer",
    "delta_norm", "residual_norm", "delta_residual_ratio",
    "cos_delta_residual", "cos_delta_embedding", "cos_delta_prev_delta",
    "delta_norm_std", "residual_norm_std",
]


class ResidualDecompositionExperiment(BaseExperiment):
    experiment_id = "e1"
    experiment_name = "residual"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"residual": RESIDUAL_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "residual"

    @classmethod
    def build_sweep_jobs(cls, config: Dict[str, Any]) -> List[ExperimentJob]:
        sweep = config["sweep"]
        context_lengths = sweep["context_lengths"]
        num_prompts = sweep.get("num_prompts_per_length", 3)
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
            jobs.append(ExperimentJob(job_id=make_experiment_id(params), params=params))

        logger.info("Built %d residual sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run(self, model, tokenizer, prompt_text: str, config: Dict[str, Any], **kwargs):
        residual_cfg = config.get("residual", {})
        num_decode = residual_cfg.get("num_decode_tokens", 128)
        do_sample = residual_cfg.get("do_sample", False)
        temperature = residual_cfg.get("temperature", 1.0)
        act_dtype = resolve_dtype(residual_cfg.get("activation_dtype", "float32"))

        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        batch = prefill(model, input_tensor, use_cache=True)
        all_hidden = batch.hidden_states
        num_all_layers = batch.num_layers

        # Prefill phase: compute deltas for all tokens
        prefill_records: List[Dict[str, Any]] = []
        for layer_idx in range(1, num_all_layers + 1):
            h_curr = all_hidden[layer_idx].squeeze(0).to(act_dtype)
            h_prev = all_hidden[layer_idx - 1].squeeze(0).to(act_dtype)
            h_embed = all_hidden[0].squeeze(0).to(act_dtype)

            if layer_idx >= 2:
                h_prev_prev = all_hidden[layer_idx - 2].squeeze(0).to(act_dtype)
                prev_delta = h_prev - h_prev_prev
            else:
                prev_delta = None

            metrics = compute_residual_update(h_curr, h_prev, h_embed, prev_delta)
            prefill_records.append({
                "phase": "prefill", "step": -1, "layer": layer_idx,
                **metrics,
            })

        past_key_values = batch.past_key_values
        logits = batch.last_logits
        del batch
        gc.collect()

        # Decode loop
        decode_records: List[Dict[str, Any]] = []
        for step in range(num_decode):
            next_token = select_next_token(logits, do_sample, temperature)
            batch = decode_step(model, next_token, past_key_values)
            all_hidden = batch.hidden_states
            h_embed = all_hidden[0].squeeze(0).squeeze(0).to(act_dtype)

            prev_delta = None
            for layer_idx in range(1, num_all_layers + 1):
                h_curr = all_hidden[layer_idx].squeeze(0).squeeze(0).to(act_dtype)
                h_prev = all_hidden[layer_idx - 1].squeeze(0).squeeze(0).to(act_dtype)

                metrics = compute_residual_update(h_curr, h_prev, h_embed, prev_delta)
                metrics.setdefault("delta_norm_std", 0.0)
                metrics.setdefault("residual_norm_std", 0.0)
                decode_records.append({
                    "phase": "decode", "step": step, "layer": layer_idx,
                    **metrics,
                })
                prev_delta = h_curr - h_prev

            past_key_values = batch.past_key_values
            logits = batch.last_logits
            del batch

        del past_key_values, logits
        gc.collect()
        torch.cuda.empty_cache()

        return {"residual": prefill_records + decode_records}
