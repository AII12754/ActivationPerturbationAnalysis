"""E0: Decode-time activation similarity experiment.

For each newly decoded token, computes token-wise cosine similarity
between its hidden-state vector and every historical token's hidden-state
vector at each layer, then extracts top-k references and aggregate stats.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List, Tuple

import torch

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill, decode_step, select_next_token, stack_tracked_layers
from ..metrics.similarity import ActivationHistoryBuffer, compute_reference_similarity
from .base import BaseExperiment

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Schemas (from src/decode_storage.py)
# -------------------------------------------------------------------
TOPK_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id",
    "context_length", "num_decode_tokens",
    "decode_step", "token_id", "token_text", "layer", "ref_rank",
    "ref_token_position", "ref_token_id", "ref_token_text",
    "ref_token_distance", "ref_similarity",
]

AGG_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id",
    "context_length", "num_decode_tokens",
    "decode_step", "token_id", "token_text", "layer",
    "mean_similarity", "median_similarity", "max_similarity", "std_similarity",
    "frac_above_090", "frac_above_095", "frac_above_098",
    "num_historical_tokens",
]


class DecodeSimilarityExperiment(BaseExperiment):
    experiment_id = "e0"
    experiment_name = "decode_similarity"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"topk": TOPK_COLUMNS, "agg": AGG_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "decode"

    @classmethod
    def build_sweep_jobs(cls, config: Dict[str, Any]) -> List[ExperimentJob]:
        sweep = config["sweep"]
        context_lengths = sweep["context_lengths"]
        num_prompts = sweep.get("num_prompts_per_length", 3)
        num_decode_list = sweep.get("num_decode_tokens", [128])
        if isinstance(num_decode_list, int):
            num_decode_list = [num_decode_list]

        dataset_specs = resolve_dataset_list(config)

        jobs: List[ExperimentJob] = []
        for ds_spec, ctx_len, prompt_idx, num_dec in product(
            dataset_specs, context_lengths, range(num_prompts), num_decode_list,
        ):
            params = {
                "dataset_name": ds_spec["name"],
                "dataset_config": ds_spec["config"],
                "dataset_split": ds_spec["split"],
                "context_length": ctx_len,
                "prompt_index": prompt_idx,
                "num_decode_tokens": num_dec,
            }
            job_id = make_experiment_id(params)
            jobs.append(ExperimentJob(job_id=job_id, params=params))

        logger.info("Built %d decode sweep jobs (%d datasets).", len(jobs), len(dataset_specs))
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
        decode_cfg = config.get("decode", {})
        num_decode = decode_cfg.get("num_decode_tokens", 128)
        top_k = decode_cfg.get("top_k_references", 10)
        thresholds = decode_cfg.get("similarity_thresholds", [0.90, 0.95, 0.98])
        do_sample = decode_cfg.get("do_sample", False)
        temperature = decode_cfg.get("temperature", 1.0)
        layer_subset = decode_cfg.get("store_layer_subset", None)
        act_dtype = resolve_dtype(decode_cfg.get("activation_dtype", "float32"))

        device = next(model.parameters()).device

        # 1. Tokenize prompt.
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        context_length = len(input_ids)

        # 2. Prefill: get hidden states + KV cache.
        logger.debug("Prefill: %d tokens", context_length)
        batch = prefill(model, input_tensor, use_cache=True)
        all_hidden = batch.hidden_states
        num_all_layers = batch.num_layers
        hidden_dim = all_hidden[0].shape[2]

        # Determine which layers to track.
        if layer_subset is not None:
            tracked_layers = [l for l in layer_subset if 0 < l <= num_all_layers]
        else:
            tracked_layers = list(range(1, num_all_layers + 1))
        num_tracked = len(tracked_layers)

        max_seq = context_length + num_decode
        history = ActivationHistoryBuffer(
            num_layers=num_tracked, max_seq_len=max_seq,
            hidden_dim=hidden_dim, device=device, dtype=act_dtype,
        )

        # Fill history with prefill hidden states.
        prefill_states = stack_tracked_layers(all_hidden, tracked_layers)
        history.add_batch(prefill_states)

        past_key_values = batch.past_key_values
        logits = batch.last_logits

        del batch
        gc.collect()

        # 3. Decode loop.
        topk_records: List[Dict[str, Any]] = []
        agg_records: List[Dict[str, Any]] = []
        decoded_token_ids: List[int] = []

        for step in range(num_decode):
            next_token = select_next_token(logits, do_sample, temperature)
            next_token_id = next_token.item()
            decoded_token_ids.append(next_token_id)

            batch = decode_step(model, next_token, past_key_values)

            # Extract new token's hidden states for tracked layers.
            new_states = torch.stack(
                [batch.hidden_states[l].squeeze(0).squeeze(0) for l in tracked_layers],
                dim=0,
            )  # (num_tracked, hidden_dim)

            # Compute similarity against history.
            topk_idx, topk_val, agg = compute_reference_similarity(
                history, new_states, top_k=top_k, thresholds=thresholds,
            )

            # Record top-k references per layer.
            current_pos = context_length + step
            for li, layer_idx in enumerate(tracked_layers):
                k_actual = topk_idx.shape[1]
                for rank in range(k_actual):
                    ref_pos = topk_idx[li, rank].item()
                    topk_records.append({
                        "decode_step": step,
                        "token_id": next_token_id,
                        "layer": layer_idx,
                        "ref_rank": rank,
                        "ref_token_position": ref_pos,
                        "ref_token_distance": current_pos - ref_pos,
                        "ref_similarity": topk_val[li, rank].item(),
                    })

                agg_records.append({
                    "decode_step": step,
                    "token_id": next_token_id,
                    "layer": layer_idx,
                    "mean_similarity": agg["mean_similarity"][li],
                    "median_similarity": agg["median_similarity"][li],
                    "max_similarity": agg["max_similarity"][li],
                    "std_similarity": agg["std_similarity"][li],
                    "frac_above_090": agg["frac_above_090"][li],
                    "frac_above_095": agg["frac_above_095"][li],
                    "frac_above_098": agg["frac_above_098"][li],
                    "num_historical_tokens": agg["num_historical_tokens"],
                })

            history.add(new_states)
            past_key_values = batch.past_key_values
            logits = batch.last_logits
            del batch

        # 4. Batch-decode all token texts.
        decoded_texts = [tokenizer.decode([tid]) for tid in decoded_token_ids]
        all_token_ids = input_ids + decoded_token_ids
        all_token_texts = [tokenizer.decode([tid]) for tid in all_token_ids]

        for rec in topk_records:
            step = rec["decode_step"]
            rec["token_text"] = decoded_texts[step]
            ref_pos = rec["ref_token_position"]
            if ref_pos < len(all_token_ids):
                rec["ref_token_id"] = all_token_ids[ref_pos]
                rec["ref_token_text"] = all_token_texts[ref_pos]
            else:
                rec["ref_token_id"] = None
                rec["ref_token_text"] = None

        for rec in agg_records:
            step = rec["decode_step"]
            rec["token_text"] = decoded_texts[step]

        # Clean up.
        del history, past_key_values, logits
        gc.collect()
        torch.cuda.empty_cache()

        return {"topk": topk_records, "agg": agg_records}
