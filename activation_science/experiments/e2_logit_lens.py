"""E2: Logit lens analysis experiment.

Projects intermediate hidden states through norm + lm_head to see what
token each layer would predict. Reveals when the model "decides" on
its output and how certainty builds across layers.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list
from ..core.extraction import prefill, decode_step, select_next_token, project_logit_lens
from ..metrics.information import compute_logit_lens_metrics
from .base import BaseExperiment

logger = logging.getLogger(__name__)

PERSTEP_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "decode_step", "layer",
    "entropy", "kl_from_final", "rank_of_correct", "cross_entropy_correct",
    "max_prob", "top1_token_id", "correct_token_id", "generated_token_id",
]

TOPK_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "decode_step", "layer", "rank", "token_id", "logit", "probability",
]


class LogitLensExperiment(BaseExperiment):
    experiment_id = "e2"
    experiment_name = "logit_lens"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"perstep": PERSTEP_COLUMNS, "topk": TOPK_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "logit_lens"

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

        logger.info("Built %d logit lens sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run(self, model, tokenizer, prompt_text: str, config: Dict[str, Any], **kwargs):
        lens_cfg = config.get("logit_lens", {})
        num_decode = lens_cfg.get("num_decode_tokens", 128)
        top_k = lens_cfg.get("top_k_predictions", 5)
        do_sample = lens_cfg.get("do_sample", False)
        temperature = lens_cfg.get("temperature", 1.0)
        layer_stride = lens_cfg.get("layer_stride", 1)

        device = next(model.parameters()).device
        norm_layer = model.model.norm
        lm_head = model.lm_head

        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        batch = prefill(model, input_tensor, use_cache=True)
        num_all_layers = batch.num_layers

        tracked_layers = list(range(0, num_all_layers + 1, layer_stride))
        if num_all_layers not in tracked_layers:
            tracked_layers.append(num_all_layers)

        past_key_values = batch.past_key_values
        logits = batch.last_logits
        del batch
        gc.collect()

        perstep_records: List[Dict[str, Any]] = []
        topk_records: List[Dict[str, Any]] = []

        for step in range(num_decode):
            next_token = select_next_token(logits, do_sample, temperature)
            next_token_id = next_token.item()

            batch = decode_step(model, next_token, past_key_values)
            all_hidden = batch.hidden_states

            # Final layer reference distribution
            final_hidden = all_hidden[num_all_layers].squeeze(0).squeeze(0)
            final_logits_vec = project_logit_lens(final_hidden, norm_layer, lm_head)
            final_log_probs = F.log_softmax(final_logits_vec, dim=0)
            final_pred_token = final_logits_vec.argmax().item()

            for layer_idx in tracked_layers:
                h = all_hidden[layer_idx].squeeze(0).squeeze(0)
                layer_logits = project_logit_lens(h, norm_layer, lm_head)

                metrics = compute_logit_lens_metrics(
                    layer_logits, final_log_probs, final_pred_token,
                )
                perstep_records.append({
                    "decode_step": step, "layer": layer_idx,
                    "correct_token_id": final_pred_token,
                    "generated_token_id": next_token_id,
                    **metrics,
                })

                topk_vals, topk_idx = layer_logits.topk(top_k)
                topk_probs = F.softmax(topk_vals, dim=0)
                for rank in range(top_k):
                    topk_records.append({
                        "decode_step": step, "layer": layer_idx, "rank": rank,
                        "token_id": topk_idx[rank].item(),
                        "logit": topk_vals[rank].item(),
                        "probability": topk_probs[rank].item(),
                    })

                del layer_logits

            del final_logits_vec, final_log_probs
            past_key_values = batch.past_key_values
            logits = batch.last_logits
            del batch

        del past_key_values, logits
        gc.collect()
        torch.cuda.empty_cache()

        return {"perstep": perstep_records, "topk": topk_records}
