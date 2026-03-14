"""E6: Causal intervention experiment.

Registers forward hooks to replace hidden states at target layers,
measures output KL divergence, prediction flip rate, downstream propagation.
Sweeps over: layers x intervention types (skip, mean-ablate, zero).
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list
from ..metrics.intervention import (
    activation_patch, mean_ablation, compute_intervention_metrics,
    _get_transformer_layers,
)
from .base import BaseExperiment

logger = logging.getLogger(__name__)

CAUSAL_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer", "intervention_type", "position",
    "kl_divergence", "prediction_flip", "js_divergence",
]


class CausalInterventionExperiment(BaseExperiment):
    experiment_id = "e6"
    experiment_name = "causal_intervention"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"causal": CAUSAL_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "causal"

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

        logger.info("Built %d causal intervention sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run(self, model, tokenizer, prompt_text: str, config: Dict[str, Any], **kwargs):
        causal_cfg = config.get("causal", {})
        intervention_types = causal_cfg.get("intervention_types", ["zero", "mean_ablate"])
        layer_stride = causal_cfg.get("layer_stride", 4)
        measure_position = causal_cfg.get("measure_position", -1)

        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Get original outputs + hidden states
        outputs = model(input_ids=input_tensor, output_hidden_states=True, use_cache=False)
        original_logits = outputs.logits
        all_hidden = outputs.hidden_states
        num_layers = len(all_hidden) - 1
        del outputs

        # Compute per-layer mean activations (for mean ablation)
        layer_means = {}
        for l in range(1, num_layers + 1):
            layer_means[l] = all_hidden[l].squeeze(0).mean(dim=0)  # (hidden_dim,)

        # Select layers to intervene on
        target_layers = list(range(1, num_layers + 1, layer_stride))
        if num_layers not in target_layers:
            target_layers.append(num_layers)

        records: List[Dict[str, Any]] = []

        for layer_idx in target_layers:
            for intervention in intervention_types:
                if intervention == "zero":
                    # Zero ablation: replace with zeros
                    zero_states = torch.zeros_like(all_hidden[layer_idx])
                    patched_logits = activation_patch(
                        model, input_tensor, layer_idx, zero_states,
                    )
                elif intervention == "mean_ablate":
                    patched_logits = mean_ablation(
                        model, input_tensor, layer_idx, layer_means[layer_idx],
                    )
                elif intervention == "skip":
                    # Skip: replace with previous layer's output
                    prev_states = all_hidden[layer_idx - 1] if layer_idx > 1 else all_hidden[0]
                    patched_logits = activation_patch(
                        model, input_tensor, layer_idx, prev_states,
                    )
                else:
                    continue

                metrics = compute_intervention_metrics(
                    original_logits, patched_logits, position=measure_position,
                )
                records.append({
                    "layer": layer_idx,
                    "intervention_type": intervention,
                    "position": measure_position,
                    **metrics,
                })
                del patched_logits

        del original_logits, all_hidden
        gc.collect()
        torch.cuda.empty_cache()

        return {"causal": records}
