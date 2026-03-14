"""E7: Perturbation sensitivity experiment.

Adds Gaussian noise along SVD directions vs random directions,
measures output KL divergence to quantify sensitivity to structured
vs unstructured perturbations.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..metrics.intervention import directional_perturb, compute_intervention_metrics
from ..metrics.spectral import compute_svd_metrics
from .base import BaseExperiment

logger = logging.getLogger(__name__)

PERTURBATION_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer", "perturbation_type", "direction_index", "magnitude",
    "kl_divergence", "prediction_flip", "js_divergence",
]


class PerturbationSensitivityExperiment(BaseExperiment):
    experiment_id = "e7"
    experiment_name = "perturbation_sensitivity"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"perturbation": PERTURBATION_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "perturbation"

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

        logger.info("Built %d perturbation sensitivity sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run(self, model, tokenizer, prompt_text: str, config: Dict[str, Any], **kwargs):
        perturb_cfg = config.get("perturbation", {})
        layer_stride = perturb_cfg.get("layer_stride", 4)
        magnitudes = perturb_cfg.get("magnitudes", [0.1, 0.5, 1.0, 2.0])
        num_svd_directions = perturb_cfg.get("num_svd_directions", 5)
        num_random_directions = perturb_cfg.get("num_random_directions", 5)
        act_dtype = resolve_dtype(perturb_cfg.get("activation_dtype", "float32"))

        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Get original outputs + hidden states
        outputs = model(input_ids=input_tensor, output_hidden_states=True, use_cache=False)
        original_logits = outputs.logits
        all_hidden = outputs.hidden_states
        num_layers = len(all_hidden) - 1
        del outputs

        target_layers = list(range(1, num_layers + 1, layer_stride))
        if num_layers not in target_layers:
            target_layers.append(num_layers)

        records: List[Dict[str, Any]] = []

        for layer_idx in target_layers:
            h = all_hidden[layer_idx].squeeze(0).to(act_dtype)
            h_centered = h - h.mean(dim=0, keepdim=True)
            hidden_dim = h.shape[1]

            # Get SVD directions
            q = min(num_svd_directions, h.shape[0] - 1, hidden_dim)
            if q >= 1:
                U, S, V = torch.pca_lowrank(h_centered, q=q, niter=3)
                svd_directions = V[:, :num_svd_directions]  # (hidden_dim, k)
            else:
                svd_directions = torch.randn(hidden_dim, num_svd_directions, device=device)

            # Random directions
            random_directions = torch.randn(hidden_dim, num_random_directions, device=device)

            for mag in magnitudes:
                # SVD directions
                for d_idx in range(min(num_svd_directions, svd_directions.shape[1])):
                    direction = svd_directions[:, d_idx]
                    perturbed_logits = directional_perturb(
                        model, input_tensor, layer_idx, direction, magnitude=mag,
                    )
                    metrics = compute_intervention_metrics(original_logits, perturbed_logits)
                    records.append({
                        "layer": layer_idx,
                        "perturbation_type": "svd",
                        "direction_index": d_idx,
                        "magnitude": mag,
                        **metrics,
                    })
                    del perturbed_logits

                # Random directions
                for d_idx in range(num_random_directions):
                    direction = random_directions[:, d_idx]
                    perturbed_logits = directional_perturb(
                        model, input_tensor, layer_idx, direction, magnitude=mag,
                    )
                    metrics = compute_intervention_metrics(original_logits, perturbed_logits)
                    records.append({
                        "layer": layer_idx,
                        "perturbation_type": "random",
                        "direction_index": d_idx,
                        "magnitude": mag,
                        **metrics,
                    })
                    del perturbed_logits

        del original_logits, all_hidden
        gc.collect()
        torch.cuda.empty_cache()

        return {"perturbation": records}
