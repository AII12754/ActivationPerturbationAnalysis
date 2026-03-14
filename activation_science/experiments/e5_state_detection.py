"""E5: Activation state detection experiment.

Tracks hidden-state trajectories in PCA space during generation and
detects regime changes.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill, decode_step, select_next_token
from ..metrics.dynamics import fit_pca_basis, project_to_pca
from .base import BaseExperiment

logger = logging.getLogger(__name__)

TRAJECTORY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "decode_step", "layer", "token_id", "norm", "centroid_distance", "step_cosine",
    "pc0", "pc1", "pc2", "pc3", "pc4",
    "pc5", "pc6", "pc7", "pc8", "pc9",
    "pc10", "pc11", "pc12", "pc13", "pc14",
    "pc15", "pc16", "pc17", "pc18", "pc19",
]


class StateDetectionExperiment(BaseExperiment):
    experiment_id = "e5"
    experiment_name = "state_detection"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"trajectory": TRAJECTORY_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "state_detection"

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

        logger.info("Built %d state detection sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run(self, model, tokenizer, prompt_text: str, config: Dict[str, Any], **kwargs):
        state_cfg = config.get("state_detection", {})
        num_decode = state_cfg.get("num_decode_tokens", 256)
        n_components = state_cfg.get("pca_components", 20)
        do_sample = state_cfg.get("do_sample", False)
        temperature = state_cfg.get("temperature", 1.0)
        act_dtype = resolve_dtype(state_cfg.get("activation_dtype", "float32"))

        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        batch = prefill(model, input_tensor, use_cache=True)
        all_hidden = batch.hidden_states
        num_all_layers = batch.num_layers

        # Pick tracked layers
        tracked_layers = state_cfg.get("tracked_layers", None)
        if tracked_layers is None:
            step_size = max(1, num_all_layers // 8)
            tracked_layers = list(range(1, num_all_layers + 1, step_size))
            if num_all_layers not in tracked_layers:
                tracked_layers.append(num_all_layers)

        # Build PCA basis from prefill hidden states
        pca_bases: Dict[int, tuple] = {}
        centroids: Dict[int, torch.Tensor] = {}
        for layer_idx in tracked_layers:
            h = all_hidden[layer_idx].squeeze(0).to(act_dtype)
            V, h_mean = fit_pca_basis(h, n_components)
            pca_bases[layer_idx] = (V, h_mean)
            projected = project_to_pca(h, V, h_mean)
            centroids[layer_idx] = projected.mean(dim=0)

        past_key_values = batch.past_key_values
        logits = batch.last_logits
        del batch
        gc.collect()

        trajectory_records: List[Dict[str, Any]] = []
        prev_projected: Dict[int, Optional[torch.Tensor]] = {
            layer_idx: None for layer_idx in tracked_layers
        }

        for step_idx in range(num_decode):
            next_token = select_next_token(logits, do_sample, temperature)
            next_token_id = next_token.item()

            batch = decode_step(model, next_token, past_key_values)
            all_hidden = batch.hidden_states

            for layer_idx in tracked_layers:
                h = all_hidden[layer_idx].squeeze(0).squeeze(0).to(act_dtype)
                V, h_mean = pca_bases[layer_idx]

                projected = project_to_pca(h, V, h_mean)
                h_norm = h.norm().item()
                centroid_dist = (projected - centroids[layer_idx]).norm().item()

                if prev_projected[layer_idx] is not None:
                    step_cosine = F.cosine_similarity(
                        projected.unsqueeze(0),
                        prev_projected[layer_idx].unsqueeze(0),
                    ).item()
                else:
                    step_cosine = float("nan")

                coords = projected.cpu().tolist()
                rec: Dict[str, Any] = {
                    "decode_step": step_idx,
                    "layer": layer_idx,
                    "token_id": next_token_id,
                    "norm": h_norm,
                    "centroid_distance": centroid_dist,
                    "step_cosine": step_cosine,
                }
                for i, c in enumerate(coords[:n_components]):
                    rec[f"pc{i}"] = c

                trajectory_records.append(rec)
                prev_projected[layer_idx] = projected.clone()

            past_key_values = batch.past_key_values
            logits = batch.last_logits
            del batch

        del past_key_values, logits, pca_bases, centroids, prev_projected
        gc.collect()
        torch.cuda.empty_cache()

        return {"trajectory": trajectory_records}
