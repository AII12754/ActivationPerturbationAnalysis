"""E3: Representation geometry analysis experiment.

For each layer, computes SVD-based geometry metrics (effective rank,
participation ratio, anisotropy, explained variance, norm statistics)
on the hidden-state matrix from a single forward pass.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List

import torch

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill
from ..metrics.spectral import compute_svd_metrics, compute_isotropy
from .base import BaseExperiment

logger = logging.getLogger(__name__)

SUMMARY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer", "effective_rank", "participation_ratio", "mean_pairwise_cosine",
    "norm_mean", "norm_std", "norm_min", "norm_max", "num_tokens",
    "top1_explained_var", "top5_explained_var", "top10_explained_var",
    "top20_explained_var", "top50_explained_var",
]

SPECTRUM_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer", "sv_index", "singular_value",
    "explained_variance_ratio", "cumulative_variance_ratio",
]


class RepresentationGeometryExperiment(BaseExperiment):
    experiment_id = "e3"
    experiment_name = "geometry"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {"summary": SUMMARY_COLUMNS, "spectrum": SPECTRUM_COLUMNS}

    @classmethod
    def default_config_section(cls) -> str:
        return "geometry"

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

        logger.info("Built %d geometry sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run(self, model, tokenizer, prompt_text: str, config: Dict[str, Any], **kwargs):
        geo_cfg = config.get("geometry", {})
        pca_components = geo_cfg.get("pca_components", 100)
        isotropy_samples = geo_cfg.get("isotropy_samples", 500)
        act_dtype = resolve_dtype(geo_cfg.get("activation_dtype", "float32"))

        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        batch = prefill(model, input_tensor, use_cache=False)
        all_hidden = batch.hidden_states
        num_all_layers = batch.num_layers

        summary_records: List[Dict[str, Any]] = []
        spectrum_records: List[Dict[str, Any]] = []

        for layer_idx in range(num_all_layers + 1):
            h = all_hidden[layer_idx].squeeze(0).to(act_dtype)
            seq_len, hidden_dim = h.shape
            norms = h.norm(dim=1)

            h_centered = h - h.mean(dim=0, keepdim=True)
            svd_results = compute_svd_metrics(h_centered, q=pca_components)
            mean_pairwise_cosine = compute_isotropy(h, num_samples=isotropy_samples)

            explained = svd_results["explained_variance"]
            summary_records.append({
                "layer": layer_idx,
                "effective_rank": svd_results["effective_rank"],
                "participation_ratio": svd_results["participation_ratio"],
                "mean_pairwise_cosine": mean_pairwise_cosine,
                "norm_mean": norms.mean().item(),
                "norm_std": norms.std().item() if seq_len > 1 else 0.0,
                "norm_min": norms.min().item(),
                "norm_max": norms.max().item(),
                "num_tokens": seq_len,
                "top1_explained_var": explained.get("top1", 1.0),
                "top5_explained_var": explained.get("top5", 1.0),
                "top10_explained_var": explained.get("top10", 1.0),
                "top20_explained_var": explained.get("top20", 1.0),
                "top50_explained_var": explained.get("top50", 1.0),
            })

            sv_sq = svd_results.get("sv_squared", [])
            total_var = svd_results.get("total_variance", 1e-10)
            cum = 0.0
            for sv_idx, sv_val in enumerate(svd_results["singular_values"]):
                sq_val = sv_sq[sv_idx] if sv_idx < len(sv_sq) else 0.0
                cum += sq_val
                spectrum_records.append({
                    "layer": layer_idx,
                    "sv_index": sv_idx,
                    "singular_value": sv_val,
                    "explained_variance_ratio": sq_val / (total_var + 1e-10),
                    "cumulative_variance_ratio": cum / (total_var + 1e-10),
                })

        del batch
        gc.collect()
        torch.cuda.empty_cache()

        return {"summary": summary_records, "spectrum": spectrum_records}
