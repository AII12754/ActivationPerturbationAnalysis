"""E8: Token type analysis experiment.

Groups tokens by type (function words, content words, punctuation, etc.),
computes per-group geometry metrics (norms, effective rank, pairwise cosine),
and Fisher discriminant separability between groups.
"""

from __future__ import annotations

import gc
import logging
from itertools import product
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import numpy as np

from ..core.types import ExperimentJob, make_experiment_id, resolve_dataset_list, resolve_dtype
from ..core.extraction import prefill
from ..metrics.spectral import compute_svd_metrics, compute_isotropy
from ..metrics.clustering import compute_fisher_discriminant
from .base import BaseExperiment

logger = logging.getLogger(__name__)

TOKEN_TYPE_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer", "token_type", "num_tokens",
    "norm_mean", "norm_std", "effective_rank",
    "mean_pairwise_cosine", "centroid_norm",
]

SEPARABILITY_COLUMNS = [
    "experiment_id", "model_id", "dataset_name", "prompt_id", "context_length",
    "layer", "fisher_discriminant",
]

# Simple token type classification based on token text
_PUNCTUATION = set('.,;:!?-()[]{}"\'/\\@#$%^&*_+=<>~`')
_FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'must', 'need',
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about',
    'and', 'or', 'but', 'not', 'if', 'then', 'that', 'this', 'which',
    'who', 'what', 'where', 'when', 'how', 'why',
    'it', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her',
    'them', 'us', 'my', 'his', 'its', 'our', 'your', 'their',
}


def _classify_token(text: str) -> str:
    """Classify a token into a type category."""
    stripped = text.strip()
    if not stripped:
        return "whitespace"
    if all(c in _PUNCTUATION for c in stripped):
        return "punctuation"
    if stripped.isdigit():
        return "number"
    if stripped.lower() in _FUNCTION_WORDS:
        return "function_word"
    return "content_word"


class TokenTypeAnalysisExperiment(BaseExperiment):
    experiment_id = "e8"
    experiment_name = "token_type"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]:
        return {
            "token_type": TOKEN_TYPE_COLUMNS,
            "separability": SEPARABILITY_COLUMNS,
        }

    @classmethod
    def default_config_section(cls) -> str:
        return "token_type"

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

        logger.info("Built %d token type analysis sweep jobs.", len(jobs))
        return jobs

    @torch.inference_mode()
    def run(self, model, tokenizer, prompt_text: str, config: Dict[str, Any], **kwargs):
        tt_cfg = config.get("token_type", {})
        act_dtype = resolve_dtype(tt_cfg.get("activation_dtype", "float32"))

        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Classify each token
        token_texts = [tokenizer.decode([tid]) for tid in input_ids]
        token_types = [_classify_token(t) for t in token_texts]
        unique_types = sorted(set(token_types))

        # Build type-to-position mapping
        type_to_positions: Dict[str, List[int]] = {t: [] for t in unique_types}
        for i, tt in enumerate(token_types):
            type_to_positions[tt].append(i)

        # Forward pass
        batch = prefill(model, input_tensor, use_cache=False)
        all_hidden = batch.hidden_states
        num_all_layers = batch.num_layers

        token_type_records: List[Dict[str, Any]] = []
        separability_records: List[Dict[str, Any]] = []

        for layer_idx in range(num_all_layers + 1):
            h = all_hidden[layer_idx].squeeze(0).to(act_dtype)  # (seq_len, hidden_dim)

            # Per-type metrics
            for tt in unique_types:
                positions = type_to_positions[tt]
                if len(positions) < 2:
                    continue
                h_group = h[positions]
                norms = h_group.norm(dim=1)
                h_centered = h_group - h_group.mean(dim=0, keepdim=True)

                svd = compute_svd_metrics(h_centered, q=min(20, len(positions) - 1))
                isotropy = compute_isotropy(h_group, num_samples=min(200, len(positions) * 10))

                token_type_records.append({
                    "layer": layer_idx,
                    "token_type": tt,
                    "num_tokens": len(positions),
                    "norm_mean": norms.mean().item(),
                    "norm_std": norms.std().item(),
                    "effective_rank": svd["effective_rank"],
                    "mean_pairwise_cosine": isotropy,
                    "centroid_norm": h_group.mean(dim=0).norm().item(),
                })

            # Fisher discriminant separability
            if len(unique_types) >= 2:
                features = h.cpu().numpy()
                labels = np.array([unique_types.index(tt) for tt in token_types])
                fisher = compute_fisher_discriminant(features, labels)
                separability_records.append({
                    "layer": layer_idx,
                    "fisher_discriminant": fisher,
                })

        del batch
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "token_type": token_type_records,
            "separability": separability_records,
        }
