"""
Core experiment logic for representation geometry analysis.

For each layer, computes SVD-based geometry metrics (effective rank,
participation ratio, anisotropy, explained variance, norm statistics)
on the hidden-state matrix from a single forward pass.
"""

from __future__ import annotations

import gc
import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.inference_mode()
def run_geometry_experiment(
    model,
    tokenizer,
    prompt_text: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run geometry analysis on a single prompt.

    Parameters
    ----------
    model :
        HuggingFace causal LM (output_hidden_states=True).
    tokenizer :
        Corresponding tokenizer.
    prompt_text :
        The prompt to analyse.
    config :
        Full experiment config dict.

    Returns
    -------
    (summary_records, spectrum_records) — lists of dicts for storage.
        summary_records: one dict per layer with all scalar geometry metrics.
        spectrum_records: one dict per (layer, sv_index) with singular value info.
    """
    geo_cfg = config.get("geometry", {})
    pca_components = geo_cfg.get("pca_components", 100)
    isotropy_samples = geo_cfg.get("isotropy_samples", 500)
    act_dtype_str = geo_cfg.get("activation_dtype", "float32")
    act_dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}.get(act_dtype_str, torch.float32)

    device = next(model.parameters()).device

    # 1. Tokenize prompt.
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 2. Forward pass — get all hidden states.
    logger.debug("Forward pass: %d tokens", len(input_ids))
    outputs = model(
        input_ids=input_tensor,
        output_hidden_states=True,
        use_cache=False,
    )

    # hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, hidden_dim).
    # Layer 0 = embedding, layers 1..num_layers = transformer layers.
    all_hidden = outputs.hidden_states
    num_all_layers = len(all_hidden) - 1  # exclude embedding layer

    summary_records: List[Dict[str, Any]] = []
    spectrum_records: List[Dict[str, Any]] = []

    for layer_idx in range(num_all_layers + 1):  # include embedding layer 0
        h = all_hidden[layer_idx].squeeze(0).to(act_dtype)  # (seq_len, hidden_dim)
        seq_len, hidden_dim = h.shape

        # --- Norms ---
        norms = h.norm(dim=1)  # (seq_len,)

        # --- Center the data ---
        h_centered = h - h.mean(dim=0, keepdim=True)

        # --- SVD via pca_lowrank (much cheaper than full SVD) ---
        q = min(pca_components, seq_len - 1, hidden_dim)
        if q < 1:
            # Degenerate case: single token
            summary_records.append({
                "layer": layer_idx,
                "effective_rank": 1.0,
                "participation_ratio": 1.0,
                "mean_pairwise_cosine": 1.0,
                "norm_mean": norms.mean().item(),
                "norm_std": 0.0,
                "norm_min": norms.min().item(),
                "norm_max": norms.max().item(),
                "num_tokens": seq_len,
                "top1_explained_var": 1.0,
                "top5_explained_var": 1.0,
                "top10_explained_var": 1.0,
                "top20_explained_var": 1.0,
                "top50_explained_var": 1.0,
            })
            continue

        U, S, V = torch.pca_lowrank(h_centered, q=q, niter=3)
        # S are singular values, shape (q,)

        # --- Effective rank ---
        sv_norm = S / S.sum()
        sv_norm = sv_norm.clamp(min=1e-10)
        entropy = -(sv_norm * sv_norm.log()).sum().item()
        effective_rank = math.exp(entropy)

        # --- Participation ratio ---
        participation_ratio = (S.sum().item() ** 2) / (S.pow(2).sum().item() + 1e-10)

        # --- Explained variance ---
        sv_sq = S.pow(2)
        total_var = sv_sq.sum().item()
        explained = {}
        for k in [1, 5, 10, 20, 50]:
            if k <= len(S):
                explained[f"top{k}_explained_var"] = sv_sq[:k].sum().item() / (total_var + 1e-10)
            else:
                explained[f"top{k}_explained_var"] = 1.0

        # --- Isotropy: mean pairwise cosine (sample random pairs) ---
        num_samples = min(isotropy_samples, seq_len * (seq_len - 1) // 2)
        h_normed = F.normalize(h, dim=1)
        if seq_len > 1 and num_samples > 0:
            idx1 = torch.randint(0, seq_len, (num_samples,), device=device)
            idx2 = torch.randint(0, seq_len, (num_samples,), device=device)
            # Ensure different indices
            mask = idx1 == idx2
            idx2[mask] = (idx2[mask] + 1) % seq_len
            cosines = (h_normed[idx1] * h_normed[idx2]).sum(dim=1)
            mean_pairwise_cosine = cosines.mean().item()
        else:
            mean_pairwise_cosine = 1.0

        summary_records.append({
            "layer": layer_idx,
            "effective_rank": effective_rank,
            "participation_ratio": participation_ratio,
            "mean_pairwise_cosine": mean_pairwise_cosine,
            "norm_mean": norms.mean().item(),
            "norm_std": norms.std().item(),
            "norm_min": norms.min().item(),
            "norm_max": norms.max().item(),
            "num_tokens": seq_len,
            **explained,
        })

        # --- Spectrum records ---
        for sv_idx in range(len(S)):
            spectrum_records.append({
                "layer": layer_idx,
                "sv_index": sv_idx,
                "singular_value": S[sv_idx].item(),
                "explained_variance_ratio": sv_sq[sv_idx].item() / (total_var + 1e-10),
                "cumulative_variance_ratio": sv_sq[:sv_idx + 1].sum().item() / (total_var + 1e-10),
            })

    del outputs
    gc.collect()
    torch.cuda.empty_cache()

    return summary_records, spectrum_records
