"""Dynamics metrics: residual stream updates, PCA trajectory projection.

Extracted from:
  - src/residual_experiment.py (delta computation)
  - src/state_experiment.py (PCA fitting and projection)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_residual_update(
    h_curr: torch.Tensor,
    h_prev: torch.Tensor,
    h_embed: torch.Tensor,
    prev_delta: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute residual stream decomposition metrics for one layer.

    Parameters
    ----------
    h_curr:
        Current layer hidden state. Shape: ``(hidden_dim,)`` or ``(seq_len, hidden_dim)``.
    h_prev:
        Previous layer hidden state. Same shape as h_curr.
    h_embed:
        Embedding layer hidden state. Same shape as h_curr.
    prev_delta:
        Delta from previous layer pair (for cos_delta_prev_delta). Optional.

    Returns
    -------
    Dict with keys: delta_norm, residual_norm, delta_residual_ratio,
    cos_delta_residual, cos_delta_embedding, cos_delta_prev_delta.
    """
    delta = h_curr - h_prev

    if delta.dim() == 1:
        # Single token
        d_norm = delta.norm().item()
        r_norm = h_curr.norm().item()
        cos_dr = F.cosine_similarity(delta.unsqueeze(0), h_curr.unsqueeze(0)).item()
        cos_de = F.cosine_similarity(delta.unsqueeze(0), h_embed.unsqueeze(0)).item()

        if prev_delta is not None:
            cos_dp = F.cosine_similarity(delta.unsqueeze(0), prev_delta.unsqueeze(0)).item()
        else:
            cos_dp = float('nan')

        return {
            "delta_norm": d_norm,
            "residual_norm": r_norm,
            "delta_residual_ratio": d_norm / max(r_norm, 1e-8),
            "cos_delta_residual": cos_dr,
            "cos_delta_embedding": cos_de,
            "cos_delta_prev_delta": cos_dp,
        }
    else:
        # Batch of tokens — compute means
        delta_norm = delta.norm(dim=1)
        residual_norm = h_curr.norm(dim=1)
        ratio = delta_norm / residual_norm.clamp(min=1e-8)
        cos_dr = F.cosine_similarity(delta, h_curr, dim=1)
        cos_de = F.cosine_similarity(delta, h_embed, dim=1)

        if prev_delta is not None:
            cos_dp = F.cosine_similarity(delta, prev_delta, dim=1)
        else:
            cos_dp = torch.full((h_curr.shape[0],), float('nan'), device=h_curr.device)

        return {
            "delta_norm": delta_norm.mean().item(),
            "residual_norm": residual_norm.mean().item(),
            "delta_residual_ratio": ratio.mean().item(),
            "cos_delta_residual": cos_dr.mean().item(),
            "cos_delta_embedding": cos_de.mean().item(),
            "cos_delta_prev_delta": cos_dp.nanmean().item(),
            "delta_norm_std": delta_norm.std().item(),
            "residual_norm_std": residual_norm.std().item(),
        }


def fit_pca_basis(
    h: torch.Tensor,
    n_components: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit PCA basis from hidden states.

    Parameters
    ----------
    h:
        Shape ``(seq_len, hidden_dim)``.
    n_components:
        Number of PCA components.

    Returns
    -------
    (V, h_mean) where V is ``(hidden_dim, n_components)`` and h_mean is ``(hidden_dim,)``.
    """
    h_mean = h.mean(dim=0)
    h_centered = h - h_mean
    q = min(n_components, h.shape[0] - 1, h.shape[1])
    U, S, V = torch.pca_lowrank(h_centered, q=q, niter=3)
    return V[:, :n_components].clone(), h_mean.clone()


def project_to_pca(
    h: torch.Tensor,
    V: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Project hidden states onto PCA basis.

    Parameters
    ----------
    h:
        Shape ``(hidden_dim,)`` or ``(seq_len, hidden_dim)``.
    V:
        PCA directions, shape ``(hidden_dim, n_components)``.
    mean:
        Mean vector, shape ``(hidden_dim,)``.

    Returns
    -------
    Projected coordinates, shape ``(n_components,)`` or ``(seq_len, n_components)``.
    """
    return (h - mean) @ V
