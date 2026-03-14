"""Spectral metrics: SVD, effective rank, isotropy.

Extracted from src/geometry_experiment.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def compute_svd_metrics(
    h_centered: torch.Tensor,
    q: int = 100,
) -> Dict[str, Any]:
    """Compute SVD-based geometry metrics on centered hidden states.

    Parameters
    ----------
    h_centered:
        Shape ``(seq_len, hidden_dim)`` — mean-subtracted activations.
    q:
        Number of PCA components to compute.

    Returns
    -------
    Dict with keys: effective_rank, participation_ratio, singular_values,
    explained_variance (dict of top-k explained variance ratios).
    """
    seq_len, hidden_dim = h_centered.shape
    q = min(q, seq_len - 1, hidden_dim)
    if q < 1:
        return {
            "effective_rank": 1.0,
            "participation_ratio": 1.0,
            "singular_values": [],
            "explained_variance": {f"top{k}": 1.0 for k in [1, 5, 10, 20, 50]},
        }

    U, S, V = torch.pca_lowrank(h_centered, q=q, niter=3)

    # Effective rank
    sv_norm = S / S.sum()
    sv_norm = sv_norm.clamp(min=1e-10)
    entropy = -(sv_norm * sv_norm.log()).sum().item()
    effective_rank = math.exp(entropy)

    # Participation ratio
    participation_ratio = (S.sum().item() ** 2) / (S.pow(2).sum().item() + 1e-10)

    # Explained variance
    sv_sq = S.pow(2)
    total_var = sv_sq.sum().item()
    explained = {}
    for k in [1, 5, 10, 20, 50]:
        if k <= len(S):
            explained[f"top{k}"] = sv_sq[:k].sum().item() / (total_var + 1e-10)
        else:
            explained[f"top{k}"] = 1.0

    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "singular_values": S.cpu().tolist(),
        "explained_variance": explained,
        "sv_squared": sv_sq.cpu().tolist(),
        "total_variance": total_var,
    }


def compute_isotropy(
    h: torch.Tensor,
    num_samples: int = 500,
) -> float:
    """Compute isotropy as mean pairwise cosine similarity (sampled).

    Parameters
    ----------
    h:
        Shape ``(seq_len, hidden_dim)`` — raw activations (not centered).
    num_samples:
        Number of random pairs to sample.

    Returns
    -------
    Mean pairwise cosine similarity. Lower = more isotropic.
    """
    seq_len = h.shape[0]
    if seq_len <= 1:
        return 1.0

    device = h.device
    num_samples = min(num_samples, seq_len * (seq_len - 1) // 2)
    if num_samples <= 0:
        return 1.0

    h_normed = F.normalize(h, dim=1)
    idx1 = torch.randint(0, seq_len, (num_samples,), device=device)
    idx2 = torch.randint(0, seq_len, (num_samples,), device=device)
    mask = idx1 == idx2
    idx2[mask] = (idx2[mask] + 1) % seq_len
    cosines = (h_normed[idx1] * h_normed[idx2]).sum(dim=1)
    return cosines.mean().item()
