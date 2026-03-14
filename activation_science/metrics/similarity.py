"""Similarity metrics: cosine similarity, activation history buffer, CKA, Procrustes, subspace overlap.

Consolidates metrics from:
  - src/decode_experiment.py (ActivationHistoryBuffer, compute_reference_similarity)
  - src/metrics.py (cosine_similarity_per_token, mean_cosine_similarity, compute_distance_buckets)
  - src/cross_sequence_experiment.py (linear_cka, procrustes_distance, subspace_overlap)

All heavy computation stays on GPU; only final aggregated scalars are
moved to CPU.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Per-token cosine similarity (from src/metrics.py)
# ===================================================================
def cosine_similarity_per_token(
    states_a: torch.Tensor,
    states_b: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between corresponding token vectors.

    Parameters
    ----------
    states_a, states_b:
        Tensors of shape ``(seq_len, hidden_dim)`` or
        ``(num_layers, seq_len, hidden_dim)``.

    Returns
    -------
    Tensor of shape ``(seq_len,)`` or ``(num_layers, seq_len)``
    with cosine similarities in [-1, 1].
    """
    return F.cosine_similarity(states_a, states_b, dim=-1)


def mean_cosine_similarity(
    states_a: torch.Tensor,
    states_b: torch.Tensor,
    start: int = 0,
    end: Optional[int] = None,
) -> float:
    """Average cosine similarity over a token range [start, end)."""
    sim = cosine_similarity_per_token(states_a, states_b)
    return sim[start:end].mean().item()


# ===================================================================
# Distance-bucket helpers (from src/metrics.py)
# ===================================================================
@torch.no_grad()
def compute_distance_buckets(
    original_states: torch.Tensor,
    perturbed_states: torch.Tensor,
    post_perturbation_start_orig: int,
    post_perturbation_start_pert: int,
    immediate_window: int = 10,
    short_window: int = 100,
    periodic_step: int = 100,
    periodic_width: int = 10,
) -> List[Dict]:
    """Compute similarity in various distance buckets for all layers.

    Parameters
    ----------
    original_states, perturbed_states:
        Tensors of shape ``(num_layers, seq_len, hidden_dim)`` **on GPU**.
    """
    num_layers = original_states.shape[0]

    tokens_after_orig = original_states.shape[1] - post_perturbation_start_orig
    tokens_after_pert = perturbed_states.shape[1] - post_perturbation_start_pert
    max_offset = min(tokens_after_orig, tokens_after_pert)

    if max_offset <= 0:
        return []

    orig_post = original_states[
        :, post_perturbation_start_orig : post_perturbation_start_orig + max_offset, :,
    ]
    pert_post = perturbed_states[
        :, post_perturbation_start_pert : post_perturbation_start_pert + max_offset, :,
    ]

    all_sim = F.cosine_similarity(orig_post, pert_post, dim=-1)

    windows: List[Tuple[str, int, int]] = []
    windows.append(("immediate", 0, min(immediate_window, max_offset)))
    windows.append(("short", 0, min(short_window, max_offset)))
    windows.append(("all_remaining", 0, max_offset))

    offset = 0
    while offset < max_offset:
        wend = min(offset + periodic_width, max_offset)
        windows.append((f"periodic_{offset}", offset, wend))
        offset += periodic_step

    per_token_count = min(immediate_window, max_offset)

    all_sim_cpu = all_sim.cpu()

    records: List[Dict] = []
    for layer_idx in range(num_layers):
        layer_sim = all_sim_cpu[layer_idx]

        for bucket_name, wstart, wend in windows:
            sim_slice = layer_sim[wstart:wend]
            length = sim_slice.numel()
            if length == 0:
                continue
            if length == 1:
                v = sim_slice.item()
                mean_v, min_v, max_v, std_v = v, v, v, 0.0
            else:
                mean_v = sim_slice.mean().item()
                min_v = sim_slice.min().item()
                max_v = sim_slice.max().item()
                std_v = sim_slice.std().item()

            records.append({
                "layer": layer_idx,
                "bucket": bucket_name,
                "offset_start": wstart,
                "offset_end": wstart + length,
                "mean_similarity": mean_v,
                "min_similarity": min_v,
                "max_similarity": max_v,
                "std_similarity": std_v,
                "num_tokens": length,
            })

        for t in range(per_token_count):
            v = layer_sim[t].item()
            records.append({
                "layer": layer_idx,
                "bucket": "per_token",
                "offset_start": t,
                "offset_end": t + 1,
                "mean_similarity": v,
                "min_similarity": v,
                "max_similarity": v,
                "std_similarity": 0.0,
                "num_tokens": 1,
            })

    return records


# ===================================================================
# Activation History Buffer (from src/decode_experiment.py)
# ===================================================================
class ActivationHistoryBuffer:
    """Pre-allocated GPU tensor for storing per-layer hidden states.

    Shape: ``(num_layers, max_seq_len, hidden_dim)``.
    Avoids repeated concatenation by writing into a pre-allocated buffer.
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

        self.buffer = torch.zeros(
            (num_layers, max_seq_len, hidden_dim),
            device=device,
            dtype=dtype,
        )
        self.length = 0

    def add(self, states: torch.Tensor):
        """Append a single token's hidden states across all layers.

        Parameters
        ----------
        states : Tensor of shape ``(num_layers, hidden_dim)``
        """
        if self.length >= self.max_seq_len:
            logger.warning("History buffer full (%d). Ignoring.", self.max_seq_len)
            return
        self.buffer[:, self.length, :] = states.to(self.dtype)
        self.length += 1

    def add_batch(self, states: torch.Tensor):
        """Append multiple tokens' hidden states at once.

        Parameters
        ----------
        states : Tensor of shape ``(num_layers, seq_len, hidden_dim)``
        """
        seq_len = states.shape[1]
        end = min(self.length + seq_len, self.max_seq_len)
        actual = end - self.length
        if actual <= 0:
            logger.warning("History buffer full. Ignoring batch.")
            return
        self.buffer[:, self.length:end, :] = states[:, :actual, :].to(self.dtype)
        self.length = end

    def get_all(self) -> torch.Tensor:
        """Return filled portion: ``(num_layers, length, hidden_dim)``."""
        return self.buffer[:, :self.length, :]


# ===================================================================
# Token-wise reference similarity (from src/decode_experiment.py)
# ===================================================================
def compute_reference_similarity(
    history: ActivationHistoryBuffer,
    new_token_states: torch.Tensor,
    top_k: int = 10,
    thresholds: List[float] = (0.90, 0.95, 0.98),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute token-wise cosine similarity between a new token and all history.

    For each layer, the new token's hidden-state vector is compared against
    every historical token's hidden-state vector via cosine similarity.

    Returns
    -------
    topk_indices : ``(num_layers, top_k)`` — positions in the history
    topk_values  : ``(num_layers, top_k)`` — corresponding similarities
    agg_stats    : dict with keys mean/median/max/std_similarity,
                   frac_above_*, num_historical_tokens (per layer lists)
    """
    num_layers = history.num_layers
    hist_len = history.length
    k = min(top_k, hist_len)

    history_states = history.get_all()
    query = new_token_states.to(history.dtype)

    history_norm = F.normalize(history_states, dim=2)
    query_norm = F.normalize(query, dim=1)

    similarities = torch.bmm(
        history_norm, query_norm.unsqueeze(2)
    ).squeeze(2)

    topk_values, topk_indices = similarities.topk(k, dim=1)

    mean_sim = similarities.mean(dim=1)
    std_sim = similarities.std(dim=1)
    max_sim = similarities.max(dim=1).values
    median_sim = similarities.median(dim=1).values

    frac_above = {}
    for thr in thresholds:
        key = f"frac_above_{int(round(thr * 100)):03d}"
        frac_above[key] = (similarities > thr).float().mean(dim=1)

    agg_stats = {
        "mean_similarity": mean_sim.cpu().tolist(),
        "median_similarity": median_sim.cpu().tolist(),
        "max_similarity": max_sim.cpu().tolist(),
        "std_similarity": std_sim.cpu().tolist(),
        "num_historical_tokens": hist_len,
    }
    for key, vals in frac_above.items():
        agg_stats[key] = vals.cpu().tolist()

    return topk_indices.cpu(), topk_values.cpu(), agg_stats


# ===================================================================
# Cross-sequence alignment metrics (from src/cross_sequence_experiment.py)
# ===================================================================
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between X (n, d1) and Y (n, d2).

    Both must have the same number of rows (aligned samples).
    For unequal lengths, truncate to min length.
    """
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n].float(), Y[:n].float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtX = X @ X.T
    YtY = Y @ Y.T

    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()


def procrustes_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Orthogonal Procrustes distance between centered, normalized X and Y."""
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n].float(), Y[:n].float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    X = X / (X.norm() + 1e-10)
    Y = Y / (Y.norm() + 1e-10)
    M = Y.T @ X
    U, S, Vt = torch.linalg.svd(M)
    dist = max(0.0, 2.0 - 2.0 * S.sum().item())
    return dist


def subspace_overlap(X: torch.Tensor, Y: torch.Tensor, k: int = 10) -> float:
    """Overlap of top-k principal subspaces."""
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n].float(), Y[:n].float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    k = min(k, n - 1, X.shape[1])
    if k <= 0:
        return 0.0
    _, _, Vx = torch.pca_lowrank(X, q=k, niter=3)
    _, _, Vy = torch.pca_lowrank(Y, q=k, niter=3)
    overlap = (Vx.T @ Vy).pow(2).sum().item() / k
    return overlap
