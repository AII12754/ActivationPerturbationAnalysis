"""
Similarity metrics for comparing hidden states.

Primary metric: token-level cosine similarity.
All heavy computation stays on GPU; only final aggregated scalars are
moved to CPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


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


# -------------------------------------------------------------------
# Distance-bucket helpers
# -------------------------------------------------------------------

def _window_stats(sim_slice: torch.Tensor) -> Tuple[float, float, float, float]:
    """Return (mean, min, max, std) from a 1-D similarity tensor on GPU.

    Calls ``.item()`` once per stat → 4 host-device syncs total (minimal).
    """
    if sim_slice.numel() == 0:
        return (0.0, 0.0, 0.0, 0.0)
    if sim_slice.numel() == 1:
        v = sim_slice.item()
        return (v, v, v, 0.0)
    return (
        sim_slice.mean().item(),
        sim_slice.min().item(),
        sim_slice.max().item(),
        sim_slice.std().item(),
    )


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
    post_perturbation_start_orig, post_perturbation_start_pert:
        First token position after the perturbation in each sequence.

    Returns a flat list of dicts, each describing one measurement.
    """
    num_layers = original_states.shape[0]

    tokens_after_orig = original_states.shape[1] - post_perturbation_start_orig
    tokens_after_pert = perturbed_states.shape[1] - post_perturbation_start_pert
    max_offset = min(tokens_after_orig, tokens_after_pert)

    if max_offset <= 0:
        return []

    # Slice the post-perturbation region for all layers at once.
    # Shape: (num_layers, max_offset, hidden_dim)
    orig_post = original_states[
        :,
        post_perturbation_start_orig : post_perturbation_start_orig + max_offset,
        :,
    ]
    pert_post = perturbed_states[
        :,
        post_perturbation_start_pert : post_perturbation_start_pert + max_offset,
        :,
    ]

    # Vectorized cosine similarity across ALL layers and ALL tokens in one call.
    # Shape: (num_layers, max_offset)
    all_sim = F.cosine_similarity(orig_post, pert_post, dim=-1)

    # --- Build window definitions ------------------------------------
    # Each window: (bucket_name, offset_start, offset_end)
    windows: List[Tuple[str, int, int]] = []

    windows.append(("immediate", 0, min(immediate_window, max_offset)))
    windows.append(("short", 0, min(short_window, max_offset)))
    windows.append(("all_remaining", 0, max_offset))

    offset = 0
    while offset < max_offset:
        wend = min(offset + periodic_width, max_offset)
        windows.append((f"periodic_{offset}", offset, wend))
        offset += periodic_step

    # Per-token windows (first N tokens).
    per_token_count = min(immediate_window, max_offset)

    # --- Extract stats per layer: batch the work ----------------------
    # Move the full similarity matrix to CPU in one transfer.
    all_sim_cpu = all_sim.cpu()  # (num_layers, max_offset) — one transfer

    records: List[Dict] = []

    for layer_idx in range(num_layers):
        layer_sim = all_sim_cpu[layer_idx]  # (max_offset,) on CPU

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

        # Per-token entries (first N tokens) — no loop over GPU tensors.
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
