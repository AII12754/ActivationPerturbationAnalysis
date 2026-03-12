"""
Similarity metrics for comparing hidden states.

Primary metric: token-level cosine similarity.
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
        Tensors of shape ``(seq_len, hidden_dim)``.
        ``seq_len`` must match.

    Returns
    -------
    Tensor of shape ``(seq_len,)`` with cosine similarities in [-1, 1].
    """
    assert states_a.shape == states_b.shape, (
        f"Shape mismatch: {states_a.shape} vs {states_b.shape}"
    )
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

@torch.no_grad()
def compute_distance_buckets(
    original_states: List[torch.Tensor],
    perturbed_states: List[torch.Tensor],
    post_perturbation_start_orig: int,
    post_perturbation_start_pert: int,
    immediate_window: int = 10,
    short_window: int = 100,
    periodic_step: int = 100,
    periodic_width: int = 10,
) -> List[Dict]:
    """Compute similarity in various distance buckets for all layers.

    Both *original_states* and *perturbed_states* are lists indexed by
    layer.  Tokens before the perturbation are assumed to be aligned
    (same positions).  Tokens *after* the perturbation are compared
    by their *offset from the perturbation end* in each respective
    sequence.

    Returns a flat list of dicts, each describing one measurement.
    """
    num_layers = len(original_states)
    records: List[Dict] = []

    for layer_idx in range(num_layers):
        orig = original_states[layer_idx]   # (seq_len_orig, hidden_dim)
        pert = perturbed_states[layer_idx]  # (seq_len_pert, hidden_dim)

        tokens_after_orig = orig.shape[0] - post_perturbation_start_orig
        tokens_after_pert = pert.shape[0] - post_perturbation_start_pert
        max_offset = min(tokens_after_orig, tokens_after_pert)

        if max_offset <= 0:
            continue

        # --- Helper to record a window --------------------------------
        def _record(bucket_name: str, offset_start: int, offset_end: int):
            ostart = post_perturbation_start_orig + offset_start
            oend   = post_perturbation_start_orig + offset_end
            pstart = post_perturbation_start_pert + offset_start
            pend   = post_perturbation_start_pert + offset_end

            o_slice = orig[ostart:oend]
            p_slice = pert[pstart:pend]
            length = min(o_slice.shape[0], p_slice.shape[0])
            if length == 0:
                return
            o_slice = o_slice[:length]
            p_slice = p_slice[:length]

            sim = cosine_similarity_per_token(o_slice, p_slice)
            records.append({
                "layer": layer_idx,
                "bucket": bucket_name,
                "offset_start": offset_start,
                "offset_end": offset_start + length,
                "mean_similarity": sim.mean().item(),
                "min_similarity": sim.min().item(),
                "max_similarity": sim.max().item(),
                "std_similarity": sim.std().item() if length > 1 else 0.0,
                "num_tokens": length,
            })

        # --- Named windows -------------------------------------------
        _record("immediate", 0, min(immediate_window, max_offset))
        _record("short", 0, min(short_window, max_offset))
        _record("all_remaining", 0, max_offset)

        # --- Periodic sampling ----------------------------------------
        offset = 0
        while offset < max_offset:
            wstart = offset
            wend = min(offset + periodic_width, max_offset)
            _record(f"periodic_{offset}", wstart, wend)
            offset += periodic_step

        # --- Per-token (first 10) for fine-grained curves -------------
        for t in range(min(immediate_window, max_offset)):
            opos = post_perturbation_start_orig + t
            ppos = post_perturbation_start_pert + t
            sim_val = F.cosine_similarity(
                orig[opos].unsqueeze(0), pert[ppos].unsqueeze(0)
            ).item()
            records.append({
                "layer": layer_idx,
                "bucket": "per_token",
                "offset_start": t,
                "offset_end": t + 1,
                "mean_similarity": sim_val,
                "min_similarity": sim_val,
                "max_similarity": sim_val,
                "std_similarity": 0.0,
                "num_tokens": 1,
            })

    return records
