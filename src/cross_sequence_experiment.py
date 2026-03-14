"""
Core experiment logic for cross-sequence alignment analysis.

For a pair of prompts, runs both through the model, extracts hidden states,
then compares per layer using five complementary representation similarity
measures: CKA, Procrustes distance, subspace overlap, centroid cosine,
and shared-token cosine.

Theory: When different prompts produce similar outputs, do they converge
to similar internal representations? Tests canonical vs context-dependent
representations.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Representation similarity metrics
# ===================================================================
def linear_cka(X, Y):
    """Compute linear CKA between X (n, d1) and Y (n, d2).
    Both must have the same number of rows (aligned samples).
    For unequal lengths, truncate to min length.
    """
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n].float(), Y[:n].float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # HSIC
    XtX = X @ X.T  # (n, n)
    YtY = Y @ Y.T  # (n, n)

    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()


def procrustes_distance(X, Y):
    """Orthogonal Procrustes distance between centered, normalized X and Y."""
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n].float(), Y[:n].float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    # Normalize
    X = X / (X.norm() + 1e-10)
    Y = Y / (Y.norm() + 1e-10)
    # Optimal rotation: U, _, Vt = svd(Y.T @ X)
    M = Y.T @ X
    U, S, Vt = torch.linalg.svd(M)
    # Procrustes distance = ||X - Y @ R||^2 where R = U @ Vt
    # = 2 - 2 * trace(S) since both are normalized
    dist = max(0.0, 2.0 - 2.0 * S.sum().item())
    return dist


def subspace_overlap(X, Y, k=10):
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
    # Overlap = ||Vx.T @ Vy||_F^2 / k
    overlap = (Vx.T @ Vy).pow(2).sum().item() / k
    return overlap


# ===================================================================
# Main cross-sequence experiment runner
# ===================================================================
@torch.inference_mode()
def run_cross_sequence_experiment(
    model,
    tokenizer,
    prompt_text_a: str,
    prompt_text_b: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compare hidden states between two prompts.

    Parameters
    ----------
    model :
        HuggingFace causal LM (output_hidden_states=True).
    tokenizer :
        Corresponding tokenizer.
    prompt_text_a :
        First prompt text.
    prompt_text_b :
        Second prompt text.
    config :
        Full experiment config dict.

    Returns
    -------
    records — list of dicts, one per layer, with all five metrics.
    """
    cross_cfg = config.get("cross_sequence", {})
    act_dtype_str = cross_cfg.get("activation_dtype", "float32")
    act_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(act_dtype_str, torch.float32)
    subspace_k = cross_cfg.get("subspace_k", 10)

    device = next(model.parameters()).device

    # Forward pass A
    ids_a = tokenizer.encode(prompt_text_a, add_special_tokens=False)
    tensor_a = torch.tensor([ids_a], dtype=torch.long, device=device)
    outputs_a = model(input_ids=tensor_a, output_hidden_states=True, use_cache=False)
    hidden_a = outputs_a.hidden_states
    num_all_layers = len(hidden_a) - 1
    # Keep hidden states on CPU to free GPU for second forward pass
    hidden_a_cpu = [h.squeeze(0).to(act_dtype).cpu() for h in hidden_a]
    del outputs_a
    gc.collect()
    torch.cuda.empty_cache()

    # Forward pass B
    ids_b = tokenizer.encode(prompt_text_b, add_special_tokens=False)
    tensor_b = torch.tensor([ids_b], dtype=torch.long, device=device)
    outputs_b = model(input_ids=tensor_b, output_hidden_states=True, use_cache=False)
    hidden_b = outputs_b.hidden_states
    hidden_b_cpu = [h.squeeze(0).to(act_dtype).cpu() for h in hidden_b]
    del outputs_b
    gc.collect()
    torch.cuda.empty_cache()

    # Compare per layer
    records = []
    for layer_idx in range(num_all_layers + 1):
        ha = hidden_a_cpu[layer_idx]  # (seq_len_a, hidden_dim)
        hb = hidden_b_cpu[layer_idx]  # (seq_len_b, hidden_dim)

        # CKA (truncate to min length)
        cka = linear_cka(ha, hb)

        # Procrustes
        proc_dist = procrustes_distance(ha, hb)

        # Subspace overlap
        sub_overlap = subspace_overlap(ha, hb, k=subspace_k)

        # Centroid cosine
        centroid_a = ha.mean(dim=0)
        centroid_b = hb.mean(dim=0)
        centroid_cos = F.cosine_similarity(
            centroid_a.unsqueeze(0), centroid_b.unsqueeze(0)
        ).item()

        # Shared token cosine: find token IDs in common
        set_a = set(ids_a)
        set_b = set(ids_b)
        shared_ids = set_a & set_b
        shared_cosines = []
        if shared_ids:
            for tid in list(shared_ids)[:50]:  # cap at 50 to avoid too much compute
                pos_a = [i for i, t in enumerate(ids_a) if t == tid]
                pos_b = [i for i, t in enumerate(ids_b) if t == tid]
                # Use first occurrence
                if pos_a and pos_b and pos_a[0] < ha.shape[0] and pos_b[0] < hb.shape[0]:
                    cos = F.cosine_similarity(
                        ha[pos_a[0]].unsqueeze(0), hb[pos_b[0]].unsqueeze(0)
                    ).item()
                    shared_cosines.append(cos)
        shared_token_cosine = (
            sum(shared_cosines) / len(shared_cosines)
            if shared_cosines
            else float("nan")
        )

        records.append({
            "layer": layer_idx,
            "cka": cka,
            "procrustes_distance": proc_dist,
            "subspace_overlap": sub_overlap,
            "centroid_cosine": centroid_cos,
            "shared_token_cosine": shared_token_cosine,
            "num_shared_tokens": len(shared_ids),
            "seq_len_a": ha.shape[0],
            "seq_len_b": hb.shape[0],
        })

    del hidden_a_cpu, hidden_b_cpu
    gc.collect()
    return records
