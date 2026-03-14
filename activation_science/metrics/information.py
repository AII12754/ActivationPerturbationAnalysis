"""Information-theoretic metrics: logit lens entropy, KL divergence, rank.

Extracted from src/logit_lens_experiment.py.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def compute_logit_lens_metrics(
    layer_logits: torch.Tensor,
    final_log_probs: torch.Tensor,
    correct_token_id: int,
) -> Dict[str, Any]:
    """Compute information-theoretic metrics for a single layer's logit lens projection.

    Parameters
    ----------
    layer_logits:
        Shape ``(vocab_size,)`` — logits from projecting a layer's hidden state
        through norm + lm_head (in float32).
    final_log_probs:
        Shape ``(vocab_size,)`` — log-softmax of the final layer's logits.
    correct_token_id:
        The token predicted by the final layer (ground truth for this analysis).

    Returns
    -------
    Dict with keys: entropy, kl_from_final, rank_of_correct,
    cross_entropy_correct, max_prob, top1_token_id.
    """
    layer_log_probs = F.log_softmax(layer_logits, dim=0)
    layer_probs = layer_log_probs.exp()

    # Entropy: H = -sum p * log p
    entropy = -(layer_probs * layer_log_probs).sum().item()

    # KL divergence: KL(final || layer)
    kl_div = F.kl_div(
        layer_log_probs, final_log_probs.exp(),
        reduction='sum', log_target=False,
    ).item()

    # Rank of correct token
    sorted_indices = layer_logits.argsort(descending=True)
    rank_of_correct = (sorted_indices == correct_token_id).nonzero(as_tuple=True)[0].item()

    # Cross-entropy of correct token
    ce_correct = -layer_log_probs[correct_token_id].item()

    # Max probability and top-1 token
    max_prob = layer_probs.max().item()
    top1_token_id = sorted_indices[0].item()

    return {
        "entropy": entropy,
        "kl_from_final": kl_div,
        "rank_of_correct": rank_of_correct,
        "cross_entropy_correct": ce_correct,
        "max_prob": max_prob,
        "top1_token_id": top1_token_id,
    }
