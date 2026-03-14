"""
Core experiment logic for logit lens analysis.

Projects intermediate hidden states through model.model.norm + model.lm_head
to see what token each layer would predict. Reveals when the model "decides"
on its output and how certainty builds across layers.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Main logit lens experiment runner
# ===================================================================
@torch.inference_mode()
def run_logit_lens_experiment(
    model,
    tokenizer,
    prompt_text: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run a single logit lens experiment.

    For each decode step, projects every tracked layer's hidden state
    through norm + lm_head to obtain a full vocabulary distribution,
    then computes entropy, KL divergence from the final layer,
    rank of the correct token, and top-k predictions.

    Parameters
    ----------
    model :
        HuggingFace causal LM (output_hidden_states=True, KV-cache enabled).
    tokenizer :
        Corresponding tokenizer.
    prompt_text :
        The prompt to start decoding from.
    config :
        Full experiment config dict.

    Returns
    -------
    (per_step_records, topk_records) — lists of dicts for storage.
    """
    lens_cfg = config.get("logit_lens", {})
    num_decode = lens_cfg.get("num_decode_tokens", 128)
    top_k = lens_cfg.get("top_k_predictions", 5)
    do_sample = lens_cfg.get("do_sample", False)
    temperature = lens_cfg.get("temperature", 1.0)
    # Select subset of layers to project (to save compute)
    layer_stride = lens_cfg.get("layer_stride", 1)

    device = next(model.parameters()).device
    # Get norm and lm_head from model
    norm_layer = model.model.norm   # Qwen2RMSNorm
    lm_head = model.lm_head         # Linear(5120, 152064)

    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    context_length = len(input_ids)

    # Prefill
    logger.debug("Prefill: %d tokens", context_length)
    outputs = model(
        input_ids=input_tensor,
        output_hidden_states=True,
        use_cache=True,
    )

    # Extract hidden states: tuple of (num_layers+1,) tensors each (1, seq_len, hidden_dim).
    # Layer 0 = embedding, layers 1..num_layers = transformer layers.
    all_hidden = outputs.hidden_states
    num_all_layers = len(all_hidden) - 1  # exclude embedding layer

    # Determine which layers to project
    tracked_layers = list(range(0, num_all_layers + 1, layer_stride))
    if num_all_layers not in tracked_layers:
        tracked_layers.append(num_all_layers)

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]  # (1, vocab_size)

    del outputs
    gc.collect()

    per_step_records: List[Dict[str, Any]] = []
    topk_records: List[Dict[str, Any]] = []

    for step in range(num_decode):
        # Select next token
        if do_sample and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        next_token_id = next_token.item()

        # Forward with KV cache
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        all_hidden = outputs.hidden_states

        # Get the final layer's distribution as reference
        # Each hidden state: (1, 1, hidden_dim) -> (hidden_dim,)
        final_hidden = all_hidden[num_all_layers].squeeze(0).squeeze(0)
        final_normed = norm_layer(final_hidden.unsqueeze(0)).squeeze(0)
        final_logits_vec = lm_head(final_normed.unsqueeze(0)).squeeze(0).float()
        final_log_probs = F.log_softmax(final_logits_vec, dim=0)

        # The "correct" token is what the final layer predicts
        final_pred_token = final_logits_vec.argmax().item()

        # Process one layer at a time to avoid OOM on vocab_size=152064
        for layer_idx in tracked_layers:
            h = all_hidden[layer_idx].squeeze(0).squeeze(0)  # (hidden_dim,)
            # Project through norm + lm_head
            h_normed = norm_layer(h.unsqueeze(0)).squeeze(0)
            layer_logits = lm_head(h_normed.unsqueeze(0)).squeeze(0).float()

            layer_log_probs = F.log_softmax(layer_logits, dim=0)
            layer_probs = layer_log_probs.exp()

            # Entropy: H = -sum p * log p
            entropy = -(layer_probs * layer_log_probs).sum().item()

            # KL divergence: KL(final || layer) = sum final_p * (log final_p - log layer_p)
            kl_div = F.kl_div(
                layer_log_probs, final_log_probs.exp(),
                reduction='sum', log_target=False,
            ).item()

            # Rank of correct token (final layer's prediction) in this layer's distribution
            sorted_indices = layer_logits.argsort(descending=True)
            rank_of_correct = (sorted_indices == final_pred_token).nonzero(as_tuple=True)[0].item()

            # Cross-entropy of correct token: -log p(correct)
            ce_correct = -layer_log_probs[final_pred_token].item()

            # Max probability and top-1 token
            max_prob = layer_probs.max().item()
            top1_token_id = sorted_indices[0].item()

            per_step_records.append({
                "decode_step": step,
                "layer": layer_idx,
                "entropy": entropy,
                "kl_from_final": kl_div,
                "rank_of_correct": rank_of_correct,
                "cross_entropy_correct": ce_correct,
                "max_prob": max_prob,
                "top1_token_id": top1_token_id,
                "correct_token_id": final_pred_token,
                "generated_token_id": next_token_id,
            })

            # Top-k predictions
            topk_vals, topk_idx = layer_logits.topk(top_k)
            topk_probs = F.softmax(topk_vals, dim=0)
            for rank in range(top_k):
                topk_records.append({
                    "decode_step": step,
                    "layer": layer_idx,
                    "rank": rank,
                    "token_id": topk_idx[rank].item(),
                    "logit": topk_vals[rank].item(),
                    "probability": topk_probs[rank].item(),
                })

            del layer_logits, layer_log_probs, layer_probs, h_normed

        del final_logits_vec, final_log_probs
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        del outputs

    # Clean up
    del past_key_values, logits
    gc.collect()
    torch.cuda.empty_cache()

    return per_step_records, topk_records
