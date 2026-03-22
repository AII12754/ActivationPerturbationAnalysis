from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def run_remaining_layers(model, hidden_states: torch.Tensor, attention_mask: torch.Tensor, start_layer: int = 0) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    cache_position = torch.arange(seq_len, device=hidden_states.device)
    causal_mask = model.model._update_causal_mask(attention_mask, hidden_states, cache_position, None, False)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    hidden = hidden_states
    for layer in model.model.layers[start_layer:]:
        hidden = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)


def compute_logit_drift_metrics(orig_logits: torch.Tensor, recon_logits: torch.Tensor) -> Dict[str, float]:
    seq_len = orig_logits.shape[1]
    logit_cos = F.cosine_similarity(orig_logits.float(), recon_logits.float(), dim=-1).squeeze(0)
    orig_top1 = orig_logits.argmax(dim=-1).squeeze(0)
    recon_top1 = recon_logits.argmax(dim=-1).squeeze(0)
    mismatch = orig_top1 != recon_top1
    first_top1 = int(torch.where(mismatch)[0][0].item()) if mismatch.any() else seq_len
    below = logit_cos < 0.999
    first_cos = int(torch.where(below)[0][0].item()) if below.any() else seq_len
    kl = F.kl_div(
        F.log_softmax(recon_logits.float(), dim=-1),
        F.softmax(orig_logits.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1).squeeze(0)
    return {
        "top1_match_rate": float((~mismatch).float().mean().item()),
        "first_top1_drift_pos": first_top1,
        "first_logit_cos_below_0_999": first_cos,
        "logit_cosine_mean": float(logit_cos.mean().item()),
        "logit_cosine_min": float(logit_cos.min().item()),
        "kl_mean": float(kl.mean().item()),
        "kl_max": float(kl.max().item()),
    }