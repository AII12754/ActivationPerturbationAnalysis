"""
Core experiment logic for residual stream decomposition.

Each transformer layer adds delta[L] = h[L] - h[L-1] to the residual stream.
This experiment measures which layers do real work (large delta) vs skip
(small delta), and how the deltas relate to the residual stream and embeddings.

Operates in two phases: prefill (single forward, averaged over tokens) and
decode (autoregressive loop, per-step per-layer measurements).
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Main residual experiment runner
# ===================================================================
@torch.inference_mode()
def run_residual_experiment(
    model,
    tokenizer,
    prompt_text: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run a single residual stream decomposition experiment.

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
    records — list of dicts (prefill + decode) for storage.
    """
    residual_cfg = config.get("residual", {})
    num_decode = residual_cfg.get("num_decode_tokens", 128)
    do_sample = residual_cfg.get("do_sample", False)
    temperature = residual_cfg.get("temperature", 1.0)
    act_dtype_str = residual_cfg.get("activation_dtype", "float32")
    act_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(act_dtype_str, torch.float32)

    device = next(model.parameters()).device

    # 1. Tokenize prompt.
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    context_length = len(input_ids)

    # 2. Prefill: get hidden states + KV cache.
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

    # --- Prefill phase: compute deltas for all tokens, store mean per layer ---
    prefill_records: List[Dict[str, Any]] = []
    for layer_idx in range(1, num_all_layers + 1):
        h_curr = all_hidden[layer_idx].squeeze(0).to(act_dtype)   # (seq_len, hidden_dim)
        h_prev = all_hidden[layer_idx - 1].squeeze(0).to(act_dtype)
        h_embed = all_hidden[0].squeeze(0).to(act_dtype)

        delta = h_curr - h_prev  # (seq_len, hidden_dim)

        delta_norm = delta.norm(dim=1)      # (seq_len,)
        residual_norm = h_curr.norm(dim=1)  # (seq_len,)

        ratio = delta_norm / residual_norm.clamp(min=1e-8)
        cos_dr = F.cosine_similarity(delta, h_curr, dim=1)
        cos_de = F.cosine_similarity(delta, h_embed, dim=1)

        # Cosine with previous layer's delta.
        if layer_idx >= 2:
            h_prev_prev = all_hidden[layer_idx - 2].squeeze(0).to(act_dtype)
            prev_delta = h_prev - h_prev_prev
            cos_dp = F.cosine_similarity(delta, prev_delta, dim=1)
        else:
            cos_dp = torch.full((h_curr.shape[0],), float('nan'), device=device)

        # Store mean across all prefill tokens (not per-token to save space).
        prefill_records.append({
            "phase": "prefill",
            "step": -1,  # -1 means averaged over prefill
            "layer": layer_idx,
            "delta_norm": delta_norm.mean().item(),
            "residual_norm": residual_norm.mean().item(),
            "delta_residual_ratio": ratio.mean().item(),
            "cos_delta_residual": cos_dr.mean().item(),
            "cos_delta_embedding": cos_de.mean().item(),
            "cos_delta_prev_delta": cos_dp.nanmean().item(),
            "delta_norm_std": delta_norm.std().item(),
            "residual_norm_std": residual_norm.std().item(),
        })

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]  # (1, vocab_size)

    del outputs
    gc.collect()

    # 3. Decode loop.
    decode_records: List[Dict[str, Any]] = []

    for step in range(num_decode):
        # Select next token.
        if do_sample and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        # Forward with KV cache.
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )

        all_hidden = outputs.hidden_states
        h_embed = all_hidden[0].squeeze(0).squeeze(0).to(act_dtype)  # (hidden_dim,)

        prev_delta = None
        for layer_idx in range(1, num_all_layers + 1):
            h_curr = all_hidden[layer_idx].squeeze(0).squeeze(0).to(act_dtype)
            h_prev = all_hidden[layer_idx - 1].squeeze(0).squeeze(0).to(act_dtype)

            delta = h_curr - h_prev
            d_norm = delta.norm().item()
            r_norm = h_curr.norm().item()

            cos_dr = F.cosine_similarity(delta.unsqueeze(0), h_curr.unsqueeze(0)).item()
            cos_de = F.cosine_similarity(delta.unsqueeze(0), h_embed.unsqueeze(0)).item()

            if prev_delta is not None:
                cos_dp = F.cosine_similarity(delta.unsqueeze(0), prev_delta.unsqueeze(0)).item()
            else:
                cos_dp = float('nan')

            decode_records.append({
                "phase": "decode",
                "step": step,
                "layer": layer_idx,
                "delta_norm": d_norm,
                "residual_norm": r_norm,
                "delta_residual_ratio": d_norm / max(r_norm, 1e-8),
                "cos_delta_residual": cos_dr,
                "cos_delta_embedding": cos_de,
                "cos_delta_prev_delta": cos_dp,
                "delta_norm_std": 0.0,
                "residual_norm_std": 0.0,
            })
            prev_delta = delta

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        del outputs

    # 4. Clean up.
    del past_key_values, logits
    gc.collect()
    torch.cuda.empty_cache()

    return prefill_records + decode_records
