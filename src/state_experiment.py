"""
Core experiment logic for activation state detection.

Tracks hidden-state trajectories in PCA space during generation and
detects regime changes.  For each tracked layer, fits a PCA basis from
prefill hidden states, then projects each decoded token's hidden state
onto that basis and records coordinates, step-to-step cosine similarity,
centroid distance, and norm.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Main state-detection experiment runner
# ===================================================================
@torch.inference_mode()
def run_state_experiment(
    model,
    tokenizer,
    prompt_text: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run a single activation-state detection experiment.

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
    trajectory_records — list of dicts for storage.
    """
    state_cfg = config.get("state_detection", {})
    num_decode = state_cfg.get("num_decode_tokens", 256)
    n_components = state_cfg.get("pca_components", 20)
    do_sample = state_cfg.get("do_sample", False)
    temperature = state_cfg.get("temperature", 1.0)
    act_dtype_str = state_cfg.get("activation_dtype", "float32")
    act_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(act_dtype_str, torch.float32)

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    context_length = len(input_ids)

    # 1. Prefill: get hidden states + KV cache.
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

    # 2. Pick 9 evenly spaced layers across the transformer stack.
    tracked_layers = state_cfg.get("tracked_layers", None)
    if tracked_layers is None:
        step = max(1, num_all_layers // 8)
        tracked_layers = list(range(1, num_all_layers + 1, step))
        if num_all_layers not in tracked_layers:
            tracked_layers.append(num_all_layers)

    # 3. Build PCA basis from prefill hidden states for each tracked layer.
    pca_bases: Dict[int, tuple] = {}  # layer -> (V, mean)
    for layer_idx in tracked_layers:
        h = all_hidden[layer_idx].squeeze(0).to(act_dtype)  # (seq_len, hidden_dim)
        h_mean = h.mean(dim=0)
        h_centered = h - h_mean
        q = min(n_components, h.shape[0] - 1, h.shape[1])
        U, S, V = torch.pca_lowrank(h_centered, q=q, niter=3)
        # V: (hidden_dim, q) — PCA directions
        pca_bases[layer_idx] = (V[:, :n_components].clone(), h_mean.clone())

    # 4. Compute prefill centroid for each layer (in PCA space).
    centroids: Dict[int, torch.Tensor] = {}
    for layer_idx in tracked_layers:
        h = all_hidden[layer_idx].squeeze(0).to(act_dtype)
        V, h_mean = pca_bases[layer_idx]
        projected = (h - h_mean) @ V  # (seq_len, n_components)
        centroids[layer_idx] = projected.mean(dim=0)  # (n_components,)

    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]  # (1, vocab_size)

    del outputs
    gc.collect()

    # 5. Decode loop — project each new token onto PCA basis and record trajectory.
    trajectory_records: List[Dict[str, Any]] = []
    prev_projected: Dict[int, Optional[torch.Tensor]] = {
        layer_idx: None for layer_idx in tracked_layers
    }

    for step_idx in range(num_decode):
        # Select next token.
        if do_sample and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        next_token_id = next_token.item()

        # Forward with KV cache.
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )

        all_hidden = outputs.hidden_states

        for layer_idx in tracked_layers:
            h = all_hidden[layer_idx].squeeze(0).squeeze(0).to(act_dtype)  # (hidden_dim,)
            V, h_mean = pca_bases[layer_idx]

            projected = (h - h_mean) @ V  # (n_components,)
            h_norm = h.norm().item()
            centroid_dist = (projected - centroids[layer_idx]).norm().item()

            # Step-to-step cosine similarity.
            if prev_projected[layer_idx] is not None:
                step_cosine = F.cosine_similarity(
                    projected.unsqueeze(0),
                    prev_projected[layer_idx].unsqueeze(0),
                ).item()
            else:
                step_cosine = float("nan")

            # Build record with PCA coordinates as separate columns.
            coords = projected.cpu().tolist()
            rec: Dict[str, Any] = {
                "decode_step": step_idx,
                "layer": layer_idx,
                "token_id": next_token_id,
                "norm": h_norm,
                "centroid_distance": centroid_dist,
                "step_cosine": step_cosine,
            }
            for i, c in enumerate(coords[:n_components]):
                rec[f"pc{i}"] = c

            trajectory_records.append(rec)
            prev_projected[layer_idx] = projected.clone()

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        del outputs

    # Clean up.
    del past_key_values, logits, pca_bases, centroids, prev_projected
    gc.collect()
    torch.cuda.empty_cache()

    return trajectory_records
