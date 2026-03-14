"""
Core experiment logic for decode-time activation similarity.

For each newly decoded token, computes token-wise cosine similarity
between its hidden-state vector and every historical token's hidden-state
vector at each layer, then extracts top-k references and aggregate stats.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Activation History Buffer
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
# Token-wise cosine similarity computation
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

    Parameters
    ----------
    history :
        Buffer containing hidden states for all previous tokens.
    new_token_states :
        Shape ``(num_layers, hidden_dim)`` — the new token's activations.
    top_k :
        Number of most-similar reference tokens to return per layer.
    thresholds :
        Similarity thresholds for computing fraction-above stats.

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

    # history_states: (num_layers, hist_len, hidden_dim)
    history_states = history.get_all()

    # new_token_states: (num_layers, hidden_dim) -> (num_layers, hidden_dim, 1)
    query = new_token_states.to(history.dtype)

    # Normalize along hidden_dim for cosine similarity.
    # history_norm: (num_layers, hist_len, hidden_dim)
    history_norm = F.normalize(history_states, dim=2)
    # query_norm: (num_layers, hidden_dim)
    query_norm = F.normalize(query, dim=1)

    # Token-wise cosine similarity via batched matmul:
    # (num_layers, hist_len, hidden_dim) @ (num_layers, hidden_dim, 1)
    # -> (num_layers, hist_len, 1) -> (num_layers, hist_len)
    similarities = torch.bmm(
        history_norm, query_norm.unsqueeze(2)
    ).squeeze(2)  # (num_layers, hist_len)

    # Top-k per layer.
    topk_values, topk_indices = similarities.topk(k, dim=1)  # (num_layers, k)

    # Aggregate stats per layer.
    mean_sim = similarities.mean(dim=1)       # (num_layers,)
    std_sim = similarities.std(dim=1)         # (num_layers,)
    max_sim = similarities.max(dim=1).values  # (num_layers,)
    median_sim = similarities.median(dim=1).values  # (num_layers,)

    frac_above = {}
    for thr in thresholds:
        # Format threshold as 3-digit integer: 0.90 -> "090", 0.95 -> "095"
        key = f"frac_above_{int(round(thr * 100)):03d}"
        frac_above[key] = (similarities > thr).float().mean(dim=1)  # (num_layers,)

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
# Main decode experiment runner
# ===================================================================
@torch.inference_mode()
def run_decode_experiment(
    model,
    tokenizer,
    prompt_text: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run a single decode-time similarity experiment.

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
    (topk_records, aggregate_records) — lists of dicts for storage.
    """
    decode_cfg = config.get("decode", {})
    num_decode = decode_cfg.get("num_decode_tokens", 128)
    top_k = decode_cfg.get("top_k_references", 10)
    thresholds = decode_cfg.get("similarity_thresholds", [0.90, 0.95, 0.98])
    do_sample = decode_cfg.get("do_sample", False)
    temperature = decode_cfg.get("temperature", 1.0)
    layer_subset = decode_cfg.get("store_layer_subset", None)
    act_dtype_str = decode_cfg.get("activation_dtype", "float32")
    act_dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}.get(act_dtype_str, torch.float32)

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
    # Skip layer 0 (embedding) — use layers 1..num_layers.
    all_hidden = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
    num_all_layers = len(all_hidden) - 1  # exclude embedding layer
    hidden_dim = all_hidden[0].shape[2]

    # Determine which layers to track.
    if layer_subset is not None:
        tracked_layers = [l for l in layer_subset if 0 < l <= num_all_layers]
    else:
        tracked_layers = list(range(1, num_all_layers + 1))
    num_tracked = len(tracked_layers)

    max_seq = context_length + num_decode
    history = ActivationHistoryBuffer(
        num_layers=num_tracked,
        max_seq_len=max_seq,
        hidden_dim=hidden_dim,
        device=device,
        dtype=act_dtype,
    )

    # Fill history with prefill hidden states.
    # Stack tracked layers: (num_tracked, seq_len, hidden_dim)
    prefill_states = torch.stack(
        [all_hidden[l].squeeze(0) for l in tracked_layers], dim=0
    )
    history.add_batch(prefill_states)

    past_key_values = outputs.past_key_values

    # Get next token from prefill logits.
    logits = outputs.logits[:, -1, :]  # (1, vocab_size)

    del outputs
    gc.collect()

    # 3. Decode loop.
    topk_records: List[Dict[str, Any]] = []
    agg_records: List[Dict[str, Any]] = []
    decoded_token_ids: List[int] = []

    for step in range(num_decode):
        # Select next token.
        if do_sample and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        next_token_id = next_token.item()
        decoded_token_ids.append(next_token_id)

        # Forward with KV cache.
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )

        # Extract new token's hidden states for tracked layers.
        # Each hidden state: (1, 1, hidden_dim) -> (hidden_dim,)
        new_states = torch.stack(
            [outputs.hidden_states[l].squeeze(0).squeeze(0) for l in tracked_layers],
            dim=0,
        )  # (num_tracked, hidden_dim)

        # Compute similarity against history.
        topk_idx, topk_val, agg = compute_reference_similarity(
            history, new_states, top_k=top_k, thresholds=thresholds,
        )

        # Record top-k references per layer.
        current_pos = context_length + step
        for li, layer_idx in enumerate(tracked_layers):
            k_actual = topk_idx.shape[1]
            for rank in range(k_actual):
                ref_pos = topk_idx[li, rank].item()
                topk_records.append({
                    "decode_step": step,
                    "token_id": next_token_id,
                    "layer": layer_idx,
                    "ref_rank": rank,
                    "ref_token_position": ref_pos,
                    "ref_token_distance": current_pos - ref_pos,
                    "ref_similarity": topk_val[li, rank].item(),
                })

            # Record aggregate stats.
            agg_records.append({
                "decode_step": step,
                "token_id": next_token_id,
                "layer": layer_idx,
                "mean_similarity": agg["mean_similarity"][li],
                "median_similarity": agg["median_similarity"][li],
                "max_similarity": agg["max_similarity"][li],
                "std_similarity": agg["std_similarity"][li],
                "frac_above_090": agg["frac_above_090"][li],
                "frac_above_095": agg["frac_above_095"][li],
                "frac_above_098": agg["frac_above_098"][li],
                "num_historical_tokens": agg["num_historical_tokens"],
            })

        # Update history and prepare for next step.
        history.add(new_states)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        del outputs

    # 4. Batch-decode all token texts.
    decoded_texts = [tokenizer.decode([tid]) for tid in decoded_token_ids]

    # Also decode reference token IDs — collect all unique positions.
    # Build a mapping: position -> token_id from input_ids + decoded_token_ids.
    all_token_ids = input_ids + decoded_token_ids
    all_token_texts = [tokenizer.decode([tid]) for tid in all_token_ids]

    # Fill in token texts for topk records.
    for rec in topk_records:
        step = rec["decode_step"]
        rec["token_text"] = decoded_texts[step]
        ref_pos = rec["ref_token_position"]
        if ref_pos < len(all_token_ids):
            rec["ref_token_id"] = all_token_ids[ref_pos]
            rec["ref_token_text"] = all_token_texts[ref_pos]
        else:
            rec["ref_token_id"] = None
            rec["ref_token_text"] = None

    # Fill in token texts for aggregate records.
    for rec in agg_records:
        step = rec["decode_step"]
        rec["token_text"] = decoded_texts[step]

    # Clean up.
    del history, past_key_values, logits
    gc.collect()
    torch.cuda.empty_cache()

    return topk_records, agg_records
