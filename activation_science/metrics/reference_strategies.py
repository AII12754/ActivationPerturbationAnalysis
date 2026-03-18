"""Reference strategy builders and shared evaluation for E11.

Each builder returns ``(ref_acts, info_dict)`` where ``ref_acts`` has shape
``(seq_len, hidden_dim)`` and ``info_dict`` carries strategy-specific metadata
(e.g. number of unique n-gram entries, window size used).

``evaluate_strategy`` runs the full affine + delta + quantize + reconstruct
pipeline on a (real_acts, ref_acts) pair and returns a flat metrics dict.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .delta_coding import (
    apply_affine,
    compute_affine_params,
    compute_delta,
    compute_reconstruction_quality,
    compute_transfer_size,
    groupwise_int4_dequantize_topk,
    groupwise_int4_quantize_topk,
    reconstruct_activation,
    DeltaPacket,
)
from .static_delta import (
    StaticTokenTable,
    compute_affine_param_stats,
    compute_delta_magnitude_stats,
)

logger = logging.getLogger(__name__)


# ===================================================================
# 1. Static (baseline) — single-token, no context
# ===================================================================
def build_static_references(
    token_ids: torch.Tensor,
    static_table: StaticTokenTable,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Look up per-token static activations from a pre-built table.

    Parameters
    ----------
    token_ids : (seq_len,)
    static_table : StaticTokenTable with dense_table populated

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict
    """
    ref_acts = static_table.lookup(token_ids)
    num_unique = token_ids.unique().shape[0]
    return ref_acts, {"num_unique_refs": num_unique}


# ===================================================================
# 2 & 3. N-gram (bigram / trigram)
# ===================================================================
@torch.inference_mode()
def build_ngram_references(
    model,
    token_ids: torch.Tensor,
    layer_idx: int,
    n: int = 2,
    batch_size: int = 128,
    dtype: torch.dtype = torch.float16,
    static_table: StaticTokenTable | None = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build n-gram reference activations.

    For each position *i* in the sequence, the reference is the hidden state
    at the last position of the n-gram ``token_ids[i-n+1 : i+1]`` run through
    the model.  Positions with insufficient left context (``i < n-1``) fall
    back to a shorter n-gram or unigram (static table).

    Parameters
    ----------
    model : HuggingFace causal LM
    token_ids : (seq_len,) on model device
    layer_idx : int
    n : 2 for bigram, 3 for trigram
    batch_size : forward-pass batch size
    dtype : activation dtype
    static_table : fallback for positions without enough context

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict with num_unique_ngrams, fallback_count
    """
    device = next(model.parameters()).device
    seq_len = token_ids.shape[0]
    ids_cpu = token_ids.cpu().tolist()

    # Collect unique n-grams and their positions
    ngram_to_positions: Dict[tuple, list] = {}
    fallback_positions: list = []

    for i in range(seq_len):
        if i < n - 1:
            fallback_positions.append(i)
        else:
            gram = tuple(ids_cpu[i - n + 1 : i + 1])
            ngram_to_positions.setdefault(gram, []).append(i)

    # Forward-pass unique n-grams in batches
    unique_grams = list(ngram_to_positions.keys())
    ngram_acts: Dict[tuple, torch.Tensor] = {}

    for start in range(0, len(unique_grams), batch_size):
        batch_grams = unique_grams[start : start + batch_size]
        input_tensor = torch.tensor(batch_grams, dtype=torch.long, device=device)
        # input_tensor shape: (batch, n)
        outputs = model(
            input_ids=input_tensor,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[layer_idx]  # (batch, n, hidden_dim)
        # Take last-position hidden state
        last_hidden = hidden[:, -1, :].to(dtype).detach()  # (batch, hidden_dim)
        for j, gram in enumerate(batch_grams):
            ngram_acts[gram] = last_hidden[j]

    # Assemble ref_acts
    hidden_dim = next(iter(ngram_acts.values())).shape[0] if ngram_acts else 0
    if hidden_dim == 0 and static_table is not None and static_table.dense_table is not None:
        hidden_dim = static_table.dense_table.shape[1]
    ref_acts = torch.zeros(seq_len, hidden_dim, device=device, dtype=dtype)

    for gram, positions in ngram_to_positions.items():
        act = ngram_acts[gram]
        for pos in positions:
            ref_acts[pos] = act

    # Fallback for positions with insufficient context
    if fallback_positions and static_table is not None:
        fb_ids = token_ids[fallback_positions]
        fb_acts = static_table.lookup(fb_ids)
        for idx_in_fb, pos in enumerate(fallback_positions):
            ref_acts[pos] = fb_acts[idx_in_fb]

    info = {
        "num_unique_refs": len(unique_grams),
        "fallback_count": len(fallback_positions),
    }
    return ref_acts, info


# ===================================================================
# 4. Cluster Mean — per-token mean of all real occurrences (oracle)
# ===================================================================
def build_cluster_mean_references(
    real_acts: torch.Tensor,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Per-token mean activation across all occurrences in the prompt.

    Parameters
    ----------
    real_acts : (seq_len, hidden_dim)
    token_ids : (seq_len,)

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict
    """
    seq_len, hidden_dim = real_acts.shape
    device = real_acts.device
    dtype = real_acts.dtype

    unique_ids = token_ids.unique()
    mean_table = torch.zeros(unique_ids.max().item() + 1, hidden_dim, device=device, dtype=torch.float32)
    count_table = torch.zeros(unique_ids.max().item() + 1, device=device, dtype=torch.float32)

    # Accumulate
    ids_long = token_ids.long()
    mean_table.index_add_(0, ids_long, real_acts.float())
    count_table.index_add_(0, ids_long, torch.ones(seq_len, device=device, dtype=torch.float32))

    # Avoid division by zero
    count_table = count_table.clamp(min=1.0)
    mean_table = mean_table / count_table.unsqueeze(1)

    ref_acts = mean_table[ids_long].to(dtype)
    return ref_acts, {"num_unique_refs": unique_ids.shape[0]}


# ===================================================================
# 5. Sequential — previous token's real activation (semi-deployable)
# ===================================================================
def build_sequential_references(
    real_acts: torch.Tensor,
    static_table: StaticTokenTable,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Use the previous position's real activation as reference.

    Position 0 falls back to static lookup.

    Parameters
    ----------
    real_acts : (seq_len, hidden_dim)
    static_table : for position-0 fallback
    token_ids : (seq_len,)

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict
    """
    seq_len = real_acts.shape[0]
    ref_acts = torch.empty_like(real_acts)

    # Position 0: static fallback
    ref_acts[0] = static_table.lookup(token_ids[0:1]).squeeze(0)
    # Positions 1..N-1: previous real activation
    ref_acts[1:] = real_acts[:-1]

    return ref_acts, {"num_unique_refs": seq_len}


# ===================================================================
# 6. Sliding Window — mean of W preceding real activations (semi)
# ===================================================================
def build_sliding_window_references(
    real_acts: torch.Tensor,
    window_size: int,
    static_table: StaticTokenTable,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Mean of the W preceding real activations via cumsum (O(seq_len)).

    Positions with fewer than W predecessors use whatever is available.
    Position 0 falls back to static lookup.

    Parameters
    ----------
    real_acts : (seq_len, hidden_dim)
    window_size : W
    static_table : for position-0 fallback
    token_ids : (seq_len,)

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict
    """
    seq_len, hidden_dim = real_acts.shape
    device = real_acts.device
    dtype = real_acts.dtype

    # Cumulative sum in float32 for precision
    real_f = real_acts.float()
    cumsum = torch.zeros(seq_len + 1, hidden_dim, device=device, dtype=torch.float32)
    cumsum[1:] = real_f.cumsum(dim=0)

    ref_acts = torch.empty(seq_len, hidden_dim, device=device, dtype=dtype)

    # Position 0: static fallback
    ref_acts[0] = static_table.lookup(token_ids[0:1]).squeeze(0)

    # Positions 1..N-1: mean of preceding W (or fewer) activations
    positions = torch.arange(1, seq_len, device=device)
    window_starts = (positions - window_size).clamp(min=0)
    # sum of real_acts[window_start:pos] = cumsum[pos] - cumsum[window_start]
    window_sums = cumsum[positions] - cumsum[window_starts]
    window_counts = (positions - window_starts).float().unsqueeze(1)  # (N-1, 1)
    ref_acts[1:] = (window_sums / window_counts).to(dtype)

    return ref_acts, {"num_unique_refs": seq_len, "window_size": window_size}


# ===================================================================
# 7. Global Mean — mean of all real activations at this layer (oracle)
# ===================================================================
def build_global_mean_references(
    real_acts: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Broadcast the global mean activation to every position.

    Parameters
    ----------
    real_acts : (seq_len, hidden_dim)

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict
    """
    global_mean = real_acts.float().mean(dim=0, keepdim=True).to(real_acts.dtype)
    ref_acts = global_mean.expand_as(real_acts).contiguous()
    return ref_acts, {"num_unique_refs": 1}


# ===================================================================
# 8. EMA — exponential moving average of preceding activations (semi)
# ===================================================================
def build_ema_references(
    real_acts: torch.Tensor,
    alpha: float,
    static_table: StaticTokenTable,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Exponential moving average: ema[i] = alpha * real[i-1] + (1-alpha) * ema[i-1].

    Position 0 falls back to static lookup.

    Parameters
    ----------
    real_acts : (seq_len, hidden_dim)
    alpha : EMA weight for most-recent observation
    static_table : for position-0 fallback
    token_ids : (seq_len,)

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict
    """
    seq_len, hidden_dim = real_acts.shape
    device = real_acts.device
    dtype = real_acts.dtype

    ref_acts = torch.empty(seq_len, hidden_dim, device=device, dtype=dtype)

    # Position 0: static fallback
    static_ref = static_table.lookup(token_ids[0:1]).squeeze(0)
    ref_acts[0] = static_ref

    # EMA computation (sequential — cannot vectorize due to recurrence)
    ema = static_ref.float()
    real_f = real_acts.float()
    for i in range(1, seq_len):
        ema = alpha * real_f[i - 1] + (1.0 - alpha) * ema
        ref_acts[i] = ema.to(dtype)

    return ref_acts, {"num_unique_refs": seq_len, "alpha": alpha}


# ===================================================================
# 9. Low-Rank SVD — rank-r truncated SVD of full activation matrix (oracle)
# ===================================================================
def build_lowrank_references(
    real_acts: torch.Tensor,
    rank: int = 32,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Rank-r truncated SVD reconstruction as reference.

    Parameters
    ----------
    real_acts : (seq_len, hidden_dim)
    rank : number of singular values to keep

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict
    """
    real_f = real_acts.float()
    U, S, Vh = torch.linalg.svd(real_f, full_matrices=False)
    # Truncate to rank r
    U_r = U[:, :rank]       # (seq_len, r)
    S_r = S[:rank]           # (r,)
    Vh_r = Vh[:rank, :]      # (r, hidden_dim)
    reconstructed = (U_r * S_r.unsqueeze(0)) @ Vh_r  # (seq_len, hidden_dim)
    ref_acts = reconstructed.to(real_acts.dtype)

    energy_ratio = (S_r ** 2).sum().item() / ((S ** 2).sum().item() + 1e-12)
    return ref_acts, {"num_unique_refs": rank, "rank": rank, "energy_ratio": energy_ratio}


# ===================================================================
# 10. Hybrid — static + sequential with threshold switch
# ===================================================================
def build_hybrid_references(
    real_acts: torch.Tensor,
    token_ids: torch.Tensor,
    static_table: StaticTokenTable,
    threshold: float = 0.95,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Hybrid: use sequential reference when raw cosine with static >= threshold,
    else fall back to static.

    Position 0 always uses static.

    Parameters
    ----------
    real_acts : (seq_len, hidden_dim)
    token_ids : (seq_len,)
    static_table : StaticTokenTable
    threshold : cosine threshold for switching to sequential

    Returns
    -------
    ref_acts : (seq_len, hidden_dim)
    info : dict with num_sequential, num_static
    """
    seq_len = real_acts.shape[0]

    # Build both candidate references
    static_refs = static_table.lookup(token_ids)
    sequential_refs = torch.empty_like(real_acts)
    sequential_refs[0] = static_refs[0]
    sequential_refs[1:] = real_acts[:-1]

    # Cosine similarity between real and static (raw quality indicator)
    raw_cos = F.cosine_similarity(real_acts.float(), static_refs.float(), dim=-1)  # (seq_len,)

    # Where static is good enough (high raw cosine), keep static.
    # Where it's poor, sequential (previous real) is more informative.
    use_sequential = raw_cos < threshold
    use_sequential[0] = False  # position 0 has no predecessor

    ref_acts = torch.where(
        use_sequential.unsqueeze(1).expand_as(real_acts),
        sequential_refs,
        static_refs,
    )

    num_seq = use_sequential.sum().item()
    return ref_acts, {
        "num_unique_refs": seq_len,
        "num_sequential": int(num_seq),
        "num_static": int(seq_len - num_seq),
        "threshold": threshold,
    }


# ===================================================================
# Shared Evaluation Pipeline
# ===================================================================
def evaluate_strategy(
    real_acts: torch.Tensor,
    ref_acts: torch.Tensor,
    group_size: int = 128,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Run full affine + delta + quantize + reconstruct pipeline and return metrics.

    Parameters
    ----------
    real_acts : (seq_len, hidden_dim), the ground-truth prefill activations
    ref_acts : (seq_len, hidden_dim), the strategy's reference activations
    group_size : quantization group size
    top_k : outlier count per group

    Returns
    -------
    dict with raw_cosine_*, raw_mse_*, cosine_similarity_*, mse_*, max_abs_error_*,
    delta_*, affine_*, transfer_bytes, compression_ratio.
    """
    seq_len, hidden_dim = real_acts.shape

    # --- Raw cosine (before affine) ---
    raw_cos = F.cosine_similarity(real_acts.float(), ref_acts.float(), dim=-1)
    raw_diff = real_acts.float() - ref_acts.float()
    raw_mse_per_pos = (raw_diff ** 2).mean(dim=-1)

    raw_stats = {
        "raw_cosine_mean": raw_cos.mean().item(),
        "raw_cosine_min": raw_cos.min().item(),
        "raw_cosine_max": raw_cos.max().item(),
        "raw_cosine_std": raw_cos.std().item() if seq_len > 1 else 0.0,
        "raw_mse_mean": raw_mse_per_pos.mean().item(),
        "raw_mse_max": raw_mse_per_pos.max().item(),
    }

    # --- Affine transform ---
    scale, bias = compute_affine_params(real_acts, ref_acts)
    ref_transformed = apply_affine(ref_acts, scale, bias)
    delta = compute_delta(real_acts, ref_transformed)

    # --- Quantize ---
    packed, scales, zeros, topk_vals, topk_idx = groupwise_int4_quantize_topk(
        delta, group_size, top_k,
    )

    # --- Dequantize + reconstruct ---
    dequant = groupwise_int4_dequantize_topk(
        packed, scales, zeros, topk_vals, topk_idx, group_size, hidden_dim,
    )
    reconstructed = reconstruct_activation(dequant, ref_acts, scale.float(), bias.float())

    # --- Quality ---
    quality = compute_reconstruction_quality(real_acts, reconstructed)

    # --- Delta / affine stats ---
    delta_stats = compute_delta_magnitude_stats(delta)
    affine_stats = compute_affine_param_stats(scale, bias)

    # --- Transfer size ---
    packet = DeltaPacket(
        quantized_data=packed,
        scales=scales,
        zero_points=zeros,
        topk_values=topk_vals,
        topk_indices=topk_idx,
        affine_scale=scale.to(torch.float16),
        affine_bias=bias.to(torch.float16),
        ref_indices=torch.zeros(seq_len, dtype=torch.long, device=real_acts.device),
        group_size=group_size,
        top_k=top_k,
    )
    transfer_bytes = compute_transfer_size(packet)
    baseline_bytes = seq_len * hidden_dim * 2  # fp16
    compression_ratio = baseline_bytes / max(transfer_bytes, 1)

    result: Dict[str, Any] = {}
    result.update(raw_stats)
    result.update(quality)
    result.update(delta_stats)
    result.update(affine_stats)
    result["transfer_bytes"] = transfer_bytes
    result["compression_ratio"] = compression_ratio
    return result
