"""Delta-coding primitives for pipeline-parallel activation transfer.

Implements the encode/decode pipeline:
  Sender:  reference lookup -> affine transform -> delta -> Int4 quantize + top-k outliers
  Receiver: dequantize + outlier overlay -> affine reference -> reconstruct

All operations are GPU-native with CUDA event-based timing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===================================================================
# Dataclasses
# ===================================================================
@dataclass
class DeltaPacket:
    """Packed delta-coded activation for inter-stage transfer."""
    quantized_data: torch.Tensor    # (batch, hidden_dim // 2) uint8
    scales: torch.Tensor            # (batch, num_groups) float16
    zero_points: torch.Tensor       # (batch, num_groups) float16
    topk_values: torch.Tensor       # (batch, num_groups, k) float16
    topk_indices: torch.Tensor      # (batch, num_groups, k) uint8 — intra-group indices
    affine_scale: torch.Tensor      # (batch,) float16
    affine_bias: torch.Tensor       # (batch,) float16
    ref_indices: torch.Tensor       # (batch,) int64
    group_size: int
    top_k: int


@dataclass
class TimingResult:
    """GPU kernel timing statistics."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    num_runs: int


@dataclass
class LookupStats:
    """Statistics from optimized reference lookup strategies."""
    tokens_searched_mean: float   # avg history positions examined (early-exit)
    tokens_searched_max: float    # worst-case positions examined
    early_exit_rate: float        # fraction of batch that exited early
    bucket_hit_rate: float        # fraction with candidates in LSH bucket
    num_candidates_mean: float    # avg LSH candidate set size


# ===================================================================
# GPU Timing Utility
# ===================================================================
def gpu_timed(
    fn,
    *args,
    num_warmup: int = 3,
    num_runs: int = 10,
    **kwargs,
) -> Tuple[Any, TimingResult]:
    """Time a GPU function using CUDA events.

    Parameters
    ----------
    fn : callable
        Function to time. Must operate on GPU tensors.
    num_warmup : int
        Number of warmup iterations (results discarded).
    num_runs : int
        Number of timed iterations.

    Returns
    -------
    (last_result, TimingResult)
    """
    torch.cuda.synchronize()

    # Warmup
    result = None
    for _ in range(num_warmup):
        result = fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Timed runs
    times_ms: List[float] = []
    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()

        times_ms.append(start_event.elapsed_time(end_event))

    t = torch.tensor(times_ms)
    timing = TimingResult(
        mean_ms=t.mean().item(),
        std_ms=t.std().item() if len(times_ms) > 1 else 0.0,
        min_ms=t.min().item(),
        max_ms=t.max().item(),
        num_runs=num_runs,
    )
    return result, timing


# ===================================================================
# Per-Request History Buffer
# ===================================================================
class PerRequestHistoryBuffer:
    """Pre-allocated GPU buffer for per-request activation history.

    Shape: ``(num_requests, max_history_len, hidden_dim)`` on GPU.
    Used for cosine-similarity reference lookup during delta coding.
    """

    def __init__(
        self,
        num_requests: int,
        max_history_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_requests = num_requests
        self.max_history_len = max_history_len
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

        self.buffer = torch.zeros(
            num_requests, max_history_len, hidden_dim,
            device=device, dtype=dtype,
        )
        # Pre-normalized buffer — updated incrementally on add/add_batch_prefill
        self.buffer_norm = torch.zeros(
            num_requests, max_history_len, hidden_dim,
            device=device, dtype=dtype,
        )
        # Track how many tokens have been added per request
        self.lengths = torch.zeros(num_requests, dtype=torch.long, device=device)

    def add(self, request_idx: int, activation: torch.Tensor):
        """Add a single activation vector for one request.

        Parameters
        ----------
        request_idx : int
            Index of the request in the batch.
        activation : torch.Tensor
            Shape ``(hidden_dim,)``.
        """
        pos = self.lengths[request_idx].item()
        if pos < self.max_history_len:
            act = activation.to(self.dtype)
            self.buffer[request_idx, pos] = act
            self.buffer_norm[request_idx, pos] = F.normalize(act.unsqueeze(0), dim=-1).squeeze(0)
            self.lengths[request_idx] = pos + 1

    def add_batch_prefill(self, request_idx: int, activations: torch.Tensor):
        """Add a batch of prefill activations for one request.

        Parameters
        ----------
        request_idx : int
            Index of the request in the batch.
        activations : torch.Tensor
            Shape ``(seq_len, hidden_dim)``.
        """
        seq_len = activations.shape[0]
        current = self.lengths[request_idx].item()
        space = self.max_history_len - current
        n = min(seq_len, space)
        if n > 0:
            acts = activations[:n].to(self.dtype)
            self.buffer[request_idx, current:current + n] = acts
            self.buffer_norm[request_idx, current:current + n] = F.normalize(acts, dim=-1)
            self.lengths[request_idx] = current + n

    def find_best_references(
        self,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the most similar historical activation for each query.

        Parameters
        ----------
        queries : torch.Tensor
            Shape ``(batch, hidden_dim)``.

        Returns
        -------
        ref_indices : torch.Tensor
            Shape ``(batch,)`` — index into history for each request.
        ref_activations : torch.Tensor
            Shape ``(batch, hidden_dim)`` — the matched reference activations.
        """
        batch = queries.shape[0]

        # Use pre-normalized buffer for cosine similarity via bmm
        buf_norm = self.buffer_norm[:batch]  # (batch, max_hist, D)
        q_norm = F.normalize(queries.unsqueeze(1), dim=-1)  # (batch, 1, D)

        # Cosine similarity: (batch, 1, max_hist)
        sim = torch.bmm(q_norm, buf_norm.transpose(1, 2)).squeeze(1)  # (batch, max_hist)

        # Mask out unfilled positions with -inf
        mask = torch.arange(self.max_history_len, device=self.device).unsqueeze(0)
        mask = mask >= self.lengths[:batch].unsqueeze(1)  # True for invalid positions
        sim.masked_fill_(mask, float("-inf"))

        # Best match per request
        ref_indices = sim.argmax(dim=-1)  # (batch,)

        # Gather reference activations
        ref_activations = self.buffer[:batch].gather(
            1,
            ref_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim),
        ).squeeze(1)  # (batch, hidden_dim)

        return ref_indices, ref_activations

    def get_lookup_stats(self) -> Optional[LookupStats]:
        """Return lookup statistics if available. Base class returns None."""
        return None


# ===================================================================
# Optimized History Buffers
# ===================================================================
class EarlyExitHistoryBuffer(PerRequestHistoryBuffer):
    """History buffer with early-exit reference lookup.

    Searches history in reverse-recency chunks and exits early when
    all queries exceed the similarity threshold. Exploits temporal
    locality at early layers.
    """

    def __init__(
        self,
        num_requests: int,
        max_history_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        similarity_threshold: float = 0.95,
        chunk_size: int = 64,
    ):
        super().__init__(num_requests, max_history_len, hidden_dim, device, dtype)
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self._lookup_stats: Optional[LookupStats] = None

    def find_best_references(
        self,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = queries.shape[0]
        q_norm = F.normalize(queries, dim=-1)  # (batch, D)

        best_sim = torch.full((batch,), float("-inf"), device=self.device)
        best_idx = torch.zeros(batch, dtype=torch.long, device=self.device)
        found = torch.zeros(batch, dtype=torch.bool, device=self.device)

        # Per-request history lengths
        lengths = self.lengths[:batch]  # (batch,)
        max_len = lengths.max().item()

        # Track how many tokens each request searched
        tokens_searched = torch.zeros(batch, dtype=torch.long, device=self.device)

        # Process chunks in reverse order (most recent first)
        chunk_starts = list(range(max_len - self.chunk_size, -1, -self.chunk_size))
        if not chunk_starts or chunk_starts[-1] > 0:
            chunk_starts.append(0)

        for start in chunk_starts:
            if found.all():
                break

            end = min(start + self.chunk_size, max_len)
            chunk_norm = self.buffer_norm[:batch, start:end]  # (batch, chunk_len, D)
            # (batch, 1, D) x (batch, D, chunk_len) -> (batch, 1, chunk_len)
            sim = torch.bmm(
                q_norm.unsqueeze(1), chunk_norm.transpose(1, 2)
            ).squeeze(1)  # (batch, chunk_len)

            # Mask invalid positions
            pos_range = torch.arange(start, end, device=self.device).unsqueeze(0)
            invalid = pos_range >= lengths.unsqueeze(1)
            sim.masked_fill_(invalid, float("-inf"))

            # Update best per request
            chunk_best_sim, chunk_best_local = sim.max(dim=-1)
            chunk_best_global = chunk_best_local + start

            improved = chunk_best_sim > best_sim
            best_sim = torch.where(improved, chunk_best_sim, best_sim)
            best_idx = torch.where(improved, chunk_best_global, best_idx)

            # Track tokens searched for non-found requests
            valid_count = (~invalid).sum(dim=1)  # (batch,)
            tokens_searched = tokens_searched + torch.where(found, torch.zeros_like(valid_count), valid_count)

            # Check threshold
            found = found | (best_sim >= self.similarity_threshold)

        # Requests that never hit threshold searched everything
        # (tokens_searched already accumulated for them)

        # Gather reference activations
        ref_activations = self.buffer[:batch].gather(
            1,
            best_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim),
        ).squeeze(1)

        # Cache stats
        ts_float = tokens_searched.float()
        self._lookup_stats = LookupStats(
            tokens_searched_mean=ts_float.mean().item(),
            tokens_searched_max=ts_float.max().item(),
            early_exit_rate=found.float().mean().item(),
            bucket_hit_rate=0.0,
            num_candidates_mean=0.0,
        )

        return best_idx, ref_activations

    def get_lookup_stats(self) -> Optional[LookupStats]:
        return self._lookup_stats


class RecentWindowHistoryBuffer(PerRequestHistoryBuffer):
    """History buffer that searches only the most recent `window_size` positions.

    Exploits temporal locality by restricting cosine-similarity search to a
    recent window, yielding a single small bmm on (B, window_size, D) instead
    of the full (B, max_history_len, D).
    """

    def __init__(
        self,
        num_requests: int,
        max_history_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        window_size: int = 256,
    ):
        super().__init__(num_requests, max_history_len, hidden_dim, device, dtype)
        self.window_size = window_size
        self._lookup_stats: Optional[LookupStats] = None

    def find_best_references(
        self,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = queries.shape[0]
        lengths = self.lengths[:batch]  # (batch,)

        # Determine per-request window start: max(0, length - window_size)
        starts = torch.clamp(lengths - self.window_size, min=0)  # (batch,)
        max_len = lengths.max().item()
        window_start = max(0, int(max_len) - self.window_size)
        window_end = int(max_len)
        actual_window = window_end - window_start

        if actual_window <= 0:
            # Empty history
            ref_indices = torch.zeros(batch, dtype=torch.long, device=self.device)
            ref_activations = self.buffer[:batch, 0]
            self._lookup_stats = LookupStats(
                tokens_searched_mean=0.0, tokens_searched_max=0.0,
                early_exit_rate=0.0, bucket_hit_rate=0.0, num_candidates_mean=0.0,
            )
            return ref_indices, ref_activations

        # Slice the recent window from pre-normalized buffer
        buf_window = self.buffer_norm[:batch, window_start:window_end]  # (batch, actual_window, D)
        q_norm = F.normalize(queries.unsqueeze(1), dim=-1)  # (batch, 1, D)

        # Single bmm: (batch, 1, D) x (batch, D, actual_window) -> (batch, 1, actual_window)
        sim = torch.bmm(q_norm, buf_window.transpose(1, 2)).squeeze(1)  # (batch, actual_window)

        # Mask invalid positions (positions beyond each request's length)
        pos_range = torch.arange(window_start, window_end, device=self.device).unsqueeze(0)
        invalid = pos_range >= lengths.unsqueeze(1)
        # Also mask positions before each request's start
        invalid = invalid | (pos_range < starts.unsqueeze(1))
        sim.masked_fill_(invalid, float("-inf"))

        # Best match within window, offset back to global index
        best_local = sim.argmax(dim=-1)  # (batch,)
        ref_indices = best_local + window_start  # (batch,)

        # Gather reference activations from the original buffer
        ref_activations = self.buffer[:batch].gather(
            1,
            ref_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim),
        ).squeeze(1)

        # Cache stats
        tokens_searched = torch.clamp(lengths - starts, min=0).float()
        self._lookup_stats = LookupStats(
            tokens_searched_mean=tokens_searched.mean().item(),
            tokens_searched_max=tokens_searched.max().item(),
            early_exit_rate=0.0,
            bucket_hit_rate=0.0,
            num_candidates_mean=0.0,
        )

        return ref_indices, ref_activations

    def get_lookup_stats(self) -> Optional[LookupStats]:
        return self._lookup_stats


class LSHHistoryBuffer(PerRequestHistoryBuffer):
    """History buffer with LSH-based approximate reference lookup.

    Uses SimHash (random hyperplane) to hash activations into buckets,
    then only searches candidate matches — producing a compact-gather
    bmm that is much smaller than full-history search.
    """

    def __init__(
        self,
        num_requests: int,
        max_history_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        num_tables: int = 8,
        num_planes: int = 12,
        max_hamming: int = 0,
    ):
        super().__init__(num_requests, max_history_len, hidden_dim, device, dtype)
        self.num_tables = num_tables
        self.num_planes = num_planes
        self.max_hamming = max_hamming
        self._lookup_stats: Optional[LookupStats] = None

        # Random hyperplanes for SimHash, normalized
        hyperplanes = torch.randn(num_tables, hidden_dim, num_planes, device=device, dtype=torch.float32)
        self.hyperplanes = F.normalize(hyperplanes, dim=1)

        # Hash codes storage
        self.hash_codes = torch.zeros(
            num_requests, max_history_len, num_tables,
            device=device, dtype=torch.int32,
        )

    def _compute_hash(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute SimHash codes.

        Parameters
        ----------
        activations : torch.Tensor
            Shape ``(..., hidden_dim)``.

        Returns
        -------
        hash_codes : torch.Tensor
            Shape ``(..., num_tables)`` int32.
        """
        # (..., D) x (T, D, P) -> (..., T, P)
        proj = torch.einsum("...d,tdp->...tp", activations.float(), self.hyperplanes)
        signs = (proj > 0).int()  # (..., T, P)
        # Pack bits: sum of sign_i * 2^i
        powers = (2 ** torch.arange(self.num_planes, device=activations.device)).int()
        codes = (signs * powers.unsqueeze(0)).sum(dim=-1)  # (..., T)
        return codes.to(torch.int32)

    def add(self, request_idx: int, activation: torch.Tensor):
        pos = self.lengths[request_idx].item()
        super().add(request_idx, activation)
        if pos < self.max_history_len:
            self.hash_codes[request_idx, pos] = self._compute_hash(activation)

    def add_batch_prefill(self, request_idx: int, activations: torch.Tensor):
        current = self.lengths[request_idx].item()
        super().add_batch_prefill(request_idx, activations)
        new_len = self.lengths[request_idx].item()
        n = new_len - current
        if n > 0:
            codes = self._compute_hash(activations[:n])  # (n, T)
            self.hash_codes[request_idx, current:current + n] = codes

    def find_best_references(
        self,
        queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = queries.shape[0]
        lengths = self.lengths[:batch]

        # Hash queries: (batch, num_tables)
        query_codes = self._compute_hash(queries)

        # Compare with stored codes: (batch, max_history, num_tables)
        stored = self.hash_codes[:batch]

        if self.max_hamming == 0:
            # Exact match (original behavior)
            matches = (query_codes.unsqueeze(1) == stored).any(dim=-1)  # (batch, H)
        else:
            # Multi-probe: match if Hamming distance <= max_hamming in any table
            xor = query_codes.unsqueeze(1) ^ stored  # (batch, H, T) int32
            # Popcount via bit-unpack for small num_planes (<=16)
            bit_positions = torch.arange(self.num_planes, device=self.device)
            bits = (xor.unsqueeze(-1) >> bit_positions) & 1  # (batch, H, T, P)
            hamming = bits.sum(dim=-1)  # (batch, H, T)
            matches = (hamming <= self.max_hamming).any(dim=-1)  # (batch, H)

        # Mask invalid positions
        pos_range = torch.arange(self.max_history_len, device=self.device).unsqueeze(0)
        invalid = pos_range >= lengths.unsqueeze(1)
        matches.masked_fill_(invalid, False)

        # Count candidates per request
        cand_counts = matches.sum(dim=1)  # (batch,)

        # Fallback: requests with zero candidates use full history
        zero_cands = cand_counts == 0
        if zero_cands.any():
            full_valid = ~invalid  # (batch, H)
            matches = torch.where(zero_cands.unsqueeze(1), full_valid, matches)
            cand_counts = matches.sum(dim=1)

        max_cands = cand_counts.max().item()
        if max_cands == 0:
            # Edge case: empty history
            ref_indices = torch.zeros(batch, dtype=torch.long, device=self.device)
            ref_activations = self.buffer[:batch, 0]
            self._lookup_stats = LookupStats(
                tokens_searched_mean=0.0, tokens_searched_max=0.0,
                early_exit_rate=0.0, bucket_hit_rate=0.0, num_candidates_mean=0.0,
            )
            return ref_indices, ref_activations

        # Compact gather: get top-max_cands candidate indices per request
        # Use topk on float matches to get indices
        _, cand_indices = torch.topk(
            matches.float(), min(max_cands, matches.shape[1]), dim=1,
        )  # (batch, max_cands)

        # Gather candidate activations (pre-normalized): (batch, max_cands, D)
        cand_norm = self.buffer_norm[:batch].gather(
            1,
            cand_indices.unsqueeze(2).expand(-1, -1, self.hidden_dim),
        )

        # bmm on compact set: (batch, 1, D) x (batch, D, max_cands)
        q_norm = F.normalize(queries.unsqueeze(1), dim=-1)
        sim = torch.bmm(q_norm, cand_norm.transpose(1, 2)).squeeze(1)  # (batch, max_cands)

        # Mask out padding candidates
        cand_valid = torch.arange(sim.shape[1], device=self.device).unsqueeze(0) < cand_counts.unsqueeze(1)
        sim.masked_fill_(~cand_valid, float("-inf"))

        # Best match
        best_local = sim.argmax(dim=-1)  # (batch,)
        ref_indices = cand_indices.gather(1, best_local.unsqueeze(1)).squeeze(1)

        # Gather reference activations
        ref_activations = self.buffer[:batch].gather(
            1,
            ref_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_dim),
        ).squeeze(1)

        # Cache stats
        bucket_hit_rate = (~zero_cands).float().mean().item()
        self._lookup_stats = LookupStats(
            tokens_searched_mean=cand_counts.float().mean().item(),
            tokens_searched_max=cand_counts.float().max().item(),
            early_exit_rate=0.0,
            bucket_hit_rate=bucket_hit_rate,
            num_candidates_mean=cand_counts.float().mean().item(),
        )

        return ref_indices, ref_activations

    def get_lookup_stats(self) -> Optional[LookupStats]:
        return self._lookup_stats


def create_history_buffer(
    strategy: str,
    num_requests: int,
    max_history_len: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    **kwargs,
) -> PerRequestHistoryBuffer:
    """Factory for history buffer creation.

    Parameters
    ----------
    strategy : str
        One of ``"exact"``, ``"early_exit"``, ``"lsh"``.
    """
    if strategy == "exact":
        return PerRequestHistoryBuffer(num_requests, max_history_len, hidden_dim, device, dtype)
    elif strategy == "early_exit":
        return EarlyExitHistoryBuffer(
            num_requests, max_history_len, hidden_dim, device, dtype,
            similarity_threshold=kwargs.get("similarity_threshold", 0.95),
            chunk_size=kwargs.get("chunk_size", 64),
        )
    elif strategy == "recent_window":
        return RecentWindowHistoryBuffer(
            num_requests, max_history_len, hidden_dim, device, dtype,
            window_size=kwargs.get("window_size", 256),
        )
    elif strategy == "lsh":
        return LSHHistoryBuffer(
            num_requests, max_history_len, hidden_dim, device, dtype,
            num_tables=kwargs.get("num_tables", 8),
            num_planes=kwargs.get("num_planes", 12),
            max_hamming=kwargs.get("max_hamming", 0),
        )
    else:
        raise ValueError(f"Unknown lookup strategy: {strategy!r}")


# ===================================================================
# Core Operations
# ===================================================================
def compute_affine_params(
    new_acts: torch.Tensor,
    ref_acts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute affine transform parameters: scale and bias.

    ``scale = dot(new, ref) / dot(ref, ref)``
    ``bias = mean(new - scale * ref)``

    Parameters
    ----------
    new_acts : torch.Tensor
        Shape ``(batch, hidden_dim)``.
    ref_acts : torch.Tensor
        Shape ``(batch, hidden_dim)``.

    Returns
    -------
    scale : torch.Tensor — shape ``(batch,)``
    bias : torch.Tensor — shape ``(batch,)``
    """
    # Upcast to float32 to avoid overflow when activations have large magnitudes
    # (e.g. static single-token activations at deeper layers can exceed fp16 range)
    new_f = new_acts.float()
    ref_f = ref_acts.float()
    dot_nr = (new_f * ref_f).sum(dim=-1)  # (batch,)
    dot_rr = (ref_f * ref_f).sum(dim=-1)  # (batch,)
    scale = dot_nr / (dot_rr + 1e-8)  # (batch,)
    bias = (new_f - scale.unsqueeze(-1) * ref_f).mean(dim=-1)  # (batch,)
    return scale, bias


def apply_affine(
    ref_acts: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply affine transform: ``scale * ref + bias``.

    Parameters
    ----------
    ref_acts : torch.Tensor — shape ``(batch, hidden_dim)``
    scale : torch.Tensor — shape ``(batch,)``
    bias : torch.Tensor — shape ``(batch,)``

    Returns
    -------
    Tensor of shape ``(batch, hidden_dim)``.
    """
    return scale.unsqueeze(-1) * ref_acts + bias.unsqueeze(-1)


def compute_delta(
    new_acts: torch.Tensor,
    ref_transformed: torch.Tensor,
) -> torch.Tensor:
    """Compute delta: ``new - ref_transformed``.

    Parameters
    ----------
    new_acts : torch.Tensor — shape ``(batch, hidden_dim)``
    ref_transformed : torch.Tensor — shape ``(batch, hidden_dim)``

    Returns
    -------
    Tensor of shape ``(batch, hidden_dim)``.
    """
    return new_acts - ref_transformed


def groupwise_int4_quantize_topk(
    delta: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise Int4 quantization with top-k fp16 outlier extraction.

    Divides delta into groups of ``group_size`` elements. Within each group,
    extracts ``top_k`` largest-magnitude values as fp16 outliers, then
    quantizes the remainder to 4-bit symmetric (scale = (max-min)/15,
    two nibbles packed per uint8).

    Parameters
    ----------
    delta : torch.Tensor
        Shape ``(batch, hidden_dim)``.
    group_size : int
        Number of elements per quantization group.
    top_k : int
        Number of outlier values to keep in fp16 per group.

    Returns
    -------
    packed : torch.Tensor — ``(batch, hidden_dim // 2)`` uint8
    scales : torch.Tensor — ``(batch, num_groups)`` float16
    zero_points : torch.Tensor — ``(batch, num_groups)`` float16
    topk_values : torch.Tensor — ``(batch, num_groups, top_k)`` float16
    topk_indices : torch.Tensor — ``(batch, num_groups, top_k)`` uint8 (intra-group)
    """
    batch, hidden_dim = delta.shape
    num_groups = hidden_dim // group_size

    # Reshape to groups: (batch, num_groups, group_size)
    grouped = delta.reshape(batch, num_groups, group_size)

    # Extract top-k outliers per group
    abs_grouped = grouped.abs()
    topk_vals_abs, topk_idx = torch.topk(abs_grouped, top_k, dim=-1)  # (B, G, k)
    topk_values = grouped.gather(-1, topk_idx).to(torch.float16)  # actual signed values
    # Indices are intra-group (0..group_size-1), so uint8 suffices for group_size <= 256
    topk_indices = topk_idx.to(torch.uint8)

    # Zero out outlier positions for quantization
    grouped_zeroed = grouped.clone()
    grouped_zeroed.scatter_(-1, topk_idx, 0.0)

    # Per-group min/max for symmetric quantization
    g_min = grouped_zeroed.min(dim=-1).values  # (batch, num_groups)
    g_max = grouped_zeroed.max(dim=-1).values  # (batch, num_groups)
    scales = ((g_max - g_min) / 15.0).to(torch.float16)  # (batch, num_groups)
    zero_points = g_min.to(torch.float16)  # (batch, num_groups)

    # Quantize to [0, 15]
    scales_f = scales.float().unsqueeze(-1)  # (batch, num_groups, 1)
    zeros_f = zero_points.float().unsqueeze(-1)  # (batch, num_groups, 1)
    q = torch.clamp(
        torch.round((grouped_zeroed - zeros_f) / (scales_f + 1e-10)),
        0, 15,
    ).to(torch.uint8)  # (batch, num_groups, group_size)

    # Pack two nibbles per uint8: high << 4 | low
    q_flat = q.reshape(batch, hidden_dim)  # (batch, hidden_dim)
    even = q_flat[:, 0::2]  # high nibbles
    odd = q_flat[:, 1::2]   # low nibbles
    packed = (even << 4) | odd  # (batch, hidden_dim // 2)

    return packed, scales, zero_points, topk_values, topk_indices


def groupwise_int4_dequantize_topk(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    topk_values: torch.Tensor,
    topk_indices: torch.Tensor,
    group_size: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Dequantize Int4 groups and overlay top-k fp16 outliers.

    Parameters
    ----------
    packed : torch.Tensor — ``(batch, hidden_dim // 2)`` uint8
    scales : torch.Tensor — ``(batch, num_groups)`` float16
    zero_points : torch.Tensor — ``(batch, num_groups)`` float16
    topk_values : torch.Tensor — ``(batch, num_groups, top_k)`` float16
    topk_indices : torch.Tensor — ``(batch, num_groups, top_k)`` uint8 (intra-group)
    group_size : int
    hidden_dim : int

    Returns
    -------
    Tensor of shape ``(batch, hidden_dim)`` float16.
    """
    batch = packed.shape[0]
    num_groups = hidden_dim // group_size

    # Unpack uint8 -> two int4 values
    even = (packed >> 4).to(torch.uint8)  # high nibble
    odd = (packed & 0x0F).to(torch.uint8)  # low nibble

    # Interleave back to (batch, hidden_dim)
    q_flat = torch.zeros(batch, hidden_dim, dtype=torch.uint8, device=packed.device)
    q_flat[:, 0::2] = even
    q_flat[:, 1::2] = odd

    # Reshape to groups
    q_grouped = q_flat.reshape(batch, num_groups, group_size)

    # Dequantize: x = q * scale + zero_point
    scales_f = scales.float().unsqueeze(-1)  # (batch, num_groups, 1)
    zeros_f = zero_points.float().unsqueeze(-1)  # (batch, num_groups, 1)
    dequant = q_grouped.float() * scales_f + zeros_f  # (batch, num_groups, group_size)

    # Scatter top-k fp16 outliers at their original indices
    topk_idx_long = topk_indices.long()
    dequant.scatter_(-1, topk_idx_long, topk_values.float())

    return dequant.reshape(batch, hidden_dim).to(torch.float16)


@dataclass
class Int8Packet:
    """Packed Int8-quantized activation with optional Int4 residual."""
    quantized: torch.Tensor             # (batch, hidden_dim) uint8
    scales: torch.Tensor                # (batch, num_groups) float16
    zero_points: torch.Tensor           # (batch, num_groups) float16
    # Optional Int4 residual fields (None if no residual)
    residual_packed: Optional[torch.Tensor]
    residual_scales: Optional[torch.Tensor]
    residual_zero_points: Optional[torch.Tensor]
    residual_topk_values: Optional[torch.Tensor]
    residual_topk_indices: Optional[torch.Tensor]
    group_size: int
    residual_group_size: Optional[int]
    residual_top_k: Optional[int]


def groupwise_int8_quantize(
    tensor: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise Int8 quantization (256 levels, 0-255).

    Parameters
    ----------
    tensor : torch.Tensor — shape ``(batch, hidden_dim)``
    group_size : int

    Returns
    -------
    quantized : (batch, hidden_dim) uint8
    scales : (batch, num_groups) float16
    zero_points : (batch, num_groups) float16
    """
    batch, hidden_dim = tensor.shape
    num_groups = hidden_dim // group_size

    grouped = tensor.float().reshape(batch, num_groups, group_size)

    g_min = grouped.min(dim=-1).values  # (batch, num_groups)
    g_max = grouped.max(dim=-1).values
    scales = ((g_max - g_min) / 255.0).to(torch.float16)
    zero_points = g_min.to(torch.float16)

    scales_f = scales.float().unsqueeze(-1)      # (batch, num_groups, 1)
    zeros_f = zero_points.float().unsqueeze(-1)
    q = torch.clamp(
        torch.round((grouped - zeros_f) / (scales_f + 1e-10)),
        0, 255,
    ).to(torch.uint8)  # (batch, num_groups, group_size)

    quantized = q.reshape(batch, hidden_dim)
    return quantized, scales, zero_points


def groupwise_int8_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Dequantize Int8 groups back to float16.

    Parameters
    ----------
    quantized : (batch, hidden_dim) uint8
    scales : (batch, num_groups) float16
    zero_points : (batch, num_groups) float16
    group_size : int
    hidden_dim : int

    Returns
    -------
    Tensor of shape ``(batch, hidden_dim)`` float16.
    """
    batch = quantized.shape[0]
    num_groups = hidden_dim // group_size

    q_grouped = quantized.reshape(batch, num_groups, group_size)
    scales_f = scales.float().unsqueeze(-1)
    zeros_f = zero_points.float().unsqueeze(-1)
    dequant = q_grouped.float() * scales_f + zeros_f
    return dequant.reshape(batch, hidden_dim).to(torch.float16)


def int8_with_residual_encode(
    tensor: torch.Tensor,
    group_size: int,
    residual_top_k: int = 4,
) -> Int8Packet:
    """Int8 quantization with optional Int4+topk residual coding.

    Quantizes ``tensor`` to Int8, computes the residual (original - dequant),
    then applies Int4 group quantization with top-k outliers on the residual.

    Parameters
    ----------
    tensor : (batch, hidden_dim)
    group_size : int
    residual_top_k : int — per-group top-k outliers for the residual

    Returns
    -------
    Int8Packet
    """
    hidden_dim = tensor.shape[1]

    quantized, scales, zero_points = groupwise_int8_quantize(tensor, group_size)
    dequant = groupwise_int8_dequantize(quantized, scales, zero_points, group_size, hidden_dim)
    residual = tensor - dequant

    r_packed, r_scales, r_zeros, r_topk_vals, r_topk_idx = groupwise_int4_quantize_topk(
        residual, group_size, residual_top_k,
    )

    return Int8Packet(
        quantized=quantized,
        scales=scales,
        zero_points=zero_points,
        residual_packed=r_packed,
        residual_scales=r_scales,
        residual_zero_points=r_zeros,
        residual_topk_values=r_topk_vals,
        residual_topk_indices=r_topk_idx,
        group_size=group_size,
        residual_group_size=group_size,
        residual_top_k=residual_top_k,
    )


def int8_with_residual_decode(packet: Int8Packet) -> torch.Tensor:
    """Decode an Int8Packet back to float16 activations.

    Parameters
    ----------
    packet : Int8Packet

    Returns
    -------
    Tensor of shape ``(batch, hidden_dim)`` float16.
    """
    hidden_dim = packet.quantized.shape[1]
    base = groupwise_int8_dequantize(
        packet.quantized, packet.scales, packet.zero_points,
        packet.group_size, hidden_dim,
    )
    if packet.residual_packed is not None:
        residual = groupwise_int4_dequantize_topk(
            packet.residual_packed, packet.residual_scales, packet.residual_zero_points,
            packet.residual_topk_values, packet.residual_topk_indices,
            packet.group_size, hidden_dim,
        )
        base = base + residual
    return base


def compute_transfer_size_int8(packet: Int8Packet) -> int:
    """Compute total transfer size in bytes for an Int8Packet."""
    total = 0
    total += packet.quantized.nelement() * packet.quantized.element_size()
    total += packet.scales.nelement() * packet.scales.element_size()
    total += packet.zero_points.nelement() * packet.zero_points.element_size()
    if packet.residual_packed is not None:
        total += packet.residual_packed.nelement() * packet.residual_packed.element_size()
        total += packet.residual_scales.nelement() * packet.residual_scales.element_size()
        total += packet.residual_zero_points.nelement() * packet.residual_zero_points.element_size()
        total += packet.residual_topk_values.nelement() * packet.residual_topk_values.element_size()
        total += packet.residual_topk_indices.nelement() * packet.residual_topk_indices.element_size()
    return total


@dataclass
class Int8OutlierPacket:
    """Packed Int8-quantized activation with top-k fp16 outlier extraction (single pass)."""
    quantized: torch.Tensor             # (batch, hidden_dim) uint8
    scales: torch.Tensor                # (batch, num_groups) float16
    zero_points: torch.Tensor           # (batch, num_groups) float16
    topk_values: torch.Tensor           # (batch, num_groups, top_k) float16
    topk_indices: torch.Tensor          # (batch, num_groups, top_k) uint8 — intra-group
    group_size: int
    top_k: int


def groupwise_int8_quantize_topk(
    tensor: torch.Tensor,
    group_size: int,
    top_k: int,
) -> Int8OutlierPacket:
    """Group-wise Int8 quantization with top-k fp16 outlier extraction.

    Single-pass encoding: within each group, extract top-k largest-magnitude
    values as fp16 outliers, zero them out, then Int8-quantize the rest.
    On decode, dequantize Int8 and overlay outliers.

    Parameters
    ----------
    tensor : (batch, hidden_dim)
    group_size : int
    top_k : int — per-group outliers kept in fp16

    Returns
    -------
    Int8OutlierPacket
    """
    batch, hidden_dim = tensor.shape
    num_groups = hidden_dim // group_size

    grouped = tensor.float().reshape(batch, num_groups, group_size)

    # Extract top-k outliers per group
    abs_vals = grouped.abs()
    _, tk_idx = abs_vals.topk(top_k, dim=-1)  # (batch, num_groups, top_k)
    tk_vals = torch.gather(grouped, -1, tk_idx)  # (batch, num_groups, top_k)

    # Zero out outliers before quantizing
    masked = grouped.clone()
    masked.scatter_(-1, tk_idx, 0.0)

    # Int8 quantize the masked tensor
    g_min = masked.min(dim=-1).values  # (batch, num_groups)
    g_max = masked.max(dim=-1).values
    scales = ((g_max - g_min) / 255.0).to(torch.float16)
    zero_points = g_min.to(torch.float16)

    scales_f = scales.float().unsqueeze(-1)
    zeros_f = zero_points.float().unsqueeze(-1)
    q = torch.clamp(
        torch.round((masked - zeros_f) / (scales_f + 1e-10)),
        0, 255,
    ).to(torch.uint8)

    quantized = q.reshape(batch, hidden_dim)

    return Int8OutlierPacket(
        quantized=quantized,
        scales=scales,
        zero_points=zero_points,
        topk_values=tk_vals.to(torch.float16),
        topk_indices=tk_idx.to(torch.uint8),
        group_size=group_size,
        top_k=top_k,
    )


def groupwise_int8_dequantize_topk(packet: Int8OutlierPacket) -> torch.Tensor:
    """Dequantize Int8 + overlay top-k outliers.

    Returns
    -------
    Tensor of shape ``(batch, hidden_dim)`` float16.
    """
    hidden_dim = packet.quantized.shape[1]
    batch = packet.quantized.shape[0]
    num_groups = hidden_dim // packet.group_size

    q_grouped = packet.quantized.reshape(batch, num_groups, packet.group_size)
    scales_f = packet.scales.float().unsqueeze(-1)
    zeros_f = packet.zero_points.float().unsqueeze(-1)
    dequant = q_grouped.float() * scales_f + zeros_f  # (batch, num_groups, group_size)

    # Overlay outliers
    tk_idx = packet.topk_indices.long()
    tk_vals = packet.topk_values.float()
    dequant.scatter_(-1, tk_idx, tk_vals)

    return dequant.reshape(batch, hidden_dim).to(torch.float16)


def compute_transfer_size_int8_outlier(packet: Int8OutlierPacket) -> int:
    """Compute total transfer size in bytes for an Int8OutlierPacket."""
    total = 0
    total += packet.quantized.nelement() * packet.quantized.element_size()   # uint8
    total += packet.scales.nelement() * packet.scales.element_size()         # fp16
    total += packet.zero_points.nelement() * packet.zero_points.element_size()  # fp16
    total += packet.topk_values.nelement() * packet.topk_values.element_size()  # fp16
    total += packet.topk_indices.nelement() * packet.topk_indices.element_size()  # uint8
    return total


def reconstruct_activation(
    dequant_delta: torch.Tensor,
    ref_acts: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct activation: ``affine(ref) + dequantized_delta``.

    Parameters
    ----------
    dequant_delta : torch.Tensor — shape ``(batch, hidden_dim)``
    ref_acts : torch.Tensor — shape ``(batch, hidden_dim)``
    scale : torch.Tensor — shape ``(batch,)``
    bias : torch.Tensor — shape ``(batch,)``

    Returns
    -------
    Tensor of shape ``(batch, hidden_dim)``.
    """
    ref_transformed = apply_affine(ref_acts, scale, bias)
    return ref_transformed + dequant_delta


def compute_reconstruction_quality(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    """Compute quality metrics between original and reconstructed activations.

    Parameters
    ----------
    original : torch.Tensor — shape ``(batch, hidden_dim)``
    reconstructed : torch.Tensor — shape ``(batch, hidden_dim)``

    Returns
    -------
    Dict with keys: cosine_similarity_mean, cosine_similarity_min,
    mse_mean, mse_max, max_abs_error_mean, max_abs_error_max.
    """
    orig_f = original.float()
    recon_f = reconstructed.float()

    # Per-sample cosine similarity
    cos_sim = F.cosine_similarity(orig_f, recon_f, dim=-1)  # (batch,)

    # Per-sample MSE
    mse = ((orig_f - recon_f) ** 2).mean(dim=-1)  # (batch,)

    # Per-sample max absolute error
    max_abs = (orig_f - recon_f).abs().max(dim=-1).values  # (batch,)

    return {
        "cosine_similarity_mean": cos_sim.mean().item(),
        "cosine_similarity_min": cos_sim.min().item(),
        "mse_mean": mse.mean().item(),
        "mse_max": mse.max().item(),
        "max_abs_error_mean": max_abs.mean().item(),
        "max_abs_error_max": max_abs.max().item(),
    }


def compute_transfer_size(packet: DeltaPacket) -> int:
    """Compute total transfer size in bytes for a DeltaPacket.

    Sums: packed_data + scales + zero_points + topk_values + topk_indices
          + affine_scale + affine_bias + ref_indices.
    """
    total = 0
    total += packet.quantized_data.nelement() * packet.quantized_data.element_size()
    total += packet.scales.nelement() * packet.scales.element_size()
    total += packet.zero_points.nelement() * packet.zero_points.element_size()
    total += packet.topk_values.nelement() * packet.topk_values.element_size()
    total += packet.topk_indices.nelement() * packet.topk_indices.element_size()
    total += packet.affine_scale.nelement() * packet.affine_scale.element_size()
    total += packet.affine_bias.nelement() * packet.affine_bias.element_size()
    total += packet.ref_indices.nelement() * packet.ref_indices.element_size()
    return total


# ===================================================================
# Timed Pipelines
# ===================================================================
def run_timed_encode_pipeline(
    new_acts: torch.Tensor,
    history_buf: PerRequestHistoryBuffer,
    group_size: int,
    top_k: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Tuple[DeltaPacket, Dict[str, TimingResult]]:
    """Run the full encode pipeline with per-operation GPU timing.

    Parameters
    ----------
    new_acts : torch.Tensor
        Shape ``(batch, hidden_dim)`` — new activations to encode.
    history_buf : PerRequestHistoryBuffer
        Pre-filled history buffer for reference lookup.
    group_size : int
    top_k : int
    num_warmup : int
    num_runs : int

    Returns
    -------
    (DeltaPacket, Dict[str, TimingResult])
        The packet and per-operation timings.
    """
    timings: Dict[str, TimingResult] = {}

    # 1. Reference lookup
    (ref_indices, ref_acts), timings["reference_lookup"] = gpu_timed(
        history_buf.find_best_references, new_acts,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 2. Affine parameter computation
    (scale, bias), timings["affine_compute"] = gpu_timed(
        compute_affine_params, new_acts, ref_acts,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 3. Delta computation
    def _delta_fn():
        ref_t = apply_affine(ref_acts, scale, bias)
        return compute_delta(new_acts, ref_t)

    delta, timings["delta_compute"] = gpu_timed(
        _delta_fn,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 4. Quantize + top-k
    quant_result, timings["quantize_topk"] = gpu_timed(
        groupwise_int4_quantize_topk, delta, group_size, top_k,
        num_warmup=num_warmup, num_runs=num_runs,
    )
    packed, scales, zeros, topk_vals, topk_idx = quant_result

    # 5. Pack into DeltaPacket (trivial — struct creation)
    def _pack_fn():
        return DeltaPacket(
            quantized_data=packed,
            scales=scales,
            zero_points=zeros,
            topk_values=topk_vals,
            topk_indices=topk_idx,
            affine_scale=scale.to(torch.float16),
            affine_bias=bias.to(torch.float16),
            ref_indices=ref_indices,
            group_size=group_size,
            top_k=top_k,
        )

    packet, timings["pack"] = gpu_timed(
        _pack_fn,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 6. Total encode (end-to-end)
    def _full_encode():
        ri, ra = history_buf.find_best_references(new_acts)
        s, b = compute_affine_params(new_acts, ra)
        rt = apply_affine(ra, s, b)
        d = compute_delta(new_acts, rt)
        pk, sc, zp, tv, ti = groupwise_int4_quantize_topk(d, group_size, top_k)
        return DeltaPacket(
            quantized_data=pk, scales=sc, zero_points=zp,
            topk_values=tv, topk_indices=ti,
            affine_scale=s.to(torch.float16), affine_bias=b.to(torch.float16),
            ref_indices=ri, group_size=group_size, top_k=top_k,
        )

    _, timings["total_encode"] = gpu_timed(
        _full_encode,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    return packet, timings


def run_timed_decode_pipeline(
    packet: DeltaPacket,
    history_buf: PerRequestHistoryBuffer,
    hidden_dim: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Tuple[torch.Tensor, Dict[str, TimingResult]]:
    """Run the full decode pipeline with per-operation GPU timing.

    Parameters
    ----------
    packet : DeltaPacket
    history_buf : PerRequestHistoryBuffer
    hidden_dim : int
    num_warmup : int
    num_runs : int

    Returns
    -------
    (reconstructed, Dict[str, TimingResult])
    """
    timings: Dict[str, TimingResult] = {}
    batch = packet.quantized_data.shape[0]

    # Look up reference activations from buffer using packet indices
    ref_acts = history_buf.buffer[:batch].gather(
        1,
        packet.ref_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_dim),
    ).squeeze(1)  # (batch, hidden_dim)

    # 1. Dequantize + top-k overlay
    dequant, timings["dequantize_topk"] = gpu_timed(
        groupwise_int4_dequantize_topk,
        packet.quantized_data, packet.scales, packet.zero_points,
        packet.topk_values, packet.topk_indices,
        packet.group_size, hidden_dim,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 2. Reconstruction
    scale_f = packet.affine_scale.float()
    bias_f = packet.affine_bias.float()

    reconstructed, timings["reconstruction"] = gpu_timed(
        reconstruct_activation, dequant, ref_acts, scale_f, bias_f,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    # 3. Total decode (end-to-end)
    def _full_decode():
        dq = groupwise_int4_dequantize_topk(
            packet.quantized_data, packet.scales, packet.zero_points,
            packet.topk_values, packet.topk_indices,
            packet.group_size, hidden_dim,
        )
        ra = history_buf.buffer[:batch].gather(
            1,
            packet.ref_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_dim),
        ).squeeze(1)
        return reconstruct_activation(dq, ra, scale_f, bias_f)

    _, timings["total_decode"] = gpu_timed(
        _full_decode,
        num_warmup=num_warmup, num_runs=num_runs,
    )

    return reconstructed, timings


def run_baseline_memcpy(
    activations: torch.Tensor,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> TimingResult:
    """Baseline: raw GPU memcpy of full fp16 activations.

    Parameters
    ----------
    activations : torch.Tensor
        Shape ``(batch, hidden_dim)``.

    Returns
    -------
    TimingResult for the memcpy operation.
    """
    def _memcpy():
        dst = torch.empty_like(activations)
        dst.copy_(activations)
        return dst

    _, timing = gpu_timed(_memcpy, num_warmup=num_warmup, num_runs=num_runs)
    return timing


def run_timed_encode_pipeline_with_stats(
    new_acts: torch.Tensor,
    history_buf: PerRequestHistoryBuffer,
    group_size: int,
    top_k: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Tuple[DeltaPacket, Dict[str, TimingResult], Optional[LookupStats]]:
    """Run encode pipeline and return lookup stats as third element.

    Thin wrapper around ``run_timed_encode_pipeline`` that also retrieves
    ``history_buf.get_lookup_stats()`` after the pipeline completes.
    """
    packet, timings = run_timed_encode_pipeline(
        new_acts, history_buf, group_size, top_k,
        num_warmup=num_warmup, num_runs=num_runs,
    )
    lookup_stats = history_buf.get_lookup_stats()
    return packet, timings, lookup_stats
