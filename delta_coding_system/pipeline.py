"""OverlappedPipeline: Core delta-coding system with CPU/GPU overlap.

Prefill pipeline:
  1. classify (CPU, async) + prefill forward (GPU, concurrent)
  2. Collect classify → batch encode per tier (GPU)
  3. "Send" compressed data
  4. Table update (CPU, async — not on critical path)

Decode pipeline (per token):
  1. classify step N (CPU, async) + model forward step N (GPU, concurrent)
  2. Collect classify → encode (GPU)
  3. "Send"
  4. Table update (CPU, async)
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from delta_coding_system.table import DomainTableManager, NgramTable
from delta_coding_system.topic_router import DEFAULT_TOPIC_KEYWORDS, LightweightTopicRouter, RoutingDecision
from delta_coding_system.codec import (
    DeltaPacket,
    compute_affine_params,
    apply_affine,
    compute_delta,
    compute_transfer_size,
    compute_transfer_size_int8_outlier,
    entropy_coded_num_bytes,
    encode_decode_single,
    groupwise_int2_dequantize_topk,
    groupwise_int2_quantize_topk,
    groupwise_int4_dequantize_topk,
    groupwise_int4_quantize_topk,
    groupwise_int8_dequantize_topk,
    groupwise_int8_quantize_topk,
    reconstruct_activation,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Result dataclasses
# ===================================================================
@dataclass
class PrefillResult:
    """Results from prefill phase processing."""
    seq_len: int
    # Tier counts
    num_trigram: int = 0
    num_bigram: int = 0
    num_self_ref: int = 0
    num_unigram: int = 0
    # Quality
    raw_cosine_mean: float = 0.0
    raw_cosine_min: float = 0.0
    recon_cosine_mean: float = 0.0
    recon_cosine_min: float = 0.0
    mse_mean: float = 0.0
    mse_max: float = 0.0
    # Compression
    total_transfer_bytes: int = 0
    raw_fp16_bytes: int = 0
    compression_ratio: float = 1.0
    transfer_bytes_by_tier: Dict[str, int] = field(default_factory=dict)
    # Per-tier quality
    tier_detail: List[Dict[str, Any]] = field(default_factory=list)
    # Timing
    prefill_fwd_ms: float = 0.0
    classify_ms: float = 0.0
    encode_delta_ms: float = 0.0
    encode_self_ref_ms: float = 0.0
    encode_unigram_ms: float = 0.0
    table_update_ms: float = 0.0
    total_ms: float = 0.0
    reconstructed_hidden: Optional[torch.Tensor] = None


@dataclass
class DecodeStepRecord:
    """Per-step decode metrics."""
    step: int
    tier: str
    raw_cosine: float
    recon_cosine: float
    transfer_bytes: int
    raw_fp16_bytes: int
    fwd_ms: float
    classify_ms: float
    encode_ms: float
    table_update_ms: float


@dataclass
class DecodeResult:
    """Aggregate decode phase results."""
    decode_tokens: int
    step_records: List[DecodeStepRecord] = field(default_factory=list)
    # Aggregate tier counts
    num_trigram: int = 0
    num_bigram: int = 0
    num_self_ref: int = 0
    num_unigram: int = 0
    # Aggregate quality
    recon_cosine_mean: float = 0.0
    recon_cosine_min: float = 0.0
    # Aggregate compression
    total_transfer_bytes: int = 0
    raw_fp16_bytes: int = 0
    compression_ratio: float = 1.0
    # Timing
    total_fwd_ms: float = 0.0
    total_classify_ms: float = 0.0
    total_encode_ms: float = 0.0
    total_table_update_ms: float = 0.0
    total_ms: float = 0.0
    generated_token_ids: List[int] = field(default_factory=list)
    reconstructed_hidden: Optional[torch.Tensor] = None


# ===================================================================
# OverlappedPipeline
# ===================================================================
class OverlappedPipeline:
    """Production delta-coding system with overlapped CPU/GPU pipeline."""

    def __init__(
        self,
        model,
        tokenizer,
        layer_boundary: int = 6,
        table_dtype: torch.dtype = torch.float16,
        max_table_entries: int = 0,
        group_size: int = 128,
        top_k: int = 1,
        int8_group_size: int = 128,
        int8_outlier_top_k: int = 1,
        decode_tokens: int = 128,
        max_seq_len: int = 4096,
        device: torch.device = None,
        domain_aware: bool = False,
        max_gpu_tables: int = 3,
        max_active_tables_per_request: int = 4,
        table_placement: str = "cpu",
        pin_cpu_output_copy: bool = True,
        enable_async_cpu_output_copy: bool = True,
        gpu_hot_cache_entries: int = 4096,
        auto_topic_routing: bool = True,
        topic_keywords: Optional[Dict[str, Sequence[str]]] = None,
        delta_strategy: str = "delta_noaffine_int4_k1",
        unigram_strategy: str = "unigram_int4_k4",
        track_transfer_bytes: bool = True,
        extra_stop_token_ids: Optional[List[int]] = None,
        decode_use_raw_fp16: bool = True,
        prefill_use_raw_fp16: bool = False,
        low_latency_multidomain_classify: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_boundary = layer_boundary
        self.group_size = group_size
        self.top_k = top_k
        self.int8_group_size = int8_group_size
        self.int8_outlier_top_k = int8_outlier_top_k
        self.decode_tokens = decode_tokens
        self.max_seq_len = max_seq_len
        self.hidden_dim = model.config.hidden_size
        self.delta_strategy = delta_strategy
        self.unigram_strategy = unigram_strategy
        self.track_transfer_bytes = track_transfer_bytes
        self.decode_use_raw_fp16 = decode_use_raw_fp16
        self.prefill_use_raw_fp16 = prefill_use_raw_fp16
        self.low_latency_multidomain_classify = low_latency_multidomain_classify
        self.extra_stop_token_ids: Set[int] = set(extra_stop_token_ids or [])
        self.max_active_tables_per_request = max(1, max_active_tables_per_request)
        self.table_placement = table_placement
        self.pin_cpu_output_copy = pin_cpu_output_copy
        self.enable_async_cpu_output_copy = enable_async_cpu_output_copy
        self.gpu_hot_cache_entries = max(0, gpu_hot_cache_entries)
        self.auto_topic_routing = auto_topic_routing
        self.topic_keywords = {
            domain: tuple(keywords)
            for domain, keywords in (topic_keywords or DEFAULT_TOPIC_KEYWORDS).items()
        }
        self.topic_router = LightweightTopicRouter(
            topic_keywords=self.topic_keywords,
            max_active_domains=max_active_tables_per_request,
            max_write_domains=max(1, min(3, max_active_tables_per_request)),
        )

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # Domain-aware mode: multiple tables with GPU/CPU tiering
        self.domain_aware = domain_aware
        self._table_dtype = table_dtype
        self._max_table_entries = max_table_entries
        if domain_aware:
            self.table_manager = DomainTableManager(
                gpu_device=device,
                table_dtype=table_dtype,
                max_entries_per_table=max_table_entries,
                max_gpu_tables=max_gpu_tables,
                table_storage_format="int8",
                int8_group_size=int8_group_size,
                int8_top_k=int8_outlier_top_k,
                table_placement=table_placement,
                pin_cpu_output_copy=pin_cpu_output_copy,
                enable_async_cpu_output_copy=enable_async_cpu_output_copy,
                gpu_hot_cache_entries=gpu_hot_cache_entries,
            )
            self.table = None  # set per-request via select_domains()
            self._current_domains: List[str] = []
            self._current_domain_weights: Dict[str, float] = {}
            self._current_write_domains: List[str] = []
            self._active_table_cache: List[NgramTable] = []
            self._write_table_cache: List[NgramTable] = []
            self._manual_domains: List[str] = []
            self._last_routing_info: Dict[str, Any] = {}
            self._last_routing_decision: Optional[RoutingDecision] = None
            self._last_request_domain_hits: Dict[str, int] = {}
        else:
            self.table_manager = None
            self._current_domains = []
            self._current_domain_weights = {}
            self._current_write_domains = []
            self._active_table_cache = []
            self._write_table_cache = []
            self._manual_domains = []
            self._last_routing_info = {}
            self._last_routing_decision = None
            self._last_request_domain_hits = {}
            self.table = NgramTable(
                device=torch.device("cpu") if table_placement == "cpu" else device,
                dtype=table_dtype,
                max_entries=max_table_entries,
                storage_format="int8",
                int8_group_size=int8_group_size,
                int8_top_k=int8_outlier_top_k,
                pin_cpu_output_copy=pin_cpu_output_copy,
                enable_async_cpu_output_copy=enable_async_cpu_output_copy,
                gpu_hot_cache_entries=gpu_hot_cache_entries if table_placement == "cpu" else 0,
                gpu_hot_cache_device=device if table_placement == "cpu" and gpu_hot_cache_entries > 0 else None,
            )
        self.classify_executor = ThreadPoolExecutor(max_workers=1)
        self.update_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_prefill_update: Optional[Future] = None
        self._pending_decode_updates: List[Future] = []
        self._transfer_kernels_warmed = False

        # Import extraction helpers
        from activation_science.core.extraction import select_next_token
        self._select_next_token = select_next_token

        self._warmup_transfer_kernels()

    def _cache_seq_len(self, cache: DynamicCache) -> int:
        for key_state in cache.key_cache:
            if isinstance(key_state, torch.Tensor) and key_state.ndim >= 2 and key_state.numel() > 0:
                return int(key_state.shape[-2])
        return 0

    def _run_model_segment(
        self,
        hidden_states: torch.Tensor,
        cache: DynamicCache,
        start_layer: int,
        end_layer: int,
    ) -> torch.Tensor:
        batch_size, step_len = hidden_states.shape[:2]
        past_len = self._cache_seq_len(cache)
        total_len = past_len + step_len
        attention_mask = torch.ones((batch_size, total_len), dtype=torch.long, device=hidden_states.device)
        cache_position = torch.arange(past_len, total_len, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask = self.model.model._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            cache,
            False,
        )
        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
        hidden = hidden_states
        for layer in self.model.model.layers[start_layer:end_layer]:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        return hidden

    def _warmup_transfer_kernels(self) -> None:
        if self._transfer_kernels_warmed or self.device.type != "cuda":
            return
        warm_batch = min(8, max(1, self.group_size))
        real = torch.randn(warm_batch, self.hidden_dim, device=self.device, dtype=torch.float16)
        ref = torch.randn(warm_batch, self.hidden_dim, device=self.device, dtype=torch.float16)
        with torch.inference_mode():
            self._encode_delta_batch(real, ref, include_ref_idx=True)
            self._encode_unigram_batch(real)
            self._encode_prev_unigram_batch(real[:1], ref[:1])
        torch.cuda.synchronize(self.device)
        self._transfer_kernels_warmed = True

    def _run_prefix_prefill(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, DynamicCache]:
        hidden = self.model.model.embed_tokens(input_tensor)
        prefix_cache = DynamicCache()
        boundary_hidden = self._run_model_segment(hidden, prefix_cache, 0, self.layer_boundary)
        return boundary_hidden.squeeze(0).to(torch.float16), prefix_cache

    def _run_suffix_prefill(self, boundary_hidden: torch.Tensor) -> Tuple[torch.Tensor, DynamicCache]:
        suffix_cache = DynamicCache()
        suffix_hidden = self._run_model_segment(
            boundary_hidden.unsqueeze(0),
            suffix_cache,
            self.layer_boundary,
            len(self.model.model.layers),
        )
        suffix_hidden = self.model.model.norm(suffix_hidden)
        logits = self.model.lm_head(suffix_hidden)
        return logits[:, -1, :], suffix_cache

    def _run_prefix_decode_step(self, next_tok: torch.Tensor, prefix_cache: DynamicCache) -> torch.Tensor:
        hidden = self.model.model.embed_tokens(next_tok)
        boundary_hidden = self._run_model_segment(hidden, prefix_cache, 0, self.layer_boundary)
        return boundary_hidden[0, 0, :].to(torch.float16)

    def _run_suffix_decode_step(self, boundary_hidden: torch.Tensor, suffix_cache: DynamicCache) -> torch.Tensor:
        suffix_hidden = self._run_model_segment(
            boundary_hidden.unsqueeze(0).unsqueeze(0),
            suffix_cache,
            self.layer_boundary,
            len(self.model.model.layers),
        )
        suffix_hidden = self.model.model.norm(suffix_hidden)
        logits = self.model.lm_head(suffix_hidden)
        return logits[:, -1, :]

    def _activate_request_tables(
        self,
        text: str,
        task_name: Optional[str] = None,
        request_domains: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if not self.domain_aware:
            return []

        manual_domains = list(self._manual_domains)
        if request_domains:
            manual_domains.extend(request_domains)
        decision = self.topic_router.route(text, task_name=task_name, manual_domains=manual_domains)
        tables = self.table_manager.get_many(decision.active_domains)
        self._current_domains = list(tables.keys())
        self._current_domain_weights = {
            domain: decision.domain_weights.get(domain, 1.0)
            for domain in self._current_domains
        }
        self._current_write_domains = [domain for domain in decision.write_domains if domain in tables]
        self._active_table_cache = [tables[domain] for domain in self._current_domains]
        self._write_table_cache = [tables[domain] for domain in self._current_write_domains]
        self.table = tables[self._current_domains[0]] if self._current_domains else None
        self._last_routing_decision = decision
        self._last_routing_info = {
            "manual_domains": manual_domains,
            "resolved_domains": list(self._current_domains),
            "write_domains": list(self._current_write_domains),
            "domain_weights": dict(self._current_domain_weights),
            "task_name": task_name,
            "topic_scores": decision.topic_scores,
            "merge_domains": decision.merge_domains,
        }
        return self._current_domains

    def select_domains(self, domains: Sequence[str]):
        if not self.domain_aware:
            return [self.table]
        self._manual_domains = [re.sub(r"[^a-z0-9_:+-]+", "_", domain.strip().lower()).strip("_") or "misc" for domain in domains if domain]
        tables = self.table_manager.get_many(self._manual_domains)
        self._current_domains = list(tables.keys())
        self._current_domain_weights = {domain: 1.0 for domain in self._current_domains}
        self._current_write_domains = list(self._current_domains)
        self._active_table_cache = [tables[domain] for domain in self._current_domains]
        self._write_table_cache = [tables[domain] for domain in self._current_write_domains]
        self.table = tables[self._current_domains[0]] if self._current_domains else None
        return [tables[domain] for domain in self._current_domains]

    def select_domain(self, domain: str) -> NgramTable:
        selected = self.select_domains([domain])
        return selected[0] if selected else self.table

    def release_domains(self) -> None:
        if self.domain_aware and self._current_domains:
            self.table_manager.release_many(self._current_domains)
        self._current_domains = []
        self._current_domain_weights = {}
        self._current_write_domains = []
        self._active_table_cache = []
        self._write_table_cache = []
        self._manual_domains = []
        self._last_routing_decision = None
        self._last_request_domain_hits = {}

    def release_domain(self) -> None:
        self.release_domains()

    def _materialize_stored_hidden(self, stored_hidden: Any) -> torch.Tensor:
        if isinstance(stored_hidden, torch.Tensor):
            non_blocking = bool(
                stored_hidden.device.type == "cpu"
                and stored_hidden.is_pinned()
                and self.device.type == "cuda"
                and self.enable_async_cpu_output_copy
            )
            return stored_hidden.to(device=self.device, dtype=torch.float16, non_blocking=non_blocking)
        packet = stored_hidden
        non_blocking = bool(
            packet.quantized.device.type == "cpu"
            and packet.quantized.is_pinned()
            and self.device.type == "cuda"
            and self.enable_async_cpu_output_copy
        )
        packet = type(packet)(
            quantized=packet.quantized.to(self.device, non_blocking=non_blocking),
            scales=packet.scales.to(self.device, non_blocking=non_blocking),
            zero_points=packet.zero_points.to(self.device, non_blocking=non_blocking),
            topk_values=packet.topk_values.to(self.device, non_blocking=non_blocking),
            topk_indices=packet.topk_indices.to(self.device, non_blocking=non_blocking),
            group_size=packet.group_size,
            top_k=packet.top_k,
        )
        return groupwise_int8_dequantize_topk(packet)[0]

    def _active_tables(self) -> List[NgramTable]:
        if not self.domain_aware:
            return [self.table]
        if self._active_table_cache:
            return self._active_table_cache
        return [self.table_manager.get(domain) for domain in self._current_domains]

    def _write_tables(self) -> List[NgramTable]:
        if not self.domain_aware:
            return [self.table]
        if self._write_table_cache:
            return self._write_table_cache
        return [self.table_manager.get(domain) for domain in self._current_write_domains]

    def _update_active_tables_from_hidden_states(
        self,
        token_ids: List[int],
        hidden_states: torch.Tensor,
    ) -> Tuple[int, int, float]:
        total_new_trigrams = 0
        total_new_bigrams = 0
        t0 = time.perf_counter()
        for table in self._write_tables():
            new_tri, new_bi, _ = table.update_from_hidden_states(token_ids, hidden_states)
            total_new_trigrams += new_tri
            total_new_bigrams += new_bi
        return total_new_trigrams, total_new_bigrams, (time.perf_counter() - t0) * 1000.0

    def shutdown(self):
        """Shutdown the thread pool executors."""
        if self._pending_prefill_update is not None:
            self._pending_prefill_update.result()
            self._pending_prefill_update = None
        self._drain_decode_updates(wait=True)
        self.classify_executor.shutdown(wait=False)
        self.update_executor.shutdown(wait=False)
        if self.domain_aware and self.table_manager is not None:
            self.table_manager.offload_all()

    def _drain_decode_updates(self, wait: bool = False) -> None:
        if not self._pending_decode_updates:
            return
        remaining: List[Future] = []
        for future in self._pending_decode_updates:
            if wait:
                future.result()
                continue
            if future.done():
                future.result()
                continue
            remaining.append(future)
        self._pending_decode_updates = remaining

    def _submit_decode_table_update(
        self,
        running_token_ids: List[int],
        decode_pos: int,
        h: torch.Tensor,
        prefill_hidden: torch.Tensor,
        input_ids: List[int],
        decode_hidden_by_pos: Dict[int, torch.Tensor],
    ) -> None:
        if len(running_token_ids) < 3:
            return
        a = running_token_ids[-3]
        b = running_token_ids[-2]
        c = running_token_ids[-1]
        b_abs_pos = decode_pos - 1
        if b_abs_pos < len(input_ids):
            bi_hidden = prefill_hidden[b_abs_pos]
        else:
            bi_hidden = decode_hidden_by_pos.get(b_abs_pos, h)
        tables = tuple(self._write_tables() if self.domain_aware else [self.table])
        future = self.update_executor.submit(
            self._update_table_step,
            tables,
            a,
            b,
            c,
            bi_hidden,
            h,
        )
        self._pending_decode_updates.append(future)

    def _delta_uses_affine(self) -> bool:
        if self.delta_strategy in {"direct_int8", "direct_int4", "direct_int2"}:
            return False
        return "noaffine" not in self.delta_strategy

    def _delta_uses_entropy(self) -> bool:
        if self.delta_strategy in {"direct_int8", "direct_int4", "direct_int2"}:
            return False
        return "entropy" in self.delta_strategy

    def _delta_params(self) -> Tuple[int, int]:
        bits = 2 if "int2" in self.delta_strategy else 4
        match = re.search(r"_k(\d+)", self.delta_strategy)
        top_k = int(match.group(1)) if match is not None else self.top_k
        return bits, top_k

    def _unigram_uses_direct_int4(self) -> bool:
        return self.unigram_strategy == "unigram_int4_k4"

    def _unigram_uses_direct_int2(self) -> bool:
        return self.unigram_strategy == "unigram_int2_k4"

    def _unigram_uses_prev_ref(self) -> bool:
        return self.unigram_strategy in {"prev_int4_k2", "prev_gs256_k2"}

    def _unigram_prev_params(self) -> Tuple[int, int]:
        if self.unigram_strategy == "prev_gs256_k2":
            return 256, 2
        return self.group_size, 2

    def _delta_transfer_size(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
        batch_size: int,
        include_affine: bool,
        include_ref_idx: bool,
    ) -> int:
        if not self.track_transfer_bytes:
            return 0

        if self._delta_uses_entropy():
            entropy_bytes = entropy_coded_num_bytes([
                packed,
                scales,
                zeros,
                topk_values,
                topk_indices,
            ])
            if include_affine:
                entropy_bytes += batch_size * 2
                entropy_bytes += batch_size * 2
            if include_ref_idx:
                entropy_bytes += batch_size * 8
            return entropy_bytes

        total = 0
        total += packed.nelement() * packed.element_size()
        total += scales.nelement() * scales.element_size()
        total += zeros.nelement() * zeros.element_size()
        total += topk_values.nelement() * topk_values.element_size()
        total += topk_indices.nelement() * topk_indices.element_size()
        if include_affine:
            total += batch_size * 2
            total += batch_size * 2
        if include_ref_idx:
            total += batch_size * 8
        return total

    def _direct_int4_transfer_size(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> int:
        if not self.track_transfer_bytes:
            return 0
        return (
            packed.nelement() * packed.element_size()
            + scales.nelement() * scales.element_size()
            + zeros.nelement() * zeros.element_size()
            + topk_values.nelement() * topk_values.element_size()
            + topk_indices.nelement() * topk_indices.element_size()
        )

    def _direct_int2_transfer_size(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> int:
        if not self.track_transfer_bytes:
            return 0
        return (
            packed.nelement() * packed.element_size()
            + scales.nelement() * scales.element_size()
            + zeros.nelement() * zeros.element_size()
            + topk_values.nelement() * topk_values.element_size()
            + topk_indices.nelement() * topk_indices.element_size()
        )

    def _encode_direct_int2_batch(self, real_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(
            real_batch.clone(), self.group_size, 4,
        )
        recon = groupwise_int2_dequantize_topk(
            packed, scales, zeros, tv, ti, self.group_size, self.hidden_dim,
        )
        transfer = self._direct_int2_transfer_size(packed, scales, zeros, tv, ti)
        return recon, transfer

    def _encode_direct_int4_batch(self, real_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(
            real_batch.clone(), self.group_size, 4,
        )
        recon = groupwise_int4_dequantize_topk(
            packed, scales, zeros, tv, ti, self.group_size, self.hidden_dim,
        )
        transfer = self._direct_int4_transfer_size(packed, scales, zeros, tv, ti)
        return recon, transfer

    def _encode_direct_int8_batch(self, real_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        int8_pkt = groupwise_int8_quantize_topk(
            real_batch, self.int8_group_size, self.int8_outlier_top_k,
        )
        recon = groupwise_int8_dequantize_topk(int8_pkt)
        transfer = compute_transfer_size_int8_outlier(int8_pkt) if self.track_transfer_bytes else 0
        return recon, transfer

    def _build_local_prompt_refs(
        self,
        token_ids: List[int],
        hidden_states: torch.Tensor,
    ) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]:
        trigram_refs: Dict[Tuple[int, int, int], torch.Tensor] = {}
        bigram_refs: Dict[Tuple[int, int], torch.Tensor] = {}
        hidden_fp = hidden_states.to(torch.float16).detach()
        for i in range(len(token_ids)):
            if i >= 1:
                key_bi = (token_ids[i - 1], token_ids[i])
                bigram_refs.setdefault(key_bi, hidden_fp[i])
            if i >= 2:
                key_tri = (token_ids[i - 2], token_ids[i - 1], token_ids[i])
                trigram_refs.setdefault(key_tri, hidden_fp[i])
        return trigram_refs, bigram_refs

    def _encode_delta_batch(
        self,
        real_batch: torch.Tensor,
        ref_batch: torch.Tensor,
        include_ref_idx: bool,
    ) -> Tuple[torch.Tensor, int]:
        if self.delta_strategy == "direct_int2":
            return self._encode_direct_int2_batch(real_batch)
        if self.delta_strategy == "direct_int8":
            return self._encode_direct_int8_batch(real_batch)
        if self.delta_strategy == "direct_int4":
            return self._encode_direct_int4_batch(real_batch)
        use_affine = self._delta_uses_affine()
        if use_affine:
            scale, bias = compute_affine_params(real_batch, ref_batch)
            ref_t = apply_affine(ref_batch, scale, bias)
        else:
            scale = torch.ones(real_batch.shape[0], dtype=torch.float16, device=self.device)
            bias = torch.zeros(real_batch.shape[0], dtype=torch.float16, device=self.device)
            ref_t = ref_batch
        delta = compute_delta(real_batch, ref_t)
        quant_bits, top_k = self._delta_params()
        if quant_bits == 2:
            packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(
                delta, self.group_size, top_k,
            )
            dequant = groupwise_int2_dequantize_topk(
                packed, scales, zeros, tv, ti, self.group_size, self.hidden_dim,
            )
        else:
            packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(
                delta, self.group_size, top_k,
            )
            dequant = groupwise_int4_dequantize_topk(
                packed, scales, zeros, tv, ti, self.group_size, self.hidden_dim,
            )
        if use_affine:
            recon = reconstruct_activation(dequant, ref_batch, scale, bias).to(torch.float16)
        else:
            recon = (ref_batch + dequant).to(torch.float16)
        transfer = self._delta_transfer_size(
            packed, scales, zeros, tv, ti, real_batch.shape[0], use_affine, include_ref_idx,
        )
        return recon, transfer

    def _encode_unigram_batch(self, real_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self._unigram_uses_direct_int2():
            return self._encode_direct_int2_batch(real_batch)
        if self._unigram_uses_direct_int4():
            return self._encode_direct_int4_batch(real_batch)
        return self._encode_direct_int8_batch(real_batch)

    def _encode_prev_unigram_batch(
        self,
        real_batch: torch.Tensor,
        ref_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        group_size, top_k = self._unigram_prev_params()
        scale, bias = compute_affine_params(real_batch, ref_batch)
        ref_t = apply_affine(ref_batch, scale, bias)
        delta = compute_delta(real_batch, ref_t)
        packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(
            delta, group_size, top_k,
        )
        dequant = groupwise_int4_dequantize_topk(
            packed, scales, zeros, tv, ti, group_size, self.hidden_dim,
        )
        recon = reconstruct_activation(dequant, ref_batch, scale, bias).to(torch.float16)
        transfer = self._delta_transfer_size(
            packed, scales, zeros, tv, ti, real_batch.shape[0], True, False,
        )
        return recon, transfer

    def _encode_decode_step(
        self,
        real_h: torch.Tensor,
        ref_h: Optional[torch.Tensor],
        tier: str,
    ) -> Tuple[torch.Tensor, int]:
        if self.decode_use_raw_fp16:
            recon = real_h.to(torch.float16).clone()
            transfer = int(real_h.numel() * 2) if self.track_transfer_bytes else 0
            return recon, transfer
        if tier in ("trigram", "bigram", "self_ref") and ref_h is not None:
            recon, transfer = self._encode_delta_batch(real_h, ref_h, include_ref_idx=True)
            return recon, transfer
        recon, transfer = self._encode_unigram_batch(real_h)
        return recon, transfer

    # ------------------------------------------------------------------
    # Prefill phase
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_prefill(
        self,
        text: str,
        phase: str = "test",
        task_name: Optional[str] = None,
        request_domains: Optional[Sequence[str]] = None,
    ) -> Tuple[
        PrefillResult,
        DynamicCache,
        DynamicCache,
        torch.Tensor,
        List[int],
        torch.Tensor,
        Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor]],
    ]:
        """Process prefill phase with overlapped classify + forward.

        Returns
        -------
        result : PrefillResult
        prefix_cache : prefix-half KV cache for decode
        suffix_cache : suffix-half KV cache for decode
        next_tok : next token tensor for decode
        input_ids : token id list
        prefill_hidden : (seq_len, hidden_dim) hidden states
        """
        t_total_start = time.perf_counter()
        is_test = (phase == "test")

        # Set CUDA device so synchronize()/Event.record() target the correct GPU
        torch.cuda.set_device(self.device)

        # Tokenize
        if self._pending_prefill_update is not None:
            self._pending_prefill_update.result()
            self._pending_prefill_update = None
        self._drain_decode_updates(wait=False)

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        seq_len = len(input_ids)

        if self.domain_aware:
            self._activate_request_tables(text, task_name=task_name, request_domains=request_domains)

        # 1. Launch classify async (CPU) — only needs token_ids
        classify_future = None
        if is_test:
            if self.domain_aware:
                classify_future = self.classify_executor.submit(
                    self.table_manager.classify_and_build_refs_from_domains,
                    list(self._current_domains),
                    input_ids,
                    self.hidden_dim,
                    dict(self._current_domain_weights),
                    self.low_latency_multidomain_classify,
                )
            else:
                classify_future = self.classify_executor.submit(
                    self.table.classify_and_build_refs,
                    input_ids,
                    self.hidden_dim,
                    self.device,
                )

        # 2. Prefill forward (GPU, concurrent with classify)
        torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        prefill_hidden, prefix_cache = self._run_prefix_prefill(input_tensor)
        torch.cuda.synchronize()
        prefill_fwd_ms = (time.perf_counter() - t_fwd_start) * 1000.0

        result = PrefillResult(seq_len=seq_len, prefill_fwd_ms=prefill_fwd_ms)

        if not is_test:
            # Warmup: just update table, no encoding
            next_logits, suffix_cache = self._run_suffix_prefill(prefill_hidden)
            next_tok = self._select_next_token(next_logits, do_sample=False)
            t_upd_start = time.perf_counter()
            if self.domain_aware:
                self._update_active_tables_from_hidden_states(input_ids, prefill_hidden)
            else:
                self.table.update_from_hidden_states(input_ids, prefill_hidden)
            result.table_update_ms = (time.perf_counter() - t_upd_start) * 1000.0
            result.total_ms = (time.perf_counter() - t_total_start) * 1000.0
            result.reconstructed_hidden = prefill_hidden
            local_prompt_refs = self._build_local_prompt_refs(input_ids, prefill_hidden)
            return result, prefix_cache, suffix_cache, next_tok, input_ids, prefill_hidden, local_prompt_refs

        # 3. Collect classify result
        t_classify_start = time.perf_counter()
        if self.domain_aware:
            tiers, ref_acts, self_ref_sources, first_occ_map, ref_stats = classify_future.result()
            self._last_request_domain_hits = dict(ref_stats.domain_hits)
        else:
            tiers, ref_acts, self_ref_sources, first_occ_map = classify_future.result()
            self._last_request_domain_hits = {}
        classify_ms = (time.perf_counter() - t_classify_start) * 1000.0
        result.classify_ms = classify_ms

        # Separate indices by tier
        trigram_indices = [i for i, t in enumerate(tiers) if t == "trigram"]
        bigram_indices = [i for i, t in enumerate(tiers) if t == "bigram"]
        self_ref_indices = [i for i, t in enumerate(tiers) if t == "self_ref"]
        unigram_indices = [i for i, t in enumerate(tiers) if t == "unigram"]

        result.num_trigram = len(trigram_indices)
        result.num_bigram = len(bigram_indices)
        result.num_self_ref = len(self_ref_indices)
        result.num_unigram = len(unigram_indices)

        real_acts = prefill_hidden
        reconstructed = torch.zeros_like(real_acts)
        transfer_bytes_by_tier: Dict[str, int] = {
            "trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0,
        }

        if self.prefill_use_raw_fp16:
            reconstructed = real_acts.to(torch.float16).clone()
            bytes_per_pos = self.hidden_dim * 2 if self.track_transfer_bytes else 0
            transfer_bytes_by_tier["trigram"] = len(trigram_indices) * bytes_per_pos
            transfer_bytes_by_tier["bigram"] = len(bigram_indices) * bytes_per_pos
            transfer_bytes_by_tier["self_ref"] = len(self_ref_indices) * bytes_per_pos
            transfer_bytes_by_tier["unigram"] = len(unigram_indices) * bytes_per_pos

            if self_ref_indices:
                sorted_self_ref = sorted(self_ref_indices)
                idx_sr = torch.tensor(sorted_self_ref, dtype=torch.long, device=self.device)
                source_positions = [self_ref_sources[pos] for pos in sorted_self_ref]
                src_t = torch.tensor(source_positions, dtype=torch.long, device=self.device)
                ref_sr = reconstructed[src_t]
                ref_acts[idx_sr] = ref_sr

            result.encode_delta_ms = 0.0
            result.encode_unigram_ms = 0.0
            result.encode_self_ref_ms = 0.0
        else:

            # 4. Encode all tiers — use CUDA events for timing (no intermediate syncs)
            # Sync to drain GPU ops queued by classify thread (FP8→FP16 conversions)
            # so that encode events measure only encode work
            torch.cuda.synchronize()
            evt_enc_start = torch.cuda.Event(enable_timing=True)
            evt_after_delta = torch.cuda.Event(enable_timing=True)
            evt_after_unigram = torch.cuda.Event(enable_timing=True)
            evt_after_self_ref = torch.cuda.Event(enable_timing=True)

            evt_enc_start.record()

            # 4a. Encode TRIGRAM + BIGRAM (batch)
            delta_indices = trigram_indices + bigram_indices
            if delta_indices:
                idx_t = torch.tensor(delta_indices, dtype=torch.long, device=self.device)
                real_batch = real_acts[idx_t]
                ref_batch = ref_acts[idx_t]

                recon_batch, total_delta_bytes = self._encode_delta_batch(
                    real_batch, ref_batch, include_ref_idx=True,
                )
                reconstructed[idx_t] = recon_batch

                n_delta = len(delta_indices)
                if n_delta > 0:
                    per_pos = total_delta_bytes / n_delta
                    transfer_bytes_by_tier["trigram"] = int(per_pos * len(trigram_indices))
                    transfer_bytes_by_tier["bigram"] = int(per_pos * len(bigram_indices))

            evt_after_delta.record()

            # 4b. Encode UNIGRAM (Int8 + outliers)
            if unigram_indices:
                if self._unigram_uses_prev_ref():
                    unigram_transfer = 0
                    for pos in sorted(unigram_indices):
                        real_uni = real_acts[pos].unsqueeze(0)
                        if pos > 0:
                            prev_ref = reconstructed[pos - 1].unsqueeze(0)
                            if torch.count_nonzero(prev_ref).item() == 0:
                                prev_ref = real_acts[pos - 1].unsqueeze(0)
                            recon_uni, xfer = self._encode_prev_unigram_batch(real_uni, prev_ref)
                        else:
                            recon_uni, xfer = self._encode_unigram_batch(real_uni)
                        reconstructed[pos] = recon_uni.squeeze(0)
                        unigram_transfer += xfer
                    transfer_bytes_by_tier["unigram"] = unigram_transfer
                else:
                    idx_u = torch.tensor(unigram_indices, dtype=torch.long, device=self.device)
                    real_uni = real_acts[idx_u]
                    recon_uni, transfer_bytes_by_tier["unigram"] = self._encode_unigram_batch(real_uni)
                    reconstructed[idx_u] = recon_uni

            evt_after_unigram.record()

            # 4c. Encode SELF_REF (uses already-reconstructed positions as references)
            if self_ref_indices:
                sorted_self_ref = sorted(self_ref_indices)
                idx_sr = torch.tensor(sorted_self_ref, dtype=torch.long, device=self.device)
                real_sr = real_acts[idx_sr]

                source_positions = [self_ref_sources[pos] for pos in sorted_self_ref]
                src_t = torch.tensor(source_positions, dtype=torch.long, device=self.device)
                ref_sr = reconstructed[src_t]
                ref_acts[idx_sr] = ref_sr

                recon_sr, transfer_bytes_by_tier["self_ref"] = self._encode_delta_batch(
                    real_sr, ref_sr, include_ref_idx=True,
                )
                reconstructed[idx_sr] = recon_sr

            evt_after_self_ref.record()

            # Single sync — needed before quality metrics that read tensor values
            torch.cuda.synchronize()
            result.encode_delta_ms = evt_enc_start.elapsed_time(evt_after_delta)
            result.encode_unigram_ms = evt_after_delta.elapsed_time(evt_after_unigram)
            result.encode_self_ref_ms = evt_after_unigram.elapsed_time(evt_after_self_ref)

        # "Send" — record transfer bytes
        result.transfer_bytes_by_tier = transfer_bytes_by_tier
        result.total_transfer_bytes = sum(transfer_bytes_by_tier.values())
        result.raw_fp16_bytes = seq_len * self.hidden_dim * 2
        result.compression_ratio = result.raw_fp16_bytes / max(result.total_transfer_bytes, 1)

        # Quality metrics
        overall_cos = F.cosine_similarity(real_acts.float(), reconstructed.float(), dim=-1)
        overall_mse = ((real_acts.float() - reconstructed.float()) ** 2).mean(dim=-1)
        result.recon_cosine_mean = overall_cos.mean().item()
        result.recon_cosine_min = overall_cos.min().item()
        result.mse_mean = overall_mse.mean().item()
        result.mse_max = overall_mse.max().item()

        # Raw cosine for non-unigram
        non_uni_indices = trigram_indices + bigram_indices + sorted(self_ref_indices)
        if non_uni_indices:
            idx_nu = torch.tensor(non_uni_indices, dtype=torch.long, device=self.device)
            raw_cos = F.cosine_similarity(
                real_acts[idx_nu].float(), ref_acts[idx_nu].float(), dim=-1,
            )
            result.raw_cosine_mean = raw_cos.mean().item()
            result.raw_cosine_min = raw_cos.min().item()

        # Per-tier detail
        for tier_name, tier_indices in [
            ("trigram", trigram_indices), ("bigram", bigram_indices),
            ("self_ref", self_ref_indices), ("unigram", unigram_indices),
        ]:
            if not tier_indices:
                result.tier_detail.append({
                    "tier": tier_name, "count": 0,
                    "raw_cosine_mean": 0.0, "recon_cosine_mean": 0.0,
                    "recon_cosine_min": 0.0, "mse_mean": 0.0, "mse_max": 0.0,
                    "transfer_bytes": 0,
                })
                continue
            idx_t = torch.tensor(tier_indices, dtype=torch.long, device=self.device)
            real_tier = real_acts[idx_t]
            ref_tier = ref_acts[idx_t]
            recon_tier = reconstructed[idx_t]
            if tier_name == "unigram":
                raw_cos_t = torch.zeros(len(tier_indices), device=self.device)
            else:
                raw_cos_t = F.cosine_similarity(real_tier.float(), ref_tier.float(), dim=-1)
            recon_cos_t = F.cosine_similarity(real_tier.float(), recon_tier.float(), dim=-1)
            mse_t = ((real_tier.float() - recon_tier.float()) ** 2).mean(dim=-1)
            result.tier_detail.append({
                "tier": tier_name, "count": len(tier_indices),
                "raw_cosine_mean": raw_cos_t.mean().item(),
                "recon_cosine_mean": recon_cos_t.mean().item(),
                "recon_cosine_min": recon_cos_t.min().item(),
                "mse_mean": mse_t.mean().item(),
                "mse_max": mse_t.max().item(),
                "transfer_bytes": transfer_bytes_by_tier[tier_name],
            })

        # 5. Prefill table update: launch asynchronously and let decode hide it.
        local_prompt_refs = self._build_local_prompt_refs(input_ids, prefill_hidden)
        t_upd_start = time.perf_counter()
        if self.domain_aware:
            self._pending_prefill_update = self.update_executor.submit(
                self._update_active_tables_from_hidden_states,
                input_ids,
                prefill_hidden,
            )
        else:
            self._pending_prefill_update = self.update_executor.submit(
                self.table.update_from_hidden_states,
                input_ids,
                prefill_hidden,
            )
        result.table_update_ms = (time.perf_counter() - t_upd_start) * 1000.0

        result.total_ms = (time.perf_counter() - t_total_start) * 1000.0
        result.reconstructed_hidden = reconstructed
        next_logits, suffix_cache = self._run_suffix_prefill(reconstructed)
        next_tok = self._select_next_token(next_logits, do_sample=False)
        return result, prefix_cache, suffix_cache, next_tok, input_ids, prefill_hidden, local_prompt_refs

    # ------------------------------------------------------------------
    # Decode phase
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_decode(
        self,
        prefix_cache: DynamicCache,
        suffix_cache: DynamicCache,
        next_tok: torch.Tensor,
        input_ids: List[int],
        prefill_hidden: torch.Tensor,
        prefill_reconstructed_hidden: torch.Tensor,
        local_prompt_refs: Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor]],
        phase: str = "test",
    ) -> DecodeResult:
        """Process decode phase with overlapped classify + forward per step.

        Parameters
        ----------
        prefix_cache : prefix-half KV cache from prefill
        suffix_cache : suffix-half KV cache from prefill
        next_tok : (1,1) next token from prefill
        input_ids : prefill token ids
        prefill_hidden : (seq_len, hidden_dim) prefill hidden states
        phase : "warmup" or "test"
        """
        t_total_start = time.perf_counter()
        is_test = (phase == "test")

        # Set CUDA device so synchronize()/Event.record() target the correct GPU
        torch.cuda.set_device(self.device)

        decode_result = DecodeResult(decode_tokens=0)

        running_token_ids: List[int] = list(input_ids)
        first_occ_map: Dict[Tuple[int, int, int], int] = {}
        reconstructed_hiddens: Dict[int, torch.Tensor] = {}
        decode_hidden_by_pos: Dict[int, torch.Tensor] = {}
        recon_sequence: List[torch.Tensor] = []
        generated_token_ids: List[int] = []

        tier_counts = {"trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0}
        all_cosines: List[float] = []
        total_transfer_bytes = 0
        total_raw_bytes = 0

        local_prompt_trigrams, local_prompt_bigrams = local_prompt_refs

        # Pre-allocate CUDA events for decode loop timing
        evt_fwd_start = torch.cuda.Event(enable_timing=True)
        evt_fwd_end = torch.cuda.Event(enable_timing=True)
        evt_enc_start = torch.cuda.Event(enable_timing=True)
        evt_enc_end = torch.cuda.Event(enable_timing=True)

        for step in range(self.decode_tokens):
            tok_id = next_tok.item()
            running_token_ids.append(tok_id)
            generated_token_ids.append(tok_id)
            decode_pos = len(running_token_ids) - 1

            # Launch classify (CPU, concurrent with forward)
            classify_future = self.classify_executor.submit(
                self._classify_decode_step,
                running_token_ids, decode_pos, first_occ_map, reconstructed_hiddens,
                local_prompt_trigrams, local_prompt_bigrams,
            )

            # Prefix-half decode (GPU) — CUDA events, no sync
            evt_fwd_start.record()
            h = self._run_prefix_decode_step(next_tok, prefix_cache)
            evt_fwd_end.record()
            decode_hidden_by_pos[decode_pos] = h

            # Collect classify result
            t_classify_start = time.perf_counter()
            tier, ref_h, raw_cos = classify_future.result()
            classify_ms = (time.perf_counter() - t_classify_start) * 1000.0

            # Encode (GPU, on critical path) — CUDA events, no sync
            real_h_2d = h.unsqueeze(0)
            if tier == "unigram" and self._unigram_uses_prev_ref():
                if decode_pos - 1 < len(input_ids):
                    prev_ref = prefill_hidden[decode_pos - 1].unsqueeze(0)
                elif (decode_pos - 1) in reconstructed_hiddens:
                    prev_ref = reconstructed_hiddens[decode_pos - 1].unsqueeze(0)
                else:
                    prev_ref = None
            else:
                prev_ref = None
            evt_enc_start.record()
            if tier == "unigram" and prev_ref is not None and self._unigram_uses_prev_ref():
                recon, xfer_bytes = self._encode_prev_unigram_batch(real_h_2d, prev_ref)
            else:
                recon, xfer_bytes = self._encode_decode_step(real_h_2d, ref_h, tier)
            evt_enc_end.record()

            # Sync once per step to read cosine similarity .item()
            torch.cuda.synchronize()
            fwd_ms = evt_fwd_start.elapsed_time(evt_fwd_end)
            encode_ms = evt_enc_start.elapsed_time(evt_enc_end)

            recon_cos = F.cosine_similarity(
                real_h_2d.float(), recon.float(), dim=-1,
            ).item()
            # Compute raw cosine (ref vs real) for non-unigram tiers
            if ref_h is not None:
                raw_cos = F.cosine_similarity(
                    real_h_2d.float(), ref_h.float(), dim=-1,
                ).item()
            raw_fp16_bytes = self.hidden_dim * 2

            # Store reconstructed for self-ref
            reconstructed_hiddens[decode_pos] = recon.squeeze(0)
            recon_sequence.append(recon.squeeze(0))

            # "Send" compressed data

            table_update_ms = 0.0
            self._drain_decode_updates(wait=False)

            # Table update (fully async, after send)
            self._submit_decode_table_update(
                running_token_ids,
                decode_pos,
                h,
                prefill_hidden,
                input_ids,
                decode_hidden_by_pos,
            )

            # Record metrics
            tier_counts[tier] += 1
            all_cosines.append(recon_cos)
            total_transfer_bytes += xfer_bytes
            total_raw_bytes += raw_fp16_bytes

            if is_test:
                decode_result.step_records.append(DecodeStepRecord(
                    step=step, tier=tier,
                    raw_cosine=raw_cos, recon_cosine=recon_cos,
                    transfer_bytes=xfer_bytes, raw_fp16_bytes=raw_fp16_bytes,
                    fwd_ms=fwd_ms, classify_ms=classify_ms,
                    encode_ms=encode_ms, table_update_ms=table_update_ms,
                ))

            suffix_logits = self._run_suffix_decode_step(recon.squeeze(0), suffix_cache)
            next_tok = self._select_next_token(suffix_logits, do_sample=False)

            if self.tokenizer.eos_token_id is not None and tok_id == self.tokenizer.eos_token_id:
                break
            if tok_id in self.extra_stop_token_ids:
                break

        self._drain_decode_updates(wait=False)

        del next_tok

        # Aggregate
        decode_result.num_trigram = tier_counts["trigram"]
        decode_result.num_bigram = tier_counts["bigram"]
        decode_result.num_self_ref = tier_counts["self_ref"]
        decode_result.num_unigram = tier_counts["unigram"]
        decode_result.decode_tokens = len(generated_token_ids)
        decode_result.generated_token_ids = generated_token_ids
        decode_result.total_transfer_bytes = total_transfer_bytes
        decode_result.raw_fp16_bytes = total_raw_bytes
        decode_result.compression_ratio = total_raw_bytes / max(total_transfer_bytes, 1)
        if all_cosines:
            decode_result.recon_cosine_mean = sum(all_cosines) / len(all_cosines)
            decode_result.recon_cosine_min = min(all_cosines)
        if recon_sequence:
            decode_result.reconstructed_hidden = torch.stack(recon_sequence, dim=0)

        # Timing aggregation from step records
        if decode_result.step_records:
            decode_result.total_fwd_ms = sum(s.fwd_ms for s in decode_result.step_records)
            decode_result.total_classify_ms = sum(s.classify_ms for s in decode_result.step_records)
            decode_result.total_encode_ms = sum(s.encode_ms for s in decode_result.step_records)
            decode_result.total_table_update_ms += sum(s.table_update_ms for s in decode_result.step_records)
        decode_result.total_ms = (time.perf_counter() - t_total_start) * 1000.0

        return decode_result

    # ------------------------------------------------------------------
    # Internal helpers for decode
    # ------------------------------------------------------------------
    def _classify_decode_step(
        self,
        running_token_ids: List[int],
        decode_pos: int,
        first_occ_map: Dict[Tuple[int, int, int], int],
        reconstructed_hiddens: Dict[int, torch.Tensor],
        local_prompt_trigrams: Dict[Tuple[int, int, int], torch.Tensor],
        local_prompt_bigrams: Dict[Tuple[int, int], torch.Tensor],
    ) -> Tuple[str, Optional[torch.Tensor], float]:
        """Classify a single decode position. Runs on CPU thread."""
        tier = "unigram"
        ref_h = None
        raw_cos = 0.0
        single_active_table = None
        if self.domain_aware and len(self._current_domains) == 1 and self.table is not None:
            single_active_table = self.table
            active_tables: List[NgramTable] = [single_active_table]
        else:
            active_tables = self._active_tables()

        if len(running_token_ids) >= 3:
            a = running_token_ids[-3]
            b = running_token_ids[-2]
            c = running_token_ids[-1]
            trigram_key = (a, b, c)

            # 1. Trigram table lookup
            tri_ref = None
            if single_active_table is not None:
                tri_ref = single_active_table.get_trigram(a, b, c)
            else:
                for table in active_tables:
                    tri_ref = table.get_trigram(a, b, c)
                    if tri_ref is not None:
                        break
            if tri_ref is not None:
                tier = "trigram"
                ref_h = self._materialize_stored_hidden(tri_ref).unsqueeze(0)
                first_occ_map.setdefault(trigram_key, decode_pos)
            elif trigram_key in local_prompt_trigrams:
                tier = "trigram"
                ref_h = local_prompt_trigrams[trigram_key].unsqueeze(0).to(torch.float16)
                first_occ_map.setdefault(trigram_key, decode_pos)
            # 2. Self-ref
            elif trigram_key in first_occ_map:
                src_pos = first_occ_map[trigram_key]
                if src_pos in reconstructed_hiddens:
                    tier = "self_ref"
                    ref_h = reconstructed_hiddens[src_pos].unsqueeze(0).to(torch.float16)
            # 3. Bigram
            if tier == "unigram":
                if len(running_token_ids) >= 2:
                    b_tok = running_token_ids[-2]
                    c_tok = running_token_ids[-1]
                    bi_ref = None
                    if single_active_table is not None:
                        bi_ref = single_active_table.get_bigram(b_tok, c_tok)
                    else:
                        for table in active_tables:
                            bi_ref = table.get_bigram(b_tok, c_tok)
                            if bi_ref is not None:
                                break
                    if bi_ref is not None:
                        tier = "bigram"
                        ref_h = self._materialize_stored_hidden(bi_ref).unsqueeze(0)
                    elif (b_tok, c_tok) in local_prompt_bigrams:
                        tier = "bigram"
                        ref_h = local_prompt_bigrams[(b_tok, c_tok)].unsqueeze(0).to(torch.float16)
                first_occ_map.setdefault(trigram_key, decode_pos)

        return tier, ref_h, raw_cos

    def _update_table_step(
        self,
        tables: Sequence[NgramTable],
        a: int,
        b: int,
        c: int,
        bi_hidden: torch.Tensor,
        trigram_hidden: torch.Tensor,
    ) -> None:
        """Update table with a single decode step's trigram. Runs on CPU thread."""
        for table in tables:
            if table.has_trigram(a, b, c):
                continue
            stored_bigram = table._encode_hidden_batch_for_storage(bi_hidden.unsqueeze(0))[0]
            stored_trigram = table._encode_hidden_batch_for_storage(trigram_hidden.unsqueeze(0))[0]
            node = table._get_or_create_node(a, b, stored_bigram)
            if c not in node.suffixes:
                node.suffixes[c] = stored_trigram
                table._num_trigrams += 1
            table._request_counter += 1
            table._last_evicted = table.evict()

    # ------------------------------------------------------------------
    # Full request processing
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_request(
        self,
        text: str,
        phase: str = "test",
        task_name: Optional[str] = None,
        request_domains: Optional[Sequence[str]] = None,
    ) -> Tuple[PrefillResult, DecodeResult, Dict[str, Any]]:
        """Process a complete request: prefill + decode.

        Returns (prefill_result, decode_result, table_stats).
        """
        prefill_outputs = self.process_prefill(
            text,
            phase=phase,
            task_name=task_name,
            request_domains=request_domains,
        )
        if len(prefill_outputs) == 6:
            prefill_result, prefix_cache, next_tok, input_ids, prefill_hidden, local_prompt_refs = prefill_outputs
            _suffix_logits, suffix_cache = self._run_suffix_prefill(prefill_hidden)
        else:
            prefill_result, prefix_cache, suffix_cache, next_tok, input_ids, prefill_hidden, local_prompt_refs = prefill_outputs

        decode_result = self.process_decode(
            prefix_cache,
            suffix_cache,
            next_tok,
            input_ids,
            prefill_hidden,
            prefill_result.reconstructed_hidden,
            local_prompt_refs,
            phase=phase,
        )

        table_stats = self.table.stats if self.table is not None else {}
        if self.table is not None:
            table_stats["last_evicted"] = self.table._last_evicted
        if self.domain_aware:
            table_stats["domains"] = list(self._current_domains)
            table_stats["routing"] = dict(self._last_routing_info)
            table_stats["domain_hits"] = dict(self._last_request_domain_hits)
            table_stats["manager"] = self.table_manager.stats

            if self._last_routing_decision is not None:
                self.topic_router.observe(
                    text,
                    self._last_routing_decision,
                    self._last_request_domain_hits,
                )

        return prefill_result, decode_result, table_stats
