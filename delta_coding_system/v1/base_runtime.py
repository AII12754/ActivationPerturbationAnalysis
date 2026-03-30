from __future__ import annotations

import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from transformers.cache_utils import DynamicCache

from delta_coding_system.codec import (
    DeltaPacket,
    apply_affine,
    compute_affine_params,
    compute_delta,
    compute_transfer_size_int8_outlier,
    entropy_coded_num_bytes,
    fused_int4_affine_delta_encode,
    fused_int4_delta_encode,
    fused_int4_quantize_dequantize,
    fused_int8_quantize_dequantize,
    groupwise_int2_dequantize_topk,
    groupwise_int2_quantize_topk,
    groupwise_int4_dequantize_topk,
    groupwise_int4_quantize_topk,
    groupwise_int8_dequantize_topk,
    groupwise_int8_quantize_topk,
    reconstruct_activation,
)
from delta_coding_system.fused_kernels import (
    fused_int4_qdq,
    fused_int4_delta_qdq,
    fused_int8_qdq,
)
from delta_coding_system.table import NgramTable, create_activation_table

logger = logging.getLogger(__name__)


class V1RuntimeBase:
    """Standalone runtime substrate for the v1 latency-first pipeline."""

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
        table_placement: str = "cpu",
        pin_cpu_output_copy: bool = True,
        enable_async_cpu_output_copy: bool = True,
        gpu_hot_cache_entries: int = 0,
        enable_disk_offload: bool = False,
        disk_offload_dir: Optional[str] = None,
        table_backend: str = "trie",
        block_size: int = 256,
        enable_async_block_paging: bool = False,
        max_resident_blocks: int = 0,
        block_pager_workers: int = 1,
        pinned_block_budget: int = 2,
        delta_strategy: str = "delta_noaffine_int4_k1",
        unigram_strategy: str = "unigram_int4_k4",
        track_transfer_bytes: bool = True,
        compute_cosine_similarity: bool = False,
        extra_stop_token_ids: Optional[List[int]] = None,
        decode_use_raw_fp16: bool = True,
        prefill_use_raw_fp16: bool = False,
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
        self.compute_cosine_similarity = compute_cosine_similarity
        self.decode_use_raw_fp16 = True  # Always raw FP16 for decode
        self.prefill_use_raw_fp16 = prefill_use_raw_fp16
        self.extra_stop_token_ids: Set[int] = set(extra_stop_token_ids or [])
        self.table_placement = table_placement
        self.pin_cpu_output_copy = pin_cpu_output_copy
        self.enable_async_cpu_output_copy = enable_async_cpu_output_copy
        self.gpu_hot_cache_entries = max(0, gpu_hot_cache_entries)
        self.enable_disk_offload = enable_disk_offload
        self.disk_offload_dir = disk_offload_dir
        self.table_backend = table_backend
        self.block_size = max(16, block_size)
        self.enable_async_block_paging = enable_async_block_paging
        self.max_resident_blocks = max(0, max_resident_blocks)
        self.block_pager_workers = max(1, block_pager_workers)
        self.pinned_block_budget = max(0, pinned_block_budget)

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        self._table_dtype = table_dtype
        self._max_table_entries = max_table_entries
        self.table = create_activation_table(
            backend=table_backend,
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
            block_size=block_size,
            enable_async_paging=enable_async_block_paging,
            page_directory=disk_offload_dir,
            max_resident_blocks=max_resident_blocks,
            pager_workers=block_pager_workers,
            pinned_block_budget=pinned_block_budget,
        )

        self.classify_executor = ThreadPoolExecutor(max_workers=1)
        self.update_executor = ThreadPoolExecutor(max_workers=1)
        self._pending_prefill_update: Optional[Future] = None
        self._pending_decode_updates: List[Future] = []
        self._decode_update_batch: List[Tuple] = []
        self._decode_update_flush_every: int = 16
        self._transfer_kernels_warmed = False

        # Dedicated CUDA stream for loading hidden states to HBM during
        # CPU-side classify (prefill path).  Allows CPU table lookup and
        # GPU hidden-state materialization to overlap.
        self.hidden_load_stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self.hidden_load_stream = torch.cuda.Stream(self.device)

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
        packet = DeltaPacket(
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
        return [self.table]

    def _write_tables(self) -> List[NgramTable]:
        return [self.table]

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
        if self._pending_prefill_update is not None:
            self._pending_prefill_update.result()
            self._pending_prefill_update = None
        self._flush_decode_update_batch()
        self._drain_decode_updates(wait=True)
        self.classify_executor.shutdown(wait=False)
        self.update_executor.shutdown(wait=False)
        if self.table is not None:
            shutdown = getattr(self.table, "shutdown", None)
            if callable(shutdown):
                shutdown()

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
        return fused_int4_qdq(real_batch, self.group_size, 4)

    def _encode_direct_int8_batch(self, real_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return fused_int8_qdq(
            real_batch, self.int8_group_size, self.int8_outlier_top_k,
        )

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
        quant_bits, top_k = self._delta_params()
        if quant_bits == 2:
            # Int2 path: keep original (no fused version yet)
            delta = compute_delta(real_batch, ref_batch)
            packed, scales, zeros, tv, ti = groupwise_int2_quantize_topk(
                delta, self.group_size, top_k,
            )
            dequant = groupwise_int2_dequantize_topk(
                packed, scales, zeros, tv, ti, self.group_size, self.hidden_dim,
            )
            recon = (ref_batch + dequant).to(torch.float16)
            transfer = self._direct_int2_transfer_size(packed, scales, zeros, tv, ti)
            return recon, transfer
        # Int4 path: fused
        if use_affine:
            return fused_int4_affine_delta_encode(
                real_batch, ref_batch, self.group_size, top_k,
            )
        return fused_int4_delta_qdq(
            real_batch, ref_batch, self.group_size, top_k,
        )

    def _encode_unigram_batch(self, real_batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self._unigram_uses_direct_int2():
            return self._encode_direct_int2_batch(real_batch)
        if self._unigram_uses_direct_int4():
            return self._encode_direct_int4_batch(real_batch)
        return self._encode_direct_int8_batch(real_batch)

    def _encode_decode_step(
        self,
        real_h: torch.Tensor,
        ref_h: Optional[torch.Tensor],
        tier: str,
    ) -> Tuple[torch.Tensor, int]:
        """Decode always sends raw FP16.  Kept for legacy callers."""
        recon = real_h.to(torch.float16).clone()
        transfer = int(real_h.numel() * 2) if self.track_transfer_bytes else 0
        return recon, transfer

    def _update_table_step(
        self,
        tables: Sequence[Any],
        a: int,
        b: int,
        c: int,
        bi_hidden: torch.Tensor,
        trigram_hidden: torch.Tensor,
    ) -> None:
        for table in tables:
            table.update_with_decode_step(a, b, c, bi_hidden, trigram_hidden)

    def _submit_decode_table_update_v2(
        self,
        running_token_ids: List[int],
        decode_pos: int,
        h: torch.Tensor,
        prev_h: Optional[torch.Tensor],
        curr_h: Optional[torch.Tensor],
        prefill_hidden: torch.Tensor,
        input_ids: List[int],
    ) -> None:
        """Buffer a decode table update for batched submission.

        Updates are accumulated and flushed to the update_executor every
        ``_decode_update_flush_every`` steps to amortise Future creation
        overhead across multiple tokens (OPT-8).
        """
        if len(running_token_ids) < 3:
            return
        a = running_token_ids[-3]
        b = running_token_ids[-2]
        c = running_token_ids[-1]
        b_abs_pos = decode_pos - 1
        if b_abs_pos < len(input_ids):
            bi_hidden = prefill_hidden[b_abs_pos]
        elif curr_h is not None:
            bi_hidden = curr_h
        else:
            bi_hidden = h
        self._decode_update_batch.append((a, b, c, bi_hidden, h))
        if len(self._decode_update_batch) >= self._decode_update_flush_every:
            self._flush_decode_update_batch()

    def _update_table_batch(
        self,
        tables: Sequence[Any],
        updates: List[Tuple],
    ) -> None:
        """Worker: apply a batch of buffered table updates."""
        for a, b, c, bi_hidden, tri_hidden in updates:
            for table in tables:
                table.update_with_decode_step(a, b, c, bi_hidden, tri_hidden)

    def _flush_decode_update_batch(self) -> None:
        """Submit all buffered decode table updates as a single Future."""
        self._drain_decode_updates(wait=False)
        if not self._decode_update_batch:
            return
        batch = list(self._decode_update_batch)
        self._decode_update_batch.clear()
        tables = (self.table,)
        future = self.update_executor.submit(self._update_table_batch, tables, batch)
        self._pending_decode_updates.append(future)