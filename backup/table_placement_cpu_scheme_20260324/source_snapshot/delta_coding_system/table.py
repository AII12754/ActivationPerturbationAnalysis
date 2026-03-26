"""Persistent trigram/bigram activation table for a single PP boundary layer.

Stores pre-computed hidden states from n-gram forward passes and supports:
  - Batch table construction from a corpus
  - Online updates from new requests
  - Tiered classification: trigram → self-ref → bigram → unigram

DAG storage:
  Hidden states are organized as a two-level trie (DAG):
    dag[A][B] = (bigram_hidden, {C: trigram_hidden, ...})
  This shares the bigram prefix lookup across all trigrams with the same
  (A,B) prefix, reducing lookup overhead and key storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from delta_coding_system.codec import Int8OutlierPacket, groupwise_int8_dequantize_topk, groupwise_int8_quantize_topk

logger = logging.getLogger(__name__)


StoredHidden = Union[torch.Tensor, Int8OutlierPacket]


@dataclass
class SparseReferenceBatch:
    positions: List[int]
    refs: torch.Tensor
    position_to_offset: Dict[int, int]

    def gather(self, positions: List[int]) -> torch.Tensor:
        if not positions:
            return self.refs[:0]
        offsets = [self.position_to_offset[pos] for pos in positions]
        idx = torch.tensor(offsets, dtype=torch.long, device=self.refs.device)
        return self.refs[idx]


class _BigramNode:
    """DAG node for a bigram prefix (A, B).

    Stores the bigram hidden state and a dict mapping suffix token C
    to the trigram hidden state for (A, B, C).
    """
    __slots__ = ("bigram_hidden", "suffixes", "last_access", "hit_count")

    def __init__(self, bigram_hidden: StoredHidden):
        self.bigram_hidden = bigram_hidden
        self.suffixes: Dict[int, StoredHidden] = {}
        self.last_access = 0
        self.hit_count = 0


class NgramTable:
    """Persistent trigram/bigram activation table for a single PP boundary layer.

    Internal storage is a two-level DAG (trie):
      ``_dag[token_A][token_B]`` → ``_BigramNode``
        - ``.bigram_hidden``: hidden state at position B from a 3-token forward
        - ``.suffixes[token_C]``: hidden state at position C (trigram)
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        max_entries: int = 0,
    ):
        self.device = device
        self.dtype = dtype
        self.max_entries = max_entries  # 0 = unlimited (backward compat)
        self._request_counter = 0
        self._last_evicted = 0
        # Two-level trie: A -> B -> _BigramNode
        self._dag: Dict[int, Dict[int, _BigramNode]] = {}
        # Counts for fast stats
        self._num_trigrams = 0
        self._num_bigrams = 0
        self._last_classify_stats: Dict[str, Any] = {}
        self.pin_cpu_output_copy = False
        self.enable_async_cpu_output_copy = False
        self._output_copy_streams: Dict[str, torch.cuda.Stream] = {}
        self.gpu_hot_cache: Optional[NgramTable] = None
        self.storage_format = "raw"
        self.int8_group_size = 128
        self.int8_top_k = 4

    def enable_int8_storage(self, group_size: int = 128, top_k: int = 4) -> None:
        self.storage_format = "int8"
        self.int8_group_size = group_size
        self.int8_top_k = top_k

    def disable_int8_storage(self) -> None:
        self.storage_format = "raw"

    def _packet_num_bytes(self, packet: Int8OutlierPacket) -> int:
        total = 0
        total += packet.quantized.nelement() * packet.quantized.element_size()
        total += packet.scales.nelement() * packet.scales.element_size()
        total += packet.zero_points.nelement() * packet.zero_points.element_size()
        total += packet.topk_values.nelement() * packet.topk_values.element_size()
        total += packet.topk_indices.nelement() * packet.topk_indices.element_size()
        return total

    def _slice_int8_packet(self, packet: Int8OutlierPacket, idx: int) -> Int8OutlierPacket:
        sl = slice(idx, idx + 1)
        return Int8OutlierPacket(
            quantized=packet.quantized[sl].clone(),
            scales=packet.scales[sl].clone(),
            zero_points=packet.zero_points[sl].clone(),
            topk_values=packet.topk_values[sl].clone(),
            topk_indices=packet.topk_indices[sl].clone(),
            group_size=packet.group_size,
            top_k=packet.top_k,
        )

    def _merge_int8_packets(self, stored_batch: List[Int8OutlierPacket]) -> Int8OutlierPacket:
        first = stored_batch[0]
        return Int8OutlierPacket(
            quantized=torch.cat([item.quantized for item in stored_batch], dim=0),
            scales=torch.cat([item.scales for item in stored_batch], dim=0),
            zero_points=torch.cat([item.zero_points for item in stored_batch], dim=0),
            topk_values=torch.cat([item.topk_values for item in stored_batch], dim=0),
            topk_indices=torch.cat([item.topk_indices for item in stored_batch], dim=0),
            group_size=first.group_size,
            top_k=first.top_k,
        )

    def _pin_int8_packet(self, packet: Int8OutlierPacket) -> Int8OutlierPacket:
        return Int8OutlierPacket(
            quantized=packet.quantized.pin_memory(),
            scales=packet.scales.pin_memory(),
            zero_points=packet.zero_points.pin_memory(),
            topk_values=packet.topk_values.pin_memory(),
            topk_indices=packet.topk_indices.pin_memory(),
            group_size=packet.group_size,
            top_k=packet.top_k,
        )

    def _move_int8_packet(
        self,
        packet: Int8OutlierPacket,
        device: torch.device,
        non_blocking: bool,
    ) -> Int8OutlierPacket:
        return Int8OutlierPacket(
            quantized=packet.quantized.to(device, non_blocking=non_blocking),
            scales=packet.scales.to(device, non_blocking=non_blocking),
            zero_points=packet.zero_points.to(device, non_blocking=non_blocking),
            topk_values=packet.topk_values.to(device, non_blocking=non_blocking),
            topk_indices=packet.topk_indices.to(device, non_blocking=non_blocking),
            group_size=packet.group_size,
            top_k=packet.top_k,
        )

    def _encode_hidden_batch_for_storage(self, hidden_states: torch.Tensor) -> List[StoredHidden]:
        if self.storage_format != "int8":
            return [hidden_states[i].to(device=self.device, dtype=self.dtype).detach() for i in range(hidden_states.shape[0])]
        packet = groupwise_int8_quantize_topk(
            hidden_states.to(device=self.device, dtype=torch.float16),
            self.int8_group_size,
            self.int8_top_k,
        )
        return [self._slice_int8_packet(packet, i) for i in range(hidden_states.shape[0])]

    def _decode_stored_batch(
        self,
        stored_batch: List[StoredHidden],
        output_device: torch.device,
    ) -> torch.Tensor:
        if not stored_batch:
            raise ValueError("stored_batch must be non-empty")
        first = stored_batch[0]
        if isinstance(first, torch.Tensor):
            stacked = torch.stack(stored_batch).to(torch.float16)
            if stacked.device != output_device:
                stacked = stacked.to(output_device, non_blocking=False)
            return stacked

        packet = self._merge_int8_packets(stored_batch)
        packet = self._move_int8_packet(packet, output_device, non_blocking=False)
        return groupwise_int8_dequantize_topk(packet)

    def enable_gpu_hot_cache(self, device: torch.device, max_entries: int) -> None:
        self.gpu_hot_cache = NgramTable(
            device=device,
            dtype=self.dtype,
            max_entries=max_entries,
        )

    def disable_gpu_hot_cache(self) -> None:
        self.gpu_hot_cache = None

    def _get_output_copy_stream(self, device: torch.device) -> torch.cuda.Stream:
        key = str(device)
        stream = self._output_copy_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._output_copy_streams[key] = stream
        return stream

    # ------------------------------------------------------------------
    # DAG access helpers
    # ------------------------------------------------------------------
    def _get_node(self, a: int, b: int) -> Optional[_BigramNode]:
        """Look up the bigram node for (a, b), or None."""
        level_b = self._dag.get(a)
        if level_b is None:
            return None
        return level_b.get(b)

    def _get_or_create_node(self, a: int, b: int, bigram_hidden: StoredHidden) -> _BigramNode:
        """Get existing node for (a, b) or create one with the given hidden state."""
        level_b = self._dag.get(a)
        if level_b is None:
            level_b = {}
            self._dag[a] = level_b
        node = level_b.get(b)
        if node is None:
            node = _BigramNode(bigram_hidden)
            level_b[b] = node
            self._num_bigrams += 1
        return node

    def has_trigram(self, a: int, b: int, c: int) -> bool:
        node = self._get_node(a, b)
        return node is not None and c in node.suffixes

    def get_trigram(self, a: int, b: int, c: int) -> Optional[torch.Tensor]:
        node = self._get_node(a, b)
        if node is None:
            return None
        return node.suffixes.get(c)

    def get_bigram(self, a: int, b: int) -> Optional[torch.Tensor]:
        node = self._get_node(a, b)
        if node is None:
            return None
        return node.bigram_hidden

    # ------------------------------------------------------------------
    # Internal: batched 3-token forward
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _forward_trigrams(
        self,
        model,
        trigram_list: List[Tuple[int, int, int]],
        layer_idx: int,
        batch_size: int,
    ) -> None:
        """Forward trigrams as flat (batch, 3) batches, storing into DAG."""
        for start in range(0, len(trigram_list), batch_size):
            batch_tris = trigram_list[start : start + batch_size]
            input_tensor = torch.tensor(batch_tris, dtype=torch.long, device=self.device)
            outputs = model(
                input_ids=input_tensor,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[layer_idx]  # (batch, 3, hidden_dim)
            tri_hidden_list = self._encode_hidden_batch_for_storage(hidden[:, -1, :])
            bi_hidden_list = self._encode_hidden_batch_for_storage(hidden[:, -2, :])

            for j, tri in enumerate(batch_tris):
                a, b, c = tri
                node = self._get_or_create_node(a, b, bi_hidden_list[j])
                if c not in node.suffixes:
                    node.suffixes[c] = tri_hidden_list[j]
                    self._num_trigrams += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def build_from_corpus(
        self,
        model,
        token_ids_list: List[List[int]],
        layer_idx: int,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """Build table from one or more token sequences."""
        t0 = time.perf_counter()

        unique_trigrams: Dict[Tuple[int, int, int], None] = {}
        for ids in token_ids_list:
            for i in range(2, len(ids)):
                tri = (ids[i - 2], ids[i - 1], ids[i])
                if not self.has_trigram(*tri):
                    unique_trigrams.setdefault(tri, None)

        trigram_list = list(unique_trigrams.keys())
        if not trigram_list:
            return {"num_trigrams": self._num_trigrams, "num_bigrams": self._num_bigrams, "build_time_ms": 0.0}

        self._forward_trigrams(model, trigram_list, layer_idx, batch_size)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "NgramTable built: %d trigrams, %d bigrams in %.1f ms",
            self._num_trigrams, self._num_bigrams, elapsed_ms,
        )
        return {
            "num_trigrams": self._num_trigrams,
            "num_bigrams": self._num_bigrams,
            "build_time_ms": elapsed_ms,
        }

    @torch.inference_mode()
    def update_from_request(
        self,
        model,
        token_ids: List[int],
        layer_idx: int,
        batch_size: int = 128,
    ) -> Tuple[int, int, float]:
        """Add new trigrams from a request not already in table."""
        t0 = time.perf_counter()

        old_tri = self._num_trigrams
        old_bi = self._num_bigrams

        new_trigrams: Dict[Tuple[int, int, int], None] = {}
        for i in range(2, len(token_ids)):
            tri = (token_ids[i - 2], token_ids[i - 1], token_ids[i])
            if not self.has_trigram(*tri) and tri not in new_trigrams:
                new_trigrams[tri] = None

        if not new_trigrams:
            return 0, 0, (time.perf_counter() - t0) * 1000.0

        tri_list = list(new_trigrams.keys())
        self._forward_trigrams(model, tri_list, layer_idx, batch_size)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._num_trigrams - old_tri, self._num_bigrams - old_bi, elapsed_ms

    def classify_and_build_refs(
        self,
        token_ids: List[int],
        hidden_dim: int,
        output_device: Optional[torch.device] = None,
    ) -> Tuple[List[str], SparseReferenceBatch, List[Optional[int]], Dict[Tuple, int]]:
        """Classify each position into tier and return reference info.

        Uses DAG trie for efficient prefix-shared lookups.

        Returns
        -------
        tiers : list[str] — "trigram"|"bigram"|"self_ref"|"unigram" per position
        sparse_refs : matched positions + decoded references on output_device
        self_ref_sources : list[Optional[int]] — source position index for self_ref tier
        first_occurrence_map : dict — trigram → first position
        """
        if output_device is None:
            output_device = self.device

        seq_len = len(token_ids)
        tiers: List[str] = []
        self_ref_sources: List[Optional[int]] = []
        first_occurrence_map: Dict[Tuple[int, int, int], int] = {}
        lookup_start = time.perf_counter()

        # Collect (position, raw_tensor) pairs to batch FP8→FP16 conversion
        ref_positions_gpu: List[int] = []
        ref_tensors_gpu: List[StoredHidden] = []
        ref_positions_cpu: List[int] = []
        ref_tensors_cpu: List[StoredHidden] = []
        gpu_hot_hits = 0

        for i in range(seq_len):
            trigram = None
            if i >= 2:
                a, b, c = token_ids[i - 2], token_ids[i - 1], token_ids[i]
                trigram = (a, b, c)

                hot_cache = self.gpu_hot_cache
                if hot_cache is not None and output_device.type == "cuda":
                    hot_tri_ref = hot_cache.get_trigram(a, b, c)
                    if hot_tri_ref is not None:
                        tiers.append("trigram")
                        ref_positions_gpu.append(i)
                        ref_tensors_gpu.append(hot_tri_ref)
                        self_ref_sources.append(None)
                        first_occurrence_map.setdefault(trigram, i)
                        gpu_hot_hits += 1
                        continue

                # 1. Trigram table lookup (highest priority)
                node_ab = self._get_node(a, b)
                if node_ab is not None and c in node_ab.suffixes:
                    node_ab.hit_count += 1
                    node_ab.last_access = self._request_counter
                    tiers.append("trigram")
                    if output_device == self.device:
                        ref_positions_gpu.append(i)
                        ref_tensors_gpu.append(node_ab.suffixes[c])
                    else:
                        ref_positions_cpu.append(i)
                        ref_tensors_cpu.append(node_ab.suffixes[c])
                    self_ref_sources.append(None)
                    first_occurrence_map.setdefault(trigram, i)
                    continue

                # 2. Self-ref: repeated trigram from earlier in this sequence
                if trigram in first_occurrence_map:
                    tiers.append("self_ref")
                    self_ref_sources.append(first_occurrence_map[trigram])
                    continue

            # 3. Bigram lookup: _dag[B][C].bigram_hidden
            if i >= 1:
                b_tok, c_tok = token_ids[i - 1], token_ids[i]
                hot_cache = self.gpu_hot_cache
                if hot_cache is not None and output_device.type == "cuda":
                    hot_bi_ref = hot_cache.get_bigram(b_tok, c_tok)
                    if hot_bi_ref is not None:
                        tiers.append("bigram")
                        ref_positions_gpu.append(i)
                        ref_tensors_gpu.append(hot_bi_ref)
                        self_ref_sources.append(None)
                        if trigram is not None:
                            first_occurrence_map.setdefault(trigram, i)
                        gpu_hot_hits += 1
                        continue
                node_bc = self._get_node(b_tok, c_tok)
                if node_bc is not None:
                    node_bc.hit_count += 1
                    node_bc.last_access = self._request_counter
                    tiers.append("bigram")
                    if output_device == self.device:
                        ref_positions_gpu.append(i)
                        ref_tensors_gpu.append(node_bc.bigram_hidden)
                    else:
                        ref_positions_cpu.append(i)
                        ref_tensors_cpu.append(node_bc.bigram_hidden)
                    self_ref_sources.append(None)
                    if trigram is not None:
                        first_occurrence_map.setdefault(trigram, i)
                    continue

            # 4. Unigram fallback
            tiers.append("unigram")
            self_ref_sources.append(None)
            if trigram is not None:
                first_occurrence_map.setdefault(trigram, i)

        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0

        # Batch FP8→FP16 conversion: single stack + cast instead of per-element loop
        materialize_start = time.perf_counter()
        matched_positions: List[int] = []
        matched_refs: List[torch.Tensor] = []
        output_copy_ms = 0.0
        pin_memory_ms = 0.0
        used_async_copy = False
        if ref_tensors_gpu:
            stacked_gpu = self._decode_stored_batch(ref_tensors_gpu, output_device)
            matched_positions.extend(ref_positions_gpu)
            matched_refs.append(stacked_gpu)
        if ref_tensors_cpu:
            cpu_first = ref_tensors_cpu[0]
            if isinstance(cpu_first, Int8OutlierPacket) and output_device.type == "cuda":
                packet = self._merge_int8_packets(ref_tensors_cpu)
                if self.pin_cpu_output_copy and packet.quantized.device.type == "cpu":
                    pin_start = time.perf_counter()
                    packet = self._pin_int8_packet(packet)
                    pin_memory_ms = (time.perf_counter() - pin_start) * 1000.0
                copy_start = time.perf_counter()
                non_blocking = bool(packet.quantized.device.type == "cpu" and packet.quantized.is_pinned())
                if self.enable_async_cpu_output_copy:
                    copy_stream = self._get_output_copy_stream(output_device)
                    with torch.cuda.stream(copy_stream):
                        packet = self._move_int8_packet(packet, output_device, non_blocking=non_blocking)
                        converted = groupwise_int8_dequantize_topk(packet)
                    copy_stream.synchronize()
                    used_async_copy = True
                else:
                    packet = self._move_int8_packet(packet, output_device, non_blocking=non_blocking)
                    converted = groupwise_int8_dequantize_topk(packet)
                output_copy_ms = (time.perf_counter() - copy_start) * 1000.0
            else:
                decode_device = output_device if isinstance(cpu_first, Int8OutlierPacket) else self.device
                converted = self._decode_stored_batch(ref_tensors_cpu, decode_device)
            if converted.device != output_device:
                if (
                    self.pin_cpu_output_copy
                    and converted.device.type == "cpu"
                    and output_device.type == "cuda"
                    and not converted.is_pinned()
                ):
                    pin_start = time.perf_counter()
                    converted = converted.pin_memory()
                    pin_memory_ms = (time.perf_counter() - pin_start) * 1000.0
                copy_start = time.perf_counter()
                if (
                    self.enable_async_cpu_output_copy
                    and converted.device.type == "cpu"
                    and output_device.type == "cuda"
                ):
                    copy_stream = self._get_output_copy_stream(output_device)
                    with torch.cuda.stream(copy_stream):
                        converted = converted.to(output_device, non_blocking=True)
                    used_async_copy = True
                else:
                    converted = converted.to(
                        output_device,
                        non_blocking=bool(converted.device.type == "cpu" and converted.is_pinned() and output_device.type == "cuda"),
                    )
                output_copy_ms = (time.perf_counter() - copy_start) * 1000.0
            matched_positions.extend(ref_positions_cpu)
            matched_refs.append(converted)

        materialize_ms = (time.perf_counter() - materialize_start) * 1000.0
        if matched_refs:
            refs = torch.cat(matched_refs, dim=0)
            position_to_offset = {pos: idx for idx, pos in enumerate(matched_positions)}
        else:
            refs = torch.empty((0, hidden_dim), device=output_device, dtype=torch.float16)
            position_to_offset = {}
        sparse_refs = SparseReferenceBatch(
            positions=matched_positions,
            refs=refs,
            position_to_offset=position_to_offset,
        )
        self._last_classify_stats = {
            "table_device": str(self.device),
            "output_device": str(output_device),
            "lookup_ms": lookup_ms,
            "materialize_ms": materialize_ms,
            "output_copy_ms": output_copy_ms,
            "pin_memory_ms": pin_memory_ms,
            "used_async_copy": used_async_copy,
            "num_refs": len(ref_tensors_gpu) + len(ref_tensors_cpu),
            "gpu_hot_hits": gpu_hot_hits,
            "seq_len": seq_len,
        }

        return tiers, sparse_refs, self_ref_sources, first_occurrence_map

    def update_from_hidden_states(
        self,
        token_ids: List[int],
        hidden_states: torch.Tensor,
    ) -> Tuple[int, int, float]:
        """Update table directly from full-sequence prefill hidden states.

        Instead of running separate 3-token forwards, reuse the hidden states
        already computed during prefill.
        """
        t0 = time.perf_counter()
        old_tri = self._num_trigrams
        old_bi = self._num_bigrams

        # Batch dtype conversion
        hidden_list = self._encode_hidden_batch_for_storage(hidden_states)

        for i in range(len(token_ids)):
            h_i = hidden_list[i]

            # Store bigram reference
            if i >= 1:
                b_tok, c_tok = token_ids[i - 1], token_ids[i]
                self._get_or_create_node(b_tok, c_tok, h_i)

            # Store trigram reference
            if i >= 2:
                a, b, c = token_ids[i - 2], token_ids[i - 1], token_ids[i]
                node = self._get_node(a, b)
                if node is None:
                    node = self._get_or_create_node(a, b, hidden_list[i - 1])
                if c not in node.suffixes:
                    node.suffixes[c] = h_i
                    self._num_trigrams += 1

        self._request_counter += 1
        self._last_evicted = self.evict()

        if self.gpu_hot_cache is not None:
            self.gpu_hot_cache.update_from_hidden_states(token_ids, hidden_states)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._num_trigrams - old_tri, self._num_bigrams - old_bi, elapsed_ms

    def evict(self) -> int:
        """Evict lowest-scoring entries if table exceeds max_entries."""
        total = self._num_trigrams + self._num_bigrams
        if self.max_entries <= 0 or total <= self.max_entries:
            return 0

        target = int(self.max_entries * 0.9)

        scored_trigrams = []
        for a, level_b in self._dag.items():
            for b, node in level_b.items():
                score = node.hit_count * 10 + node.last_access
                for c in node.suffixes:
                    scored_trigrams.append((score, a, b, c))

        scored_trigrams.sort(key=lambda x: x[0])

        evicted = 0
        to_evict = total - target
        for score, a, b, c in scored_trigrams:
            if evicted >= to_evict:
                break
            node = self._dag[a][b]
            del node.suffixes[c]
            self._num_trigrams -= 1
            evicted += 1
            if not node.suffixes:
                del self._dag[a][b]
                self._num_bigrams -= 1
                if not self._dag[a]:
                    del self._dag[a]

        return evicted

    # ------------------------------------------------------------------
    # Device transfer (GPU ↔ CPU offload)
    # ------------------------------------------------------------------
    def to_device(self, device: torch.device) -> "NgramTable":
        """Move all stored tensors to *device* and update self.device.

        Returns self for chaining.
        """
        if device == self.device:
            return self
        for level_b in self._dag.values():
            for node in level_b.values():
                if isinstance(node.bigram_hidden, torch.Tensor):
                    node.bigram_hidden = node.bigram_hidden.to(device, non_blocking=True)
                else:
                    node.bigram_hidden = Int8OutlierPacket(
                        quantized=node.bigram_hidden.quantized.to(device, non_blocking=True),
                        scales=node.bigram_hidden.scales.to(device, non_blocking=True),
                        zero_points=node.bigram_hidden.zero_points.to(device, non_blocking=True),
                        topk_values=node.bigram_hidden.topk_values.to(device, non_blocking=True),
                        topk_indices=node.bigram_hidden.topk_indices.to(device, non_blocking=True),
                        group_size=node.bigram_hidden.group_size,
                        top_k=node.bigram_hidden.top_k,
                    )
                moved_suffixes: Dict[int, StoredHidden] = {}
                for c, h in node.suffixes.items():
                    if isinstance(h, torch.Tensor):
                        moved_suffixes[c] = h.to(device, non_blocking=True)
                    else:
                        moved_suffixes[c] = Int8OutlierPacket(
                            quantized=h.quantized.to(device, non_blocking=True),
                            scales=h.scales.to(device, non_blocking=True),
                            zero_points=h.zero_points.to(device, non_blocking=True),
                            topk_values=h.topk_values.to(device, non_blocking=True),
                            topk_indices=h.topk_indices.to(device, non_blocking=True),
                            group_size=h.group_size,
                            top_k=h.top_k,
                        )
                node.suffixes = moved_suffixes
        self.device = device
        return self

    @property
    def stats(self) -> Dict[str, Any]:
        """Return table statistics."""
        sample = None
        for level_b in self._dag.values():
            for node in level_b.values():
                sample = node.bigram_hidden
                break
            if sample is not None:
                break
        if sample is not None:
            if isinstance(sample, torch.Tensor):
                per_entry_bytes = sample.nelement() * sample.element_size()
            else:
                per_entry_bytes = self._packet_num_bytes(sample)
        else:
            per_entry_bytes = 0
        memory_bytes = (self._num_trigrams + self._num_bigrams) * per_entry_bytes
        return {
            "num_trigrams": self._num_trigrams,
            "num_bigrams": self._num_bigrams,
            "memory_bytes": memory_bytes,
            "device": str(self.device),
            "gpu_hot_cache": None if self.gpu_hot_cache is None else self.gpu_hot_cache.stats,
        }

    @property
    def last_classify_stats(self) -> Dict[str, Any]:
        return dict(self._last_classify_stats)


# ======================================================================
# DomainTableManager — domain-aware table pool with GPU/CPU tiering
# ======================================================================
class DomainTableManager:
    """Manages multiple NgramTables keyed by domain/task.

    Active tables live on *gpu_device* for fast lookups.
    When the number of GPU-resident tables exceeds *max_gpu_tables*,
    the least-recently-used table is offloaded to CPU memory.
    Accessing a CPU-resident table transparently promotes it back to GPU.

    Usage::

        mgr = DomainTableManager(gpu_device=torch.device("cuda:0"))
        tbl = mgr.get("gsm8k")          # creates or activates
        tbl.classify_and_build_refs(...)  # runs on GPU
        mgr.release("gsm8k")            # optional hint: done for now
    """

    def __init__(
        self,
        gpu_device: torch.device,
        table_dtype: torch.dtype = torch.float16,
        max_entries_per_table: int = 0,
        max_gpu_tables: int = 3,
    ):
        self.gpu_device = gpu_device
        self.cpu_device = torch.device("cpu")
        self.table_dtype = table_dtype
        self.max_entries_per_table = max_entries_per_table
        self.max_gpu_tables = max(max_gpu_tables, 1)

        # domain_key → NgramTable
        self._tables: Dict[str, NgramTable] = {}
        # Ordered list of domain keys on GPU (most-recently-used last)
        self._gpu_lru: List[str] = []
        self._access_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, domain: str) -> NgramTable:
        """Return the NgramTable for *domain*, creating or promoting as needed.

        If the table is on CPU it is moved to GPU first.
        If GPU budget is exceeded the LRU table is offloaded to CPU.
        """
        tbl = self._tables.get(domain)
        already_on_gpu = domain in self._gpu_lru

        if tbl is None:
            # Brand-new domain — needs a GPU slot
            self._evict_one_if_full()
            tbl = NgramTable(
                device=self.gpu_device,
                dtype=self.table_dtype,
                max_entries=self.max_entries_per_table,
            )
            self._tables[domain] = tbl
            self._gpu_lru.append(domain)
        elif not already_on_gpu:
            # Promote from CPU → GPU
            self._evict_one_if_full()
            tbl.to_device(self.gpu_device)
            self._gpu_lru.append(domain)
        else:
            # Already on GPU — just refresh LRU position
            self._gpu_lru.remove(domain)
            self._gpu_lru.append(domain)

        self._access_counter += 1
        return tbl

    def release(self, domain: str) -> None:
        """Optional hint that *domain* is not immediately needed.

        Does NOT offload eagerly — just deprioritizes in LRU.
        """
        if domain in self._gpu_lru:
            self._gpu_lru.remove(domain)
            self._gpu_lru.insert(0, domain)  # move to front (oldest)

    def offload(self, domain: str) -> None:
        """Explicitly offload *domain* table to CPU."""
        if domain in self._gpu_lru:
            tbl = self._tables.get(domain)
            if tbl is not None:
                tbl.to_device(self.cpu_device)
            self._gpu_lru.remove(domain)

    def offload_all(self) -> None:
        """Move every table to CPU."""
        for domain in list(self._gpu_lru):
            self.offload(domain)

    def delete(self, domain: str) -> None:
        """Permanently remove a domain table."""
        self._tables.pop(domain, None)
        if domain in self._gpu_lru:
            self._gpu_lru.remove(domain)

    def domains(self) -> List[str]:
        """Return all registered domain keys."""
        return list(self._tables.keys())

    def gpu_domains(self) -> List[str]:
        """Return domain keys currently on GPU (LRU order, oldest first)."""
        return list(self._gpu_lru)

    def cpu_domains(self) -> List[str]:
        """Return domain keys currently on CPU."""
        return [d for d in self._tables if d not in self._gpu_lru]

    @property
    def stats(self) -> Dict[str, Any]:
        per_domain = {}
        for d, tbl in self._tables.items():
            s = tbl.stats
            s["on_gpu"] = (d in self._gpu_lru)
            per_domain[d] = s
        return {
            "num_domains": len(self._tables),
            "gpu_resident": len(self._gpu_lru),
            "max_gpu_tables": self.max_gpu_tables,
            "per_domain": per_domain,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evict_one_if_full(self) -> None:
        """Offload the LRU GPU table if at capacity (to free one slot)."""
        if len(self._gpu_lru) >= self.max_gpu_tables:
            victim = self._gpu_lru.pop(0)
            victim_tbl = self._tables.get(victim)
            if victim_tbl is not None:
                logger.info("Offloading domain '%s' table to CPU (%d tri, %d bi)",
                            victim, victim_tbl._num_trigrams, victim_tbl._num_bigrams)
                victim_tbl.to_device(self.cpu_device)
