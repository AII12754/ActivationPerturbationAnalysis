"""Persistent trigram/bigram activation table for a single PP boundary layer.

Stores pre-computed hidden states from n-gram forward passes and supports:
  - Batch table construction from a corpus
  - Online updates from new requests
  - Tiered classification: trigram -> self-ref -> bigram -> unigram

DAG storage:
  Hidden states are organized as a two-level trie (DAG):
    dag[A][B] = (bigram_hidden, {C: trigram_hidden, ...})
  This shares the bigram prefix lookup across all trigrams with the same
  (A,B) prefix, reducing lookup overhead and key storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from delta_coding_system.codec import Int8OutlierPacket, groupwise_int8_dequantize_topk, groupwise_int8_quantize_topk

logger = logging.getLogger(__name__)


StoredHidden = Union[torch.Tensor, Int8OutlierPacket]


@dataclass
class MultiTableRefStats:
    matched_domains: List[Optional[str]]
    domain_hits: Dict[str, int]


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
      ``_dag[token_A][token_B]`` -> ``_BigramNode``
        - ``.bigram_hidden``: hidden state at position B from a 3-token forward
        - ``.suffixes[token_C]``: hidden state at position C (trigram)
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        max_entries: int = 0,
        storage_format: str = "int8",
        int8_group_size: int = 128,
        int8_top_k: int = 4,
        pin_cpu_output_copy: bool = False,
        enable_async_cpu_output_copy: bool = False,
        gpu_hot_cache_entries: int = 0,
        gpu_hot_cache_device: Optional[torch.device] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.max_entries = max_entries  # 0 = unlimited (backward compat)
        self.storage_format = storage_format
        self.int8_group_size = int8_group_size
        self.int8_top_k = int8_top_k
        self.pin_cpu_output_copy = pin_cpu_output_copy
        self.enable_async_cpu_output_copy = enable_async_cpu_output_copy
        self.gpu_hot_cache_entries = max(0, gpu_hot_cache_entries)
        self.gpu_hot_cache_device = gpu_hot_cache_device
        self._request_counter = 0
        self._last_evicted = 0
        # Two-level trie: A -> B -> _BigramNode
        self._dag: Dict[int, Dict[int, _BigramNode]] = {}
        # Counts for fast stats
        self._num_trigrams = 0
        self._num_bigrams = 0
        self._last_classify_stats: Dict[str, Any] = {}
        self.gpu_hot_cache: Optional[NgramTable] = None
        if self.gpu_hot_cache_entries > 0 and self.gpu_hot_cache_device is not None:
            self.enable_gpu_hot_cache(self.gpu_hot_cache_device, self.gpu_hot_cache_entries)

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
        batch_slice = slice(idx, idx + 1)
        return Int8OutlierPacket(
            quantized=packet.quantized[batch_slice].clone(),
            scales=packet.scales[batch_slice].clone(),
            zero_points=packet.zero_points[batch_slice].clone(),
            topk_values=packet.topk_values[batch_slice].clone(),
            topk_indices=packet.topk_indices[batch_slice].clone(),
            group_size=packet.group_size,
            top_k=packet.top_k,
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

    def _prepare_stored_hidden(self, stored_hidden: StoredHidden) -> StoredHidden:
        if isinstance(stored_hidden, torch.Tensor):
            tensor = stored_hidden.to(device=self.device, dtype=self.dtype, non_blocking=True).detach()
            if self.device.type == "cpu" and self.pin_cpu_output_copy and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor

        packet = stored_hidden
        non_blocking = bool(
            packet.quantized.device.type == "cpu"
            and getattr(packet.quantized, "is_pinned", lambda: False)()
            and self.device.type == "cuda"
        )
        packet = self._move_int8_packet(packet, self.device, non_blocking=non_blocking)
        if self.device.type == "cpu" and self.pin_cpu_output_copy:
            packet = self._pin_int8_packet(packet)
        return packet

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

    def _encode_hidden_batch_for_storage(self, hidden_states: torch.Tensor) -> List[StoredHidden]:
        if self.storage_format != "int8":
            return [
                self._prepare_stored_hidden(hidden_states[i].to(dtype=torch.float16))
                for i in range(hidden_states.shape[0])
            ]

        packet = groupwise_int8_quantize_topk(
            hidden_states.to(dtype=torch.float16),
            self.int8_group_size,
            self.int8_top_k,
        )
        return [self._prepare_stored_hidden(self._slice_int8_packet(packet, i)) for i in range(hidden_states.shape[0])]

    def _decode_stored_batch(self, stored_batch: List[StoredHidden], output_device: torch.device) -> torch.Tensor:
        if not stored_batch:
            raise ValueError("stored_batch must be non-empty")

        first = stored_batch[0]
        if isinstance(first, torch.Tensor):
            tensor_batch = [item for item in stored_batch if isinstance(item, torch.Tensor)]
            stacked = torch.stack(tensor_batch).to(torch.float16)
            if stacked.device != output_device:
                non_blocking = bool(
                    stacked.device.type == "cpu"
                    and stacked.is_pinned()
                    and output_device.type == "cuda"
                    and self.enable_async_cpu_output_copy
                )
                stacked = stacked.to(output_device, non_blocking=non_blocking)
            return stacked

        packet_batch = [item for item in stored_batch if isinstance(item, Int8OutlierPacket)]
        packet = self._merge_int8_packets(packet_batch)
        non_blocking = bool(
            packet.quantized.device.type == "cpu"
            and packet.quantized.is_pinned()
            and output_device.type == "cuda"
            and self.enable_async_cpu_output_copy
        )
        packet = self._move_int8_packet(packet, output_device, non_blocking=non_blocking)
        return groupwise_int8_dequantize_topk(packet)

    def materialize_hidden(self, stored_hidden: StoredHidden, output_device: torch.device) -> torch.Tensor:
        return self._decode_stored_batch([stored_hidden], output_device)[0]

    def enable_gpu_hot_cache(self, device: torch.device, max_entries: int) -> None:
        self.gpu_hot_cache = NgramTable(
            device=device,
            dtype=self.dtype,
            max_entries=max_entries,
            storage_format=self.storage_format,
            int8_group_size=self.int8_group_size,
            int8_top_k=self.int8_top_k,
        )

    def disable_gpu_hot_cache(self) -> None:
        self.gpu_hot_cache = None

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

    def get_trigram(self, a: int, b: int, c: int) -> Optional[StoredHidden]:
        node = self._get_node(a, b)
        if node is None:
            return None
        return node.suffixes.get(c)

    def get_bigram(self, a: int, b: int) -> Optional[StoredHidden]:
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
    ) -> Tuple[List[str], torch.Tensor, List[Optional[int]], Dict[Tuple, int]]:
        """Classify each position into tier and return reference info.

        Uses DAG trie for efficient prefix-shared lookups.

        Returns
        -------
        tiers : list[str] - "trigram"|"bigram"|"self_ref"|"unigram" per position
        ref_acts : (seq_len, hidden_dim) - reference activation per position
        self_ref_sources : list[Optional[int]] - source position index for self_ref tier
        first_occurrence_map : dict - trigram -> first position
        """
        if output_device is None:
            output_device = self.device

        seq_len = len(token_ids)
        tiers: List[str] = []
        self_ref_sources: List[Optional[int]] = []
        first_occurrence_map: Dict[Tuple[int, int, int], int] = {}

        # Collect matched references and decode in one batch.
        ref_positions: List[int] = []
        ref_tensors: List[StoredHidden] = []
        gpu_hot_hits = 0
        lookup_start = time.perf_counter()

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
                        ref_positions.append(i)
                        ref_tensors.append(hot_tri_ref)
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
                    ref_positions.append(i)
                    ref_tensors.append(node_ab.suffixes[c])
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
                        ref_positions.append(i)
                        ref_tensors.append(hot_bi_ref)
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
                    ref_positions.append(i)
                    ref_tensors.append(node_bc.bigram_hidden)
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

        # Decode all matched references in one batch.
        materialize_start = time.perf_counter()
        ref_acts = torch.zeros(seq_len, hidden_dim, device=output_device, dtype=torch.float16)
        if ref_tensors:
            converted = self._decode_stored_batch(ref_tensors, output_device)
            idx = torch.tensor(ref_positions, dtype=torch.long, device=output_device)
            ref_acts[idx] = converted

        self._last_classify_stats = {
            "table_device": str(self.device),
            "output_device": str(output_device),
            "lookup_ms": lookup_ms,
            "materialize_ms": (time.perf_counter() - materialize_start) * 1000.0,
            "num_refs": len(ref_tensors),
            "gpu_hot_hits": gpu_hot_hits,
            "seq_len": seq_len,
        }

        return tiers, ref_acts, self_ref_sources, first_occurrence_map

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
    # Device transfer (GPU <-> CPU offload)
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
                    if device.type == "cpu" and self.pin_cpu_output_copy and not node.bigram_hidden.is_pinned():
                        node.bigram_hidden = node.bigram_hidden.pin_memory()
                else:
                    node.bigram_hidden = self._move_int8_packet(node.bigram_hidden, device, non_blocking=True)
                    if device.type == "cpu" and self.pin_cpu_output_copy:
                        node.bigram_hidden = self._pin_int8_packet(node.bigram_hidden)
                moved_suffixes: Dict[int, StoredHidden] = {}
                for c, h in node.suffixes.items():
                    if isinstance(h, torch.Tensor):
                        moved_h = h.to(device, non_blocking=True)
                        if device.type == "cpu" and self.pin_cpu_output_copy and not moved_h.is_pinned():
                            moved_h = moved_h.pin_memory()
                        moved_suffixes[c] = moved_h
                    else:
                        moved_h = self._move_int8_packet(h, device, non_blocking=True)
                        if device.type == "cpu" and self.pin_cpu_output_copy:
                            moved_h = self._pin_int8_packet(moved_h)
                        moved_suffixes[c] = moved_h
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
        if sample is None:
            per_entry_bytes = 0
        elif isinstance(sample, torch.Tensor):
            per_entry_bytes = sample.nelement() * sample.element_size()
        else:
            per_entry_bytes = self._packet_num_bytes(sample)
        memory_bytes = (self._num_trigrams + self._num_bigrams) * per_entry_bytes
        return {
            "num_trigrams": self._num_trigrams,
            "num_bigrams": self._num_bigrams,
            "memory_bytes": memory_bytes,
            "device": str(self.device),
            "storage_format": self.storage_format,
            "int8_group_size": self.int8_group_size,
            "int8_top_k": self.int8_top_k,
            "gpu_hot_cache": None if self.gpu_hot_cache is None else self.gpu_hot_cache.stats,
        }

    @property
    def last_classify_stats(self) -> Dict[str, Any]:
        return dict(self._last_classify_stats)


# =====================================================================
# DomainTableManager - domain-aware table pool with GPU/CPU tiering
# =====================================================================
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
        table_storage_format: str = "int8",
        int8_group_size: int = 128,
        int8_top_k: int = 4,
        table_placement: str = "gpu",
        pin_cpu_output_copy: bool = False,
        enable_async_cpu_output_copy: bool = False,
        gpu_hot_cache_entries: int = 0,
    ):
        self.gpu_device = gpu_device
        self.cpu_device = torch.device("cpu")
        self.table_dtype = table_dtype
        self.max_entries_per_table = max_entries_per_table
        self.max_gpu_tables = max(max_gpu_tables, 1)
        self.table_storage_format = table_storage_format
        self.int8_group_size = int8_group_size
        self.int8_top_k = int8_top_k
        self.table_placement = table_placement
        self.pin_cpu_output_copy = pin_cpu_output_copy
        self.enable_async_cpu_output_copy = enable_async_cpu_output_copy
        self.gpu_hot_cache_entries = max(0, gpu_hot_cache_entries)

        # domain_key -> NgramTable
        self._tables: Dict[str, NgramTable] = {}
        # Ordered list of domain keys on GPU (most-recently-used last)
        self._gpu_lru: List[str] = []
        self._access_counter = 0

    def _create_table(self, device: torch.device) -> NgramTable:
        return NgramTable(
            device=device,
            dtype=self.table_dtype,
            max_entries=self.max_entries_per_table,
            storage_format=self.table_storage_format,
            int8_group_size=self.int8_group_size,
            int8_top_k=self.int8_top_k,
            pin_cpu_output_copy=self.pin_cpu_output_copy,
            enable_async_cpu_output_copy=self.enable_async_cpu_output_copy,
            gpu_hot_cache_entries=self.gpu_hot_cache_entries if device.type == "cpu" else 0,
            gpu_hot_cache_device=self.gpu_device if device.type == "cpu" and self.gpu_hot_cache_entries > 0 else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, domain: str) -> NgramTable:
        """Return the NgramTable for *domain*, creating or promoting as needed.

        If the table is on CPU it is moved to GPU first.
        If GPU budget is exceeded the LRU table is offloaded to CPU.
        """
        if self.table_placement == "cpu":
            tbl = self._tables.get(domain)
            if tbl is None:
                tbl = self._create_table(self.cpu_device)
                self._tables[domain] = tbl
            self._access_counter += 1
            return tbl

        tbl = self._tables.get(domain)
        already_on_gpu = domain in self._gpu_lru

        if tbl is None:
            # Brand-new domain - needs a GPU slot
            self._evict_one_if_full()
            tbl = self._create_table(self.gpu_device)
            self._tables[domain] = tbl
            self._gpu_lru.append(domain)
        elif not already_on_gpu:
            # Promote from CPU -> GPU
            self._evict_one_if_full()
            tbl.to_device(self.gpu_device)
            self._gpu_lru.append(domain)
        else:
            # Already on GPU - just refresh LRU position
            self._gpu_lru.remove(domain)
            self._gpu_lru.append(domain)

        self._access_counter += 1
        return tbl

    def get_many(self, domains: Sequence[str]) -> Dict[str, NgramTable]:
        ordered_domains: List[str] = []
        seen = set()
        for domain in domains:
            if domain and domain not in seen:
                seen.add(domain)
                ordered_domains.append(domain)

        if not ordered_domains:
            return {}

        if self.table_placement == "cpu":
            return {domain: self.get(domain) for domain in ordered_domains}

        resident_requested = {domain for domain in ordered_domains if domain in self._gpu_lru}
        required_new_slots = len([domain for domain in ordered_domains if domain not in self._gpu_lru])

        while len(self._gpu_lru) - len(resident_requested) + required_new_slots > self.max_gpu_tables:
            victim = next((domain for domain in self._gpu_lru if domain not in resident_requested), None)
            if victim is None:
                logger.warning(
                    "Requested %d active tables, exceeding max_gpu_tables=%d; keeping all requested tables resident",
                    len(ordered_domains),
                    self.max_gpu_tables,
                )
                break
            self.offload(victim)

        return {domain: self.get(domain) for domain in ordered_domains}

    def release(self, domain: str) -> None:
        """Optional hint that *domain* is not immediately needed.

        Does NOT offload eagerly - just deprioritizes in LRU.
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

    def release_many(self, domains: Sequence[str]) -> None:
        for domain in domains:
            self.release(domain)

    def classify_and_build_refs_from_domains(
        self,
        domains: Sequence[str],
        token_ids: List[int],
        hidden_dim: int,
        domain_weights: Optional[Dict[str, float]] = None,
        prefer_first_hit: bool = False,
    ) -> Tuple[List[str], torch.Tensor, List[Optional[int]], Dict[Tuple, int], MultiTableRefStats]:
        tables = self.get_many(domains)
        ordered_domains = [domain for domain in domains if domain in tables]
        if len(ordered_domains) == 1:
            domain = ordered_domains[0]
            table = tables[domain]
            tiers, ref_acts, self_ref_sources, first_occurrence_map = table.classify_and_build_refs(
                token_ids,
                hidden_dim,
                output_device=self.gpu_device,
            )
            matched_domains: List[Optional[str]] = [None] * len(token_ids)
            domain_hits = {domain: 0}
            for index, tier in enumerate(tiers):
                if tier in {"trigram", "bigram"}:
                    matched_domains[index] = domain
                    domain_hits[domain] += 1
            return (
                tiers,
                ref_acts,
                self_ref_sources,
                first_occurrence_map,
                MultiTableRefStats(matched_domains=matched_domains, domain_hits=domain_hits),
            )

        if prefer_first_hit and ordered_domains:
            seq_len = len(token_ids)
            tiers: List[str] = []
            self_ref_sources: List[Optional[int]] = []
            first_occurrence_map: Dict[Tuple[int, int, int], int] = {}
            matched_domains: List[Optional[str]] = [None] * seq_len
            domain_hits = {domain: 0 for domain in ordered_domains}
            ref_positions: List[int] = []
            ref_hidden_batch: List[StoredHidden] = []

            for i in range(seq_len):
                trigram = None
                if i >= 2:
                    a, b, c = token_ids[i - 2], token_ids[i - 1], token_ids[i]
                    trigram = (a, b, c)

                    trigram_hit = False
                    for domain in ordered_domains:
                        table = tables[domain]
                        node_ab = table._get_node(a, b)
                        if node_ab is not None and c in node_ab.suffixes:
                            node_ab.hit_count += 1
                            node_ab.last_access = table._request_counter
                            tiers.append("trigram")
                            ref_positions.append(i)
                            ref_hidden_batch.append(node_ab.suffixes[c])
                            self_ref_sources.append(None)
                            first_occurrence_map.setdefault(trigram, i)
                            matched_domains[i] = domain
                            domain_hits[domain] += 1
                            trigram_hit = True
                            break
                    if trigram_hit:
                        continue

                    if trigram in first_occurrence_map:
                        tiers.append("self_ref")
                        self_ref_sources.append(first_occurrence_map[trigram])
                        continue

                if i >= 1:
                    b_tok, c_tok = token_ids[i - 1], token_ids[i]
                    bigram_hit = False
                    for domain in ordered_domains:
                        table = tables[domain]
                        node_bc = table._get_node(b_tok, c_tok)
                        if node_bc is not None:
                            node_bc.hit_count += 1
                            node_bc.last_access = table._request_counter
                            tiers.append("bigram")
                            ref_positions.append(i)
                            ref_hidden_batch.append(node_bc.bigram_hidden)
                            self_ref_sources.append(None)
                            if trigram is not None:
                                first_occurrence_map.setdefault(trigram, i)
                            matched_domains[i] = domain
                            domain_hits[domain] += 1
                            bigram_hit = True
                            break
                    if bigram_hit:
                        continue

                tiers.append("unigram")
                self_ref_sources.append(None)
                if trigram is not None:
                    first_occurrence_map.setdefault(trigram, i)

            ref_acts = torch.zeros(seq_len, hidden_dim, device=self.gpu_device, dtype=torch.float16)
            if ref_hidden_batch:
                converted = next(iter(tables.values()))._decode_stored_batch(ref_hidden_batch, self.gpu_device)
                idx = torch.tensor(ref_positions, dtype=torch.long, device=self.gpu_device)
                ref_acts[idx] = converted

            return (
                tiers,
                ref_acts,
                self_ref_sources,
                first_occurrence_map,
                MultiTableRefStats(matched_domains=matched_domains, domain_hits=domain_hits),
            )

        seq_len = len(token_ids)
        tiers: List[str] = []
        self_ref_sources: List[Optional[int]] = []
        first_occurrence_map: Dict[Tuple[int, int, int], int] = {}
        matched_domains: List[Optional[str]] = [None] * seq_len
        domain_hits = {domain: 0 for domain in ordered_domains}
        ref_positions: List[int] = []
        ref_tensors: List[torch.Tensor] = []
        domain_weights = domain_weights or {}

        def materialize_hidden(table: NgramTable, stored_hidden: StoredHidden) -> torch.Tensor:
            return table._decode_stored_batch([stored_hidden], self.gpu_device)[0]

        def fuse_candidates(candidates: List[Tuple[str, NgramTable, StoredHidden]]) -> Tuple[str, torch.Tensor]:
            if len(candidates) == 1:
                domain, table, stored_hidden = candidates[0]
                return domain, materialize_hidden(table, stored_hidden)

            weighted_refs: List[torch.Tensor] = []
            weights: List[float] = []
            for domain, table, stored_hidden in candidates:
                weighted_refs.append(materialize_hidden(table, stored_hidden))
                weight = domain_weights.get(domain, 1.0)
                stats = self._tables.get(domain)
                if stats is not None:
                    weight += min(0.2, math.log1p(stats._num_trigrams + stats._num_bigrams) * 0.01)
                weights.append(max(weight, 1e-4))

            weight_tensor = torch.tensor(weights, device=self.gpu_device, dtype=torch.float32)
            weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-8)
            ref_tensor = torch.stack(weighted_refs).to(torch.float32)
            fused = torch.sum(weight_tensor.unsqueeze(-1) * ref_tensor, dim=0).to(torch.float16)
            best_domain = candidates[int(torch.argmax(weight_tensor).item())][0]
            return best_domain, fused

        for i in range(seq_len):
            trigram = None
            if i >= 2:
                a, b, c = token_ids[i - 2], token_ids[i - 1], token_ids[i]
                trigram = (a, b, c)

                trigram_candidates: List[Tuple[str, NgramTable, StoredHidden]] = []
                for domain in ordered_domains:
                    table = tables[domain]
                    node_ab = table._get_node(a, b)
                    if node_ab is not None and c in node_ab.suffixes:
                        node_ab.hit_count += 1
                        node_ab.last_access = table._request_counter
                        trigram_candidates.append((domain, table, node_ab.suffixes[c]))
                if trigram_candidates:
                    best_domain, fused_ref = fuse_candidates(trigram_candidates)
                    tiers.append("trigram")
                    ref_positions.append(i)
                    ref_tensors.append(fused_ref)
                    self_ref_sources.append(None)
                    first_occurrence_map.setdefault(trigram, i)
                    matched_domains[i] = best_domain
                    domain_hits[best_domain] += 1
                    continue

                if trigram in first_occurrence_map:
                    tiers.append("self_ref")
                    self_ref_sources.append(first_occurrence_map[trigram])
                    continue

            if i >= 1:
                b_tok, c_tok = token_ids[i - 1], token_ids[i]
                bigram_candidates: List[Tuple[str, NgramTable, StoredHidden]] = []
                for domain in ordered_domains:
                    table = tables[domain]
                    node_bc = table._get_node(b_tok, c_tok)
                    if node_bc is not None:
                        node_bc.hit_count += 1
                        node_bc.last_access = table._request_counter
                        bigram_candidates.append((domain, table, node_bc.bigram_hidden))
                if bigram_candidates:
                    best_domain, fused_ref = fuse_candidates(bigram_candidates)
                    tiers.append("bigram")
                    ref_positions.append(i)
                    ref_tensors.append(fused_ref)
                    self_ref_sources.append(None)
                    if trigram is not None:
                        first_occurrence_map.setdefault(trigram, i)
                    matched_domains[i] = best_domain
                    domain_hits[best_domain] += 1
                    continue

            tiers.append("unigram")
            self_ref_sources.append(None)
            if trigram is not None:
                first_occurrence_map.setdefault(trigram, i)

        ref_acts = torch.zeros(seq_len, hidden_dim, device=self.gpu_device, dtype=torch.float16)
        if ref_tensors:
            idx = torch.tensor(ref_positions, dtype=torch.long, device=self.gpu_device)
            ref_acts[idx] = torch.stack(ref_tensors).to(device=self.gpu_device, dtype=torch.float16)

        return (
            tiers,
            ref_acts,
            self_ref_sources,
            first_occurrence_map,
            MultiTableRefStats(matched_domains=matched_domains, domain_hits=domain_hits),
        )

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
            "table_placement": self.table_placement,
            "storage_format": self.table_storage_format,
            "int8_group_size": self.int8_group_size,
            "int8_top_k": self.int8_top_k,
            "per_domain": per_domain,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evict_one_if_full(self) -> None:
        """Offload the LRU GPU table if at capacity (to free one slot)."""
        if self.table_placement == "cpu":
            return
        if len(self._gpu_lru) >= self.max_gpu_tables:
            victim = self._gpu_lru.pop(0)
            victim_tbl = self._tables.get(victim)
            if victim_tbl is not None:
                logger.info("Offloading domain '%s' table to CPU (%d tri, %d bi)",
                            victim, victim_tbl._num_trigrams, victim_tbl._num_bigrams)
                victim_tbl.to_device(self.cpu_device)
