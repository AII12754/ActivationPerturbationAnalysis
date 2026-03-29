from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from delta_coding_system.codec import Int8OutlierPacket, groupwise_int8_dequantize_topk, groupwise_int8_quantize_topk

logger = logging.getLogger(__name__)


StoredHidden = Union[torch.Tensor, Int8OutlierPacket]


@dataclass
class _EntryBlock:
    kind: str
    entries: List[Tuple[Tuple[int, ...], StoredHidden]]
    last_access: int = 0
    hit_count: int = 0


class BlockTable:
    """Production-oriented block activation table with non-blocking cold paging."""

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
        block_size: int = 256,
        enable_async_paging: bool = False,
        page_directory: Optional[str] = None,
        max_resident_blocks: int = 0,
        pager_workers: int = 1,
        pinned_block_budget: int = 2,
    ):
        self.device = device
        self.dtype = dtype
        self.max_entries = max_entries
        self.storage_format = storage_format
        self.int8_group_size = int8_group_size
        self.int8_top_k = int8_top_k
        self.pin_cpu_output_copy = pin_cpu_output_copy
        self.enable_async_cpu_output_copy = enable_async_cpu_output_copy
        self.gpu_hot_cache_entries = max(0, gpu_hot_cache_entries)
        self.gpu_hot_cache_device = gpu_hot_cache_device
        self.block_size = max(16, block_size)
        self.enable_async_paging = enable_async_paging
        self.page_directory = Path(page_directory).expanduser() if page_directory else None
        self.max_resident_blocks = max(0, max_resident_blocks)
        self.pager_workers = max(1, pager_workers)
        self.pinned_block_budget = max(0, pinned_block_budget)

        self._request_counter = 0
        self._last_evicted = 0
        self._last_classify_stats: Dict[str, Any] = {}
        self._trigram_blocks: List[_EntryBlock] = []
        self._bigram_blocks: List[_EntryBlock] = []
        self._trigram_index: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
        self._bigram_index: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._current_trigram_block: Optional[int] = None
        self._current_bigram_block: Optional[int] = None
        self._paged_trigram_blocks: Set[int] = set()
        self._paged_bigram_blocks: Set[int] = set()
        self._page_futures: Dict[Tuple[str, int], Future] = {}
        self._pinned_blocks: List[Tuple[str, int]] = []
        self._pager_executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.RLock()
        self._pager_stats = {
            "cold_miss_enqueued": 0,
            "load_completed": 0,
            "load_failed": 0,
            "spill_count": 0,
        }
        self._num_trigrams = 0
        self._num_bigrams = 0
        self.gpu_hot_cache: Optional[BlockTable] = None

        if self.enable_async_paging:
            if self.page_directory is None:
                self.page_directory = Path(".cache") / "block_table_pages"
            self.page_directory.mkdir(parents=True, exist_ok=True)
            self._pager_executor = ThreadPoolExecutor(max_workers=self.pager_workers, thread_name_prefix="block-pager")

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

    def _move_int8_packet(self, packet: Int8OutlierPacket, device: torch.device, non_blocking: bool) -> Int8OutlierPacket:
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
            return [self._prepare_stored_hidden(hidden_states[i].to(dtype=torch.float16)) for i in range(hidden_states.shape[0])]

        packet = groupwise_int8_quantize_topk(hidden_states.to(dtype=torch.float16), self.int8_group_size, self.int8_top_k)
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
        self.gpu_hot_cache = BlockTable(
            device=device,
            dtype=self.dtype,
            max_entries=max_entries,
            storage_format=self.storage_format,
            int8_group_size=self.int8_group_size,
            int8_top_k=self.int8_top_k,
            block_size=self.block_size,
            enable_async_paging=False,
        )

    def disable_gpu_hot_cache(self) -> None:
        self.gpu_hot_cache = None

    def shutdown(self) -> None:
        if self.gpu_hot_cache is not None:
            self.gpu_hot_cache.shutdown()
        if self._pager_executor is not None:
            self._pager_executor.shutdown(wait=True)
            self._pager_executor = None

    def _touch_block(self, block: _EntryBlock) -> None:
        block.hit_count += 1
        block.last_access = self._request_counter

    def _pin_block(self, kind: str, block_id: int) -> None:
        if self.pinned_block_budget <= 0:
            return
        key = (kind, block_id)
        if key in self._pinned_blocks:
            self._pinned_blocks.remove(key)
        self._pinned_blocks.append(key)
        if len(self._pinned_blocks) > self.pinned_block_budget:
            self._pinned_blocks.pop(0)

    def _allocate_block(self, kind: str) -> int:
        block = _EntryBlock(kind=kind, entries=[])
        if kind == "trigram":
            self._trigram_blocks.append(block)
            return len(self._trigram_blocks) - 1
        self._bigram_blocks.append(block)
        return len(self._bigram_blocks) - 1

    def _current_block(self, kind: str) -> Tuple[List[_EntryBlock], Optional[int]]:
        if kind == "trigram":
            return self._trigram_blocks, self._current_trigram_block
        return self._bigram_blocks, self._current_bigram_block

    def _set_current_block(self, kind: str, block_id: Optional[int]) -> None:
        if kind == "trigram":
            self._current_trigram_block = block_id
        else:
            self._current_bigram_block = block_id

    def _paged_set(self, kind: str) -> Set[int]:
        return self._paged_trigram_blocks if kind == "trigram" else self._paged_bigram_blocks

    def _blocks(self, kind: str) -> List[_EntryBlock]:
        return self._trigram_blocks if kind == "trigram" else self._bigram_blocks

    def _page_path(self, kind: str, block_id: int) -> Path:
        assert self.page_directory is not None
        return self.page_directory / f"{kind}_{block_id}.pt"

    def _block_is_resident(self, kind: str, block_id: int) -> bool:
        blocks = self._blocks(kind)
        if block_id >= len(blocks):
            return False
        return bool(blocks[block_id].entries) and block_id not in self._paged_set(kind)

    def _resident_block_count(self) -> int:
        total = 0
        for kind in ("trigram", "bigram"):
            paged = self._paged_set(kind)
            for block_id, block in enumerate(self._blocks(kind)):
                if block.entries and block_id not in paged:
                    total += 1
        return total

    def _load_block_payload(self, path: Path) -> _EntryBlock:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)

    def _schedule_block_load(self, kind: str, block_id: int) -> None:
        if not self.enable_async_paging or self.page_directory is None or self._pager_executor is None:
            return
        key = (kind, block_id)
        if key in self._page_futures:
            return
        path = self._page_path(kind, block_id)
        if not path.exists():
            return
        self._page_futures[key] = self._pager_executor.submit(self._load_block_payload, path)
        self._pager_stats["cold_miss_enqueued"] += 1

    def _maybe_activate_loaded_block(self, kind: str, block_id: int) -> bool:
        key = (kind, block_id)
        future = self._page_futures.get(key)
        if future is None or not future.done():
            return False
        try:
            block = future.result()
        except Exception:
            logger.exception("Failed to page in %s block %d", kind, block_id)
            self._pager_stats["load_failed"] += 1
            del self._page_futures[key]
            return False
        del self._page_futures[key]
        blocks = self._blocks(kind)
        blocks[block_id] = block
        self._paged_set(kind).discard(block_id)
        self._pin_block(kind, block_id)
        self._page_path(kind, block_id).unlink(missing_ok=True)
        self._pager_stats["load_completed"] += 1
        self._spill_cold_blocks_if_needed()
        return True

    def _spill_block_to_disk(self, kind: str, block_id: int) -> bool:
        if not self.enable_async_paging or self.page_directory is None:
            return False
        blocks = self._blocks(kind)
        if block_id >= len(blocks):
            return False
        block = blocks[block_id]
        if not block.entries or block_id in self._paged_set(kind):
            return False
        torch.save(block, self._page_path(kind, block_id))
        blocks[block_id] = _EntryBlock(kind=kind, entries=[], last_access=block.last_access, hit_count=block.hit_count)
        self._paged_set(kind).add(block_id)
        key = (kind, block_id)
        if key in self._pinned_blocks:
            self._pinned_blocks.remove(key)
        self._pager_stats["spill_count"] += 1
        return True

    def _spill_cold_blocks_if_needed(self) -> None:
        if not self.enable_async_paging or self.max_resident_blocks <= 0:
            return
        while self._resident_block_count() > self.max_resident_blocks:
            candidates: List[Tuple[float, str, int]] = []
            for kind in ("trigram", "bigram"):
                current = self._current_trigram_block if kind == "trigram" else self._current_bigram_block
                paged = self._paged_set(kind)
                for block_id, block in enumerate(self._blocks(kind)):
                    if not block.entries or block_id in paged or block_id == current or (kind, block_id) in self._pinned_blocks:
                        continue
                    score = block.hit_count * 10 + block.last_access
                    candidates.append((float(score), kind, block_id))
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])
            _score, kind, block_id = candidates[0]
            if not self._spill_block_to_disk(kind, block_id):
                break

    def _get_or_create_append_block(self, kind: str) -> Tuple[List[_EntryBlock], int]:
        blocks, current_id = self._current_block(kind)
        if current_id is None or current_id >= len(blocks) or len(blocks[current_id].entries) >= self.block_size or current_id in self._paged_set(kind):
            current_id = self._allocate_block(kind)
            blocks, _ = self._current_block(kind)
            self._set_current_block(kind, current_id)
        return blocks, current_id

    def _insert_entry(self, kind: str, key: Tuple[int, ...], stored_hidden: StoredHidden) -> bool:
        if kind == "trigram":
            tri_key = (int(key[0]), int(key[1]), int(key[2]))
            if tri_key in self._trigram_index:
                return False
        else:
            bi_key = (int(key[0]), int(key[1]))
            if bi_key in self._bigram_index:
                return False

        with self._lock:
            blocks, block_id = self._get_or_create_append_block(kind)
            block = blocks[block_id]
            slot = len(block.entries)
            block.entries.append((key, stored_hidden))
            block.last_access = self._request_counter
            if kind == "trigram":
                tri_key = (int(key[0]), int(key[1]), int(key[2]))
                self._trigram_index[tri_key] = (block_id, slot)
                self._num_trigrams += 1
            else:
                bi_key = (int(key[0]), int(key[1]))
                self._bigram_index[bi_key] = (block_id, slot)
                self._num_bigrams += 1
            self._spill_cold_blocks_if_needed()
            return True

    def _get_entry(self, kind: str, key: Tuple[int, ...]) -> Optional[StoredHidden]:
        with self._lock:
            if kind == "trigram":
                tri_key = (int(key[0]), int(key[1]), int(key[2]))
                addr = self._trigram_index.get(tri_key)
            else:
                bi_key = (int(key[0]), int(key[1]))
                addr = self._bigram_index.get(bi_key)
            if addr is None:
                return None
            block_id, slot = addr
            self._maybe_activate_loaded_block(kind, block_id)
            if not self._block_is_resident(kind, block_id):
                self._schedule_block_load(kind, block_id)
                return None
            block = self._blocks(kind)[block_id]
            self._touch_block(block)
            self._pin_block(kind, block_id)
            return block.entries[slot][1]

    def has_trigram(self, a: int, b: int, c: int) -> bool:
        return (a, b, c) in self._trigram_index

    def get_trigram(self, a: int, b: int, c: int) -> Optional[StoredHidden]:
        return self._get_entry("trigram", (a, b, c))

    def get_bigram(self, a: int, b: int) -> Optional[StoredHidden]:
        return self._get_entry("bigram", (a, b))

    @torch.inference_mode()
    def build_from_corpus(self, model, token_ids_list: List[List[int]], layer_idx: int, batch_size: int = 128) -> Dict[str, Any]:
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

        for start in range(0, len(trigram_list), batch_size):
            batch_tris = trigram_list[start : start + batch_size]
            input_tensor = torch.tensor(batch_tris, dtype=torch.long, device=self.device)
            outputs = model(input_ids=input_tensor, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[layer_idx]
            tri_hidden_list = self._encode_hidden_batch_for_storage(hidden[:, -1, :])
            bi_hidden_list = self._encode_hidden_batch_for_storage(hidden[:, -2, :])
            for j, tri in enumerate(batch_tris):
                a, b, c = tri
                self._insert_entry("bigram", (a, b), bi_hidden_list[j])
                self._insert_entry("trigram", (a, b, c), tri_hidden_list[j])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {"num_trigrams": self._num_trigrams, "num_bigrams": self._num_bigrams, "build_time_ms": elapsed_ms}

    @torch.inference_mode()
    def update_from_request(self, model, token_ids: List[int], layer_idx: int, batch_size: int = 128) -> Tuple[int, int, float]:
        t0 = time.perf_counter()
        old_tri = self._num_trigrams
        old_bi = self._num_bigrams
        new_trigrams: Dict[Tuple[int, int, int], None] = {}
        for i in range(2, len(token_ids)):
            tri = (token_ids[i - 2], token_ids[i - 1], token_ids[i])
            if not self.has_trigram(*tri) and tri not in new_trigrams:
                new_trigrams[tri] = None
        if new_trigrams:
            self.build_from_corpus(model, [list(tri) for tri in new_trigrams.keys()], layer_idx, batch_size)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._num_trigrams - old_tri, self._num_bigrams - old_bi, elapsed_ms

    def classify_and_build_refs(self, token_ids: List[int], hidden_dim: int, output_device: Optional[torch.device] = None) -> Tuple[List[str], torch.Tensor, List[Optional[int]], Dict[Tuple, int]]:
        if output_device is None:
            output_device = self.device

        seq_len = len(token_ids)
        tiers: List[str] = []
        self_ref_sources: List[Optional[int]] = []
        first_occurrence_map: Dict[Tuple[int, int, int], int] = {}
        ref_positions_gpu: List[int] = []
        ref_tensors_gpu: List[StoredHidden] = []
        ref_positions_cpu: List[int] = []
        ref_tensors_cpu: List[StoredHidden] = []
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
                        ref_positions_gpu.append(i)
                        ref_tensors_gpu.append(hot_tri_ref)
                        self_ref_sources.append(None)
                        first_occurrence_map.setdefault(trigram, i)
                        gpu_hot_hits += 1
                        continue

                tri_ref = self.get_trigram(a, b, c)
                if tri_ref is not None:
                    tiers.append("trigram")
                    if self.device.type == "cuda":
                        ref_positions_gpu.append(i)
                        ref_tensors_gpu.append(tri_ref)
                    else:
                        ref_positions_cpu.append(i)
                        ref_tensors_cpu.append(tri_ref)
                    self_ref_sources.append(None)
                    first_occurrence_map.setdefault(trigram, i)
                    continue

                if trigram in first_occurrence_map:
                    tiers.append("self_ref")
                    self_ref_sources.append(first_occurrence_map[trigram])
                    continue

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

                bi_ref = self.get_bigram(b_tok, c_tok)
                if bi_ref is not None:
                    tiers.append("bigram")
                    if self.device.type == "cuda":
                        ref_positions_gpu.append(i)
                        ref_tensors_gpu.append(bi_ref)
                    else:
                        ref_positions_cpu.append(i)
                        ref_tensors_cpu.append(bi_ref)
                    self_ref_sources.append(None)
                    if trigram is not None:
                        first_occurrence_map.setdefault(trigram, i)
                    continue

            tiers.append("unigram")
            self_ref_sources.append(None)
            if trigram is not None:
                first_occurrence_map.setdefault(trigram, i)

        lookup_ms = (time.perf_counter() - lookup_start) * 1000.0
        materialize_start = time.perf_counter()
        ref_acts = torch.zeros(seq_len, hidden_dim, device=output_device, dtype=torch.float16)
        if ref_tensors_gpu:
            converted = self._decode_stored_batch(ref_tensors_gpu, output_device)
            idx = torch.tensor(ref_positions_gpu, dtype=torch.long, device=output_device)
            ref_acts[idx] = converted
        if ref_tensors_cpu:
            converted = self._decode_stored_batch(ref_tensors_cpu, output_device)
            idx = torch.tensor(ref_positions_cpu, dtype=torch.long, device=output_device)
            ref_acts[idx] = converted

        self._last_classify_stats = {
            "table_device": str(self.device),
            "output_device": str(output_device),
            "lookup_ms": lookup_ms,
            "materialize_ms": (time.perf_counter() - materialize_start) * 1000.0,
            "num_refs": len(ref_tensors_gpu) + len(ref_tensors_cpu),
            "gpu_hot_hits": gpu_hot_hits,
            "seq_len": seq_len,
            "backend": "block",
            "pager": dict(self._pager_stats),
            "resident_blocks": self._resident_block_count(),
            "paged_blocks": len(self._paged_trigram_blocks) + len(self._paged_bigram_blocks),
        }
        return tiers, ref_acts, self_ref_sources, first_occurrence_map

    def update_from_hidden_states(self, token_ids: List[int], hidden_states: torch.Tensor) -> Tuple[int, int, float]:
        t0 = time.perf_counter()
        old_tri = self._num_trigrams
        old_bi = self._num_bigrams
        hidden_list = self._encode_hidden_batch_for_storage(hidden_states)
        for i in range(len(token_ids)):
            if i >= 1:
                self._insert_entry("bigram", (token_ids[i - 1], token_ids[i]), hidden_list[i])
            if i >= 2:
                self._insert_entry("trigram", (token_ids[i - 2], token_ids[i - 1], token_ids[i]), hidden_list[i])
        self._request_counter += 1
        self._last_evicted = self.evict()
        if self.gpu_hot_cache is not None:
            self.gpu_hot_cache.update_from_hidden_states(token_ids, hidden_states)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._num_trigrams - old_tri, self._num_bigrams - old_bi, elapsed_ms

    def update_with_decode_step(self, a: int, b: int, c: int, bigram_hidden: torch.Tensor, trigram_hidden: torch.Tensor) -> None:
        if self.has_trigram(a, b, c):
            return
        stored_bigram = self._encode_hidden_batch_for_storage(bigram_hidden.unsqueeze(0))[0]
        stored_trigram = self._encode_hidden_batch_for_storage(trigram_hidden.unsqueeze(0))[0]
        self._insert_entry("bigram", (a, b), stored_bigram)
        self._insert_entry("trigram", (a, b, c), stored_trigram)
        self._request_counter += 1
        self._last_evicted = self.evict()

    def _evict_block(self, kind: str, block_id: int) -> int:
        blocks = self._blocks(kind)
        if block_id >= len(blocks):
            return 0
        block = blocks[block_id]
        removed = len(block.entries)
        if kind == "trigram":
            for key, _value in block.entries:
                self._trigram_index.pop((int(key[0]), int(key[1]), int(key[2])), None)
        else:
            for key, _value in block.entries:
                self._bigram_index.pop((int(key[0]), int(key[1])), None)
        self._paged_set(kind).discard(block_id)
        if self.page_directory is not None:
            self._page_path(kind, block_id).unlink(missing_ok=True)
        blocks[block_id] = _EntryBlock(kind=kind, entries=[])
        current_id = self._current_trigram_block if kind == "trigram" else self._current_bigram_block
        if current_id == block_id:
            self._set_current_block(kind, None)
        return removed

    def evict(self) -> int:
        total = self._num_trigrams + self._num_bigrams
        if self.max_entries <= 0 or total <= self.max_entries:
            return 0

        target = int(self.max_entries * 0.9)
        scored_blocks: List[Tuple[float, str, int, int]] = []
        for kind in ("trigram", "bigram"):
            for block_id, block in enumerate(self._blocks(kind)):
                if block.entries:
                    block_entries = len(block.entries)
                elif block_id in self._paged_set(kind):
                    values = self._trigram_index.values() if kind == "trigram" else self._bigram_index.values()
                    block_entries = sum(1 for (bid, _slot) in values if bid == block_id)
                else:
                    block_entries = 0
                if block_entries <= 0:
                    continue
                score = block.hit_count * 10 + block.last_access
                scored_blocks.append((float(score), kind, block_id, block_entries))
        scored_blocks.sort(key=lambda item: item[0])

        evicted = 0
        for _score, kind, block_id, block_entries in scored_blocks:
            if total - evicted <= target:
                break
            removed = self._evict_block(kind, block_id)
            evicted += removed
            if kind == "trigram":
                self._num_trigrams -= removed
            else:
                self._num_bigrams -= removed
            if removed != block_entries:
                logger.warning("BlockTable eviction removed %d entries, expected %d", removed, block_entries)
        return evicted

    def _materialize_all_paged_blocks(self) -> None:
        if not self.enable_async_paging or self.page_directory is None:
            return
        for kind in ("trigram", "bigram"):
            for block_id in list(self._paged_set(kind)):
                self._schedule_block_load(kind, block_id)
                future = self._page_futures.get((kind, block_id))
                if future is None:
                    continue
                future.result()
                self._maybe_activate_loaded_block(kind, block_id)

    def to_device(self, device: torch.device) -> "BlockTable":
        if device == self.device:
            return self
        self._materialize_all_paged_blocks()
        for blocks in (self._trigram_blocks, self._bigram_blocks):
            for block in blocks:
                if not block.entries:
                    continue
                moved_entries: List[Tuple[Tuple[int, ...], StoredHidden]] = []
                for key, hidden in block.entries:
                    if isinstance(hidden, torch.Tensor):
                        moved_h = hidden.to(device, non_blocking=True)
                        if device.type == "cpu" and self.pin_cpu_output_copy and not moved_h.is_pinned():
                            moved_h = moved_h.pin_memory()
                    else:
                        moved_h = self._move_int8_packet(hidden, device, non_blocking=True)
                        if device.type == "cpu" and self.pin_cpu_output_copy:
                            moved_h = self._pin_int8_packet(moved_h)
                    moved_entries.append((key, moved_h))
                block.entries = moved_entries
        self.device = device
        return self

    def export_state(self) -> Dict[str, Any]:
        self._materialize_all_paged_blocks()
        if self.device.type != "cpu":
            self.to_device(torch.device("cpu"))
        return {
            "trigram_blocks": self._trigram_blocks,
            "bigram_blocks": self._bigram_blocks,
            "trigram_index": self._trigram_index,
            "bigram_index": self._bigram_index,
            "current_trigram_block": self._current_trigram_block,
            "current_bigram_block": self._current_bigram_block,
            "num_trigrams": self._num_trigrams,
            "num_bigrams": self._num_bigrams,
            "request_counter": self._request_counter,
            "last_evicted": self._last_evicted,
            "block_size": self.block_size,
            "enable_async_paging": self.enable_async_paging,
            "max_resident_blocks": self.max_resident_blocks,
            "pager_workers": self.pager_workers,
            "pinned_block_budget": self.pinned_block_budget,
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        self._trigram_blocks = payload.get("trigram_blocks", [])
        self._bigram_blocks = payload.get("bigram_blocks", [])
        self._trigram_index = payload.get("trigram_index", {})
        self._bigram_index = payload.get("bigram_index", {})
        self._current_trigram_block = payload.get("current_trigram_block")
        self._current_bigram_block = payload.get("current_bigram_block")
        self._num_trigrams = int(payload.get("num_trigrams", 0))
        self._num_bigrams = int(payload.get("num_bigrams", 0))
        self._request_counter = int(payload.get("request_counter", 0))
        self._last_evicted = int(payload.get("last_evicted", 0))
        self.block_size = int(payload.get("block_size", self.block_size))
        self.enable_async_paging = bool(payload.get("enable_async_paging", self.enable_async_paging))
        self.max_resident_blocks = int(payload.get("max_resident_blocks", self.max_resident_blocks))
        self.pager_workers = int(payload.get("pager_workers", self.pager_workers))
        self.pinned_block_budget = int(payload.get("pinned_block_budget", self.pinned_block_budget))
        self._paged_trigram_blocks = set()
        self._paged_bigram_blocks = set()
        self._page_futures = {}
        self._pinned_blocks = []
        self._last_classify_stats = {}
        if self.enable_async_paging and self._pager_executor is None:
            if self.page_directory is None:
                self.page_directory = Path(".cache") / "block_table_pages"
            self.page_directory.mkdir(parents=True, exist_ok=True)
            self._pager_executor = ThreadPoolExecutor(max_workers=self.pager_workers, thread_name_prefix="block-pager")
        if self.device.type != "cpu":
            self.to_device(self.device)

    @property
    def stats(self) -> Dict[str, Any]:
        sample: Optional[StoredHidden] = None
        for blocks in (self._trigram_blocks, self._bigram_blocks):
            for block in blocks:
                if block.entries:
                    sample = block.entries[0][1]
                    break
            if sample is not None:
                break
        if sample is None:
            per_entry_bytes = 0
        elif isinstance(sample, torch.Tensor):
            per_entry_bytes = sample.nelement() * sample.element_size()
        else:
            per_entry_bytes = self._packet_num_bytes(sample)
        return {
            "num_trigrams": self._num_trigrams,
            "num_bigrams": self._num_bigrams,
            "memory_bytes": (self._num_trigrams + self._num_bigrams) * per_entry_bytes,
            "device": str(self.device),
            "storage_format": self.storage_format,
            "int8_group_size": self.int8_group_size,
            "int8_top_k": self.int8_top_k,
            "gpu_hot_cache": None if self.gpu_hot_cache is None else self.gpu_hot_cache.stats,
            "backend": "block",
            "block_size": self.block_size,
            "enable_async_paging": self.enable_async_paging,
            "max_resident_blocks": self.max_resident_blocks,
            "pinned_block_budget": self.pinned_block_budget,
            "resident_blocks": self._resident_block_count(),
            "paged_blocks": len(self._paged_trigram_blocks) + len(self._paged_bigram_blocks),
            "num_trigram_blocks": len(self._trigram_blocks),
            "num_bigram_blocks": len(self._bigram_blocks),
            "pager": dict(self._pager_stats),
        }

    @property
    def last_classify_stats(self) -> Dict[str, Any]:
        return dict(self._last_classify_stats)
