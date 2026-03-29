from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence


FINAL_DELTA_STRATEGY = "delta_noaffine_int4_k1"
FINAL_UNIGRAM_STRATEGY = "unigram_int4_k4"


@dataclass(frozen=True)
class TransportPolicy:
    """Transport policy for the latency-first production path."""

    delta_strategy: str = FINAL_DELTA_STRATEGY
    unigram_strategy: str = FINAL_UNIGRAM_STRATEGY
    decode_use_raw_fp16: bool = True
    prefill_use_raw_fp16: bool = False
    track_transfer_bytes: bool = True

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        return {
            "delta_strategy": self.delta_strategy,
            "unigram_strategy": self.unigram_strategy,
            "decode_use_raw_fp16": self.decode_use_raw_fp16,
            "prefill_use_raw_fp16": self.prefill_use_raw_fp16,
            "track_transfer_bytes": self.track_transfer_bytes,
        }


@dataclass(frozen=True)
class BlockTableConfig:
    """Only the block-table backend is supported in v1."""

    table_placement: str = "cpu"
    pin_cpu_output_copy: bool = True
    enable_async_cpu_output_copy: bool = True
    gpu_hot_cache_entries: int = 0
    enable_disk_offload: bool = False
    disk_offload_dir: Optional[str] = None
    block_size: int = 256
    enable_async_paging: bool = True
    max_resident_blocks: int = 64
    pager_workers: int = 2
    pinned_block_budget: int = 8

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        return {
            "table_backend": "block",
            "table_placement": self.table_placement,
            "pin_cpu_output_copy": self.pin_cpu_output_copy,
            "enable_async_cpu_output_copy": self.enable_async_cpu_output_copy,
            "gpu_hot_cache_entries": self.gpu_hot_cache_entries,
            "enable_disk_offload": self.enable_disk_offload,
            "disk_offload_dir": self.disk_offload_dir,
            "block_size": self.block_size,
            "enable_async_block_paging": self.enable_async_paging,
            "max_resident_blocks": self.max_resident_blocks,
            "block_pager_workers": self.pager_workers,
            "pinned_block_budget": self.pinned_block_budget,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime knobs that affect model execution rather than transport policy."""

    layer_boundary: int = 6
    decode_tokens: int = 128
    max_seq_len: int = 4096
    max_table_entries: int = 100000
    group_size: int = 128
    top_k: int = 1
    int8_group_size: int = 128
    int8_outlier_top_k: int = 1
    extra_stop_token_ids: Sequence[int] = field(default_factory=tuple)

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        return {
            "layer_boundary": self.layer_boundary,
            "decode_tokens": self.decode_tokens,
            "max_seq_len": self.max_seq_len,
            "max_table_entries": self.max_table_entries,
            "group_size": self.group_size,
            "top_k": self.top_k,
            "int8_group_size": self.int8_group_size,
            "int8_outlier_top_k": self.int8_outlier_top_k,
            "extra_stop_token_ids": list(self.extra_stop_token_ids),
        }


@dataclass(frozen=True)
class V1PipelineConfig:
    """Full latency-first configuration surface for the v1 runtime."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    transport: TransportPolicy = field(default_factory=TransportPolicy)
    table: BlockTableConfig = field(default_factory=BlockTableConfig)

    def to_legacy_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        kwargs.update(self.runtime.to_legacy_kwargs())
        kwargs.update(self.transport.to_legacy_kwargs())
        kwargs.update(self.table.to_legacy_kwargs())
        return kwargs

    def summary(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["table"]["table_backend"] = "block"
        return payload