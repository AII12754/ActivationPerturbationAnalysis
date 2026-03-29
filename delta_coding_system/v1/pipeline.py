from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .config import BlockTableConfig, RuntimeConfig, TransportPolicy, V1PipelineConfig
from .base_runtime import V1RuntimeBase
from .decode_kernel import LatencyFirstDecodeKernel
from .prefill_kernel import LatencyFirstPrefillKernel


class LatencyFirstPipeline(LatencyFirstPrefillKernel, LatencyFirstDecodeKernel, V1RuntimeBase):
    """Standalone v1 runtime that owns the validated latency-first path."""

    def _warmup_transfer_kernels(self) -> None:
        if self._transfer_kernels_warmed or self.device.type != "cuda":
            return

        candidate_batches = [1, 8, 64, 256, self.max_seq_len]
        warm_batches = []
        seen = set()
        for batch_size in candidate_batches:
            normalized = max(1, min(int(batch_size), int(self.max_seq_len)))
            if normalized not in seen:
                seen.add(normalized)
                warm_batches.append(normalized)

        with torch.inference_mode():
            for batch_size in warm_batches:
                real = torch.randn(batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
                ref = torch.randn(batch_size, self.hidden_dim, device=self.device, dtype=torch.float16)
                self._encode_delta_batch(real, ref, include_ref_idx=True)
                self._encode_unigram_batch(real)
            scalar_real = torch.randn(1, self.hidden_dim, device=self.device, dtype=torch.float16)
            scalar_ref = torch.randn(1, self.hidden_dim, device=self.device, dtype=torch.float16)
            self._encode_prev_unigram_batch(scalar_real, scalar_ref)

        torch.cuda.synchronize(self.device)
        self._transfer_kernels_warmed = True

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[V1PipelineConfig] = None,
        device: Optional[torch.device] = None,
    ):
        resolved = config or V1PipelineConfig()
        self.v1_config = resolved
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            **resolved.to_legacy_kwargs(),
        )

    @classmethod
    def from_parts(
        cls,
        model,
        tokenizer,
        *,
        runtime: Optional[RuntimeConfig] = None,
        transport: Optional[TransportPolicy] = None,
        table: Optional[BlockTableConfig] = None,
        device: Optional[torch.device] = None,
    ) -> "LatencyFirstPipeline":
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=V1PipelineConfig(
                runtime=runtime or RuntimeConfig(),
                transport=transport or TransportPolicy(),
                table=table or BlockTableConfig(),
            ),
            device=device,
        )

    @classmethod
    def from_legacy_args(
        cls,
        model,
        tokenizer,
        args,
        device: Optional[torch.device] = None,
    ) -> "LatencyFirstPipeline":
        table = BlockTableConfig(
            table_placement=getattr(args, "table_placement", "cpu"),
            pin_cpu_output_copy=not getattr(args, "disable_pinned_cpu_table_copy", False),
            enable_async_cpu_output_copy=not getattr(args, "disable_async_cpu_table_copy", False),
            gpu_hot_cache_entries=getattr(args, "gpu_hot_cache_entries", 0),
            enable_disk_offload=getattr(args, "enable_disk_offload", False),
            disk_offload_dir=getattr(args, "disk_offload_dir", None),
            block_size=getattr(args, "block_size", 256),
            enable_async_paging=True,
            max_resident_blocks=getattr(args, "max_resident_blocks", 64),
            pager_workers=getattr(args, "block_pager_workers", 2),
            pinned_block_budget=getattr(args, "pinned_block_budget", 8),
        )
        runtime = RuntimeConfig(
            layer_boundary=getattr(args, "layer_boundary", 6),
            decode_tokens=getattr(args, "max_decode_tokens", getattr(args, "decode_tokens", 128)),
            max_seq_len=getattr(args, "max_seq_len", 4096),
            max_table_entries=getattr(args, "max_table_entries", 100000),
            group_size=getattr(args, "group_size", 128),
            top_k=getattr(args, "top_k", 1),
            int8_group_size=getattr(args, "group_size", 128),
            int8_outlier_top_k=1,
            extra_stop_token_ids=tuple(getattr(args, "extra_stop_token_ids", []) or []),
        )
        return cls.from_parts(
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
            transport=TransportPolicy(),
            table=table,
            device=device,
        )

    def policy_summary(self) -> Dict[str, Any]:
        return self.v1_config.summary()

    def _collect_table_stats(self, text: str) -> Dict[str, Any]:
        del text
        table_stats = self.table.stats if self.table is not None else {}
        if self.table is not None:
            table_stats["last_evicted"] = self.table._last_evicted
        return table_stats

    def process_request(
        self,
        text: str,
        phase: str = "test",
        task_name: Optional[str] = None,
    ):
        prefill_result, prefix_cache, suffix_cache, next_tok, input_ids, prefill_hidden, local_prompt_refs = self.process_prefill(
            text,
            phase=phase,
            task_name=task_name,
        )

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
        return prefill_result, decode_result, self._collect_table_stats(text)

    def process_request_v1(
        self,
        text: str,
        phase: str = "test",
        task_name: Optional[str] = None,
    ):
        return self.process_request(
            text=text,
            phase=phase,
            task_name=task_name,
        )


def build_latency_first_pipeline(
    model,
    tokenizer,
    *,
    config: Optional[V1PipelineConfig] = None,
    device: Optional[torch.device] = None,
) -> LatencyFirstPipeline:
    return LatencyFirstPipeline(model=model, tokenizer=tokenizer, config=config, device=device)