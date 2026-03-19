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
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from delta_coding_system.table import NgramTable
from delta_coding_system.codec import (
    DeltaPacket,
    compute_affine_params,
    apply_affine,
    compute_delta,
    compute_transfer_size,
    compute_transfer_size_int8_outlier,
    encode_decode_single,
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

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        self.table = NgramTable(
            device=device,
            dtype=table_dtype,
            max_entries=max_table_entries,
        )
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Import extraction helpers
        from activation_science.core.extraction import (
            ActivationBatch,
            decode_step,
            prefill,
            select_next_token,
        )
        self._prefill = prefill
        self._decode_step = decode_step
        self._select_next_token = select_next_token

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Prefill phase
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_prefill(
        self,
        text: str,
        phase: str = "test",
    ) -> Tuple[PrefillResult, Any, torch.Tensor, List[int], torch.Tensor]:
        """Process prefill phase with overlapped classify + forward.

        Returns
        -------
        result : PrefillResult
        past_kv : KV cache for decode
        next_tok : next token tensor for decode
        input_ids : token id list
        prefill_hidden : (seq_len, hidden_dim) hidden states
        """
        t_total_start = time.perf_counter()
        is_test = (phase == "test")

        # Tokenize
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        seq_len = len(input_ids)

        # 1. Launch classify async (CPU) — only needs token_ids
        classify_future = None
        if is_test:
            classify_future = self.executor.submit(
                self.table.classify_and_build_refs,
                input_ids, self.hidden_dim,
            )

        # 2. Prefill forward (GPU, concurrent with classify)
        torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        batch = self._prefill(self.model, input_tensor, use_cache=True)
        torch.cuda.synchronize()
        prefill_fwd_ms = (time.perf_counter() - t_fwd_start) * 1000.0

        layer_idx = min(self.layer_boundary, batch.num_layers)
        prefill_hidden = batch.hidden_states[layer_idx].squeeze(0).to(torch.float16)
        past_kv = batch.past_key_values
        next_tok = self._select_next_token(batch.last_logits, do_sample=False)
        del batch

        result = PrefillResult(seq_len=seq_len, prefill_fwd_ms=prefill_fwd_ms)

        if not is_test:
            # Warmup: just update table, no encoding
            t_upd_start = time.perf_counter()
            self.table.update_from_hidden_states(input_ids, prefill_hidden)
            result.table_update_ms = (time.perf_counter() - t_upd_start) * 1000.0
            result.total_ms = (time.perf_counter() - t_total_start) * 1000.0
            return result, past_kv, next_tok, input_ids, prefill_hidden

        # 3. Collect classify result
        t_classify_start = time.perf_counter()
        tiers, ref_acts, self_ref_sources, first_occ_map = classify_future.result()
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

        # 4. Encode all tiers — use CUDA events for timing (no intermediate syncs)
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

            scale, bias = compute_affine_params(real_batch, ref_batch)
            ref_t = apply_affine(ref_batch, scale, bias)
            delta = compute_delta(real_batch, ref_t)
            packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(
                delta, self.group_size, self.top_k,
            )
            dequant = groupwise_int4_dequantize_topk(
                packed, scales, zeros, tv, ti, self.group_size, self.hidden_dim,
            )
            recon_batch = reconstruct_activation(dequant, ref_batch, scale, bias).to(torch.float16)
            reconstructed[idx_t] = recon_batch

            n_delta = len(delta_indices)
            packet = DeltaPacket(
                quantized_data=packed, scales=scales, zero_points=zeros,
                topk_values=tv, topk_indices=ti,
                affine_scale=scale.to(torch.float16),
                affine_bias=bias.to(torch.float16),
                ref_indices=torch.zeros(n_delta, dtype=torch.long, device=self.device),
                group_size=self.group_size, top_k=self.top_k,
            )
            total_delta_bytes = compute_transfer_size(packet)
            if n_delta > 0:
                per_pos = total_delta_bytes / n_delta
                transfer_bytes_by_tier["trigram"] = int(per_pos * len(trigram_indices))
                transfer_bytes_by_tier["bigram"] = int(per_pos * len(bigram_indices))

        evt_after_delta.record()

        # 4b. Encode UNIGRAM (Int8 + outliers)
        if unigram_indices:
            idx_u = torch.tensor(unigram_indices, dtype=torch.long, device=self.device)
            real_uni = real_acts[idx_u]
            int8_pkt = groupwise_int8_quantize_topk(
                real_uni, self.int8_group_size, self.int8_outlier_top_k,
            )
            recon_uni = groupwise_int8_dequantize_topk(int8_pkt)
            transfer_bytes_by_tier["unigram"] = compute_transfer_size_int8_outlier(int8_pkt)
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

            scale, bias = compute_affine_params(real_sr, ref_sr)
            ref_t = apply_affine(ref_sr, scale, bias)
            delta = compute_delta(real_sr, ref_t)
            packed, scales, zeros, tv, ti = groupwise_int4_quantize_topk(
                delta, self.group_size, self.top_k,
            )
            dequant = groupwise_int4_dequantize_topk(
                packed, scales, zeros, tv, ti, self.group_size, self.hidden_dim,
            )
            recon_sr = reconstruct_activation(dequant, ref_sr, scale, bias).to(torch.float16)
            reconstructed[idx_sr] = recon_sr

            n_sr = len(sorted_self_ref)
            pkt = DeltaPacket(
                quantized_data=packed, scales=scales, zero_points=zeros,
                topk_values=tv, topk_indices=ti,
                affine_scale=scale.to(torch.float16),
                affine_bias=bias.to(torch.float16),
                ref_indices=torch.zeros(n_sr, dtype=torch.long, device=self.device),
                group_size=self.group_size, top_k=self.top_k,
            )
            transfer_bytes_by_tier["self_ref"] = compute_transfer_size(pkt)

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

        # 5. Table update (async — not on critical path)
        t_upd_start = time.perf_counter()
        update_future = self.executor.submit(
            self.table.update_from_hidden_states,
            input_ids, prefill_hidden,
        )
        # Wait for it (in production, this would overlap with network send)
        update_future.result()
        result.table_update_ms = (time.perf_counter() - t_upd_start) * 1000.0

        result.total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return result, past_kv, next_tok, input_ids, prefill_hidden

    # ------------------------------------------------------------------
    # Decode phase
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_decode(
        self,
        past_kv,
        next_tok: torch.Tensor,
        input_ids: List[int],
        prefill_hidden: torch.Tensor,
        phase: str = "test",
    ) -> DecodeResult:
        """Process decode phase with overlapped classify + forward per step.

        Parameters
        ----------
        past_kv : KV cache from prefill
        next_tok : (1,1) next token from prefill
        input_ids : prefill token ids
        prefill_hidden : (seq_len, hidden_dim) prefill hidden states
        phase : "warmup" or "test"
        """
        t_total_start = time.perf_counter()
        is_test = (phase == "test")
        decode_result = DecodeResult(decode_tokens=self.decode_tokens)

        running_token_ids: List[int] = list(input_ids)
        first_occ_map: Dict[Tuple[int, int, int], int] = {}
        reconstructed_hiddens: Dict[int, torch.Tensor] = {}
        decode_hidden_by_pos: Dict[int, torch.Tensor] = {}

        tier_counts = {"trigram": 0, "bigram": 0, "self_ref": 0, "unigram": 0}
        all_cosines: List[float] = []
        total_transfer_bytes = 0
        total_raw_bytes = 0

        pending_table_update: Optional[Future] = None

        # Pre-allocate CUDA events for decode loop timing
        evt_fwd_start = torch.cuda.Event(enable_timing=True)
        evt_fwd_end = torch.cuda.Event(enable_timing=True)
        evt_enc_start = torch.cuda.Event(enable_timing=True)
        evt_enc_end = torch.cuda.Event(enable_timing=True)

        for step in range(self.decode_tokens):
            tok_id = next_tok.item()
            running_token_ids.append(tok_id)
            decode_pos = len(running_token_ids) - 1

            # Launch classify (CPU, concurrent with forward)
            classify_future = self.executor.submit(
                self._classify_decode_step,
                running_token_ids, decode_pos, first_occ_map, reconstructed_hiddens,
            )

            # Model forward (GPU) — CUDA events, no sync
            evt_fwd_start.record()
            dbatch = self._decode_step(self.model, next_tok, past_kv)
            evt_fwd_end.record()

            h = dbatch.hidden_states[min(self.layer_boundary, dbatch.num_layers)][0, 0].to(torch.float16)
            decode_hidden_by_pos[decode_pos] = h

            # Collect classify result
            t_classify_start = time.perf_counter()
            tier, ref_h, raw_cos = classify_future.result()
            classify_ms = (time.perf_counter() - t_classify_start) * 1000.0

            # Encode (GPU, on critical path) — CUDA events, no sync
            real_h_2d = h.unsqueeze(0)
            evt_enc_start.record()
            recon, xfer_bytes = encode_decode_single(
                real_h_2d, ref_h, tier,
                self.group_size, self.top_k,
                self.int8_group_size, self.int8_outlier_top_k,
                self.hidden_dim, self.device,
            )
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

            # "Send" compressed data

            # Collect previous table update if pending
            if pending_table_update is not None:
                pending_table_update.result()

            # Table update (async, after send)
            t_table_start = time.perf_counter()
            pending_table_update = self.executor.submit(
                self._update_table_step,
                running_token_ids, decode_pos, h,
                prefill_hidden, input_ids, decode_hidden_by_pos,
            )
            table_update_ms = 0.0  # async, measured on collection

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

            past_kv = dbatch.past_key_values
            next_tok = self._select_next_token(dbatch.last_logits, do_sample=False)
            del dbatch

        # Collect final table update
        if pending_table_update is not None:
            pending_table_update.result()

        del past_kv, next_tok

        # Aggregate
        decode_result.num_trigram = tier_counts["trigram"]
        decode_result.num_bigram = tier_counts["bigram"]
        decode_result.num_self_ref = tier_counts["self_ref"]
        decode_result.num_unigram = tier_counts["unigram"]
        decode_result.total_transfer_bytes = total_transfer_bytes
        decode_result.raw_fp16_bytes = total_raw_bytes
        decode_result.compression_ratio = total_raw_bytes / max(total_transfer_bytes, 1)
        if all_cosines:
            decode_result.recon_cosine_mean = sum(all_cosines) / len(all_cosines)
            decode_result.recon_cosine_min = min(all_cosines)

        # Timing aggregation from step records
        if decode_result.step_records:
            decode_result.total_fwd_ms = sum(s.fwd_ms for s in decode_result.step_records)
            decode_result.total_classify_ms = sum(s.classify_ms for s in decode_result.step_records)
            decode_result.total_encode_ms = sum(s.encode_ms for s in decode_result.step_records)
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
    ) -> Tuple[str, Optional[torch.Tensor], float]:
        """Classify a single decode position. Runs on CPU thread."""
        tier = "unigram"
        ref_h = None
        raw_cos = 0.0

        if len(running_token_ids) >= 3:
            a = running_token_ids[-3]
            b = running_token_ids[-2]
            c = running_token_ids[-1]
            trigram_key = (a, b, c)

            # 1. Trigram table lookup
            tri_ref = self.table.get_trigram(a, b, c)
            if tri_ref is not None:
                tier = "trigram"
                ref_h = tri_ref.unsqueeze(0).to(torch.float16)
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
                    bi_ref = self.table.get_bigram(b_tok, c_tok)
                    if bi_ref is not None:
                        tier = "bigram"
                        ref_h = bi_ref.unsqueeze(0).to(torch.float16)
                first_occ_map.setdefault(trigram_key, decode_pos)

        return tier, ref_h, raw_cos

    def _update_table_step(
        self,
        running_token_ids: List[int],
        decode_pos: int,
        h: torch.Tensor,
        prefill_hidden: torch.Tensor,
        input_ids: List[int],
        decode_hidden_by_pos: Dict[int, torch.Tensor],
    ) -> None:
        """Update table with a single decode step's trigram. Runs on CPU thread."""
        if len(running_token_ids) >= 3:
            a = running_token_ids[-3]
            b = running_token_ids[-2]
            c = running_token_ids[-1]
            if not self.table.has_trigram(a, b, c):
                b_abs_pos = decode_pos - 1
                if b_abs_pos < len(input_ids):
                    bi_hidden = prefill_hidden[b_abs_pos]
                elif b_abs_pos in decode_hidden_by_pos:
                    bi_hidden = decode_hidden_by_pos[b_abs_pos]
                else:
                    bi_hidden = h
                node = self.table._get_or_create_node(
                    a, b, bi_hidden.to(self.table.dtype).detach(),
                )
                if c not in node.suffixes:
                    node.suffixes[c] = h.to(self.table.dtype).detach()
                    self.table._num_trigrams += 1

    # ------------------------------------------------------------------
    # Full request processing
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def process_request(
        self,
        text: str,
        phase: str = "test",
    ) -> Tuple[PrefillResult, DecodeResult, Dict[str, Any]]:
        """Process a complete request: prefill + decode.

        Returns (prefill_result, decode_result, table_stats).
        """
        prefill_result, past_kv, next_tok, input_ids, prefill_hidden = \
            self.process_prefill(text, phase=phase)

        decode_result = self.process_decode(
            past_kv, next_tok, input_ids, prefill_hidden, phase=phase,
        )

        table_stats = self.table.stats
        table_stats["last_evicted"] = self.table._last_evicted

        return prefill_result, decode_result, table_stats
