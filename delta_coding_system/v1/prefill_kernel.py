from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache

from .results import PrefillResult


class LatencyFirstPrefillKernel:
    """v1-owned prefill kernel for the latency-first runtime."""

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

    @torch.inference_mode()
    def process_prefill(
        self,
        text: str,
        phase: str = "test",
        task_name: Optional[str] = None,
    ) -> Tuple[
        PrefillResult,
        DynamicCache,
        DynamicCache,
        torch.Tensor,
        List[int],
        torch.Tensor,
        Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor]],
    ]:
        t_total_start = time.perf_counter()
        is_test = phase == "test"

        torch.cuda.set_device(self.device)

        if self._pending_prefill_update is not None:
            self._pending_prefill_update.result()
            self._pending_prefill_update = None
        self._drain_decode_updates(wait=False)

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        seq_len = len(input_ids)

        # Submit CPU classify early so it overlaps with GPU prefix forward.
        classify_future = None
        if is_test:
            classify_future = self.classify_executor.submit(
                self.table.classify_and_build_refs,
                input_ids,
                self.hidden_dim,
                self.device,
            )

        torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        prefill_hidden, prefix_cache = self._run_prefix_prefill(input_tensor)
        torch.cuda.synchronize()
        prefill_fwd_ms = (time.perf_counter() - t_fwd_start) * 1000.0

        result = PrefillResult(seq_len=seq_len, prefill_fwd_ms=prefill_fwd_ms)

        if not is_test:
            next_logits, suffix_cache = self._run_suffix_prefill(prefill_hidden)
            next_tok = self._select_next_token(next_logits, do_sample=False)
            t_upd_start = time.perf_counter()
            self.table.update_from_hidden_states(input_ids, prefill_hidden)
            result.table_update_ms = (time.perf_counter() - t_upd_start) * 1000.0
            result.total_ms = (time.perf_counter() - t_total_start) * 1000.0
            result.reconstructed_hidden = prefill_hidden
            local_prompt_refs = self._build_local_prompt_refs(input_ids, prefill_hidden)
            return result, prefix_cache, suffix_cache, next_tok, input_ids, prefill_hidden, local_prompt_refs

        t_classify_start = time.perf_counter()
        tiers, ref_acts, self_ref_sources, first_occ_map = classify_future.result()
        classify_ms = (time.perf_counter() - t_classify_start) * 1000.0
        result.classify_ms = classify_ms
        del first_occ_map

        # While CPU was classifying, the GPU prefix forward already finished.
        # Now use a dedicated stream to start loading matched reference hidden
        # states to HBM, overlapping with the Python-side tier indexing below.
        if self.hidden_load_stream is not None and ref_acts is not None:
            with torch.cuda.stream(self.hidden_load_stream):
                ref_acts = ref_acts.to(device=self.device, non_blocking=True)

        trigram_indices = [i for i, tier in enumerate(tiers) if tier == "trigram"]
        bigram_indices = [i for i, tier in enumerate(tiers) if tier == "bigram"]
        self_ref_indices = [i for i, tier in enumerate(tiers) if tier == "self_ref"]
        unigram_indices = [i for i, tier in enumerate(tiers) if tier == "unigram"]

        result.num_trigram = len(trigram_indices)
        result.num_bigram = len(bigram_indices)
        result.num_self_ref = len(self_ref_indices)
        result.num_unigram = len(unigram_indices)

        real_acts = prefill_hidden
        reconstructed = torch.zeros_like(real_acts)
        transfer_bytes_by_tier: Dict[str, int] = {
            "trigram": 0,
            "bigram": 0,
            "self_ref": 0,
            "unigram": 0,
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
            # Wait for ref_acts to arrive on GPU if the stream copy is in flight.
            if self.hidden_load_stream is not None:
                torch.cuda.current_stream().wait_stream(self.hidden_load_stream)

            torch.cuda.synchronize()
            evt_enc_start = torch.cuda.Event(enable_timing=True)
            evt_after_delta = torch.cuda.Event(enable_timing=True)
            evt_after_unigram = torch.cuda.Event(enable_timing=True)
            evt_after_self_ref = torch.cuda.Event(enable_timing=True)
            evt_enc_start.record()

            delta_indices = trigram_indices + bigram_indices
            if delta_indices:
                idx_t = torch.tensor(delta_indices, dtype=torch.long, device=self.device)
                real_batch = real_acts[idx_t]
                ref_batch = ref_acts[idx_t]
                recon_batch, total_delta_bytes = self._encode_delta_batch(real_batch, ref_batch, include_ref_idx=True)
                reconstructed[idx_t] = recon_batch
                n_delta = len(delta_indices)
                if n_delta > 0:
                    per_pos = total_delta_bytes / n_delta
                    transfer_bytes_by_tier["trigram"] = int(per_pos * len(trigram_indices))
                    transfer_bytes_by_tier["bigram"] = int(per_pos * len(bigram_indices))
            evt_after_delta.record()

            if unigram_indices:
                idx_u = torch.tensor(unigram_indices, dtype=torch.long, device=self.device)
                real_uni = real_acts[idx_u]
                recon_uni, transfer_bytes_by_tier["unigram"] = self._encode_unigram_batch(real_uni)
                reconstructed[idx_u] = recon_uni
            evt_after_unigram.record()

            if self_ref_indices:
                sorted_self_ref = sorted(self_ref_indices)
                idx_sr = torch.tensor(sorted_self_ref, dtype=torch.long, device=self.device)
                real_sr = real_acts[idx_sr]
                source_positions = [self_ref_sources[pos] for pos in sorted_self_ref]
                src_t = torch.tensor(source_positions, dtype=torch.long, device=self.device)
                ref_sr = reconstructed[src_t]
                ref_acts[idx_sr] = ref_sr
                recon_sr, transfer_bytes_by_tier["self_ref"] = self._encode_delta_batch(real_sr, ref_sr, include_ref_idx=True)
                reconstructed[idx_sr] = recon_sr
            evt_after_self_ref.record()

            torch.cuda.synchronize()
            result.encode_delta_ms = evt_enc_start.elapsed_time(evt_after_delta)
            result.encode_unigram_ms = evt_after_delta.elapsed_time(evt_after_unigram)
            result.encode_self_ref_ms = evt_after_unigram.elapsed_time(evt_after_self_ref)

        result.transfer_bytes_by_tier = transfer_bytes_by_tier
        result.total_transfer_bytes = sum(transfer_bytes_by_tier.values())
        result.raw_fp16_bytes = seq_len * self.hidden_dim * 2
        result.compression_ratio = result.raw_fp16_bytes / max(result.total_transfer_bytes, 1)

        # Cosine similarity / MSE only computed when explicitly requested.
        if self.compute_cosine_similarity:
            overall_cos = F.cosine_similarity(real_acts.float(), reconstructed.float(), dim=-1)
            overall_mse = ((real_acts.float() - reconstructed.float()) ** 2).mean(dim=-1)
            result.recon_cosine_mean = overall_cos.mean().item()
            result.recon_cosine_min = overall_cos.min().item()
            result.mse_mean = overall_mse.mean().item()
            result.mse_max = overall_mse.max().item()

            non_uni_indices = trigram_indices + bigram_indices + sorted(self_ref_indices)
            if non_uni_indices:
                idx_nu = torch.tensor(non_uni_indices, dtype=torch.long, device=self.device)
                raw_cos = F.cosine_similarity(real_acts[idx_nu].float(), ref_acts[idx_nu].float(), dim=-1)
                result.raw_cosine_mean = raw_cos.mean().item()
                result.raw_cosine_min = raw_cos.min().item()

            for tier_name, tier_indices in [
                ("trigram", trigram_indices),
                ("bigram", bigram_indices),
                ("self_ref", self_ref_indices),
                ("unigram", unigram_indices),
            ]:
                if not tier_indices:
                    result.tier_detail.append(
                        {
                            "tier": tier_name,
                            "count": 0,
                            "raw_cosine_mean": 0.0,
                            "recon_cosine_mean": 0.0,
                            "recon_cosine_min": 0.0,
                            "mse_mean": 0.0,
                            "mse_max": 0.0,
                            "transfer_bytes": 0,
                        }
                    )
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
                result.tier_detail.append(
                    {
                        "tier": tier_name,
                        "count": len(tier_indices),
                        "raw_cosine_mean": raw_cos_t.mean().item(),
                        "recon_cosine_mean": recon_cos_t.mean().item(),
                        "recon_cosine_min": recon_cos_t.min().item(),
                        "mse_mean": mse_t.mean().item(),
                        "mse_max": mse_t.max().item(),
                        "transfer_bytes": transfer_bytes_by_tier[tier_name],
                    }
                )

        local_prompt_refs = self._build_local_prompt_refs(input_ids, prefill_hidden)
        t_upd_start = time.perf_counter()
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