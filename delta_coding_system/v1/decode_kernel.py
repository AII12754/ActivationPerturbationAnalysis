from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .results import DecodeClassifyResult, DecodeResult, DecodeStepRecord


class LatencyFirstDecodeKernel:
    """v1-owned decode kernel for the latency-first runtime.

    This is the first hot-path slice migrated out of the legacy pipeline module.
    Prefill still reuses the legacy implementation for now, but decode scheduling,
    classification, and transfer policy live here.
    """

    def _materialize_decode_reference(self, result: DecodeClassifyResult) -> Optional[torch.Tensor]:
        if result.stored_ref is None:
            return None
        if result.ref_is_materialized:
            return result.stored_ref.unsqueeze(0).to(device=self.device, dtype=torch.float16)
        return self._materialize_stored_hidden(result.stored_ref).unsqueeze(0)

    def _classify_decode_step(
        self,
        running_token_ids: List[int],
        decode_pos: int,
        first_occ_map: Dict[Tuple[int, int, int], int],
        reconstructed_hiddens: Dict[int, torch.Tensor],
        local_prompt_trigrams: Dict[Tuple[int, int, int], torch.Tensor],
        local_prompt_bigrams: Dict[Tuple[int, int], torch.Tensor],
    ) -> DecodeClassifyResult:
        tier = "unigram"
        stored_ref = None
        ref_is_materialized = False
        capture_ref = not self.decode_use_raw_fp16
        active_tables = self._active_tables()

        if len(running_token_ids) >= 3:
            a = running_token_ids[-3]
            b = running_token_ids[-2]
            c = running_token_ids[-1]
            trigram_key = (a, b, c)

            tri_ref = None
            for table in active_tables:
                tri_ref = table.get_trigram(a, b, c)
                if tri_ref is not None:
                    break
            if tri_ref is not None:
                tier = "trigram"
                if capture_ref:
                    stored_ref = tri_ref
                    ref_is_materialized = False
                first_occ_map.setdefault(trigram_key, decode_pos)
            elif trigram_key in local_prompt_trigrams:
                tier = "trigram"
                if capture_ref:
                    stored_ref = local_prompt_trigrams[trigram_key]
                    ref_is_materialized = True
                first_occ_map.setdefault(trigram_key, decode_pos)
            elif trigram_key in first_occ_map:
                src_pos = first_occ_map[trigram_key]
                if src_pos in reconstructed_hiddens:
                    tier = "self_ref"
                    if capture_ref:
                        stored_ref = reconstructed_hiddens[src_pos]
                        ref_is_materialized = True
            if tier == "unigram":
                b_tok = running_token_ids[-2]
                c_tok = running_token_ids[-1]
                bi_ref = None
                for table in active_tables:
                    bi_ref = table.get_bigram(b_tok, c_tok)
                    if bi_ref is not None:
                        break
                if bi_ref is not None:
                    tier = "bigram"
                    if capture_ref:
                        stored_ref = bi_ref
                        ref_is_materialized = False
                elif (b_tok, c_tok) in local_prompt_bigrams:
                    tier = "bigram"
                    if capture_ref:
                        stored_ref = local_prompt_bigrams[(b_tok, c_tok)]
                        ref_is_materialized = True
                first_occ_map.setdefault(trigram_key, decode_pos)

        return DecodeClassifyResult(
            tier=tier,
            stored_ref=stored_ref,
            ref_is_materialized=ref_is_materialized,
        )

    def _resolve_prev_unigram_reference(
        self,
        decode_pos: int,
        input_ids: List[int],
        prefill_hidden: torch.Tensor,
        reconstructed_hiddens: Dict[int, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.decode_use_raw_fp16 or not self._unigram_uses_prev_ref():
            return None
        prev_pos = decode_pos - 1
        if prev_pos < len(input_ids):
            return prefill_hidden[prev_pos].unsqueeze(0)
        if prev_pos in reconstructed_hiddens:
            return reconstructed_hiddens[prev_pos].unsqueeze(0)
        return None

    @torch.inference_mode()
    def process_decode(
        self,
        prefix_cache,
        suffix_cache,
        next_tok: torch.Tensor,
        input_ids: List[int],
        prefill_hidden: torch.Tensor,
        prefill_reconstructed_hidden: torch.Tensor,
        local_prompt_refs: Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int], torch.Tensor]],
        phase: str = "test",
    ) -> DecodeResult:
        del prefill_reconstructed_hidden
        t_total_start = time.perf_counter()
        is_test = phase == "test"

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

        evt_fwd_start = torch.cuda.Event(enable_timing=True)
        evt_fwd_end = torch.cuda.Event(enable_timing=True)
        evt_enc_start = torch.cuda.Event(enable_timing=True)
        evt_enc_end = torch.cuda.Event(enable_timing=True)

        for step in range(self.decode_tokens):
            tok_id = next_tok.item()
            running_token_ids.append(tok_id)
            generated_token_ids.append(tok_id)
            decode_pos = len(running_token_ids) - 1

            classify_future = self.classify_executor.submit(
                self._classify_decode_step,
                running_token_ids,
                decode_pos,
                first_occ_map,
                reconstructed_hiddens,
                local_prompt_trigrams,
                local_prompt_bigrams,
            )

            evt_fwd_start.record()
            h = self._run_prefix_decode_step(next_tok, prefix_cache)
            evt_fwd_end.record()
            decode_hidden_by_pos[decode_pos] = h

            t_classify_start = time.perf_counter()
            classify_result = classify_future.result()
            tier = classify_result.tier
            classify_ms = (time.perf_counter() - t_classify_start) * 1000.0

            ref_h = None
            raw_cos = 0.0
            if not self.decode_use_raw_fp16:
                ref_h = self._materialize_decode_reference(classify_result)

            real_h_2d = h.unsqueeze(0)
            prev_ref = self._resolve_prev_unigram_reference(
                decode_pos,
                input_ids,
                prefill_hidden,
                reconstructed_hiddens,
            )

            evt_enc_start.record()
            if not self.decode_use_raw_fp16 and tier == "unigram" and prev_ref is not None and self._unigram_uses_prev_ref():
                recon, xfer_bytes = self._encode_prev_unigram_batch(real_h_2d, prev_ref)
            else:
                recon, xfer_bytes = self._encode_decode_step(real_h_2d, ref_h, tier)
            evt_enc_end.record()

            torch.cuda.synchronize()
            fwd_ms = evt_fwd_start.elapsed_time(evt_fwd_end)
            encode_ms = evt_enc_start.elapsed_time(evt_enc_end)

            recon_cos = F.cosine_similarity(real_h_2d.float(), recon.float(), dim=-1).item()
            if ref_h is not None:
                raw_cos = F.cosine_similarity(real_h_2d.float(), ref_h.float(), dim=-1).item()
            raw_fp16_bytes = self.hidden_dim * 2

            reconstructed_hiddens[decode_pos] = recon.squeeze(0)
            recon_sequence.append(recon.squeeze(0))

            table_update_ms = 0.0
            self._drain_decode_updates(wait=False)
            self._submit_decode_table_update(
                running_token_ids,
                decode_pos,
                h,
                prefill_hidden,
                input_ids,
                decode_hidden_by_pos,
            )

            tier_counts[tier] += 1
            all_cosines.append(recon_cos)
            total_transfer_bytes += xfer_bytes
            total_raw_bytes += raw_fp16_bytes

            if is_test:
                decode_result.step_records.append(
                    DecodeStepRecord(
                        step=step,
                        tier=tier,
                        raw_cosine=raw_cos,
                        recon_cosine=recon_cos,
                        transfer_bytes=xfer_bytes,
                        raw_fp16_bytes=raw_fp16_bytes,
                        fwd_ms=fwd_ms,
                        classify_ms=classify_ms,
                        encode_ms=encode_ms,
                        table_update_ms=table_update_ms,
                    )
                )

            suffix_logits = self._run_suffix_decode_step(recon.squeeze(0), suffix_cache)
            next_tok = self._select_next_token(suffix_logits, do_sample=False)

            if self.tokenizer.eos_token_id is not None and tok_id == self.tokenizer.eos_token_id:
                break
            if tok_id in self.extra_stop_token_ids:
                break

        self._drain_decode_updates(wait=False)
        del next_tok

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
        if decode_result.step_records:
            decode_result.total_fwd_ms = sum(s.fwd_ms for s in decode_result.step_records)
            decode_result.total_classify_ms = sum(s.classify_ms for s in decode_result.step_records)
            decode_result.total_encode_ms = sum(s.encode_ms for s in decode_result.step_records)
            decode_result.total_table_update_ms += sum(s.table_update_ms for s in decode_result.step_records)
        decode_result.total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return decode_result