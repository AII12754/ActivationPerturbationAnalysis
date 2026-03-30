from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .results import DecodeResult, DecodeStepRecord


class LatencyFirstDecodeKernel:
    """v1-owned decode kernel for the latency-first runtime.

    Decode always sends raw FP16 boundary activations.  No quantization,
    no reference materialization, no cache reads on the decode hot path.
    Table updates are submitted asynchronously so they never block token
    generation.
    """

    @torch.inference_mode()
    def process_decode(
        self,
        prefix_cache,
        suffix_cache,
        next_tok: torch.Tensor,
        input_ids: List[int],
        prefill_hidden: torch.Tensor,
        prefill_reconstructed_hidden: torch.Tensor,
        local_prompt_refs: Any,
        phase: str = "test",
    ) -> DecodeResult:
        del prefill_reconstructed_hidden, local_prompt_refs
        t_total_start = time.perf_counter()
        is_test = phase == "test"

        torch.cuda.set_device(self.device)

        decode_result = DecodeResult(decode_tokens=0)

        running_token_ids: List[int] = list(input_ids)
        recon_buffer = torch.empty(
            self.decode_tokens, self.hidden_dim,
            device=self.device, dtype=torch.float16,
        )
        actual_steps = 0
        generated_token_ids: List[int] = []

        raw_fp16_bytes = self.hidden_dim * 2
        total_transfer_bytes = 0
        total_raw_bytes = 0

        # Ring buffer: only keep last 2 decode hidden states for table update.
        prev_h: Optional[torch.Tensor] = None
        curr_h: Optional[torch.Tensor] = None

        for step in range(self.decode_tokens):
            tok_id = next_tok.item()
            running_token_ids.append(tok_id)
            generated_token_ids.append(tok_id)
            decode_pos = len(running_token_ids) - 1

            t_fwd_start = time.perf_counter()
            h = self._run_prefix_decode_step(next_tok, prefix_cache)
            fwd_ms = (time.perf_counter() - t_fwd_start) * 1000.0

            # Raw FP16: recon == real hidden, transfer == hidden_dim * 2 bytes.
            recon = h
            xfer_bytes = raw_fp16_bytes if self.track_transfer_bytes else 0

            recon_buffer[step] = recon
            actual_steps += 1
            total_transfer_bytes += xfer_bytes
            total_raw_bytes += raw_fp16_bytes

            # Buffer async table update (flushed in batches).
            self._submit_decode_table_update_v2(
                running_token_ids,
                decode_pos,
                h,
                prev_h,
                curr_h,
                prefill_hidden,
                input_ids,
            )
            prev_h = curr_h
            curr_h = h

            if is_test:
                decode_result.step_records.append(
                    DecodeStepRecord(
                        step=step,
                        tier="raw_fp16",
                        raw_cosine=1.0,
                        recon_cosine=1.0,
                        transfer_bytes=xfer_bytes,
                        raw_fp16_bytes=raw_fp16_bytes,
                        fwd_ms=fwd_ms,
                        classify_ms=0.0,
                        encode_ms=0.0,
                        table_update_ms=0.0,
                    )
                )

            suffix_logits = self._run_suffix_decode_step(recon, suffix_cache)
            next_tok = self._select_next_token(suffix_logits, do_sample=False)

            if self.tokenizer.eos_token_id is not None and tok_id == self.tokenizer.eos_token_id:
                break
            if tok_id in self.extra_stop_token_ids:
                break

        self._flush_decode_update_batch()
        del next_tok

        decode_result.decode_tokens = len(generated_token_ids)
        decode_result.generated_token_ids = generated_token_ids
        decode_result.total_transfer_bytes = total_transfer_bytes
        decode_result.raw_fp16_bytes = total_raw_bytes
        decode_result.compression_ratio = 1.0
        decode_result.recon_cosine_mean = 1.0
        decode_result.recon_cosine_min = 1.0
        if actual_steps > 0:
            decode_result.reconstructed_hidden = recon_buffer[:actual_steps]
        if decode_result.step_records:
            decode_result.total_fwd_ms = sum(s.fwd_ms for s in decode_result.step_records)
            decode_result.total_classify_ms = 0.0
            decode_result.total_encode_ms = 0.0
            decode_result.total_table_update_ms = 0.0
        decode_result.total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return decode_result