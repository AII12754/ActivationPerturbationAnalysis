# Optimization Opportunities — APPLIED

This document records the optimizations that were identified and applied to the v1 runtime.

---

## Applied Changes

### OPT-1. Per-Token `torch.cuda.synchronize()` Removed from Decode ✅

**Status**: APPLIED

The decode loop no longer records CUDA events or calls `torch.cuda.synchronize()`.
Decode is always raw FP16, so there is no encode step to time. Forward timing uses
`time.perf_counter()` which does not block the GPU pipeline.

---

### OPT-2. Cosine Similarity Gated Behind `compute_cosine_similarity` Config ✅

**Status**: APPLIED

A new `compute_cosine_similarity: bool` field was added to `TransportPolicy` (default `False`).
Prefill only computes cosine similarity and MSE metrics when this flag is `True`.
Decode never computes cosine similarity — it always sends raw FP16 (recon == real, cosine == 1.0).

---

### OPT-3. `reconstructed_hiddens` Dict Removed from Decode ✅

**Status**: APPLIED

Since decode always sends raw FP16 and self-ref only happens during prefill,
there is no need to maintain a dict of reconstructed hidden states during decode.
The decode kernel no longer stores per-position hidden states in a dict.

---

### OPT-4. `decode_hidden_by_pos` Replaced with Ring Buffer ✅

**Status**: APPLIED

The unbounded `decode_hidden_by_pos` dict was replaced with a two-variable ring buffer
(`prev_h`, `curr_h`). Only the last two decode hidden states are retained, which is
sufficient for the bigram hidden lookup in `_submit_decode_table_update_v2`.

---

### OPT-5. Decode Classify Executor Removed ✅

**Status**: APPLIED

Since decode always sends raw FP16, there is no tier classification needed during decode.
The `classify_executor.submit()` call in the decode loop has been removed entirely.
No async future creation, no GIL contention, no result-wait overhead per token.

---

### OPT-6. Sequential `prev_ref` Unigram Loop Removed ✅

**Status**: APPLIED

The `_unigram_uses_prev_ref()`, `_unigram_prev_params()`, and `_encode_prev_unigram_batch()`
methods have been deleted. The prefill unigram path now always uses direct batched INT4/INT2
quantization (`_encode_unigram_batch`), which processes all unigram positions in a single
GPU kernel batch.

---

### OPT-7. GPU Stream for Hidden State Load During Prefill Classify ✅

**Status**: APPLIED

A dedicated `hidden_load_stream` (`torch.cuda.Stream`) is created during runtime init.
During prefill, after the CPU classify future returns `ref_acts`, the tensor is moved
to GPU on this separate stream (`ref_acts.to(device=..., non_blocking=True)`) while
the main Python thread continues with tier indexing. The default stream waits for the
load stream before encoding begins.

---

### OPT-8. Batched Decode Table Updates ✅

**Status**: APPLIED

Decode table updates are now buffered in `_decode_update_batch` and flushed every
16 steps as a single `Future` via `_flush_decode_update_batch()`. This reduces
per-token `update_executor.submit()` overhead from 128 future creates (for 128
tokens) to ~8 batch submissions. Each batch is processed by `_update_table_batch()`
which iterates the buffered `(a, b, c, bi_hidden, tri_hidden)` tuples.

The classify executor is still used once per prefill (negligible overhead since it's
only one submission per request, not per token).

---

### OPT-9. Encode Stream Separation

**Status**: NOT NEEDED in current architecture.

With decode always raw FP16 (no encode step), there is no decode-side GPU encode
work to overlap. For prefill, the encode block must complete before suffix forward
(which needs the reconstructed hidden), leaving no parallel GPU work on the default
stream. The `hidden_load_stream` (OPT-7) already covers the key overlap opportunity
(ref_acts GPU load during CPU tier indexing).

---

### OPT-10. Pre-allocated Decode Hidden Buffer ✅

**Status**: APPLIED

The per-step `recon_sequence.append()` + final `torch.stack()` has been replaced
with a single pre-allocated tensor `recon_buffer` of shape `[decode_tokens, hidden_dim]`.
Each step writes directly into `recon_buffer[step]`. On early termination (EOS),
the result is sliced as `recon_buffer[:actual_steps]` (a view, no copy).

This eliminates per-token list growth, intermediate tensor references, and the
final stack+copy operation.

---

### OPT-11. Trigram/Bigram Already Unified ✅

**Status**: ALREADY THE CASE

The `NgramTable` stores trigram and bigram in a unified two-level DAG:
`_dag[A][B]` → `_BigramNode` with `.bigram_hidden` and `.suffixes[C]`.
Storing `(a, b, c, hidden_c)` allows matching both `abc` (trigram) and `bc` (bigram)
from the same structure. No code change needed.

---

### OPT-12. `--skip-drift` Benchmark Flag ✅

**Status**: APPLIED

Added `--skip-drift` CLI flag to `run_latency_benchmark.py`. When set, the per-request
full-model forward pass for logit drift computation is skipped entirely. This roughly
halves GPU time per request in benchmark mode.

The `drift_records` list stays empty and no `decode_drift.parquet` is written.

---

### OPT-13. Warmup Simplification ✅

**Status**: ALREADY IMPROVED

The `_encode_prev_unigram_batch` warmup path was removed in the first pass.
The remaining multi-batch-size warmup (5 sizes) is a one-time ~100–500 ms startup
cost that's acceptable for production and benchmark use.

---

## Summary of Code Changes

| File | Change |
|------|--------|
| `config.py` | Removed `decode_use_raw_fp16` from `TransportPolicy` (always true). Added `compute_cosine_similarity` flag. |
| `base_runtime.py` | Added `compute_cosine_similarity` field. Added `hidden_load_stream`. Removed `_unigram_uses_prev_ref`, `_unigram_prev_params`, `_encode_prev_unigram_batch`. Removed old `_submit_decode_table_update`. Rewrote `_submit_decode_table_update_v2` to buffer updates. Added `_update_table_batch`, `_flush_decode_update_batch`. Added `_decode_update_batch` / `_decode_update_flush_every` fields. Simplified `_encode_decode_step` to always raw FP16. |
| `decode_kernel.py` | Complete rewrite. Removed all quantization, classification, reference materialization, event recording, synchronization, cosine similarity. Pre-allocated `recon_buffer` replaces per-step list append. Batched table updates via `_flush_decode_update_batch`. Decode is now: forward → raw FP16 → buffer write → batched async table update → suffix forward. |
| `prefill_kernel.py` | Removed `prev_ref` unigram path. Added `hidden_load_stream` usage for async ref_acts transfer. Gated cosine/MSE behind `compute_cosine_similarity`. |
| `pipeline.py` | Removed `_encode_prev_unigram_batch` from warmup. |
| `__init__.py` | Removed `DecodeClassifyResult` from exports. |
| `run_latency_benchmark.py` | Added `--skip-drift` flag to skip logit drift computation. |
