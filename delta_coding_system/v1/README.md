# v1 Latency-First Runtime

`delta_coding_system/v1` is the production-facing runtime surface for the final latency-first design.

This folder exists to stop the production path from sharing the same API surface as the research path. The goal is not only cleaner code, but also harder operational guarantees: one backend, one decode transport policy, one set of block-paging defaults, and one benchmark entrypoint that reflects the intended deployment mode.

## Goals

Primary goal:
- Minimize end-to-end communication latency, especially decode tail latency.

Operational goals:
- Keep the hot path narrow and predictable.
- Keep decode transport semantics strict rather than best-effort.
- Make cold misses asynchronous and non-blocking on the decode path.
- Avoid reintroducing experimental switches into the production entrypoint.
- Make the runtime modular enough that prefill, decode, config, and reporting can evolve independently.

## Hard Invariants

These are not tuning suggestions. They define the intended runtime contract.

- Block-table backend only.
- Decode sends raw FP16 boundary activations.
- Prefill may still compress according to the configured transport strategy.
- Cold block misses may enqueue async page-in work, but decode should not wait for disk.
- Tiny resident-block budgets are not production defaults because they regress latency under mixed workloads.
- New production work should target `v1` first rather than the legacy runtime.

## Directory Layout

- `config.py`
	Purpose: typed configuration surface for the final runtime.
	Owns: transport policy, block-table policy, runtime execution knobs.

- `results.py`
	Purpose: v1-owned result dataclasses.
	Owns: prefill metrics, decode metrics, per-step decode records, lazy decode classification result.

- `prefill_kernel.py`
	Purpose: v1-owned prefill hot path.
	Owns: request tokenization, async classify scheduling, prefill encode path, local prompt reference construction, prefill update scheduling.

- `decode_kernel.py`
	Purpose: v1-owned decode hot path.
	Owns: per-token classify scheduling, lazy decode reference materialization, raw decode fast path, async update submission.

- `pipeline.py`
	Purpose: v1 request orchestration layer.
	Owns: pipeline construction, config translation, request-level orchestration, stats collection.

- `run_latency_benchmark.py`
	Purpose: latency benchmark entrypoint for the final runtime surface.
	Owns: warmup/test loops, network-model E2E accounting, drift evaluation, parquet outputs.

## Runtime Architecture

Request flow:
1. Run v1 prefill kernel.
2. Run v1 decode kernel.
3. Aggregate table stats and transfer metrics.

Prefill flow:
1. Tokenize and truncate to `max_seq_len`.
2. Launch classify on CPU.
3. Run prefix-half forward on GPU.
4. Collect classify results.
5. Encode by tier.
6. Launch asynchronous table update.
7. Run suffix-half prefill to produce the first decode token.

Decode flow:
1. Launch classify for the next token on CPU.
2. Run prefix-half decode step on GPU.
3. Collect classify result.
4. If decode compression is disabled, skip reference materialization and send raw FP16.
5. Otherwise encode against trigram, bigram, self-ref, or unigram reference.
6. Submit async table update without blocking decode.
7. Run suffix-half decode step to produce the next token.

## Configuration Model

`V1PipelineConfig` is the root configuration object. It is intentionally split into three smaller units.

- `RuntimeConfig`
	Controls model execution shape such as `layer_boundary`, `decode_tokens`, `max_seq_len`, and quantizer group sizes used by the low-level helpers.

- `TransportPolicy`
	Controls transfer semantics. The production default is `delta_noaffine_int4_k1` for delta-coded positions, `unigram_int4_k4` for unigram prefill positions, and raw FP16 decode transport.

- `BlockTableConfig`
	Controls block-table placement and paging. The production default is async block paging enabled with `max_resident_blocks=64`, `pager_workers=2`, and `pinned_block_budget=8`.

## Current Independence Boundary

v1 now owns the full runtime substrate used by the latency-first production path:
- prefill execution
- decode execution
- request orchestration
- result dataclasses
- model-segment execution helpers
- transfer encode helper implementations
- table initialization plumbing
- executor and lifecycle setup

The remaining legacy module is now compatibility-oriented. v1 no longer inherits from `OverlappedPipeline`.

## Benchmarking

The benchmark entrypoint is:

```bash
/usr/bin/python delta_coding_system/v1/run_latency_benchmark.py \
	--model /root/share/models/Qwen2.5-14B-Instruct \
	--gpu 0 \
	--datasets sharegpt \
	--warmup-requests 1 \
	--test-requests 2 \
	--max-seq-len 1024 \
	--max-decode-tokens 4 \
	--output-dir results_v1_latency/sharegpt_smoke
```

What the benchmark reports:
- prefill transfer bytes
- decode transfer bytes
- same-PP FP16 baseline transfer time derived from the same layer boundary and token counts
- prefill operation breakdown: forward, classify wait, delta encode, unigram encode, self-ref encode, table-update submit
- decode operation breakdown: forward, lookup/classify wait, transfer-prepare, local communication work, table-update wait
- prefill E2E communication time under bandwidth models
- decode E2E communication time under bandwidth models
- additive communication time under bandwidth models: local cache/codec work + simulated link transfer
- per-token decode E2E communication time
- per-token decode step breakdown with local-vs-network decomposition
- decode hidden/logit drift against the original model

Decode metric note:
- In raw decode mode, `decode_encode_ms` and `decode_transfer_prepare_ms` do not mean quantization. They measure the GPU-side transfer-preparation window around `_encode_decode_step`, which in raw mode is just FP16 clone/materialization plus transfer-byte accounting.
- `decode_classify_wait_ms` and `decode_lookup_wait_ms` are the CPU wait time for the async decode lookup/classification future.
- `decode_local_comm_work_ms` is the sum of decode lookup wait and transfer-prepare time, and is the quantity compared against network time in the E2E communication model.
- `prefill_cpu_cache_lookup_ms` and `decode_cpu_cache_lookup_ms` expose the CPU-side cache/table lookup window directly, so you can see how much of the conservative additive model is coming from CPU activation-cache reads.

Output files per dataset:
- `request_summary.parquet`
- `decode_step_breakdown.parquet`
- `decode_drift.parquet`
- `latency_summary.parquet`

Important note:
- The benchmark computes communication E2E as `max(local_compute_ms, network_ms)` for prefill and decode, so the reported value is a critical-path model rather than raw DMA timing from NCCL or InfiniBand.
- The same request summary also includes `*_comm_additive_*` and `fp16_*_same_pp_*` fields. Use these when you want a conservative apples-to-apples comparison against a same-PP raw-FP16 baseline and do not want to assume CPU cache lookup fully overlaps with transfer.

## Recommended Production Defaults

Use these defaults unless a measurement says otherwise:
- `table_backend=block`
- `decode_use_raw_fp16=True`
- `prefill_use_raw_fp16=False`
- `enable_async_paging=True`
- `max_resident_blocks=64`
- `pinned_block_budget=8`
- `gpu_hot_cache_entries=0` unless a measured gain justifies it
- `table_placement=cpu` for larger steady-state tables

## Known Non-Goals

v1 does not try to keep compatibility with every research switch from the legacy runtime.

Examples intentionally excluded from the v1 surface:
- trie backend selection
- decode quantization toggles
- research-oriented ablation matrices in the production benchmark entrypoint
- multidomain routing and domain-aware table selection

## Migration Guidance

When moving code into v1, keep this order:
1. Request-kernel logic.
2. Execution helpers.
3. Transfer helpers.
4. Initialization and lifecycle plumbing.

This order keeps the hot path explicit first, which is the part most likely to affect latency regressions.

## Remaining Cleanup Targets

The main work left is no longer independence itself. It is cleanup and narrowing of the legacy path:
- keep new production changes landing in v1 first
- trim duplicated helper code from legacy runtime once compatibility needs are clear
- decide whether benchmark-only utilities should also move under the v1 namespace

## Related File

For a lower-level architectural snapshot, see [delta_coding_system/v1/ARCHITECTURE.md](delta_coding_system/v1/ARCHITECTURE.md).