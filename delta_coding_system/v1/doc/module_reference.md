# v1 Module Reference

## config.py

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `FINAL_DELTA_STRATEGY` | `"delta_noaffine_int4_k1"` | Default delta coding strategy for referenced positions |
| `FINAL_UNIGRAM_STRATEGY` | `"unigram_int4_k4"` | Default direct quantization strategy for unigram positions |

### `TransportPolicy`

Frozen dataclass controlling transfer semantics.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `delta_strategy` | str | `"delta_noaffine_int4_k1"` | Quantization strategy for delta-coded positions |
| `unigram_strategy` | str | `"unigram_int4_k4"` | Quantization strategy for unigram (no-reference) positions |
| `prefill_use_raw_fp16` | bool | `False` | Send raw FP16 during prefill |
| `track_transfer_bytes` | bool | `True` | Track transfer byte counts for benchmarking |
| `compute_cosine_similarity` | bool | `False` | Compute cosine/MSE metrics during prefill |

Note: `decode_use_raw_fp16` has been removed. Decode always uses raw FP16.

### `BlockTableConfig`

Frozen dataclass for block-table placement and paging.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `table_placement` | str | `"cpu"` | Where to store the n-gram table (`"cpu"` or `"gpu"`) |
| `pin_cpu_output_copy` | bool | `True` | Pin CPU table output tensors for async copy |
| `enable_async_cpu_output_copy` | bool | `True` | Use non-blocking CPU→GPU copies |
| `gpu_hot_cache_entries` | int | `0` | Number of hot entries cached on GPU |
| `enable_disk_offload` | bool | `False` | Enable disk offload for cold blocks |
| `disk_offload_dir` | str\|None | `None` | Directory for disk-offloaded blocks |
| `block_size` | int | `256` | Entries per block in the block table |
| `enable_async_paging` | bool | `True` | Enable async page-in/page-out |
| `max_resident_blocks` | int | `64` | Max blocks kept in memory |
| `pager_workers` | int | `2` | Background pager thread count |
| `pinned_block_budget` | int | `8` | Pinned memory blocks for fast transfer |

### `RuntimeConfig`

Frozen dataclass for model execution parameters.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `layer_boundary` | int | `6` | Layer index where the model is split |
| `decode_tokens` | int | `128` | Max tokens to generate per request |
| `max_seq_len` | int | `4096` | Max input sequence length |
| `max_table_entries` | int | `100000` | Max n-gram table entries |
| `group_size` | int | `128` | Group size for INT4/INT2 quantization |
| `top_k` | int | `1` | Top-k outliers preserved in quantization |
| `int8_group_size` | int | `128` | Group size for INT8 table storage |
| `int8_outlier_top_k` | int | `1` | Top-k outliers for INT8 storage |
| `extra_stop_token_ids` | Sequence[int] | `()` | Additional stop tokens beyond EOS |

### `V1PipelineConfig`

Root configuration combining all three sub-configs.

Methods:
- `to_legacy_kwargs() → Dict` — flatten to keyword args for `V1RuntimeBase.__init__`
- `summary() → Dict` — serializable dict with `table_backend="block"` injected

---

## base_runtime.py — `V1RuntimeBase`

### Constructor

Takes model, tokenizer, and all config fields as keyword arguments. Initializes:
- The n-gram table via `create_activation_table()`
- Two `ThreadPoolExecutor(max_workers=1)`: `classify_executor` and `update_executor`
- Pending future tracking: `_pending_prefill_update`, `_pending_decode_updates`
- `hidden_load_stream` — dedicated CUDA stream for async ref_acts GPU transfer during prefill
- `compute_cosine_similarity` — flag gating cosine/MSE metric computation
- `decode_use_raw_fp16 = True` (hardcoded, always raw FP16)
- Transfer kernel warmup

### Model Execution Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `_run_model_segment(hidden, cache, start, end)` | Hidden states + KV cache | Hidden states | Run layers[start:end] with attention mask and position embeddings |
| `_run_prefix_prefill(input_tensor)` | Token IDs [1, seq_len] | (boundary_hidden [seq_len, H], prefix_cache) | Embed + prefix layers |
| `_run_suffix_prefill(boundary_hidden)` | Hidden [seq_len, H] | (logits [1, 1, V], suffix_cache) | Suffix layers + norm + lm_head |
| `_run_prefix_decode_step(next_tok, prefix_cache)` | Token [1, 1] + cache | boundary_hidden [H] | Single-token prefix forward |
| `_run_suffix_decode_step(boundary_hidden, suffix_cache)` | Hidden [H] + cache | logits [1, 1, V] | Single-token suffix forward |

### Encoding Methods

| Method | Description |
|--------|-------------|
| `_encode_delta_batch(real, ref, include_ref_idx)` | Delta-code with INT4/INT2 quantization, optional affine transform |
| `_encode_unigram_batch(real)` | Direct INT4/INT2/INT8 quantization (no reference) |
| `_encode_decode_step(real_h, ref_h, tier)` | Always returns raw FP16 clone (decode path simplified) |
| `_encode_direct_int4_batch(real)` | Direct INT4 groupwise quantize + dequantize |
| `_encode_direct_int2_batch(real)` | Direct INT2 groupwise quantize + dequantize |
| `_encode_direct_int8_batch(real)` | Direct INT8 groupwise quantize + dequantize |

Removed: `_encode_prev_unigram_batch` — prev_ref unigram strategy has been deleted.

### Table Management

| Method | Description |
|--------|-------------|
| `_active_tables()` | Returns `[self.table]` |
| `_write_tables()` | Returns `[self.table]` |
| `_update_active_tables_from_hidden_states(token_ids, hidden)` | Batch update all write tables |
| `_submit_decode_table_update_v2(...)` | Buffer a single trigram+bigram update; auto-flushes every 16 steps |
| `_update_table_batch(tables, updates)` | Worker: apply a batch of buffered table updates |
| `_flush_decode_update_batch()` | Submit all buffered updates as a single Future |
| `_update_table_step(tables, a, b, c, bi_hidden, tri_hidden)` | Single-update worker (retained for backward compat) |
| `_drain_decode_updates(wait)` | Check/collect pending decode update futures |
| `_materialize_stored_hidden(stored)` | Convert stored table entry (Tensor or DeltaPacket) to GPU FP16 |

Removed: `_submit_decode_table_update` (old version with unbounded dict), `_unigram_uses_prev_ref`, `_unigram_prev_params`.

### Lifecycle

| Method | Description |
|--------|-------------|
| `shutdown()` | Wait for pending updates, shutdown executors and table |
| `_warmup_transfer_kernels()` | Run dummy encode operations to warm CUDA kernels |

---

## prefill_kernel.py — `LatencyFirstPrefillKernel`

### `process_prefill(text, phase, task_name)`

Main entry point. Returns a 7-tuple:
```
(PrefillResult, prefix_cache, suffix_cache, next_tok, input_ids, prefill_hidden, local_prompt_refs)
```

**Warmup path** (`phase != "test"`):
- No classification, no tier encoding
- Directly updates table and runs suffix prefill

**Test path**:
1. Submit async `classify_and_build_refs` on CPU
2. Run prefix forward on GPU
3. Collect classification results
4. Load `ref_acts` to GPU on `hidden_load_stream` (async, overlaps with tier indexing)
5. Wait for `hidden_load_stream` before encode
6. Encode each tier (delta for trigram/bigram, batched INT4 for unigram, delta for self-ref)
7. Compute cosine/MSE metrics only if `compute_cosine_similarity` is `True`
8. Submit async table update
9. Run suffix forward on reconstructed hidden states

The sequential `prev_ref` unigram loop has been removed. All unigrams are now processed
in a single batched call to `_encode_unigram_batch`.

### `_build_local_prompt_refs(token_ids, hidden_states)`

Builds bigram and trigram dicts mapping n-gram keys → hidden state tensors from the current prompt. Used as a local cache for decode-time tier classification when table entries are not yet populated.

---

## decode_kernel.py — `LatencyFirstDecodeKernel`

### `process_decode(prefix_cache, suffix_cache, next_tok, ...)`

Main entry point. Returns `DecodeResult`.

Per-token loop:
1. Run prefix forward → boundary hidden `h`
2. Raw FP16 pass-through (recon == h, no encode/classify)
3. Write `recon_buffer[step] = h` (pre-allocated buffer)
4. Update ring buffer: `prev_h ← curr_h`, `curr_h ← h`
5. Buffer async table update via `_submit_decode_table_update_v2` (auto-flushes every 16 steps)
6. Run suffix forward → next token logits

At end of loop, `_flush_decode_update_batch()` submits remaining buffered updates.

No classification, no quantization, no CUDA event recording, no cosine similarity.
Tier is always `"raw_fp16"`. Cosines are always `1.0`.

Removed: `_classify_decode_step`, `_materialize_decode_reference`, `DecodeClassifyResult` usage,
`reconstructed_hiddens` dict, `decode_hidden_by_pos` dict, `classify_executor.submit()` in decode.

---

## pipeline.py — `LatencyFirstPipeline`

Diamond MRO: `LatencyFirstPrefillKernel + LatencyFirstDecodeKernel + V1RuntimeBase`

### Construction

| Method | Description |
|--------|-------------|
| `__init__(model, tokenizer, config, device)` | From `V1PipelineConfig` |
| `from_parts(model, tokenizer, runtime, transport, table, device)` | From individual config objects |
| `from_legacy_args(model, tokenizer, args, device)` | From argparse namespace |
| `build_latency_first_pipeline(...)` | Module-level factory function |

### Request Processing

| Method | Description |
|--------|-------------|
| `process_request(text, phase, task_name)` | Full prefill + decode + table stats |
| `process_request_v1(text, phase, task_name)` | Alias for `process_request` |
| `policy_summary()` | Returns config dict |
| `_collect_table_stats(text)` | Returns table stats dict |

### Override

`_warmup_transfer_kernels()` is overridden to warm at multiple batch sizes (1, 8, 64, 256, max_seq_len) instead of just 8.

---

## results.py

### `PrefillResult`

| Field | Type | Description |
|-------|------|-------------|
| `seq_len` | int | Input sequence length |
| `num_{trigram,bigram,self_ref,unigram}` | int | Position counts per tier |
| `recon_cosine_{mean,min}` | float | Reconstruction quality |
| `total_transfer_bytes` | int | Compressed transfer size |
| `compression_ratio` | float | raw_fp16 / transfer bytes |
| `{encode_delta,encode_unigram,encode_self_ref}_ms` | float | Per-tier encode time |
| `prefill_fwd_ms` | float | Prefix forward time |
| `classify_ms` | float | CPU classification wait time |
| `table_update_ms` | float | Table update submission time |
| `reconstructed_hidden` | Tensor\|None | Reconstructed boundary activations |

### `DecodeStepRecord`

Per-token metrics: step, tier, cosines, bytes, fwd_ms, classify_ms, encode_ms, table_update_ms.

### `DecodeResult`

Aggregated decode metrics: token counts, tier counts, cosine stats, byte totals, timing totals, generated IDs, reconstructed hidden.

### `DecodeClassifyResult` (deprecated)

Lightweight classification output: tier string, optional stored reference, materialization flag.
Still defined in `results.py` for backward compatibility but no longer exported from `__init__.py`
or used in the decode kernel.

---

## run_latency_benchmark.py

CLI benchmark entrypoint.

### Key Functions

| Function | Description |
|----------|-------------|
| `main()` | Parse args, load model, iterate datasets |
| `run_dataset(model, tokenizer, device, dataset_name, args)` | Run warmup + test, compute drift, save parquet |
| `_build_latency_summary(dataset, records, bandwidths)` | Aggregate per-dataset statistics |
| `_network_ms(bytes, bw_mbps)` | Simulated network transfer time |
| `_prepare_texts(dataset, seed, warmup, test)` | Load and shuffle dataset texts |

### Output Files (per dataset)

| File | Contents |
|------|----------|
| `request_summary.parquet` | Per-request metrics, transfer bytes, timing, bandwidth models |
| `decode_step_breakdown.parquet` | Per-token decode metrics with bandwidth models |
| `decode_drift.parquet` | Logit drift vs. original model |
| `latency_summary.parquet` | Aggregated statistics (mean, p50, p95) |
