# v1 System Overview

## What This System Does

The v1 runtime is a **latency-first activation transfer system** for split-model LLM inference under pipeline parallelism. A transformer model is split at a configurable layer boundary (default: layer 6). The boundary hidden states are compressed and transferred between pipeline stages, rather than sending raw FP16 activations.

The system uses **n-gram tables** to find reference activations for delta coding: if a trigram/bigram of token IDs has been seen before, the stored hidden state serves as a prediction, and only the quantized residual (delta) is transferred.

## Module Map

```
v1/
├── config.py              # Typed configuration: V1PipelineConfig, RuntimeConfig,
│                          #   TransportPolicy, BlockTableConfig
├── base_runtime.py        # Runtime substrate: model execution, transfer encoding,
│                          #   table management, thread pools, quantization helpers
├── prefill_kernel.py      # Prefill hot path: tokenize → classify → encode → table update
├── decode_kernel.py       # Decode hot path: forward → raw FP16 → suffix → async table update
├── pipeline.py            # Diamond MRO: PrefillKernel + DecodeKernel + RuntimeBase,
│                          #   request orchestration, construction helpers
├── results.py             # Result dataclasses: PrefillResult, DecodeResult,
│                          #   DecodeStepRecord
├── run_latency_benchmark.py  # CLI benchmark: warmup/test loops, bandwidth modeling,
│                              #   drift evaluation (--skip-drift), parquet output
└── doc/                   # This documentation folder
```

## Data Flow

### Prefill Path

```
text
  │
  ▼
tokenize + truncate(max_seq_len)
  │
  ├──► classify_executor.submit(table.classify_and_build_refs)  [CPU, async]
  │
  ▼
_run_prefix_prefill(input_tensor)
  │ → embed_tokens → layers[0:boundary] → boundary_hidden [seq_len, hidden_dim]
  │
  ▼
collect classify_future.result()  [wait for CPU classification]
  │ → tiers[], ref_acts[], self_ref_sources[]
  │
  ▼
Load ref_acts to GPU on hidden_load_stream (async, overlaps with tier indexing)
  │
  ▼
Wait for hidden_load_stream before encode
  │
  ▼
Encode by tier:
  ├── trigram + bigram positions → _encode_delta_batch(real, ref)
  ├── unigram positions         → _encode_unigram_batch(real)  [always batched]
  └── self_ref positions        → _encode_delta_batch(real, reconstructed[src_pos])
  │
  ▼
Schedule async table update: update_executor.submit(table.update_from_hidden_states)
  │
  ▼
_run_suffix_prefill(reconstructed_hidden)
  │ → layers[boundary:end] → norm → lm_head → first decode token
```

### Decode Path (per token)

```
next_tok
  │
  ▼
_run_prefix_decode_step(next_tok, prefix_cache)
  │ → embed_tokens → layers[0:boundary] → boundary_hidden [hidden_dim]
  │
  ▼
Raw FP16 pass-through (recon == real, no quantization, no classification)
  │
  ▼
Write recon_buffer[step] = h  (pre-allocated, no per-step allocation)
  │
  ▼
Update ring buffer: prev_h ← curr_h, curr_h ← h
  │
  ▼
Buffer table update via _submit_decode_table_update_v2 (flushed every 16 steps)
  │
  ▼
_run_suffix_decode_step(h, suffix_cache)
  │ → layers[boundary:end] → norm → lm_head → next token logits
```

Decode has no tier classification, no quantization, no CUDA event recording, and no
cosine similarity computation. The boundary hidden state is forwarded as-is in FP16.

## Configuration Structure

```
V1PipelineConfig
├── RuntimeConfig
│   ├── layer_boundary = 6        # split point
│   ├── decode_tokens = 128       # max decode steps
│   ├── max_seq_len = 4096        # prompt truncation limit
│   ├── max_table_entries = 100000
│   └── group_size / top_k        # quantizer parameters
│
├── TransportPolicy
│   ├── delta_strategy = "delta_noaffine_int4_k1"   # delta-coded positions
│   ├── unigram_strategy = "unigram_int4_k4"        # unigram positions
│   ├── prefill_use_raw_fp16 = False                # prefill still compresses
│   └── compute_cosine_similarity = False           # gate cosine/MSE metrics
│
└── BlockTableConfig
    ├── table_placement = "cpu"
    ├── enable_async_paging = True
    ├── max_resident_blocks = 64
    ├── pager_workers = 2
    ├── pinned_block_budget = 8
    └── block_size = 256
```

## Class Hierarchy

```
V1RuntimeBase                         # model execution + encoding + table + thread pools
    │
    ├── LatencyFirstPrefillKernel     # process_prefill()
    │
    ├── LatencyFirstDecodeKernel      # process_decode()
    │
    └── LatencyFirstPipeline          # MRO diamond combining all three
             │                        # process_request() → prefill + decode + stats
             │
             ├── from_parts(...)
             ├── from_legacy_args(...)
             └── build_latency_first_pipeline(...)
```

The diamond MRO means `LatencyFirstPipeline` inherits prefill and decode kernels, which access `self.*` methods from `V1RuntimeBase` through the MRO chain. The kernels are not standalone—they depend on the runtime substrate.

## Communication Cost Model

The benchmark uses two cost models for estimating end-to-end communication latency:

**Overlap model** (optimistic): assumes local compute fully overlaps with network transfer.

$$t_{\text{e2e}} = \max(t_{\text{local}},\ t_{\text{network}})$$

**Additive model** (conservative): assumes no overlap.

$$t_{\text{additive}} = t_{\text{local}} + t_{\text{network}}$$

Where network time is:

$$t_{\text{network}} = \frac{\text{bytes} \times 8}{\text{bandwidth}_{\text{mbps}} \times 1000}$$

Both are reported per request and per decode token, at multiple bandwidth points (default: 200, 500, 1000 Mbps).

## Key Invariants

1. Block-table backend only (no trie in production).
2. Decode **always** sends raw FP16 (no quantization, no classification).
3. Prefill compresses using delta coding for trigram/bigram hits, INT4 for unigram.
4. Cold block misses schedule async page-in, never block decode.
5. Table updates are async and non-blocking on the hot path.
6. Decode uses a ring buffer (`prev_h`, `curr_h`) for table updates — no unbounded dicts.
7. Prefill uses a dedicated `hidden_load_stream` to overlap ref_acts GPU transfer with CPU tier indexing.
8. Cosine similarity / MSE metrics are only computed when `compute_cosine_similarity=True`.
