# v1 Architecture Notes

This document is the engineering-level companion to the v1 README.

## Why v1 Exists

The legacy runtime mixed research flexibility with production execution. That made the codebase harder to reason about in three concrete ways:
- too many runtime switches influenced the hot path
- production behavior depended on helper branches designed for experiments
- the final latency policy was not obvious from the public API

v1 fixes that by making the intended design explicit and narrow.

## Design Decisions

### 1. Decode Raw-FP16 Is a Contract

Decode is latency-sensitive enough that the runtime should not treat raw transport as one option among many. In v1, raw decode is the default and intended production mode.

Implications:
- decode still classifies tiers for table evolution and observability
- decode avoids reference materialization unless compressed decode is explicitly reintroduced later
- decode cold misses do not block the token path

### 2. Block Paging Must Be Asynchronous

Whole-domain migration was too coarse for latency-first serving. The production answer is block paging with async page-in/page-out and a resident-block budget that avoids obvious thrash regions.

Implications:
- cold miss means schedule load, not stall decode
- resident-block budgets must be selected from measured latency behavior, not memory intuition alone
- pinned recent blocks are part of the latency policy, not just a convenience optimization

### 3. One Pipeline Means One Table

The final production path should not mix paged block caching with a second routing layer that chooses among multiple activation tables.

Implications:
- v1 owns one block table per pipeline instance
- paged block management is the cache policy
- domain routing belongs to legacy experiments, not the production surface

### 4. Prefill And Decode Need Separate Policies

The optimal policy for prefill is not automatically the optimal policy for decode.

Implications:
- prefill can still use compressed transport where it helps
- decode can stay raw without forcing prefill to do the same
- benchmark reporting must separate prefill and decode communication paths

## Current Code Ownership

v1-owned modules:
- `base_runtime.py`
- `results.py`
- `prefill_kernel.py`
- `decode_kernel.py`
- `pipeline.py`
- `run_latency_benchmark.py`

v1-owned runtime substrate:
- model segment execution methods
- encode helper implementations
- runtime initialization and executor setup
- single-table construction and lifecycle glue

## Runtime State Model

Important mutable state that the runtime maintains across a request:
- pending prefill update future
- pending decode update futures
- block-table residency and pager state

This state now lives in `base_runtime.py` and is owned by v1 directly.

## Benchmark Interpretation

Current benchmark outputs include two useful latency views.

Local compute view:
- `prefill_encode_ms`
- `decode_encode_ms`
- `prefill_forward_ms`
- `decode_forward_ms`

Communication critical-path view:
- `prefill_comm_e2e_*`
- `decode_comm_e2e_*`
- `decode_comm_e2e_*_per_token_ms`

The communication critical-path model is:

$$
t_{e2e} = \max(t_{local\_comm\_work},\ t_{network\_transfer})
$$

where:

$$
t_{network\_transfer} = \frac{bytes \times 8}{bandwidth_{mbps} \times 1000}
$$

This is useful for comparing strategies under multiple bandwidth assumptions, but it is not a replacement for full distributed-system profiling.

## What "Fully Independent" Means Here

The phrase has two levels.

Request-kernel independence:
- v1 owns prefill, decode, orchestration, and results.

Runtime-substrate independence:
- v1 also owns initialization, helper plumbing, and low-level wrappers.

The current codebase has reached both levels for the production latency-first path.

## Refactor Order From Here

1. Keep all new production-path changes inside v1.
2. Narrow legacy runtime to compatibility-only status.
3. Remove duplicated helper implementations from legacy code when safe.
4. Decide whether remaining benchmark helpers belong in v1 or in shared utilities.
2. Narrow legacy runtime to compatibility-only status.
3. Remove duplicated helper implementations from legacy code when safe.
4. Decide whether remaining benchmark helpers belong in v1 or in shared utilities.