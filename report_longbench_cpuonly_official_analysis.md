# Official LongBench CPU-Only Results Analysis

Date: 2026-03-27

## Scope

This document analyzes the official LongBench results for the following Qwen2.5-14B-Instruct CPU-table-only configurations:

- `pure_int8`
- `pure_int4`
- `pure_int2`
- `delta_int2_k4_unigram_int4`
- `delta_int2_k2_unigram_int4`
- `delta_int4_k1_unigram_int4`
- `delta_noaffine_int2_k4_unigram_int4`
- `delta_int2_k4_entropy_unigram_int4`
- `fp16_baseline`

All compressed runs were launched with official LongBench generation and explicit CPU table only settings:

- `table_placement=cpu`
- `gpu_hot_cache_entries=0`

Result sources:

- `official_longbench/LongBench/pred/qwen14b_cpuonly_*/result.json`
- `official_longbench/LongBench/pred/qwen14b_cpuonly_*/task_results.json`
- `official_longbench/LongBench/pred/qwen14b_fp16_baseline/result.json`

## Important Completeness Note

Not every compressed configuration has the same task coverage in its final `result.json`.

- Full 21-task coverage:
  - `pure_int8`
  - `pure_int4`
  - `delta_int2_k4_unigram_int4`
  - `delta_int4_k1_unigram_int4`
- Missing only `repobench-p`:
  - `pure_int2`
  - `delta_int2_k2_unigram_int4`
  - `delta_noaffine_int2_k4_unigram_int4`
  - `delta_int2_k4_entropy_unigram_int4`

Because of that, there are two fair topline views:

- `avg_available`: average over whatever tasks are present in that config's final `result.json`
- `avg_common20`: average over the 20 tasks shared by all configurations, excluding `repobench-p`

The common-20 comparison is the fairest cross-config ranking.

## Topline Summary

| Config | Tasks | Missing | Avg Available | Avg Common20 | Delta vs FP16 on Common20 |
|---|---:|---|---:|---:|---:|
| `pure_int4` | 21 | none | 50.539 | 50.418 | +0.432 |
| `pure_int8` | 21 | none | 50.510 | 50.400 | +0.414 |
| `delta_int2_k2_unigram_int4` | 20 | `repobench-p` | 50.488 | 50.488 | +0.502 |
| `delta_int4_k1_unigram_int4` | 21 | none | 50.423 | 50.306 | +0.320 |
| `delta_int2_k4_entropy_unigram_int4` | 20 | `repobench-p` | 50.316 | 50.316 | +0.330 |
| `delta_noaffine_int2_k4_unigram_int4` | 20 | `repobench-p` | 50.263 | 50.263 | +0.277 |
| `pure_int2` | 20 | `repobench-p` | 50.221 | 50.221 | +0.236 |
| `delta_int2_k4_unigram_int4` | 21 | none | 50.509 | 50.354 | +0.368 |
| `fp16_baseline` | 21 | none | 49.930 | 49.986 | +0.000 |

### Topline Takeaways

1. Every compressed configuration beats `fp16_baseline` on the shared 20-task average.
2. Among full 21-task runs, the strongest three are:
   - `pure_int4`
   - `pure_int8`
   - `delta_int2_k4_unigram_int4`
3. `delta_int2_k2_unigram_int4` has the best common-20 average, but it is still missing `repobench-p`, so it should not yet be treated as the definitive overall winner.
4. The absolute margins over FP16 are real but small. This is a modest win, not a dramatic separation.

## Category-Level Comparison

Task groupings used in this section:

- Single-Doc QA: `narrativeqa`, `qasper`, `multifieldqa_en`, `multifieldqa_zh`
- Multi-Doc QA: `hotpotqa`, `2wikimqa`, `musique`
- Summarization: `gov_report`, `qmsum`, `multi_news`, `vcsum`, `samsum`, `dureader`
- Classification: `trec`, `lsht`
- Retrieval/Counting: `passage_retrieval_en`, `passage_retrieval_zh`, `passage_count`
- Code: `lcc`, `repobench-p`
- Knowledge QA: `triviaqa`

| Config | Single-Doc QA | Multi-Doc QA | Summarization | Classification | Retrieval/Counting | Code | Knowledge QA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pure_int8` | 44.81 | 52.93 | 29.58 | 63.34 | 70.24 | 58.89 | 90.06 |
| `pure_int4` | 44.62 | 52.68 | 29.62 | 64.12 | 70.40 | 58.80 | 90.05 |
| `pure_int2` | 42.98 | 53.09 | 29.62 | 64.75 | 70.72 | 65.75 | 88.09 |
| `delta_int2_k4_unigram_int4` | 44.71 | 53.19 | 29.45 | 63.12 | 70.30 | 59.42 | 89.60 |
| `delta_int2_k2_unigram_int4` | 44.74 | 52.83 | 29.60 | 63.59 | 70.94 | 64.97 | 89.75 |
| `delta_int4_k1_unigram_int4` | 44.62 | 52.73 | 29.52 | 63.75 | 69.90 | 58.85 | 90.21 |
| `delta_noaffine_int2_k4_unigram_int4` | 44.39 | 53.58 | 29.36 | 62.75 | 70.25 | 64.46 | 90.11 |
| `delta_int2_k4_entropy_unigram_int4` | 44.70 | 52.98 | 29.43 | 63.12 | 70.30 | 65.25 | 89.60 |
| `fp16_baseline` | 47.39 | 52.56 | 27.78 | 63.25 | 69.10 | 55.08 | 90.69 |

### Category Takeaways

1. `fp16_baseline` is still strongest on Single-Doc QA and Knowledge QA.
2. Compressed runs are clearly better on Summarization, Retrieval/Counting, and Code tasks.
3. The largest category gain versus FP16 is in Code and Summarization.
4. `pure_int2` is unusually strong on Retrieval/Counting, Classification, and Code, but that comes with clear regressions on some QA tasks.

## Full Task Matrix

| Task | pure_int8 | pure_int4 | pure_int2 | delta_int2_k4_unigram_int4 | delta_int2_k2_unigram_int4 | delta_int4_k1_unigram_int4 | delta_noaffine_int2_k4_unigram_int4 | delta_int2_k4_entropy_unigram_int4 | fp16_baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `trec` | 77.50 | 78.00 | 78.50 | 77.00 | 77.50 | 78.00 | 76.50 | 77.00 | 77.00 |
| `qasper` | 45.57 | 45.34 | 46.24 | 44.77 | 45.46 | 44.98 | 45.00 | 44.77 | 45.31 |
| `gov_report` | 32.38 | 32.53 | 32.57 | 32.48 | 32.60 | 32.50 | 32.22 | 32.67 | 29.54 |
| `vcsum` | 16.06 | 15.91 | 15.66 | 15.87 | 15.94 | 15.84 | 15.73 | 15.81 | 14.68 |
| `lcc` | 65.08 | 64.66 | 65.75 | 65.25 | 64.97 | 64.94 | 64.46 | 65.25 | 61.33 |
| `multi_news` | 24.60 | 24.62 | 24.83 | 24.55 | 24.34 | 24.59 | 24.43 | 24.41 | 21.95 |
| `dureader` | 32.10 | 31.90 | 31.89 | 31.58 | 31.80 | 31.88 | 31.33 | 31.58 | 29.39 |
| `qmsum` | 24.87 | 24.77 | 24.65 | 24.34 | 24.77 | 24.58 | 24.50 | 24.41 | 23.41 |
| `passage_retrieval_zh` | 98.58 | 98.58 | 97.58 | 99.58 | 99.08 | 98.58 | 98.75 | 99.58 | 98.50 |
| `2wikimqa` | 58.52 | 57.60 | 59.79 | 58.64 | 58.87 | 57.77 | 58.52 | 58.34 | 58.71 |
| `passage_retrieval_en` | 98.75 | 98.75 | 99.00 | 98.75 | 99.00 | 98.75 | 98.75 | 98.75 | 98.67 |
| `narrativeqa` | 16.26 | 16.40 | 16.30 | 16.24 | 15.89 | 16.12 | 15.36 | 16.18 | 28.01 |
| `samsum` | 47.45 | 47.96 | 48.14 | 47.85 | 48.15 | 47.71 | 47.93 | 47.69 | 47.69 |
| `multifieldqa_zh` | 62.84 | 62.93 | 56.09 | 63.05 | 62.99 | 62.95 | 62.57 | 63.08 | 62.57 |
| `lsht` | 49.17 | 50.25 | 51.00 | 49.25 | 49.67 | 49.50 | 49.00 | 49.25 | 49.50 |
| `multifieldqa_en` | 54.57 | 53.79 | 53.31 | 54.77 | 54.62 | 54.41 | 54.63 | 54.78 | 53.68 |
| `passage_count` | 13.38 | 13.88 | 15.57 | 12.58 | 14.75 | 12.38 | 13.25 | 12.58 | 10.13 |
| `repobench-p` | 52.71 | 52.95 | NA | 53.60 | NA | 52.76 | NA | NA | 48.82 |
| `musique` | 37.71 | 37.81 | 37.65 | 38.19 | 37.73 | 37.26 | 39.19 | 38.18 | 37.43 |
| `hotpotqa` | 62.55 | 62.63 | 61.82 | 62.74 | 61.88 | 63.17 | 63.02 | 62.41 | 61.53 |
| `triviaqa` | 90.06 | 90.05 | 88.09 | 89.60 | 89.75 | 90.21 | 90.11 | 89.60 | 90.69 |

## Best Config Per Task

| Task | Best Config | Best Score | Delta vs FP16 |
|---|---|---:|---:|
| `trec` | `pure_int2` | 78.50 | +1.50 |
| `qasper` | `pure_int2` | 46.24 | +0.93 |
| `gov_report` | `delta_int2_k4_entropy_unigram_int4` | 32.67 | +3.13 |
| `vcsum` | `pure_int8` | 16.06 | +1.38 |
| `lcc` | `pure_int2` | 65.75 | +4.42 |
| `multi_news` | `pure_int2` | 24.83 | +2.88 |
| `dureader` | `pure_int8` | 32.10 | +2.71 |
| `qmsum` | `pure_int8` | 24.87 | +1.46 |
| `passage_retrieval_zh` | `delta_int2_k4_unigram_int4` / `delta_int2_k4_entropy_unigram_int4` | 99.58 | +1.08 |
| `2wikimqa` | `pure_int2` | 59.79 | +1.08 |
| `passage_retrieval_en` | `pure_int2` / `delta_int2_k2_unigram_int4` | 99.00 | +0.33 |
| `narrativeqa` | `pure_int4` | 16.40 | -11.61 |
| `samsum` | `delta_int2_k2_unigram_int4` | 48.15 | +0.46 |
| `multifieldqa_zh` | `delta_int2_k4_entropy_unigram_int4` | 63.08 | +0.51 |
| `lsht` | `pure_int2` | 51.00 | +1.50 |
| `multifieldqa_en` | `delta_int2_k4_entropy_unigram_int4` | 54.78 | +1.10 |
| `passage_count` | `pure_int2` | 15.57 | +5.44 |
| `repobench-p` | `delta_int2_k4_unigram_int4` | 53.60 | +4.78 |
| `musique` | `delta_noaffine_int2_k4_unigram_int4` | 39.19 | +1.76 |
| `hotpotqa` | `delta_int4_k1_unigram_int4` | 63.17 | +1.64 |
| `triviaqa` | `delta_int4_k1_unigram_int4` | 90.21 | -0.48 |

## Main Findings

### 1. Compression is competitive under CPU table only

The main high-level result is positive: CPU-table-only compression does not degrade official LongBench overall. On the shared 20-task average, every compressed configuration is above FP16.

This means the earlier CPU-table architectural changes did not create an obvious quality collapse in the official benchmark setting.

### 2. The strongest overall candidates are still conservative configurations

If the goal is the safest deployment choice, the best current candidates are:

1. `pure_int4`
2. `pure_int8`
3. `delta_int2_k4_unigram_int4`

Reasoning:

- they are among the best topline scores
- they have full 21-task coverage
- they do not show extreme outlier regressions outside the common narrativeqa issue

### 3. `pure_int2` is strong but less stable

`pure_int2` is the most aggressive pure compression setting and it wins several tasks:

- `trec`
- `qasper`
- `lcc`
- `multi_news`
- `2wikimqa`
- `lsht`
- `passage_count`

But it also has the clearest secondary regressions beyond the common narrativeqa issue:

- `multifieldqa_zh`: 56.09 vs 62.57 FP16
- `triviaqa`: 88.09 vs 90.69 FP16
- `passage_retrieval_zh`: 97.58 vs 98.50 FP16

So `pure_int2` looks more like a high-risk/high-upside option than a default choice.

### 4. The single biggest problem is `narrativeqa`

This is the dominant negative result in the entire study.

`narrativeqa` scores:

- FP16: 28.01
- best compressed: 16.40 (`pure_int4`)
- worst compressed: 15.36 (`delta_noaffine_int2_k4_unigram_int4`)

Every compressed configuration drops by about 11.6 to 12.7 points here. Because this regression is universal, it is very unlikely to be caused by a single bad strategy choice. More likely causes are:

- a systematic sensitivity of this task to the compressed activation path
- task formatting or generation-style interaction under compressed decoding
- a shared weakness in long single-document QA reconstruction fidelity

This task should be treated as the highest-priority debugging target.

### 5. Summarization and code tasks benefit the most

The clearest gains over FP16 are concentrated in:

- `gov_report`
- `multi_news`
- `dureader`
- `qmsum`
- `lcc`
- `repobench-p`

Representative gains:

- `gov_report`: up to +3.13
- `lcc`: up to +4.42
- `repobench-p`: up to +4.78
- `passage_count`: up to +5.44

This suggests the compression pipeline is not merely preserving quality; in some tasks it may be acting like a regularizer or inducing slightly more task-favorable decoding behavior.

### 6. Delta variants are not clearly dominating pure variants

There is no strong evidence here that the delta-family configurations decisively beat the simpler pure quantized ones.

- `pure_int4` is essentially tied with the best delta settings
- `pure_int8` also remains very competitive
- `delta_int2_k4_unigram_int4` is the best delta variant among full 21-task runs

So if operational simplicity matters, the pure settings currently look more attractive than expected.

## Practical Recommendation

### Best default candidates

If only one or two configurations should be taken forward, use:

1. `pure_int4`
2. `pure_int8`

Why:

- best or near-best topline
- full coverage
- simple strategies
- no additional evidence yet that more complex delta variants buy a clearly better overall result

### Best delta candidate

If one delta strategy should be retained, use:

1. `delta_int2_k4_unigram_int4`

Why:

- full 21-task coverage
- strong overall score
- best `repobench-p`
- strong retrieval and multilingual QA behavior

### Highest-priority follow-up investigation

The next debugging target should be:

1. `narrativeqa`

The regression is too large and too consistent to ignore.

Secondary targets if more analysis is needed:

1. `multifieldqa_zh` under `pure_int2`
2. `triviaqa` under aggressive int2 settings
3. why some runs are still missing `repobench-p` in final `result.json`

## Bottom Line

The CPU-table-only version is working well enough to support official LongBench evaluation without an overall quality penalty. In fact, the compressed configurations are slightly better than FP16 on average across the shared official tasks.

But the result is not uniformly positive. The study currently says:

- overall benchmark quality is preserved or slightly improved
- summarization, retrieval/counting, and code tasks are the strongest wins
- `narrativeqa` is the major unresolved quality failure
- the simplest strong options, especially `pure_int4` and `pure_int8`, are currently the most defensible deployment choices