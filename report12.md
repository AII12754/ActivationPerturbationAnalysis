Total records: 1400 | Warmed records (req >= 50): 700
Datasets: ['ShareGPT', 'alpaca', 'cnndn', 'gsm8k', 'trivia_qa-rc', 'wikitext', 'xsum']
Context lengths: [512, 2048]


================================================================================
  E12: TRIGRAM DELTA-CODING PIPELINE — PERFORMANCE REPORT
  Model: Qwen2.5-32B-Instruct (32B params, hidden_dim=5120)
  Layer boundary: 6 (first 6 transformer layers)
  Warmup: 50 requests | Test: 50 requests per (dataset, context_length)
  Datasets: 7 | Context lengths: 512, 2048
================================================================================

1. SYSTEM DESIGN OVERVIEW
─────────────────────────
The trigram delta-coding pipeline compresses intermediate activations at a
pipeline-parallel (PP) boundary layer by exploiting n-gram repetition across
requests. A persistent DAG trie stores pre-computed hidden states keyed by
token n-grams, enabling cache-like reuse of activation patterns.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Sender (PP stage 0)                                           │
  │                                                                 │
  │  ① Prefill (GPU)  ──────────────────┐                          │
  │  ② Classify (CPU, overlapped with ①) │  concurrent             │
  │  ③ Encode per tier (GPU):            │                          │
  │     • Trigram/Bigram: Affine + Int4 delta + Top-K sparsity     │
  │     • Self-ref: Delta against earlier reconstructed position   │
  │     • Unigram: Int8 groupwise + outlier Top-K                  │
  │  ④ Table update (CPU, overlapped with ③)                       │
  └─────────────────────────────────────────────────────────────────┘
       │ compressed packet
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Receiver (PP stage 1)                                         │
  │                                                                 │
  │  ⑤ Decode + reconstruct activations                            │
  └─────────────────────────────────────────────────────────────────┘

Tier Hierarchy (cascading fallback):
  1. TRIGRAM — exact (A,B,C) match in DAG → Affine + Int4 delta coding
  2. BIGRAM — (B,C) prefix match → Affine + Int4 delta coding
  3. SELF-REF — same trigram seen earlier in this request → delta vs.
     already-reconstructed position
  4. UNIGRAM — no match → Int8 groupwise quantization + outlier encoding

Encoding Details:
  • Affine transform: per-position scale & bias to align ref → real
  • Int4 delta: groupwise quantization (group_size=128) of residual
  • Top-K sparsity: keep top-1 outlier indices per group
  • Int8 unigram: groupwise 8-bit quantization + top-1 FP16 outliers

CPU/GPU Overlap Optimization:
  • classify_and_build_refs runs on a background CPU thread concurrently
    with the GPU prefill pass (GIL released during CUDA kernel launches)
  • update_from_hidden_states runs on a background CPU thread concurrently
    with the GPU encoding phases
  • Both CPU ops are fully hidden behind GPU work (~0ms visible latency)

2. TIER MATCH RATES (warmed table, requests 50-99)
────────────────────────────────────────────────────────────

  Context length = 512:
    trigram   :  17.04% ± 11.80%
    bigram    :  21.81% ± 6.04%
    self_ref  :   9.45% ± 10.21%
    unigram   :  51.70% ± 12.95%

  Context length = 2048:
    trigram   :  24.47% ± 17.76%
    bigram    :  22.16% ± 7.46%
    self_ref  :  19.34% ± 13.95%
    unigram   :  34.02% ± 12.48%

  Per-dataset trigram+self_ref coverage (ctx=2048):
              trigram%  self_ref%  covered%
ds_short                                   
ShareGPT         15.65      33.31     62.88
alpaca           12.74      41.70     72.09
cnndn            12.16      10.67     52.25
gsm8k            41.21      10.36     76.23
trivia_qa-rc     54.26      11.65     79.24
wikitext         22.16      13.20     62.94
xsum             13.13      14.51     56.23

  [Saved: report_e12/tier_distribution.png]


3. RAW REFERENCE COSINE SIMILARITY (before delta coding)
────────────────────────────────────────────────────────────

  Context length = 512:
    trigram   : mean=0.955384  worst-case-min=0.789354
    bigram    : mean=0.912391  worst-case-min=0.652681
    self_ref  : mean=0.943625  worst-case-min=0.839583

  Context length = 2048:
    trigram   : mean=0.960112  worst-case-min=0.718891
    bigram    : mean=0.917180  worst-case-min=0.564802
    self_ref  : mean=0.959450  worst-case-min=0.753968
  [Saved: report_e12/raw_cosine_per_tier.png]


4. RECONSTRUCTION QUALITY (after delta coding)
────────────────────────────────────────────────────────────

  Context length = 512:
    Cosine similarity:  mean=0.999703  avg-min=0.997064
    MSE:                mean=0.00130031  avg-max=0.28263383

  Context length = 2048:
    Cosine similarity:  mean=0.999670  avg-min=0.996532
    MSE:                mean=0.00096354  avg-max=0.28249780

  Per-tier reconstruction cosine (ctx=2048):
    trigram   : mean=0.999616  avg-min=0.997555
    bigram    : mean=0.999202  avg-min=0.996577
    self_ref  : mean=0.999602  avg-min=0.997787
    unigram   : mean=0.999978  avg-min=0.999967

  [Saved: report_e12/reconstruction_cosine.png]


5. COMPRESSION RATIO
────────────────────────────────────────────────────────────

  Context length = 512:
    Compression ratio:    2.47× ± 0.21×
    Aggregate:            1835.0 MB raw → 747.2 MB compressed
    trigram    transfer:  87.11 MB (11.7%)
    bigram     transfer:  111.46 MB (14.9%)
    self_ref   transfer:  48.28 MB (6.5%)
    unigram    transfer:  500.31 MB (67.0%)

  Context length = 2048:
    Compression ratio:    2.77× ± 0.24×
    Aggregate:            7339.8 MB raw → 2665.6 MB compressed
    trigram    transfer:  500.25 MB (18.8%)
    bigram     transfer:  453.05 MB (17.0%)
    self_ref   transfer:  395.46 MB (14.8%)
    unigram    transfer:  1316.82 MB (49.4%)

  [Saved: report_e12/compression_ratio.png]


6. LATENCY BREAKDOWN (with CPU/GPU overlap)
────────────────────────────────────────────────────────────

  Context length = 512:
    prefill             :   836.32 ms ± 443.13
    classify            :     0.04 ms ± 0.01
    encode delta        :     7.38 ms ± 23.03
    encode self ref     :     5.42 ms ± 38.53
    encode unigram      :     9.00 ms ± 43.92
    decode              :     0.00 ms ± 0.00
    table update        :     0.01 ms ± 0.00
    total               :   902.00 ms ± 450.48

    classify visible (overlapped):    0.04 ms  (~0 = fully hidden)
    table_update visible (overlapped): 0.01 ms  (~0 = fully hidden)

  Context length = 2048:
    prefill             :  1515.68 ms ± 184.76
    classify            :     0.10 ms ± 0.09
    encode delta        :    12.11 ms ± 20.04
    encode self ref     :     4.16 ms ± 26.64
    encode unigram      :     2.61 ms ± 17.88
    decode              :     0.00 ms ± 0.00
    table update        :     0.01 ms ± 0.00
    total               :  1565.07 ms ± 188.43

    classify visible (overlapped):    0.10 ms  (~0 = fully hidden)
    table_update visible (overlapped): 0.01 ms  (~0 = fully hidden)

  [Saved: report_e12/latency_breakdown.png]


7. END-TO-END LATENCY vs. BANDWIDTH (pipeline-parallel transfer)
────────────────────────────────────────────────────────────

  E2E latency model (cross-node pipeline-parallel):
    Original:  T_prefill + T_transfer_raw
    Ours:      T_prefill + T_encode + T_transfer_compressed
  where T_transfer = bytes / bandwidth

  Target scenario: cross-node PP over 200 Mbps–1 Gbps links
  (e.g., cloud VPC, WAN, or bandwidth-constrained cluster interconnect)

  T_prefill ≈ prefill_ms (GPU forward pass)
  T_encode  ≈ encode_delta_ms + encode_self_ref_ms + encode_unigram_ms


  Context length = 512 (seq_len≈512):
    Raw FP16 transfer:    5120.0 KB
    Compressed transfer:  2084.7 KB
    Prefill time:         836.32 ms
    Encode overhead:      21.79 ms
         Bandwidth   Original (ms)    Ours (ms)    Speedup
    ───────────────────────────────────────────────────────
          200 Mbps         1046.04       943.51       1.11×
          500 Mbps          920.21       892.27       1.03×
            1 Gbps          878.27       875.20       1.00×
            2 Gbps          862.54       868.79       0.99×
            4 Gbps          846.81       862.39       0.98×
            8 Gbps          841.57       860.25       0.98×
           40 Gbps          837.37       858.54       0.98×
           80 Gbps          836.85       858.33       0.97×
          200 Gbps          836.53       858.20       0.97×

  Context length = 2048 (seq_len≈2048):
    Raw FP16 transfer:    20479.2 KB
    Compressed transfer:  7437.4 KB
    Prefill time:         1515.68 ms
    Encode overhead:      18.88 ms
         Bandwidth   Original (ms)    Ours (ms)    Speedup
    ───────────────────────────────────────────────────────
          200 Mbps         2354.51      1839.20       1.28×
          500 Mbps         1851.21      1656.42       1.12×
            1 Gbps         1683.45      1595.49       1.06×
            2 Gbps         1620.53      1572.64       1.03×
            4 Gbps         1557.62      1549.79       1.01×
            8 Gbps         1536.65      1542.18       1.00×
           40 Gbps         1519.87      1536.09       0.99×
           80 Gbps         1517.78      1535.32       0.99×
          200 Gbps         1516.52      1534.87       0.99×

  [Saved: report_e12/e2e_latency_vs_bandwidth.png]
  [Saved: report_e12/speedup_vs_bandwidth.png]


8. TABLE GROWTH TRAJECTORY
────────────────────────────────────────────────────────────

  Context length = 512:
    After request  0: 461 trigrams, 418 bigrams
    After request 50: 20,869 trigrams, 15,804 bigrams
    After request 99: 39,188 trigrams, 28,198 bigrams
    Memory at req 99: 690.0 MB

  Context length = 2048:
    After request  0: 1,636 trigrams, 1,400 bigrams
    After request 50: 70,049 trigrams, 47,307 bigrams
    After request 99: 126,147 trigrams, 79,695 bigrams
    Memory at req 99: 2107.8 MB

  [Saved: report_e12/table_growth.png]


9. PER-DATASET BREAKDOWN (warmed period, ctx=2048)
────────────────────────────────────────────────────────────
                 tri%      bi%    self%     uni%  cos_mean  cos_min   ratio  raw_cos
ds_short                                                                            
ShareGPT      15.6475  13.9209  33.3105  37.1211    0.9997   0.9965  2.7427   0.9453
alpaca        12.7412  17.6436  41.7031  27.9121    0.9996   0.9967  2.8759   0.9490
cnndn         12.1561  29.4180  10.6739  47.7520    0.9996   0.9962  2.5186   0.9298
gsm8k         41.2109  24.6562  10.3584  23.7744    0.9997   0.9972  2.9629   0.9592
trivia_qa-rc  54.2601  13.3227  11.6537  20.7634    0.9997   0.9965  3.0353   0.9654
wikitext      22.1562  27.5830  13.2002  37.0605    0.9996   0.9964  2.6990   0.9399
xsum          13.1260  28.5943  14.5130  43.7667    0.9997   0.9962  2.5819   0.9419


10. SUMMARY TABLE
────────────────────────────────────────────────────────────
 Context Trigram% SelfRef% Coverage% Cosine CompRatio Prefill(ms) Encode(ms) Classify(ms) Update(ms) Total(ms)
     512     17.0      9.4      48.3 0.9997     2.47×       836.3       21.8         0.04       0.01     902.0
    2048     24.5     19.3      66.0 0.9997     2.77×      1515.7       18.9         0.10       0.01    1565.1

  [Saved: report_e12/recon_cosine_boxplot.png]
  [Saved: report_e12/coverage_growth.png]
  [Saved: report_e12/cosine_over_requests.png]


================================================================================
  REPORT COMPLETE
  All figures saved to: /root/ActivationPerturbationAnalysis/report_e12/
  Data source: /root/ActivationPerturbationAnalysis/results_trigram_pipeline/
================================================================================

