# Delta-Coding System for Pipeline-Parallel Activation Compression

Production-quality delta-coding system with overlapped CPU/GPU pipeline for compressing intermediate activations in pipeline-parallel LLM inference.

## Architecture

```
Sender (PP Stage 0)                          Receiver (PP Stage 1)
┌─────────────────────────────────┐          ┌───────────────────────┐
│ ① Prefill forward (GPU)        │          │                       │
│ ② Classify (CPU, overlapped)   │ ──2.5×──→ │ ⑤ Decode + reconstruct│
│ ③ Tiered encode (GPU)          │  smaller  │                       │
│ ④ Table update (CPU, async)    │          │                       │
└─────────────────────────────────┘          └───────────────────────┘
```

### Tier Hierarchy (cascade fallback)

| Tier | Match Condition | Encoding | Quality |
|------|----------------|----------|---------|
| **Trigram** | DAG exact (A,B,C) | Affine + Int4 delta | 0.9996 |
| **Self-ref** | Same trigram in request | Affine + Int4 delta | 0.9996 |
| **Bigram** | DAG prefix (B,C) | Affine + Int4 delta | 0.9991 |
| **Unigram** | No match | Int8 + top-K outlier | 0.9999 |

### Key Features

- **FP8 table storage**: `float8_e4m3fn` halves memory with < 0.000005 cosine loss
- **LRU eviction**: Frequency-weighted scoring bounds table growth
- **CPU/GPU overlap**: Classify during forward, table update after send
- **Both prefill and decode**: Full pipeline for both phases

## Quick Start

```bash
# Run experiment on a single dataset (quick test)
python -m delta_coding_system.run_experiment \
  --gpu 1 \
  --datasets gsm8k \
  --warmup-requests 5 \
  --test-requests 5

# Full experiment across all datasets
python -m delta_coding_system.run_experiment --gpu 1

# Generate report and plots
python -m delta_coding_system.analyze --input-dir results_delta_system
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `/root/share/models/Qwen2.5-32B-Instruct` | Model path |
| `--gpu` | 0 | GPU index |
| `--layer-boundary` | 6 | PP split layer |
| `--table-dtype` | `float8_e4m3fn` | Table storage dtype |
| `--max-table-entries` | 100000 | LRU eviction threshold (0=unlimited) |
| `--group-size` | 128 | Int4 quantization group size |
| `--top-k` | 1 | Outliers per group |
| `--decode-tokens` | 128 | Tokens to generate per request |
| `--warmup-requests` | 50 | Table warmup requests |
| `--test-requests` | 50 | Measured test requests |

## Package Structure

```
delta_coding_system/
├── __init__.py          # Package exports
├── table.py             # NgramTable: DAG trie with FP8, LRU
├── codec.py             # Encode/decode: affine, Int4, Int8
├── pipeline.py          # OverlappedPipeline: the core system
├── run_experiment.py    # Experiment runner (6 datasets)
├── analyze.py           # Report + plot generation
├── README.md            # This file
└── report/              # Generated outputs
```

## API Usage

```python
from delta_coding_system.pipeline import OverlappedPipeline

pipeline = OverlappedPipeline(
    model=model,
    tokenizer=tokenizer,
    layer_boundary=6,
    table_dtype=torch.float8_e4m3fn,
    max_table_entries=100000,
    device=torch.device("cuda:0"),
)

# Process a single request (prefill + decode)
prefill_result, decode_result, table_stats = pipeline.process_request(
    text="What is the meaning of life?",
    phase="test",  # or "warmup"
)

print(f"Prefill: cos={prefill_result.recon_cosine_mean:.4f}, "
      f"ratio={prefill_result.compression_ratio:.2f}x")
print(f"Decode:  cos={decode_result.recon_cosine_mean:.4f}, "
      f"ratio={decode_result.compression_ratio:.2f}x")
```

## Output Files

Per dataset in `results_delta_system/{dataset}/`:

| File | Contents |
|------|----------|
| `prefill_quality.parquet` | Per-request prefill metrics |
| `tier_detail.parquet` | Per-tier quality breakdown |
| `decode_step_detail.parquet` | Per-step decode metrics |
| `decode_aggregate.parquet` | Per-request decode aggregates |
| `table_growth.parquet` | Table size over time |
