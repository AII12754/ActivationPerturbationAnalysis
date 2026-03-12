# Activation Perturbation Analysis

A research framework for studying **activation drift in large language models** when a portion of the input prompt is perturbed.

## Purpose

This framework measures how **intermediate hidden-state activations** change when a contiguous span of tokens in a prompt is modified. By comparing the original and perturbed forward passes layer-by-layer, we can understand:

- How perturbations propagate through the context window
- How similarity decays with distance from the perturbation site
- How different perturbation types (deletion, random replacement, paraphrasing, etc.) affect downstream representations
- How these patterns vary with context length, perturbation position, and span size

## Project Structure

```
├── config/
│   └── default.yaml          # Experiment configuration
├── src/
│   ├── prompts.py            # Prompt generation (datasets / templates)
│   ├── perturbations.py      # Perturbation strategies
│   ├── model.py              # Model loading & hidden-state extraction
│   ├── metrics.py            # Cosine similarity & distance buckets
│   ├── experiment.py         # Single-experiment runner
│   ├── sweep.py              # Full sweep orchestration
│   ├── scheduler.py          # GPU detection & parallel scheduling
│   ├── storage.py            # Parquet result storage & checkpointing
│   └── analysis.py           # Visualization & analysis
├── run_sweep.py              # CLI: run experiments
├── run_analysis.py           # CLI: generate figures
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the Model Path

Edit `config/default.yaml` and set `model.path` to a local HuggingFace model directory:

```yaml
model:
  path: "/path/to/your/model"
  dtype: "float16"
```

### 3. Run Experiments

```bash
# Sequential mode (single model copy, safest for large models):
python run_sweep.py --config config/default.yaml

# Parallel mode (one model copy per GPU, faster for smaller models):
python run_sweep.py --config config/default.yaml --parallel
```

### 4. Generate Figures

```bash
python run_analysis.py --results ./results
```

Figures are saved to `./results/figures/`.

## Configuration Reference

All settings live in `config/default.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `model` | Model path, dtype, memory limits |
| `gpu` | Free-memory threshold, max parallel workers |
| `sweep` | Context lengths, perturbation positions/types/spans |
| `measurement` | Window sizes for distance-based similarity |
| `prompts` | Source dataset, seed, number of candidates |
| `storage` | Output directory, format, checkpoint frequency |
| `logging` | Log level and file path |

## Experimental Variables

The sweep runs the Cartesian product of:

- **Context lengths**: 256, 512, 1024, 2048, 4096, 8192, 16384 tokens
- **Perturbation positions**: 10%, 20%, 80%, 90% of the sequence
- **Span lengths**: 8, 32, 128, 512 tokens
- **Perturbation types**: `random_replace`, `mask`, `delete`, `compress`, `paraphrase`, `semantic_change`

Impossible combinations (e.g., 512-token span at 90% of a 256-token sequence) are automatically skipped.

## Perturbation Types

| Type | Description |
|------|-------------|
| `random_replace` | Replace span tokens with uniformly random token IDs |
| `mask` | Replace span tokens with the model's `[UNK]` token |
| `delete` | Remove the span entirely (shorter sequence) |
| `compress` | Keep every other token in the span (~50% compression) |
| `paraphrase` | Shuffle words in the span (preserves bag-of-words) |
| `semantic_change` | Replace span with tokens from a different passage |

## GPU Handling

The system uses `pynvml` to detect GPUs and their free memory. GPUs below the configured `min_free_memory_gb` threshold are excluded. In parallel mode, each GPU group runs an independent model copy.

## Checkpointing & Resume

Results are periodically flushed to Parquet partitions. A `checkpoint.json` file tracks completed experiment IDs. Re-running the sweep automatically skips finished experiments.

## Generated Figures

- **Similarity vs Distance**: how cosine similarity recovers as tokens move away from the perturbation
- **Layer Heatmaps**: per-layer  × perturbation-type similarity
- **Context Length Comparison**: effect of sequence length
- **Position Effect**: how perturbation placement matters
- **Span Length Effect**: impact of perturbation size
- **Per-Token Curves**: fine-grained token-by-token similarity
- **Distribution Plots**: box plots across perturbation types

## Design Decisions

- **Alignment by offset**: for perturbations that change sequence length (delete, compress), tokens after the perturbation are aligned by their *offset from the perturbation end*, not by absolute position. This isolates the perturbation's propagation effect from sequence-length mismatches.
- **Periodic sampling**: rather than computing similarity for every token (expensive for long sequences), we sample 10-token windows every 100 tokens. This gives a representative curve without storing millions of records.
- **Sequential default**: the default mode loads one model and runs experiments sequentially. This is the safest option for large models that consume most of a GPU's memory. Parallel mode is opt-in for smaller models.
