# activation_science

Unified framework for analyzing LLM internal activations during autoregressive decoding. Consolidates nine experiments (E0–E8) with shared infrastructure for model loading, GPU scheduling, metric computation, and Parquet-based result storage.

## Package structure

```
activation_science/
├── core/                    # Infrastructure: model, extraction, storage, scheduling
│   ├── types.py             # DTYPE_MAP, ExperimentJob, make_experiment_id, resolve_dtype
│   ├── model.py             # load_model_and_tokenizer, extract_hidden_states
│   ├── extraction.py        # ActivationBatch, prefill, decode_step, select_next_token
│   ├── datasets.py          # PromptGenerator (HF datasets, parquet, JSON, templates)
│   ├── storage.py           # ExperimentStore — multi-table Parquet with checkpoint/resume
│   └── scheduler.py         # GPUInfo, detect_gpus, ExperimentScheduler (multi-GPU)
├── metrics/                 # Reusable metric library
│   ├── similarity.py        # Cosine similarity, ActivationHistoryBuffer, CKA, Procrustes
│   ├── spectral.py          # SVD effective rank, participation ratio, isotropy
│   ├── information.py       # Logit lens entropy, KL divergence, rank
│   ├── dynamics.py          # Residual updates, PCA basis fitting & projection
│   ├── intervention.py      # Forward-hook activation patching, mean ablation, perturbation
│   └── clustering.py        # Regime change detection, HDBSCAN/KMeans, Fisher discriminant
├── experiments/             # Experiment implementations
│   ├── base.py              # BaseExperiment ABC + GenericSweepRunner
│   ├── e0_decode_similarity.py
│   ├── e1_residual_decomposition.py
│   ├── e2_logit_lens.py
│   ├── e3_representation_geometry.py
│   ├── e4_cross_sequence_alignment.py
│   ├── e5_state_detection.py
│   ├── e6_causal_intervention.py
│   ├── e7_perturbation_sensitivity.py
│   └── e8_token_type_analysis.py
├── analysis/                # Post-hoc analysis and plotting
│   ├── base.py              # BaseAnalysis ABC
│   ├── per_experiment/
│   │   └── e0_plots.py      # 27 decode-similarity figures
│   └── cross_experiment/
│       ├── phase_boundary_detection.py
│       └── semantic_bottleneck.py
├── config/                  # YAML configuration (deep-merged)
│   ├── base.yaml            # Shared defaults (model, GPU, sweep, prompts, storage)
│   └── experiments/
│       ├── e0.yaml – e8.yaml
└── runners/                 # CLI entry points
    ├── run_experiment.py     # python -m activation_science.runners.run_experiment e0
    └── run_analysis.py       # python -m activation_science.runners.run_analysis e0
```

## Quick start

### Run an experiment

```bash
# Sequential (single GPU)
python -m activation_science.runners.run_experiment e0

# Parallel (multi-GPU)
python -m activation_science.runners.run_experiment e0 --parallel

# Custom config override
python -m activation_science.runners.run_experiment e2 --config my_overrides.yaml
```

### Run analysis

```bash
python -m activation_science.runners.run_analysis e0
python -m activation_science.runners.run_analysis e0 --results ./results_decode --figures ./figures
```

### Programmatic usage

```python
from activation_science.experiments.e0_decode_similarity import DecodeSimilarityExperiment
from activation_science.experiments.base import GenericSweepRunner
from activation_science.runners.run_experiment import load_config

config = load_config("e0")                          # base ← e0.yaml
runner = GenericSweepRunner(DecodeSimilarityExperiment)
runner.run_sequential(config)
```

### Load results

```python
from activation_science.core.storage import ExperimentStore

topk = ExperimentStore.load_table("./results_decode", "topk")   # 10M+ rows
agg  = ExperimentStore.load_table("./results_decode", "agg")
```

## Experiments

| ID | Name | Tables | What it measures |
|----|------|--------|------------------|
| E0 | Decode Similarity | `topk`, `agg` | Per-token cosine similarity to all prior hidden states during decode |
| E1 | Residual Decomposition | `residual` | Layer-wise residual stream update norms and directional alignment |
| E2 | Logit Lens | `perstep`, `topk` | Per-layer logit distributions — entropy, KL from final, prediction rank |
| E3 | Representation Geometry | `summary`, `spectrum` | SVD effective rank, participation ratio, isotropy per layer |
| E4 | Cross-Sequence Alignment | `alignment` | Linear CKA, Procrustes distance, subspace overlap between two prompts |
| E5 | State Detection | `regime`, `trajectory` | PCA-projected activation trajectories with regime change detection |
| E6 | Causal Intervention | `causal` | KL divergence and prediction flips from zeroing/ablating/skipping layers |
| E7 | Perturbation Sensitivity | `perturbation` | Output sensitivity to noise along SVD vs random directions |
| E8 | Token Type Analysis | `token_type`, `separability` | Per-token-type geometry metrics and Fisher discriminant separability |

## Architecture

### Experiment lifecycle

1. **Config loading** — `base.yaml` ← `experiments/{id}.yaml` ← CLI override (deep merge)
2. **Job construction** — `build_sweep_jobs(config)` generates parameter grid (datasets × context lengths × prompts × experiment params)
3. **Execution** — `GenericSweepRunner` loads model once, iterates jobs, calls `experiment.run()` per job
4. **Storage** — Records buffered in `ExperimentStore`, flushed to `{table}_part_{N:06d}.parquet`
5. **Checkpointing** — Completed experiment IDs written to `checkpoint.json`; interrupted runs resume automatically

### Key abstractions

**`BaseExperiment`** — every experiment implements:

```python
class MyExperiment(BaseExperiment):
    experiment_id = "eN"
    experiment_name = "My Experiment"

    @classmethod
    def table_schemas(cls) -> Dict[str, List[str]]: ...
    @classmethod
    def default_config_section(cls) -> str: ...
    @classmethod
    def build_sweep_jobs(cls, config) -> List[ExperimentJob]: ...
    def run(self, model, tokenizer, prompt_text, config) -> Dict[str, List[Dict]]: ...
```

**`ExperimentStore`** — multi-table append-only Parquet storage:

```python
store = ExperimentStore("./results_e0")
store.register_table("topk", TOPK_COLUMNS)
store.register_table("agg", AGG_COLUMNS)
store.add_records("topk", experiment_id, records)
store.mark_completed(experiment_id)   # periodic flush + checkpoint
```

**`ActivationBatch`** — wraps a single forward pass:

```python
batch = prefill(model, input_ids)          # prompt encoding
batch = decode_step(model, token, batch.past_key_values)  # one decode step
next_token = select_next_token(batch.last_logits)
```

### GPU optimization

All performance-critical patterns are preserved:

- **Tensors stay on GPU** — cosine similarity via `torch.bmm`; only final scalars transferred to CPU
- **Pre-allocated buffers** — `ActivationHistoryBuffer` avoids per-step allocations
- **KV-cache reuse** — `past_key_values` propagated through `ActivationBatch` across decode steps
- **Hidden states as tuple** — never stacked into a single tensor (avoids 2× memory)
- **Multi-GPU** — `ExperimentScheduler` spawns isolated processes with `CUDA_VISIBLE_DEVICES` pinning

## Configuration

Configs are deep-merged in order: `config/base.yaml` ← `config/experiments/{id}.yaml` ← user `--config`.

```yaml
# base.yaml (shared defaults)
model:
  path: "/path/to/model"
  dtype: "bfloat16"
gpu:
  min_free_memory_gb: 40
sweep:
  context_lengths: [256, 512, 1024, 2048]
  num_prompts_per_length: 3
prompts:
  source: "dataset"
  datasets:
    - name: "wikitext"
      config: "wikitext-103-raw-v1"
      split: "test"
storage:
  checkpoint_every: 5
```

```yaml
# experiments/e0.yaml (experiment-specific)
decode:
  num_decode_tokens: 128
  top_k_references: 10
  similarity_thresholds: [0.70, 0.90, 0.95, 0.98]
storage:
  output_dir: "./results_decode"
```

## Metrics reference

| Module | Key functions | Used by |
|--------|--------------|---------|
| `similarity` | `compute_reference_similarity`, `linear_cka`, `procrustes_distance`, `subspace_overlap` | E0, E4 |
| `spectral` | `compute_svd_metrics`, `compute_isotropy` | E3, E7 |
| `information` | `compute_logit_lens_metrics` | E2 |
| `dynamics` | `compute_residual_update`, `fit_pca_basis`, `project_to_pca` | E1, E5 |
| `intervention` | `activation_patch`, `mean_ablation`, `directional_perturb` | E6, E7 |
| `clustering` | `detect_regime_changes`, `cluster_trajectories`, `compute_fisher_discriminant` | E5, E8 |

## Backward compatibility

The old `src/` package and top-level `run_*.py` scripts continue to work. Existing results in `results_decode/` load through `ExperimentStore.load_table()` without modification — the Parquet file naming convention (`{table}_part_{N:06d}.parquet`) and checkpoint format (`{"completed_experiments": [...]}`) are identical.

Migration map:

| Old | New |
|-----|-----|
| `python run_decode_sweep.py --config config/decode_similarity.yaml` | `python -m activation_science.runners.run_experiment e0` |
| `python run_decode_analysis.py --results ./results_decode` | `python -m activation_science.runners.run_analysis e0` |
| `from src.decode_experiment import run_decode_experiment` | `from activation_science.experiments.e0_decode_similarity import DecodeSimilarityExperiment` |
| `from src.decode_storage import DecodeResultStore` | `from activation_science.core.storage import ExperimentStore` |
| `from src.metrics import cosine_similarity_per_token` | `from activation_science.metrics.similarity import cosine_similarity_per_token` |
