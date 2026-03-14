Implementation Plan: Activation Science Unified Framework                                                                                  
                                                        
 Context

 The project has 6 experiments (decode similarity completed, 5 ready to run), each with 4 duplicated modules (*_experiment, *_sweep,
 *_storage, *_analysis). The PLAN.md defines a unified "Activation Science" framework. This plan restructures the codebase into a clean
 package with shared infrastructure, deduplicated logic, and 3 new experiments (E6–E8).

 Key constraint: Must NOT break existing results_decode/ data or the ability to re-run decode analysis. All GPU optimization patterns
 (tensors on device, KV-cache, pre-allocated buffers) must be preserved exactly.

 ---
 Target Structure

 activation_science/
 ├── __init__.py
 ├── core/
 │   ├── __init__.py
 │   ├── types.py          # DTYPE_MAP, resolve_dtype, ExperimentJob, TableSchema
 │   ├── model.py          # load_model_and_tokenizer (from src/model.py)
 │   ├── extraction.py     # prefill(), decode_step(), select_next_token(), project_logit_lens()
 │   ├── datasets.py       # PromptGenerator (from src/prompts.py) + resolve_dataset_list + make_experiment_id
 │   ├── scheduler.py      # GPUInfo, detect_gpus, ExperimentScheduler (from src/scheduler.py)
 │   └── storage.py        # ExperimentStore (unified multi-table Parquet + checkpoint)
 ├── metrics/
 │   ├── __init__.py
 │   ├── similarity.py     # ActivationHistoryBuffer, compute_reference_similarity, CKA, Procrustes, subspace_overlap
 │   ├── spectral.py       # compute_svd_metrics, compute_isotropy
 │   ├── information.py    # compute_logit_lens_metrics (entropy, KL, rank)
 │   ├── dynamics.py       # compute_residual_update, fit_pca_basis, project_to_pca
 │   ├── intervention.py   # activation_patch, mean_ablation, directional_perturb (NEW for E6/E7)
 │   └── clustering.py     # detect_regime_changes, cluster_trajectories (NEW for E5/E8)
 ├── experiments/
 │   ├── __init__.py
 │   ├── base.py           # BaseExperiment ABC + GenericSweepRunner
 │   ├── e0_decode_similarity.py
 │   ├── e1_residual_decomposition.py
 │   ├── e2_logit_lens.py
 │   ├── e3_representation_geometry.py
 │   ├── e4_cross_sequence_alignment.py
 │   ├── e5_state_detection.py
 │   ├── e6_causal_intervention.py      # NEW
 │   ├── e7_perturbation_sensitivity.py # NEW
 │   └── e8_token_type_analysis.py      # NEW
 ├── analysis/
 │   ├── __init__.py
 │   ├── base.py           # BaseAnalysis ABC
 │   ├── per_experiment/
 │   │   ├── __init__.py
 │   │   └── e0_plots.py   # 27 plots from src/decode_analysis.py
 │   └── cross_experiment/
 │       ├── __init__.py
 │       ├── phase_boundary_detection.py  # NEW
 │       └── semantic_bottleneck.py       # NEW
 ├── config/
 │   ├── base.yaml
 │   └── experiments/
 │       ├── e0.yaml through e8.yaml
 └── runners/
     ├── __init__.py
     ├── run_experiment.py   # Unified: python -m activation_science.runners.run_experiment e0
     └── run_analysis.py     # Unified: python -m activation_science.runners.run_analysis e0

 ---
 Implementation Steps (21 steps, grouped into phases)

 Phase A: Core Infrastructure (Steps 1–6)

 Step 1: activation_science/core/types.py
 - DTYPE_MAP dict + resolve_dtype() (eliminates 6x duplication)
 - ExperimentJob dataclass (moved from src/scheduler.py)
 - TableSchema dataclass

 Step 2: activation_science/core/model.py
 - Copy src/model.py as-is (load_model_and_tokenizer + extract_hidden_states)
 - Only change: import path

 Step 3: activation_science/core/extraction.py — NEW, key file
 - prefill(model, input_ids, use_cache=True) -> ActivationBatch — wraps model(output_hidden_states=True)
 - decode_step(model, next_token, past_key_values) -> ActivationBatch — single decode step
 - select_next_token(logits, do_sample, temperature) -> Tensor — deduplicated from 4 experiments
 - stack_tracked_layers(hidden_states_tuple, tracked_layers) -> Tensor — utility
 - project_logit_lens(hidden, norm_layer, lm_head) -> Tensor — for E2
 - ActivationBatch dataclass: holds .hidden_states (tuple), .past_key_values, .logits, .num_layers

 Step 4: activation_science/core/datasets.py
 - Move PromptGenerator class from src/prompts.py
 - Add resolve_dataset_list(config) — deduplicated from 6 sweep files
 - Add make_experiment_id(params) — SHA256 hashing, deduplicated from 6 sweep files

 Step 5: activation_science/core/scheduler.py
 - Move from src/scheduler.py: GPUInfo, detect_gpus, estimate_model_vram_gb, estimate_gpus_per_experiment, ExperimentScheduler
 - Import ExperimentJob from types.py instead of defining locally

 Step 6: activation_science/core/storage.py — NEW, replaces 6 storage files
 - ExperimentStore class with:
   - register_table(name, columns) — register N tables with different schemas
   - add_records(table_name, experiment_id, records) — buffer records
   - mark_completed(experiment_id) — checkpoint trigger
   - flush() — write all tables to {name}_part_{N:06d}.parquet
   - is_completed(experiment_id) — check checkpoint
   - load_table(output_dir, table_name) — static method, concat partitions
 - Checkpoint format: identical to current {"completed_experiments": [...]} for backward compat

 Phase B: Metric Library (Steps 7–10)

 Step 7: activation_science/metrics/similarity.py
 - Move from src/decode_experiment.py: ActivationHistoryBuffer, compute_reference_similarity
 - Move from src/metrics.py: cosine_similarity_per_token, mean_cosine_similarity, compute_distance_buckets
 - Move from src/cross_sequence_experiment.py: linear_cka, procrustes_distance, subspace_overlap (extract these ~30-line functions)

 Step 8: activation_science/metrics/spectral.py
 - Extract from src/geometry_experiment.py: SVD computation block → compute_svd_metrics(h_centered, q=100)
 - Extract isotropy computation → compute_isotropy(h, num_samples=500)
 - Returns dict: {effective_rank, participation_ratio, explained_var_topk, singular_values}

 Step 9: activation_science/metrics/information.py
 - Extract from src/logit_lens_experiment.py: per-layer entropy/KL/rank computation → compute_logit_lens_metrics(layer_logits,
 final_log_probs, correct_token_id)

 Step 10: activation_science/metrics/dynamics.py
 - Extract from src/residual_experiment.py: delta computation → compute_residual_update(h_curr, h_prev, h_embed, prev_delta)
 - Extract from src/state_experiment.py: PCA fitting → fit_pca_basis(h, n_components), project_to_pca(h, V, mean)

 Phase C: Experiment Framework (Steps 11–13)

 Step 11: activation_science/experiments/base.py
 - BaseExperiment ABC with:
   - experiment_id: str class attr (e.g. "e0")
   - experiment_name: str class attr
   - table_schemas() -> Dict[str, List[str]] — abstract classmethod
   - default_config_section() -> str — abstract classmethod
   - build_sweep_jobs(config) -> List[ExperimentJob] — abstract classmethod
   - run(model, tokenizer, config, **kwargs) -> Dict[str, List[Dict]] — abstract method
 - GenericSweepRunner class:
   - run_sequential(config) — load model once, iterate jobs, use ExperimentStore
   - run_parallel(config) — multi-GPU via ExperimentScheduler
   - Handles prompt generation, metadata injection, checkpoint/skip logic
   - Special handling for E4 (two-prompt experiments)

 Step 12: activation_science/experiments/e0_decode_similarity.py
 - DecodeSimilarityExperiment(BaseExperiment):
   - table_schemas() returns {"topk": TOPK_COLUMNS, "agg": AGG_COLUMNS}
   - build_sweep_jobs() — logic from src/decode_sweep.py:build_decode_sweep_jobs
   - run() — body from src/decode_experiment.py:run_decode_experiment, using extraction.prefill/decode_step/select_next_token and
 metrics.similarity.compute_reference_similarity

 Step 13: E1–E5 experiment modules
 - Same pattern as E12, wrapping existing experiment functions
 - Each imports from metrics/ instead of inline computation
 - Each uses extraction.py helpers instead of raw model() calls

 Phase D: New Experiments (Steps 14–16)

 Step 14: activation_science/metrics/intervention.py + metrics/clustering.py
 - intervention.py: hook-based activation patching, mean ablation, noise injection
 - clustering.py: HDBSCAN regime detection, changepoint analysis

 Step 15: activation_science/experiments/e6_causal_intervention.py
 - Registers forward hooks to replace hidden states at target layers
 - Measures output KL divergence, prediction flip rate, downstream propagation
 - Tables: {"causal": CAUSAL_COLUMNS} with (layer, position, intervention_type, kl_div, flip_rate)
 - Sweep over: layers × intervention types (skip, mean-ablate, zero)

 Step 16: experiments/e7_perturbation_sensitivity.py + e8_token_type_analysis.py
 - E7: Gaussian noise along SVD directions vs random, measure output KL
 - E8: POS-tag tokens, compute per-group geometry metrics, Fisher discriminant separability

 Phase E: Config, Runners, Analysis (Steps 17–20)

 Step 17: Config files
 - config/base.yaml — shared model/GPU/dataset/storage/logging defaults (from existing YAMLs)
 - config/experiments/e0.yaml through e8.yaml — experiment-specific overrides only
 - Config loader with deep merge: base ← experiment ← CLI override

 Step 18: Unified runners
 - runners/run_experiment.py — registry of experiment_id → class, argparse for experiment/config/parallel
 - runners/run_analysis.py — registry of experiment_id → analysis class

 Step 19: Analysis layer
 - analysis/base.py — BaseAnalysis with load_data() + run_all()
 - analysis/per_experiment/e0_plots.py — move src/decode_analysis.py 27 plot functions
 - Backward compat: load from results_decode/ using ExperimentStore.load_table()

 Step 20: Cross-experiment analysis (stub)
 - analysis/cross_experiment/phase_boundary_detection.py — correlate E1 update profiles with E2 crystallization layers
 - analysis/cross_experiment/semantic_bottleneck.py — identify peak CKA layer from E4

 Phase F: Integration (Step 21)

 Step 21: Backward compatibility + testing
 - Verify ExperimentStore.load_table("./results_decode", "topk") loads existing data
 - Verify E0 run through new framework produces identical schema
 - Add __init__.py shims so old src/ imports still work
 - Keep old run_*.py scripts as thin wrappers (deprecation path)

 ---
 Critical Files to Modify/Create

 ┌──────────────────────────────────────────────┬─────────────────────────────────────────────────┬────────────────────────────────────┐
 │                   New File                   │                     Source                      │               Notes                │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/core/types.py             │ New                                             │ DTYPE_MAP, ExperimentJob from      │
 │                                              │                                                 │ scheduler.py                       │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/core/model.py             │ src/model.py                                    │ Copy, no changes                   │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/core/extraction.py        │ New                                             │ Deduplicated prefill/decode/token  │
 │                                              │                                                 │ selection                          │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │                                              │                                                 │ PromptGenerator +                  │
 │ activation_science/core/datasets.py          │ src/prompts.py + sweep helpers                  │ resolve_dataset_list +             │
 │                                              │                                                 │ make_experiment_id                 │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/core/scheduler.py         │ src/scheduler.py                                │ Move, import ExperimentJob from    │
 │                                              │                                                 │ types                              │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/core/storage.py           │ New (generalizes src/decode_storage.py pattern) │ Multi-table ExperimentStore        │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/metrics/similarity.py     │ src/decode_experiment.py + src/metrics.py +     │ Extract metric functions           │
 │                                              │ src/cross_sequence_experiment.py                │                                    │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/metrics/spectral.py       │ src/geometry_experiment.py                      │ Extract SVD metrics                │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/metrics/information.py    │ src/logit_lens_experiment.py                    │ Extract entropy/KL metrics         │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/metrics/dynamics.py       │ src/residual_experiment.py +                    │ Extract dynamics metrics           │
 │                                              │ src/state_experiment.py                         │                                    │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/metrics/intervention.py   │ New                                             │ For E6/E7                          │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/metrics/clustering.py     │ New                                             │ For E5/E8                          │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/experiments/base.py       │ New                                             │ BaseExperiment +                   │
 │                                              │                                                 │ GenericSweepRunner                 │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/experiments/e0–e8         │ Wrap existing src/*_experiment.py               │ E6–E8 are fully new                │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/config/base.yaml          │ Merge common sections from 6 existing YAMLs     │ Shared defaults                    │
 ├──────────────────────────────────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────┤
 │ activation_science/runners/run_experiment.py │ New (replaces 6 run_*_sweep.py)                 │ Unified CLI                        │
 └──────────────────────────────────────────────┴─────────────────────────────────────────────────┴────────────────────────────────────┘

 ---
 GPU Optimization Guarantees

 These patterns will be preserved exactly:
 1. Tensors stay on GPU — compute_reference_similarity does torch.bmm on GPU, only .cpu() on final scalars
 2. Pre-allocated ActivationHistoryBuffer — moved to metrics/similarity.py as-is
 3. KV-cache reuse — ActivationBatch.past_key_values propagated through decode loop
 4. gc.collect() + torch.cuda.empty_cache() — preserved in experiment cleanup
 5. hidden_states as tuple — NOT stacked into single tensor (avoids 2x GPU memory)

 ---
 Verification

 1. Load existing results: ExperimentStore.load_table("./results_decode", "topk") matches DecodeResultStore.load_topk("./results_decode")
 2. Run E0 through new framework on 1 config, compare output schema
 3. Run E1–E5 through new framework sequentially (same configs as existing YAMLs)
 4. Run E6–E8 on small configs (1 dataset, 1 context length, 1 prompt)
 5. Unified runner: python -m activation_science.runners.run_experiment e0 --parallel
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌