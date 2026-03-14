#!/usr/bin/env python3
"""Verify backward compatibility of the activation_science package.

Checks:
1. ExperimentStore.load_table can load existing results_decode/ data
2. New package imports resolve correctly
3. Old src/ imports still work
4. Config loading works
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


def check(name: str, fn):
    """Run a check function and record the result."""
    try:
        ok, msg = fn()
        if ok is None:
            status = SKIP
        else:
            status = PASS if ok else FAIL
    except Exception as e:
        status = FAIL
        msg = f"{type(e).__name__}: {e}"
    results.append((name, status, msg))
    print(f"  [{status}] {name}: {msg}")


# ── 1. ExperimentStore loads existing results_decode/ ──────────────────

def check_load_topk():
    for candidate in ["./results_decode", "./results_decode_similarity"]:
        if os.path.isdir(candidate):
            results_dir = candidate
            break
    else:
        return None, "No results directory found — skipping"
    from activation_science.core.storage import ExperimentStore
    df = ExperimentStore.load_table(results_dir, "topk")
    if df is None or len(df) == 0:
        return False, f"Loaded empty DataFrame (shape={getattr(df, 'shape', None)})"
    return True, f"Loaded topk table: {df.shape[0]} rows, {df.shape[1]} cols"


def check_load_agg():
    for candidate in ["./results_decode", "./results_decode_similarity"]:
        if os.path.isdir(candidate):
            results_dir = candidate
            break
    else:
        return None, "No results directory found — skipping"
    from activation_science.core.storage import ExperimentStore
    df = ExperimentStore.load_table(results_dir, "agg")
    if df is None or len(df) == 0:
        return False, f"Loaded empty DataFrame (shape={getattr(df, 'shape', None)})"
    return True, f"Loaded agg table: {df.shape[0]} rows, {df.shape[1]} cols"


# ── 2. New package imports ──────────────────────────────────────────────

def check_core_imports():
    modules = [
        "activation_science.core.types",
        "activation_science.core.model",
        "activation_science.core.extraction",
        "activation_science.core.datasets",
        "activation_science.core.storage",
    ]
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    if failed:
        return False, "; ".join(failed)
    return True, f"All {len(modules)} core modules imported OK"


def check_metric_imports():
    modules = [
        "activation_science.metrics.similarity",
        "activation_science.metrics.spectral",
        "activation_science.metrics.information",
        "activation_science.metrics.dynamics",
        "activation_science.metrics.intervention",
        "activation_science.metrics.clustering",
    ]
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    if failed:
        return False, "; ".join(failed)
    return True, f"All {len(modules)} metric modules imported OK"


def check_experiment_imports():
    modules = [
        "activation_science.experiments.base",
        "activation_science.experiments.e0_decode_similarity",
        "activation_science.experiments.e1_residual_decomposition",
        "activation_science.experiments.e2_logit_lens",
        "activation_science.experiments.e3_representation_geometry",
        "activation_science.experiments.e4_cross_sequence_alignment",
        "activation_science.experiments.e5_state_detection",
        "activation_science.experiments.e6_causal_intervention",
        "activation_science.experiments.e7_perturbation_sensitivity",
        "activation_science.experiments.e8_token_type_analysis",
    ]
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    if failed:
        return False, "; ".join(failed)
    return True, f"All {len(modules)} experiment modules imported OK"


def check_analysis_imports():
    modules = [
        "activation_science.analysis.base",
        "activation_science.analysis.per_experiment.e0_plots",
        "activation_science.analysis.cross_experiment.phase_boundary_detection",
        "activation_science.analysis.cross_experiment.semantic_bottleneck",
    ]
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    if failed:
        return False, "; ".join(failed)
    return True, f"All {len(modules)} analysis modules imported OK"


def check_runner_imports():
    modules = [
        "activation_science.runners.run_experiment",
        "activation_science.runners.run_analysis",
    ]
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    if failed:
        return False, "; ".join(failed)
    return True, f"All {len(modules)} runner modules imported OK"


# ── 3. Old src/ imports ─────────────────────────────────────────────────

def check_old_src_imports():
    modules = [
        "src.model",
        "src.prompts",
        "src.metrics",
        "src.decode_experiment",
        "src.decode_storage",
    ]
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    if failed:
        return False, "; ".join(failed)
    return True, f"All {len(modules)} old src modules still importable"


# ── 4. Config loading ──────────────────────────────────────────────────

def check_config_loading():
    from activation_science.runners.run_experiment import load_config
    config = load_config("e0")
    if not config:
        return False, "Empty config returned"
    has_model = "model" in config
    has_storage = "storage" in config
    return has_model and has_storage, f"Keys: {list(config.keys())}"


# ── 5. Table schema consistency ────────────────────────────────────────

def check_e0_schema():
    from activation_science.experiments.e0_decode_similarity import DecodeSimilarityExperiment
    schemas = DecodeSimilarityExperiment.table_schemas()
    has_topk = "topk" in schemas
    has_agg = "agg" in schemas
    return has_topk and has_agg, f"Tables: {list(schemas.keys())}, topk={len(schemas.get('topk', []))} cols, agg={len(schemas.get('agg', []))} cols"


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Backward Compatibility Verification")
    print("=" * 60)

    print("\n1. Loading existing results_decode/ data:")
    check("Load topk table", check_load_topk)
    check("Load agg table", check_load_agg)

    print("\n2. New package imports:")
    check("Core modules", check_core_imports)
    check("Metric modules", check_metric_imports)
    check("Experiment modules", check_experiment_imports)
    check("Analysis modules", check_analysis_imports)
    check("Runner modules", check_runner_imports)

    print("\n3. Old src/ imports:")
    check("Legacy src modules", check_old_src_imports)

    print("\n4. Config loading:")
    check("Load e0 config", check_config_loading)

    print("\n5. Schema consistency:")
    check("E0 table schemas", check_e0_schema)

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"Results: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print("=" * 60)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
