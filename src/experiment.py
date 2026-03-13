"""
Single-experiment runner.

Orchestrates the full pipeline for one (prompt, perturbation) pair:
  1. Tokenize the original prompt.
  2. Apply a perturbation.
  3. Run both sequences through the model.
  4. Compute distance-bucketed similarity metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from .metrics import compute_distance_buckets
from .model import extract_hidden_states
from .perturbations import PerturbationResult, get_strategy

logger = logging.getLogger(__name__)


def run_single_experiment(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    perturbation_type: str,
    perturbation_position_frac: float,
    perturbation_span_length: int,
    measurement_cfg: Dict[str, int],
    seed: int = 42,
    replacement_pool: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Execute a single perturbation experiment and return metric records.

    Parameters
    ----------
    model:
        HuggingFace causal LM with ``output_hidden_states=True``.
    tokenizer:
        Corresponding tokenizer.
    prompt_text:
        The original prompt (already at the desired token length).
    perturbation_type:
        Name of the perturbation strategy (must be in the registry).
    perturbation_position_frac:
        Where in the sequence the perturbation starts (0.0–1.0).
    perturbation_span_length:
        Number of tokens in the perturbed span.
    measurement_cfg:
        Dict with keys ``immediate_window``, ``short_window``,
        ``periodic_sampling_step``, ``periodic_sampling_width``.
    seed:
        Random seed for the perturbation strategy.
    replacement_pool:
        Optional token-id pool for ``semantic_change`` strategy.

    Returns
    -------
    List of metric records (dicts) ready for storage.
    """
    # 1. Tokenize -------------------------------------------------------
    original_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    seq_len = len(original_ids)

    # 2. Compute perturbation span bounds --------------------------------
    span_start = int(perturbation_position_frac * seq_len)
    span_len = min(perturbation_span_length, seq_len - span_start)
    if span_len <= 0:
        logger.warning(
            "Perturbation span has zero length (start=%d, seq_len=%d). Skipping.",
            span_start,
            seq_len,
        )
        return []

    # 3. Apply perturbation ----------------------------------------------
    kwargs = {}
    if perturbation_type == "semantic_change" and replacement_pool:
        kwargs["replacement_pool"] = replacement_pool
    strategy = get_strategy(perturbation_type, tokenizer, seed=seed, **kwargs)
    result: PerturbationResult = strategy.apply(original_ids, span_start, span_len)

    logger.debug(
        "Perturbation '%s': orig len=%d, pert len=%d, post_start_orig=%d, post_start_pert=%d",
        perturbation_type,
        len(result.original_ids),
        len(result.perturbed_ids),
        result.original_span_end,
        result.post_perturbation_start,
    )

    # 4. Forward passes (hidden states stay on GPU) ----------------------
    logger.debug("Running forward pass on original (%d tokens)…", len(result.original_ids))
    orig_hidden = extract_hidden_states(model, result.original_ids)

    logger.debug("Running forward pass on perturbed (%d tokens)…", len(result.perturbed_ids))
    pert_hidden = extract_hidden_states(model, result.perturbed_ids)

    # 5. Compute metrics (vectorized on GPU, only scalars come back) -----
    records = compute_distance_buckets(
        original_states=orig_hidden,
        perturbed_states=pert_hidden,
        post_perturbation_start_orig=result.original_span_end,
        post_perturbation_start_pert=result.post_perturbation_start,
        immediate_window=measurement_cfg.get("immediate_window", 10),
        short_window=measurement_cfg.get("short_window", 100),
        periodic_step=measurement_cfg.get("periodic_sampling_step", 100),
        periodic_width=measurement_cfg.get("periodic_sampling_width", 10),
    )

    # Free GPU memory eagerly.
    del orig_hidden, pert_hidden
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return records
