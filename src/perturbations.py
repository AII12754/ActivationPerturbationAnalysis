"""
Perturbation strategies.

Each strategy operates on a *token-id list* and modifies a contiguous span
[start, start+span_len).  The output is a new token-id list together with
metadata describing what changed.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Outcome of applying a perturbation."""

    original_ids: List[int]
    perturbed_ids: List[int]
    # Span in the *original* token list that was modified.
    original_span_start: int
    original_span_end: int
    # Span in the *perturbed* token list that replaced the original span.
    perturbed_span_start: int
    perturbed_span_end: int
    # The first token position *after* the perturbation in the perturbed seq.
    # Tokens before this position are either unchanged prefix or the new span.
    post_perturbation_start: int
    perturbation_type: str


# -----------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------
class PerturbationStrategy:
    """Abstract base for perturbation strategies."""

    name: str = "base"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, seed: int = 42):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.vocab_size = tokenizer.vocab_size

    def apply(
        self,
        token_ids: List[int],
        span_start: int,
        span_len: int,
    ) -> PerturbationResult:
        raise NotImplementedError


# -----------------------------------------------------------------------
# Concrete strategies
# -----------------------------------------------------------------------
class RandomReplace(PerturbationStrategy):
    """Replace the span with uniformly random token ids."""

    name = "random_replace"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        # Collect special token IDs to avoid inserting them.
        special = set(self.tokenizer.all_special_ids)
        replacements = []
        for _ in range(span_len):
            t = self.rng.randint(0, self.vocab_size - 1)
            while t in special:
                t = self.rng.randint(0, self.vocab_size - 1)
            replacements.append(t)
        perturbed = ids[:span_start] + replacements + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + span_len,
            post_perturbation_start=span_start + span_len,
            perturbation_type=self.name,
        )


class MaskReplace(PerturbationStrategy):
    """Replace every token in the span with *[UNK]* or a fixed mask token."""

    name = "mask"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        # Use unk_token_id if available, otherwise pick token id 0.
        mask_id = self.tokenizer.unk_token_id or 0
        replacements = [mask_id] * span_len
        perturbed = ids[:span_start] + replacements + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + span_len,
            post_perturbation_start=span_start + span_len,
            perturbation_type=self.name,
        )


class Delete(PerturbationStrategy):
    """Delete the span entirely. The perturbed sequence is shorter."""

    name = "delete"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        perturbed = ids[:span_start] + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start,  # zero-length span
            post_perturbation_start=span_start,
            perturbation_type=self.name,
        )


class Compress(PerturbationStrategy):
    """Compress the span to roughly half its length.

    Keeps every other token to simulate lossy compression while preserving
    some of the original content.
    """

    name = "compress"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        span = ids[span_start : span_start + span_len]
        # Keep every other token.
        compressed = span[::2]
        perturbed = ids[:span_start] + compressed + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + len(compressed),
            post_perturbation_start=span_start + len(compressed),
            perturbation_type=self.name,
        )


class Paraphrase(PerturbationStrategy):
    """Approximate paraphrasing by shuffling words within the span.

    True paraphrasing would require a generative model. This lightweight
    proxy preserves the bag of words (thus roughly preserving semantics)
    while altering surface form.
    """

    name = "paraphrase"

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        span_text = self.tokenizer.decode(
            ids[span_start : span_start + span_len], skip_special_tokens=True
        )
        words = span_text.split()
        self.rng.shuffle(words)
        shuffled_text = " ".join(words)
        new_span_ids = self.tokenizer.encode(shuffled_text, add_special_tokens=False)
        perturbed = ids[:span_start] + new_span_ids + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + len(new_span_ids),
            post_perturbation_start=span_start + len(new_span_ids),
            perturbation_type=self.name,
        )


class SemanticChange(PerturbationStrategy):
    """Replace the span with different content from elsewhere in the vocab.

    Draws token ids from a *different* random passage to produce a span that
    is grammatically plausible but semantically unrelated.
    """

    name = "semantic_change"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seed: int = 42,
        replacement_pool: Optional[List[int]] = None,
    ):
        super().__init__(tokenizer, seed)
        # If a pool of token ids is supplied (e.g. from a different passage),
        # replacements are drawn from it. Otherwise fall back to random ids.
        self._pool = replacement_pool

    def set_replacement_pool(self, pool: List[int]) -> None:
        self._pool = pool

    def apply(self, token_ids, span_start, span_len):
        ids = list(token_ids)
        if self._pool and len(self._pool) >= span_len:
            # Draw a contiguous chunk from the pool at a random offset.
            max_offset = len(self._pool) - span_len
            offset = self.rng.randint(0, max_offset)
            replacements = self._pool[offset : offset + span_len]
        else:
            # Fall back to random tokens (filtered of specials).
            special = set(self.tokenizer.all_special_ids)
            replacements = []
            for _ in range(span_len):
                t = self.rng.randint(0, self.vocab_size - 1)
                while t in special:
                    t = self.rng.randint(0, self.vocab_size - 1)
                replacements.append(t)
        perturbed = ids[:span_start] + replacements + ids[span_start + span_len :]
        return PerturbationResult(
            original_ids=token_ids,
            perturbed_ids=perturbed,
            original_span_start=span_start,
            original_span_end=span_start + span_len,
            perturbed_span_start=span_start,
            perturbed_span_end=span_start + span_len,
            post_perturbation_start=span_start + span_len,
            perturbation_type=self.name,
        )


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------
STRATEGY_REGISTRY: Dict[str, type] = {
    "random_replace": RandomReplace,
    "mask": MaskReplace,
    "delete": Delete,
    "compress": Compress,
    "paraphrase": Paraphrase,
    "semantic_change": SemanticChange,
}


def get_strategy(
    name: str,
    tokenizer: PreTrainedTokenizerBase,
    seed: int = 42,
    **kwargs,
) -> PerturbationStrategy:
    """Instantiate a perturbation strategy by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown perturbation type '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(tokenizer=tokenizer, seed=seed, **kwargs)
