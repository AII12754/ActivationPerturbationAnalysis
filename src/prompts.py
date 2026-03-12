"""
Prompt generation module.

Provides deterministic, length-controlled natural-text prompts by drawing
from a HuggingFace dataset (default: WikiText-103) or from built-in
templates.  All lengths are measured in *tokens* using the model tokenizer.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in template paragraphs (fallback when no dataset is available)
# ---------------------------------------------------------------------------
_TEMPLATE_PARAGRAPHS = [
    (
        "The history of computing is a story of exponential progress. "
        "From the earliest mechanical calculators to modern supercomputers, "
        "each generation of hardware has dramatically expanded what is "
        "possible in science, engineering, and everyday life."
    ),
    (
        "Language is arguably the most complex system that humans use on a "
        "daily basis. Its structure operates on multiple levels—phonology, "
        "morphology, syntax, semantics, and pragmatics—each interacting with "
        "the others in ways that linguists are still working to understand."
    ),
    (
        "Climate change is driven by the accumulation of greenhouse gases in "
        "the atmosphere. Carbon dioxide, methane, and nitrous oxide trap heat "
        "that would otherwise radiate into space, gradually raising the "
        "average temperature of the planet."
    ),
    (
        "The Silk Road was not a single road but a vast network of trade "
        "routes linking China to the Mediterranean. Along these routes, "
        "merchants exchanged not only silk and spices but also ideas, "
        "religions, and technologies."
    ),
    (
        "Machine learning models learn patterns from data rather than "
        "following explicit instructions. In supervised learning, a model is "
        "trained on labelled examples; in unsupervised learning, it discovers "
        "structure in data without labels."
    ),
    (
        "The ocean covers more than seventy percent of the Earth's surface "
        "and contains ninety-seven percent of the planet's water. Its "
        "currents regulate climate, its ecosystems support billions of "
        "organisms, and its depths remain largely unexplored."
    ),
    (
        "Renaissance art was characterized by a renewed interest in "
        "classical antiquity, an emphasis on naturalism, and the development "
        "of techniques such as linear perspective, chiaroscuro, and sfumato."
    ),
    (
        "Quantum mechanics describes the behavior of particles at the "
        "smallest scales. At these scales, particles can exist in "
        "superpositions of states, and measurements can be fundamentally "
        "probabilistic rather than deterministic."
    ),
]


class PromptGenerator:
    """Generate natural-text prompts of controlled token length."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        source: str = "dataset",
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        dataset_split: str = "train",
        num_candidates: int = 50,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.source = source
        self.seed = seed
        self.rng = random.Random(seed)

        if source == "dataset":
            self._passages = self._load_dataset_passages(
                dataset_name, dataset_config, dataset_split, num_candidates
            )
        else:
            self._passages = list(_TEMPLATE_PARAGRAPHS)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def _load_dataset_passages(
        self,
        name: str,
        config: str,
        split: str,
        num_candidates: int,
    ) -> List[str]:
        """Load long text passages from the specified HuggingFace dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset(name, config, split=split, trust_remote_code=True)
        except Exception as exc:
            logger.warning(
                "Failed to load dataset %s/%s (%s). Falling back to templates.",
                name,
                config,
                exc,
            )
            return list(_TEMPLATE_PARAGRAPHS)

        # Concatenate short wiki paragraphs into long passages.
        rng = random.Random(self.seed)
        text_key = "text" if "text" in ds.column_names else ds.column_names[0]
        all_texts = [t for t in ds[text_key] if t and len(t.strip()) > 100]
        rng.shuffle(all_texts)

        passages: List[str] = []
        buf: List[str] = []
        buf_chars = 0
        # Aim for passages of ~80k chars (≈20k tokens) so we can slice later.
        target_chars = 80_000
        for t in all_texts:
            buf.append(t.strip())
            buf_chars += len(t)
            if buf_chars >= target_chars:
                passages.append("\n\n".join(buf))
                buf, buf_chars = [], 0
                if len(passages) >= num_candidates:
                    break
        if buf:
            passages.append("\n\n".join(buf))

        logger.info("Loaded %d long passages from %s/%s.", len(passages), name, config)
        return passages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, target_length: int, index: int = 0) -> str:
        """Return a prompt whose token count equals *target_length*.

        Parameters
        ----------
        target_length:
            Desired number of tokens.
        index:
            Selects which base passage to start from (for variety).

        Returns
        -------
        str — the trimmed prompt text.
        """
        passage = self._pick_passage(index)
        return self._trim_to_length(passage, target_length)

    def generate_batch(
        self, target_length: int, count: int = 1
    ) -> List[str]:
        """Generate *count* distinct prompts of the given token length."""
        return [self.generate(target_length, index=i) for i in range(count)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pick_passage(self, index: int) -> str:
        """Select a base passage, cycling through available candidates."""
        if self.source == "template":
            # Repeat / concatenate templates to ensure enough material.
            repeats = (index // len(self._passages)) + 1
            combined = "\n\n".join(self._passages * repeats)
            return combined
        return self._passages[index % len(self._passages)]

    def _trim_to_length(self, text: str, target_length: int) -> str:
        """Tokenize *text* and truncate or pad to exactly *target_length* tokens."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) >= target_length:
            token_ids = token_ids[:target_length]
        else:
            # Repeat passage tokens until we reach target length.
            repeats_needed = (target_length // len(token_ids)) + 1
            token_ids = (token_ids * repeats_needed)[:target_length]

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
