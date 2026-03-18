"""Prompt generation module.

Provides deterministic, length-controlled natural-text prompts by drawing
from a HuggingFace dataset (default: WikiText-103) or from built-in
templates.  All lengths are measured in *tokens* using the model tokenizer.
"""

from __future__ import annotations

import logging
import os
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
        passage_target_chars: int = 80_000,
    ):
        self.tokenizer = tokenizer
        self.source = source
        self.seed = seed
        self.rng = random.Random(seed)
        self._passage_target_chars = passage_target_chars

        if source == "dataset":
            self._passages = self._load_dataset_passages(
                dataset_name, dataset_config, dataset_split, num_candidates
            )
        else:
            self._passages = list(_TEMPLATE_PARAGRAPHS)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    @staticmethod
    def _find_parquet_files(directory: str) -> List[str]:
        """Recursively find parquet files, preferring train splits."""
        import glob as _glob
        files = sorted(_glob.glob(os.path.join(directory, "*.parquet")))
        if files:
            return files
        all_pq = sorted(
            f for f in _glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True)
            if "/." not in f
        )
        if not all_pq:
            return []
        train_files = [f for f in all_pq if "train" in os.path.basename(f)]
        if train_files:
            return train_files
        test_files = [f for f in all_pq if "test" in os.path.basename(f)]
        if test_files:
            return test_files
        return all_pq

    @staticmethod
    def _find_json_files(directory: str) -> List[str]:
        """Find JSON files in a directory (non-recursive, skip metadata)."""
        import glob as _glob
        return sorted(
            f for f in _glob.glob(os.path.join(directory, "*.json"))
            if not os.path.basename(f).startswith(".")
        )

    @staticmethod
    def _extract_texts_from_json(path: str) -> List[str]:
        """Extract text strings from a JSON file (list of strings/dicts)."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            return []
        if isinstance(data[0], str):
            return [t for t in data if isinstance(t, str) and len(t.strip()) > 50]
        texts: List[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "conversations" in item:
                parts = [
                    turn.get("value", "")
                    for turn in item["conversations"]
                    if isinstance(turn, dict) and turn.get("value")
                ]
                if parts:
                    texts.append("\n\n".join(parts))
            else:
                str_vals = [v for v in item.values() if isinstance(v, str) and len(v) > 50]
                if str_vals:
                    texts.append(max(str_vals, key=len))
        return texts

    @staticmethod
    def _pick_text_column(columns: List[str]) -> str:
        """Heuristically pick the best text column from a dataset."""
        preferred = ["text", "article", "content", "document", "passage",
                      "context", "input", "question", "output", "instruction"]
        for col in preferred:
            if col in columns:
                return col
        return columns[0]

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

            if os.path.isdir(name):
                logger.info("Loading dataset from local directory: %s", name)
                pq_files = self._find_parquet_files(name)
                if pq_files:
                    logger.info("Found %d parquet files in %s", len(pq_files), name)
                    ds = load_dataset("parquet", data_files=pq_files, split="train")
                else:
                    json_files = self._find_json_files(name)
                    if json_files:
                        logger.info("Found %d JSON files in %s", len(json_files), name)
                        all_json_texts: List[str] = []
                        for jf in json_files:
                            all_json_texts.extend(self._extract_texts_from_json(jf))
                        if all_json_texts:
                            logger.info("Extracted %d texts from JSON files.", len(all_json_texts))
                            return self._build_passages(all_json_texts, num_candidates, self._passage_target_chars)
                    logger.info("Trying load_dataset with trust_remote_code for %s", name)
                    ds = load_dataset(name, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(name, config, split=split, trust_remote_code=True)
        except Exception as exc:
            logger.warning(
                "Failed to load dataset %s/%s (%s). Falling back to templates.",
                name, config, exc,
            )
            return list(_TEMPLATE_PARAGRAPHS)

        text_key = self._pick_text_column(ds.column_names)
        logger.info("Using column '%s' from dataset %s", text_key, name)
        all_texts = [t for t in ds[text_key] if t and isinstance(t, str) and len(t.strip()) > 100]

        return self._build_passages(all_texts, num_candidates, self._passage_target_chars)

    def _build_passages(
        self,
        all_texts: List[str],
        num_candidates: int,
        target_chars: int = 80_000,
    ) -> List[str]:
        """Shuffle texts and concatenate into long passages."""
        rng = random.Random(self.seed)
        rng.shuffle(all_texts)

        passages: List[str] = []
        buf: List[str] = []
        buf_chars = 0
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

        logger.info("Built %d long passages from %d texts.", len(passages), len(all_texts))
        return passages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, target_length: int, index: int = 0) -> str:
        """Return a prompt whose token count equals *target_length*."""
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
            repeats_needed = (target_length // len(token_ids)) + 1
            token_ids = (token_ids * repeats_needed)[:target_length]

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
