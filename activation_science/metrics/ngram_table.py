"""Persistent trigram/bigram activation table for a single PP boundary layer.

Stores pre-computed hidden states from n-gram forward passes and supports:
  - Batch table construction from a corpus
  - Online updates from new requests
  - Tiered classification: trigram → bigram → self-ref → unigram

DAG storage:
  Hidden states are organized as a two-level trie (DAG):
    dag[A][B] = (bigram_hidden, {C: trigram_hidden, ...})
  This shares the bigram prefix lookup across all trigrams with the same
  (A,B) prefix, reducing lookup overhead and key storage.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class _BigramNode:
    """DAG node for a bigram prefix (A, B).

    Stores the bigram hidden state and a dict mapping suffix token C
    to the trigram hidden state for (A, B, C).
    """
    __slots__ = ("bigram_hidden", "suffixes")

    def __init__(self, bigram_hidden: torch.Tensor):
        self.bigram_hidden = bigram_hidden
        self.suffixes: Dict[int, torch.Tensor] = {}


class NgramTable:
    """Persistent trigram/bigram activation table for a single PP boundary layer.

    Internal storage is a two-level DAG (trie):
      ``_dag[token_A][token_B]`` → ``_BigramNode``
        - ``.bigram_hidden``: hidden state at position B from a 3-token forward
        - ``.suffixes[token_C]``: hidden state at position C (trigram)
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        # Two-level trie: A -> B -> _BigramNode
        self._dag: Dict[int, Dict[int, _BigramNode]] = {}
        # Counts for fast stats
        self._num_trigrams = 0
        self._num_bigrams = 0

    # ------------------------------------------------------------------
    # DAG access helpers
    # ------------------------------------------------------------------
    def _get_node(self, a: int, b: int) -> Optional[_BigramNode]:
        """Look up the bigram node for (a, b), or None."""
        level_b = self._dag.get(a)
        if level_b is None:
            return None
        return level_b.get(b)

    def _get_or_create_node(self, a: int, b: int, bigram_hidden: torch.Tensor) -> _BigramNode:
        """Get existing node for (a, b) or create one with the given hidden state."""
        level_b = self._dag.get(a)
        if level_b is None:
            level_b = {}
            self._dag[a] = level_b
        node = level_b.get(b)
        if node is None:
            node = _BigramNode(bigram_hidden)
            level_b[b] = node
            self._num_bigrams += 1
        return node

    def has_trigram(self, a: int, b: int, c: int) -> bool:
        node = self._get_node(a, b)
        return node is not None and c in node.suffixes

    def get_trigram(self, a: int, b: int, c: int) -> Optional[torch.Tensor]:
        node = self._get_node(a, b)
        if node is None:
            return None
        return node.suffixes.get(c)

    def get_bigram(self, a: int, b: int) -> Optional[torch.Tensor]:
        node = self._get_node(a, b)
        if node is None:
            return None
        return node.bigram_hidden

    # ------------------------------------------------------------------
    # Internal: batched 3-token forward
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _forward_trigrams(
        self,
        model,
        trigram_list: List[Tuple[int, int, int]],
        layer_idx: int,
        batch_size: int,
    ) -> None:
        """Forward trigrams as flat (batch, 3) batches, storing into DAG."""
        for start in range(0, len(trigram_list), batch_size):
            batch_tris = trigram_list[start : start + batch_size]
            input_tensor = torch.tensor(batch_tris, dtype=torch.long, device=self.device)
            outputs = model(
                input_ids=input_tensor,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[layer_idx]  # (batch, 3, hidden_dim)
            tri_hidden = hidden[:, -1, :].to(self.dtype).detach()
            bi_hidden = hidden[:, -2, :].to(self.dtype).detach()

            for j, tri in enumerate(batch_tris):
                a, b, c = tri
                node = self._get_or_create_node(a, b, bi_hidden[j])
                if c not in node.suffixes:
                    node.suffixes[c] = tri_hidden[j]
                    self._num_trigrams += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def build_from_corpus(
        self,
        model,
        token_ids_list: List[List[int]],
        layer_idx: int,
        batch_size: int = 128,
    ) -> Dict[str, Any]:
        """Build table from one or more token sequences.

        For each trigram (A,B,C): forward pass → store hidden[-1] as trigram,
        hidden[-2] as bigram ref for (A,B) in the DAG.

        Parameters
        ----------
        model : HuggingFace causal LM
        token_ids_list : list of token-id lists
        layer_idx : int — which hidden-state layer to extract
        batch_size : int — forward-pass batch size

        Returns
        -------
        dict with num_trigrams, num_bigrams, build_time_ms
        """
        t0 = time.perf_counter()

        # Collect unique trigrams not already in table
        unique_trigrams: Dict[Tuple[int, int, int], None] = {}
        for ids in token_ids_list:
            for i in range(2, len(ids)):
                tri = (ids[i - 2], ids[i - 1], ids[i])
                if not self.has_trigram(*tri):
                    unique_trigrams.setdefault(tri, None)

        trigram_list = list(unique_trigrams.keys())
        if not trigram_list:
            return {"num_trigrams": self._num_trigrams, "num_bigrams": self._num_bigrams, "build_time_ms": 0.0}

        self._forward_trigrams(model, trigram_list, layer_idx, batch_size)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "NgramTable built: %d trigrams, %d bigrams in %.1f ms",
            self._num_trigrams, self._num_bigrams, elapsed_ms,
        )
        return {
            "num_trigrams": self._num_trigrams,
            "num_bigrams": self._num_bigrams,
            "build_time_ms": elapsed_ms,
        }

    @torch.inference_mode()
    def update_from_request(
        self,
        model,
        token_ids: List[int],
        layer_idx: int,
        batch_size: int = 128,
    ) -> Tuple[int, int, float]:
        """Add new trigrams from a request not already in table.

        Returns
        -------
        (num_new_trigrams, num_new_bigrams, update_time_ms)
        """
        t0 = time.perf_counter()

        old_tri = self._num_trigrams
        old_bi = self._num_bigrams

        new_trigrams: Dict[Tuple[int, int, int], None] = {}
        for i in range(2, len(token_ids)):
            tri = (token_ids[i - 2], token_ids[i - 1], token_ids[i])
            if not self.has_trigram(*tri) and tri not in new_trigrams:
                new_trigrams[tri] = None

        if not new_trigrams:
            return 0, 0, (time.perf_counter() - t0) * 1000.0

        tri_list = list(new_trigrams.keys())
        self._forward_trigrams(model, tri_list, layer_idx, batch_size)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._num_trigrams - old_tri, self._num_bigrams - old_bi, elapsed_ms

    def classify_and_build_refs(
        self,
        token_ids: List[int],
        hidden_dim: int,
    ) -> Tuple[List[str], torch.Tensor, List[Optional[int]], Dict[Tuple, int]]:
        """Classify each position into tier and return reference info.

        Uses DAG trie for efficient prefix-shared lookups: one traversal
        to check both trigram and bigram for the same (A, B) prefix.

        Parameters
        ----------
        token_ids : list of token ids
        hidden_dim : int — hidden dimension for reference tensor

        Returns
        -------
        tiers : list[str] — "trigram"|"bigram"|"self_ref"|"unigram" per position
        ref_acts : (seq_len, hidden_dim) — reference activation per position
                   (zeros for unigram, zeros for self_ref — filled later from reconstructed)
        self_ref_sources : list[Optional[int]] — source position index for self_ref tier
        first_occurrence_map : dict — trigram → first position
        """
        seq_len = len(token_ids)
        tiers: List[str] = []
        ref_acts = torch.zeros(seq_len, hidden_dim, device=self.device, dtype=self.dtype)
        self_ref_sources: List[Optional[int]] = []
        first_occurrence_map: Dict[Tuple[int, int, int], int] = {}

        for i in range(seq_len):
            trigram = None
            if i >= 2:
                a, b, c = token_ids[i - 2], token_ids[i - 1], token_ids[i]
                trigram = (a, b, c)

                if trigram in first_occurrence_map:
                    tiers.append("self_ref")
                    self_ref_sources.append(first_occurrence_map[trigram])
                    continue

                # Trigram lookup: _dag[A][B].suffixes[C]
                node_ab = self._get_node(a, b)
                if node_ab is not None and c in node_ab.suffixes:
                    tiers.append("trigram")
                    ref_acts[i] = node_ab.suffixes[c]
                    self_ref_sources.append(None)
                    first_occurrence_map[trigram] = i
                    continue

            # Bigram lookup: _dag[B][C].bigram_hidden
            # (bigram key is (token_ids[i-1], token_ids[i]) — the hidden
            #  state of token_ids[i] with 2-token context)
            if i >= 1:
                b_tok, c_tok = token_ids[i - 1], token_ids[i]
                node_bc = self._get_node(b_tok, c_tok)
                if node_bc is not None:
                    tiers.append("bigram")
                    ref_acts[i] = node_bc.bigram_hidden
                    self_ref_sources.append(None)
                    if trigram is not None:
                        first_occurrence_map.setdefault(trigram, i)
                    continue

            tiers.append("unigram")
            self_ref_sources.append(None)
            if trigram is not None:
                first_occurrence_map.setdefault(trigram, i)

        return tiers, ref_acts, self_ref_sources, first_occurrence_map

    def update_from_hidden_states(
        self,
        token_ids: List[int],
        hidden_states: torch.Tensor,
    ) -> Tuple[int, int, float]:
        """Update table directly from full-sequence prefill hidden states.

        Instead of running separate 3-token forwards, reuse the hidden states
        already computed during prefill.  For position i:
          - Trigram (ids[i-2], ids[i-1], ids[i]) → hidden_states[i]
          - Bigram  (ids[i-1], ids[i])           → hidden_states[i]

        The key difference from ``update_from_request``: trigram references
        carry full prior context (not just 3-token context), and bigram
        references carry full prior context (not just 2-token context).

        Parameters
        ----------
        token_ids : list of int — token ids of the sequence
        hidden_states : (seq_len, hidden_dim) tensor — hidden states from prefill

        Returns
        -------
        (num_new_trigrams, num_new_bigrams, update_time_ms)
        """
        t0 = time.perf_counter()
        old_tri = self._num_trigrams
        old_bi = self._num_bigrams

        # Batch dtype conversion: single GPU op instead of per-position .to()
        hidden_fp16 = hidden_states.to(self.dtype).detach()

        for i in range(len(token_ids)):
            h_i = hidden_fp16[i]

            # Store bigram reference: dag[B][C].bigram_hidden = h_C
            if i >= 1:
                b_tok, c_tok = token_ids[i - 1], token_ids[i]
                self._get_or_create_node(b_tok, c_tok, h_i)

            # Store trigram reference: dag[A][B].suffixes[C] = h_C
            if i >= 2:
                a, b, c = token_ids[i - 2], token_ids[i - 1], token_ids[i]
                node = self._get_node(a, b)
                if node is None:
                    # Create (A,B) node with bigram_hidden = h_B (from position i-1)
                    node = self._get_or_create_node(a, b, hidden_fp16[i - 1])
                if c not in node.suffixes:
                    node.suffixes[c] = h_i
                    self._num_trigrams += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._num_trigrams - old_tri, self._num_bigrams - old_bi, elapsed_ms

    @property
    def stats(self) -> Dict[str, Any]:
        """Return table statistics."""
        # Estimate memory: each entry is a hidden_dim fp16 tensor
        sample = None
        for level_b in self._dag.values():
            for node in level_b.values():
                sample = node.bigram_hidden
                break
            if sample is not None:
                break
        if sample is not None:
            per_entry_bytes = sample.nelement() * sample.element_size()
        else:
            per_entry_bytes = 0
        memory_bytes = (self._num_trigrams + self._num_bigrams) * per_entry_bytes
        return {
            "num_trigrams": self._num_trigrams,
            "num_bigrams": self._num_bigrams,
            "memory_bytes": memory_bytes,
        }
