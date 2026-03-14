"""Shared model-inference helpers: prefill, decode step, token selection.

Deduplicates logic repeated across all decode-loop experiments (E0–E5).
All GPU optimization patterns are preserved exactly:
  - Tensors stay on GPU
  - KV-cache reuse via ``past_key_values``
  - Hidden states kept as tuple (NOT stacked) to avoid 2× memory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class ActivationBatch:
    """Holds outputs from a single forward pass."""
    hidden_states: tuple          # tuple of (1, seq_len, hidden_dim) tensors
    past_key_values: object       # model KV-cache
    logits: torch.Tensor          # (1, seq_len, vocab_size) or (1, 1, vocab_size)
    num_layers: int               # number of transformer layers (excludes embedding)

    @property
    def last_logits(self) -> torch.Tensor:
        """Return logits for the last token: shape (1, vocab_size)."""
        return self.logits[:, -1, :]


@torch.inference_mode()
def prefill(
    model,
    input_ids: torch.Tensor,
    use_cache: bool = True,
) -> ActivationBatch:
    """Run prefill (prompt encoding) and return hidden states + KV-cache.

    Parameters
    ----------
    model:
        HuggingFace causal LM.
    input_ids:
        Shape ``(1, seq_len)`` on the model's device.
    use_cache:
        Whether to return ``past_key_values`` for subsequent decode steps.
    """
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=use_cache,
    )
    num_layers = len(outputs.hidden_states) - 1  # exclude embedding
    batch = ActivationBatch(
        hidden_states=outputs.hidden_states,
        past_key_values=outputs.past_key_values if use_cache else None,
        logits=outputs.logits,
        num_layers=num_layers,
    )
    return batch


@torch.inference_mode()
def decode_step(
    model,
    next_token: torch.Tensor,
    past_key_values,
) -> ActivationBatch:
    """Run a single autoregressive decode step with KV-cache.

    Parameters
    ----------
    model:
        HuggingFace causal LM.
    next_token:
        Shape ``(1, 1)`` on the model's device.
    past_key_values:
        KV-cache from previous step.
    """
    outputs = model(
        input_ids=next_token,
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    num_layers = len(outputs.hidden_states) - 1
    batch = ActivationBatch(
        hidden_states=outputs.hidden_states,
        past_key_values=outputs.past_key_values,
        logits=outputs.logits,
        num_layers=num_layers,
    )
    return batch


def select_next_token(
    logits: torch.Tensor,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Select next token from logits. Returns shape ``(1, 1)``.

    Parameters
    ----------
    logits:
        Shape ``(1, vocab_size)`` — last-position logits.
    do_sample:
        If True and temperature > 0, sample from the distribution.
    temperature:
        Sampling temperature. Ignored when ``do_sample=False``.
    """
    if do_sample and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return logits.argmax(dim=-1, keepdim=True)


def stack_tracked_layers(
    hidden_states: tuple,
    tracked_layers: List[int],
) -> torch.Tensor:
    """Stack tracked layers from a hidden_states tuple.

    Parameters
    ----------
    hidden_states:
        Tuple of (1, seq_len, hidden_dim) tensors.
    tracked_layers:
        Layer indices to extract (1-based for transformer layers, 0 for embedding).

    Returns
    -------
    Tensor of shape ``(num_tracked, seq_len, hidden_dim)``.
    """
    return torch.stack(
        [hidden_states[l].squeeze(0) for l in tracked_layers], dim=0
    )


def project_logit_lens(
    hidden: torch.Tensor,
    norm_layer,
    lm_head,
) -> torch.Tensor:
    """Project a hidden state through norm + lm_head (logit lens).

    Parameters
    ----------
    hidden:
        Shape ``(hidden_dim,)`` — single token's hidden state.
    norm_layer:
        The model's final RMSNorm / LayerNorm.
    lm_head:
        The model's output projection (lm_head).

    Returns
    -------
    Logits tensor of shape ``(vocab_size,)`` in float32.
    """
    h_normed = norm_layer(hidden.unsqueeze(0)).squeeze(0)
    return lm_head(h_normed.unsqueeze(0)).squeeze(0).float()
