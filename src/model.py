"""
Model loading and hidden-state extraction.

Wraps HuggingFace ``AutoModelForCausalLM`` and exposes a simple API that
returns per-layer hidden states for a given token-id sequence.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "float16",
    device_map: Optional[str | Dict] = None,
    max_memory: Optional[Dict] = None,
    trust_remote_code: bool = False,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer.

    Parameters
    ----------
    model_path:
        Path to a local HuggingFace model directory.
    dtype:
        One of ``"float16"``, ``"bfloat16"``, ``"float32"``.
    device_map:
        Accelerate device map.  ``"auto"`` lets Accelerate decide.
    max_memory:
        Per-device memory cap passed to ``from_pretrained``.
    trust_remote_code:
        Whether to trust remote code in the model repo.
    """
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )

    logger.info("Loading model from %s (dtype=%s)", model_path, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map or "auto",
        max_memory=max_memory,
        trust_remote_code=trust_remote_code,
        output_hidden_states=True,
    )
    model.eval()
    logger.info("Model loaded. Parameters: %s", f"{model.num_parameters():,}")
    return model, tokenizer


@torch.inference_mode()
def extract_hidden_states(
    model: AutoModelForCausalLM,
    token_ids: List[int],
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Run a forward pass and return hidden states for every layer.

    Parameters
    ----------
    model:
        A causal LM loaded with ``output_hidden_states=True``.
    token_ids:
        1-D list of token ids.
    device:
        Device to place the input tensor on.  If *None*, uses the device
        of the model's first parameter.

    Returns
    -------
    List of tensors, one per layer (including the embedding layer).
    Each tensor has shape ``(seq_len, hidden_dim)`` on **CPU**.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)

    # outputs.hidden_states is a tuple of (num_layers+1,) tensors,
    # each of shape (batch=1, seq_len, hidden_dim).
    hidden_states: List[torch.Tensor] = [
        hs.squeeze(0).cpu() for hs in outputs.hidden_states
    ]
    return hidden_states
