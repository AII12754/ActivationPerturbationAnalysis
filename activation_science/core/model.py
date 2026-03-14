"""Model loading and hidden-state extraction.

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
    from .types import resolve_dtype
    torch_dtype = resolve_dtype(dtype, default=torch.float16)

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
        attn_implementation="sdpa",
    )
    model.eval()
    logger.info("Model loaded. Parameters: %s", f"{model.num_parameters():,}")
    return model, tokenizer


@torch.inference_mode()
def extract_hidden_states(
    model: AutoModelForCausalLM,
    token_ids: List[int],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run a forward pass and return hidden states for every layer.

    Returns
    -------
    Tensor of shape ``(num_layers+1, seq_len, hidden_dim)`` on the
    **model's device** (GPU).  Kept on-device so that downstream
    similarity computation runs on GPU.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True)

    hidden_states = torch.stack(
        [hs.squeeze(0) for hs in outputs.hidden_states], dim=0
    )
    del outputs, input_ids
    return hidden_states
