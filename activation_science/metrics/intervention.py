"""Intervention metrics: activation patching, mean ablation, directional perturbation.

New metrics for E6 (causal intervention) and E7 (perturbation sensitivity).
Uses forward hooks to modify hidden states at target layers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def activation_patch(
    model,
    input_ids: torch.Tensor,
    target_layer: int,
    replacement_states: torch.Tensor,
    positions: Optional[List[int]] = None,
) -> torch.Tensor:
    """Run a forward pass with hidden states replaced at a target layer.

    Parameters
    ----------
    model:
        HuggingFace causal LM.
    input_ids:
        Shape ``(1, seq_len)`` on device.
    target_layer:
        Which transformer layer to patch (1-indexed).
    replacement_states:
        Shape ``(1, seq_len, hidden_dim)`` or ``(1, len(positions), hidden_dim)``.
    positions:
        Token positions to patch. If None, patches all positions.

    Returns
    -------
    Patched logits of shape ``(1, seq_len, vocab_size)``.
    """
    hook_handle = None

    def _hook(module, input, output):
        nonlocal hook_handle
        # output is a tuple; first element is the hidden states tensor
        hidden = output[0]
        if positions is not None:
            hidden = hidden.clone()
            for i, pos in enumerate(positions):
                hidden[0, pos, :] = replacement_states[0, i, :]
        else:
            hidden = replacement_states
        return (hidden,) + output[1:]

    # Register hook on the target layer
    layers = _get_transformer_layers(model)
    hook_handle = layers[target_layer - 1].register_forward_hook(_hook)

    try:
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_hidden_states=False)
        return outputs.logits
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def mean_ablation(
    model,
    input_ids: torch.Tensor,
    target_layer: int,
    mean_states: torch.Tensor,
    positions: Optional[List[int]] = None,
) -> torch.Tensor:
    """Replace hidden states with their mean at a target layer.

    Parameters
    ----------
    mean_states:
        Shape ``(hidden_dim,)`` — the mean activation to use.
    positions:
        Token positions to ablate. If None, ablates all.

    Returns
    -------
    Ablated logits of shape ``(1, seq_len, vocab_size)``.
    """
    hook_handle = None

    def _hook(module, input, output):
        hidden = output[0].clone()
        if positions is not None:
            for pos in positions:
                hidden[0, pos, :] = mean_states
        else:
            hidden[:, :, :] = mean_states
        return (hidden,) + output[1:]

    layers = _get_transformer_layers(model)
    hook_handle = layers[target_layer - 1].register_forward_hook(_hook)

    try:
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_hidden_states=False)
        return outputs.logits
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def directional_perturb(
    model,
    input_ids: torch.Tensor,
    target_layer: int,
    direction: torch.Tensor,
    magnitude: float = 1.0,
    positions: Optional[List[int]] = None,
) -> torch.Tensor:
    """Add a directional perturbation to hidden states at a target layer.

    Parameters
    ----------
    direction:
        Shape ``(hidden_dim,)`` — perturbation direction (will be normalized).
    magnitude:
        Scaling factor for the perturbation.
    positions:
        Token positions to perturb. If None, perturbs all.

    Returns
    -------
    Perturbed logits of shape ``(1, seq_len, vocab_size)``.
    """
    direction_normed = F.normalize(direction.unsqueeze(0), dim=1).squeeze(0) * magnitude
    hook_handle = None

    def _hook(module, input, output):
        hidden = output[0].clone()
        if positions is not None:
            for pos in positions:
                hidden[0, pos, :] += direction_normed
        else:
            hidden += direction_normed
        return (hidden,) + output[1:]

    layers = _get_transformer_layers(model)
    hook_handle = layers[target_layer - 1].register_forward_hook(_hook)

    try:
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_hidden_states=False)
        return outputs.logits
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def compute_intervention_metrics(
    original_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    position: int = -1,
) -> Dict[str, float]:
    """Compare original and patched logits at a token position.

    Returns
    -------
    Dict with kl_divergence, prediction_flip (bool as int), js_divergence.
    """
    orig = original_logits[0, position, :].float()
    patch = patched_logits[0, position, :].float()

    orig_probs = F.softmax(orig, dim=0)
    patch_probs = F.softmax(patch, dim=0)
    orig_log_probs = F.log_softmax(orig, dim=0)
    patch_log_probs = F.log_softmax(patch, dim=0)

    # KL divergence
    kl_div = F.kl_div(patch_log_probs, orig_probs, reduction='sum', log_target=False).item()

    # Prediction flip
    flip = int(orig.argmax().item() != patch.argmax().item())

    # JS divergence
    m_probs = 0.5 * (orig_probs + patch_probs)
    m_log_probs = m_probs.clamp(min=1e-10).log()
    js_div = 0.5 * (
        F.kl_div(m_log_probs, orig_probs, reduction='sum', log_target=False).item() +
        F.kl_div(m_log_probs, patch_probs, reduction='sum', log_target=False).item()
    )

    return {
        "kl_divergence": kl_div,
        "prediction_flip": flip,
        "js_divergence": js_div,
    }


def _get_transformer_layers(model):
    """Get the list of transformer layers from a HuggingFace model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError("Cannot find transformer layers in model architecture.")
