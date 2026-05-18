"""Sampling helpers for the FlashInfer custom generation path."""

from __future__ import annotations

import torch


def greedy_argmax_token(logits: torch.Tensor) -> torch.Tensor:
    """Return greedy tokens as a ``[batch, 1]`` GPU tensor."""

    if logits.ndim == 3:
        logits = logits[:, -1, :]
    if logits.ndim != 2:
        raise ValueError("greedy_argmax_token expects logits with shape [batch, vocab]")
    return torch.argmax(logits, dim=-1)[:, None]
