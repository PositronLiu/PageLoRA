from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from mini_openclaw_rl.core.constants import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds across libraries.

    Args:
        seed: Seed value.
    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Select a PyTorch device.

    Args:
        prefer_cuda: Whether to prefer CUDA when available.
    Returns:
        Torch device instance.
    """

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_torch_dtype() -> torch.dtype:
    """Select a reasonable default dtype for model weights.

    Args:
        None.
    Returns:
        Torch dtype.
    """

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def maybe_set_env(name: str, value: str) -> None:
    """Set an environment variable if it does not exist.

    Args:
        name: Variable name.
        value: Variable value.
    Returns:
        None.
    """

    if not os.getenv(name):
        os.environ[name] = value
