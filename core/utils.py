from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from core.constants import DEFAULT_SEED

LOG_TO_TERMINAL = "terminal"
LOG_TO_FILE = "file"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / ".env"

def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _resolve_config(name: str, default: str, dotenv_values: dict[str, str]) -> str:
    return os.getenv(name, dotenv_values.get(name, default))


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _build_session_log_path(base_path: Path, session_id: str | None) -> Path:
    if not session_id:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{session_id}{base_path.suffix}")


def _format_log_message(message: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f"[LOG {timestamp} UTC] {message}"


_DOTENV_VALUES = _read_dotenv(ENV_FILE_PATH)
LOG_OUTPUT_TARGET = _resolve_config("MINI_OPENCLAW_LOG_TARGET", LOG_TO_TERMINAL, _DOTENV_VALUES)
_BASE_LOG_FILE_PATH = _resolve_path(
    _resolve_config("MINI_OPENCLAW_LOG_FILE", "logs/mini_openclaw_rl.log", _DOTENV_VALUES)
)
_LOG_SESSION_ID: str | None = None


def set_log_session_id(session_id: str) -> None:
    """Bind file logging to a specific session id.

    Args:
        session_id: Session identifier.
    Returns:
        None.
    """

    global _LOG_SESSION_ID
    _LOG_SESSION_ID = session_id


def get_log_file_path() -> str:
    """Return the effective log file path.

    Args:
        None.
    Returns:
        Absolute log file path.
    """

    return str(_build_session_log_path(_BASE_LOG_FILE_PATH, _LOG_SESSION_ID))


LOG_FILE_PATH = get_log_file_path()


def LOG(message: str) -> None:
    """Write a formatted log line to terminal or a file.

    Args:
        message: Log message.
    Returns:
        None.
    """

    formatted = _format_log_message(str(message))
    if LOG_OUTPUT_TARGET == LOG_TO_FILE:
        log_path = Path(get_log_file_path())
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(formatted + "\n")
        return
    print(formatted, flush=True)


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
    LOG(f"Random seed set to {seed}")


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
