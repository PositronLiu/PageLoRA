from __future__ import annotations

from typing import Any


def build_prompt(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    """Build a chat prompt string using tokenizer templates when available."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}] {content}")
    parts.append("[assistant]")
    return "\n".join(parts)
