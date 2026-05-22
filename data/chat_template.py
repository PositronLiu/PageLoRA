from __future__ import annotations

from typing import Any


def build_chat_prompt(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, enable_thinking: bool = False) -> str:
    """Build a chat prompt using model template when available.

    Args:
        tokenizer: HuggingFace tokenizer.
        messages: Chat messages.
        tools: Optional tool schema list.
        enable_thinking: Whether to enable Qwen3 thinking mode.
    Returns:
        Prompt text string.
    """

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    # Fallback template for models without chat templates.
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}] {content}")
    parts.append("[assistant]")
    return "\n".join(parts)
