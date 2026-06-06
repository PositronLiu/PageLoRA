from __future__ import annotations

from typing import Any


def _merge_leading_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    first_non_system = 0
    for index, message in enumerate(messages):
        if message.get("role") != "system":
            first_non_system = index
            break
        content = str(message.get("content", ""))
        if content:
            system_parts.append(content)
    else:
        if not system_parts:
            return list(messages)
        return [{"role": "user", "content": "\n\n".join(system_parts)}]

    if not system_parts:
        return list(messages)

    merged = [dict(message) for message in messages[first_non_system:]]
    first_content = str(merged[0].get("content", ""))
    merged[0]["content"] = "\n\n".join(system_parts + ([first_content] if first_content else []))
    return merged


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
        try:
            return tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:  # noqa: BLE001
            merged_messages = _merge_leading_system_messages(messages)
            if merged_messages == messages:
                raise
            try:
                return tokenizer.apply_chat_template(
                    merged_messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except Exception:
                raise exc
    # Fallback template for models without chat templates.
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}] {content}")
    parts.append("[assistant]")
    return "\n".join(parts)
