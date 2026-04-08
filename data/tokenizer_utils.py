from __future__ import annotations

from typing import Any

from mini_openclaw_rl.data.chat_template import build_chat_prompt


def encode_chat(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> tuple[str, list[int]]:
    """Encode chat messages into prompt ids.

    Args:
        tokenizer: HuggingFace tokenizer.
        messages: Chat messages.
        tools: Optional tool schema list.
    Returns:
        Tuple of prompt text and prompt ids.
    """

    prompt_text = build_chat_prompt(tokenizer, messages, tools=tools)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return prompt_text, prompt_ids
