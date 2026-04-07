from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from mini_openclaw_rl.data.chat_template import build_prompt


def load_tokenizer(model_name: str, padding_side: str = "left") -> Any:
    """Load a Hugging Face tokenizer and ensure a pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def tokenize_messages(messages: list[dict[str, Any]], tokenizer: Any) -> list[int]:
    """Tokenize chat messages into prompt ids."""
    prompt = build_prompt(messages, tokenizer)
    return tokenizer.encode(prompt, add_special_tokens=False)
