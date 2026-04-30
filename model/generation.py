from __future__ import annotations


import gc
from dataclasses import dataclass
from typing import Optional


import torch
import torch.nn.functional as F

from core.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from core.utils import LOG
from model.model_manager import ModelManager


@dataclass
class GenerationOutput:
    """Generation output data.

    Args:
        response_text: Decoded response text.
        response_ids: Token ids for response.
        logprobs: Log-probabilities for response tokens.
    Returns:
        None.
    """

    response_text: str
    response_ids: list[int]
    logprobs: list[float]


@dataclass
class BatchGenerationOutput:
    """Generation output for a batch of prompts.

    Args:
        response_texts: Decoded response texts for each prompt.
        response_ids: Generated token ids for each prompt.
    Returns:
        None.
    """

    response_texts: list[str]
    response_ids: list[list[int]]


def generate_with_logprobs(
    manager: ModelManager,
    prompt_ids: list[int],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = 1.05,
    max_time: float | None = None,
) -> GenerationOutput:
    """Generate a response and compute token log-probabilities.

    Args:
        manager: ModelManager instance.
        prompt_ids: Prompt token ids.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        top_k: Top-k sampling.
        repetition_penalty: Repetition penalty used during generation.
        max_time: Optional maximum generation time in seconds.
    Returns:
        GenerationOutput with response text and log-probs.
    """

    model = manager.model
    tokenizer = manager.tokenizer
    device = manager.device

    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    do_sample = temperature > 0
    generation_temperature = max(temperature, 1e-5)

    generation_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'temperature': generation_temperature,
        'top_p': top_p,
        'top_k': top_k,
        'repetition_penalty': repetition_penalty,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'return_dict_in_generate': True,
        'output_scores': True,
    }
    if max_time is not None:
        generation_kwargs['max_time'] = max_time

    # Move model to GPU for generation (must do unconditionally for LoRA models
    # where LoRA params are on GPU but base model may be on CPU)
    model.to(manager.device)

    with manager.inference_session(enable_cache=True):
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

    # Move model back to CPU after generation
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    generated = outputs.sequences[0].tolist()
    response_ids = generated[len(prompt_ids):]

    # DEBUG: print first and last token
    eos_token_id = tokenizer.eos_token_id
    LOG(f"DEBUG gen: response_ids={response_ids[:10]} len={len(response_ids)} eos={eos_token_id} is_eos={response_ids[-1] == eos_token_id if response_ids else False}")

    logprobs = []
    for step, scores in enumerate(outputs.scores):
        if step >= len(response_ids):
            break
        token_id = response_ids[step]
        logp = F.log_softmax(scores[0], dim=-1)[token_id]
        logprobs.append(float(logp.item()))

    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return GenerationOutput(response_text=response_text, response_ids=response_ids, logprobs=logprobs)


def generate_batch(
    manager: ModelManager,
    batch_prompt_ids: list[list[int]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = 1.05,
    max_time: float | None = None,
) -> BatchGenerationOutput:
    """Generate responses for a batch of prompts.

    This batched path is intended for judge-style workloads that only need
    decoded outputs and do not require rollout token log-probabilities.
    """

    if not batch_prompt_ids:
        return BatchGenerationOutput(response_texts=[], response_ids=[])

    model = manager.model
    tokenizer = manager.tokenizer
    device = manager.device

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set for batched generation")

    max_prompt_len = max(len(prompt_ids) for prompt_ids in batch_prompt_ids)
    # for debug
    min_prompt_len = min(len(prompt_ids) for prompt_ids in batch_prompt_ids)
    LOG(f"Batch prompt len max:{max_prompt_len} - min:{min_prompt_len} = {max_prompt_len - min_prompt_len}")
    padded_prompts: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for prompt_ids in batch_prompt_ids:
        pad_len = max_prompt_len - len(prompt_ids)
        # Decoder-only batched generation should always use left padding.
        # This keeps the final non-pad token aligned across the batch and
        # avoids the right-padding warning from transformers.
        padded_prompts.append([pad_token_id] * pad_len + prompt_ids)
        attention_masks.append([0] * pad_len + [1] * len(prompt_ids))

    input_ids = torch.tensor(padded_prompts, device=device, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, device=device, dtype=torch.long)
    do_sample = temperature > 0
    generation_temperature = max(temperature, 1e-5)

    generation_kwargs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'max_new_tokens': max_new_tokens,
        'do_sample': do_sample,
        'temperature': generation_temperature,
        'top_p': top_p,
        'top_k': top_k,
        'repetition_penalty': repetition_penalty,
        'pad_token_id': pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    if max_time is not None:
        generation_kwargs['max_time'] = max_time

    # Move model to GPU for generation (must do unconditionally for LoRA models
    # where LoRA params are on GPU but base model may be on CPU)
    model.to(manager.device)

    with manager.inference_session(enable_cache=True):
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

    # Move model back to CPU after generation
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    response_ids_batch: list[list[int]] = []
    response_texts: list[str] = []
    prompt_width = input_ids.size(1)
    for sequence in outputs.tolist():
        response_ids = sequence[prompt_width:]
        response_ids_batch.append(response_ids)
        response_texts.append(tokenizer.decode(response_ids, skip_special_tokens=True))

    return BatchGenerationOutput(response_texts=response_texts, response_ids=response_ids_batch)
