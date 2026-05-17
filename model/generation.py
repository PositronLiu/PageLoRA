from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from core.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from core.utils import LOG
from multilora.runtime import AdapterRole
from model.model_manager import ModelManager


@dataclass
class GenerationOutput:
    """Decoded single-sample generation result with rollout log-probabilities."""

    response_text: str
    response_ids: list[int]
    logprobs: list[float]


@dataclass
class BatchGenerationOutput:
    """Decoded generation results for a batch of prompts."""

    response_texts: list[str]
    response_ids: list[list[int]]


def _build_left_padded_batch(
    token_ids_batch: list[list[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left-padded ``input_ids`` and ``attention_mask`` tensors.

    Left padding keeps the final non-pad token aligned across the batch, which
    is the safest layout for decoder-only generation.
    """

    max_prompt_len = max(len(token_ids) for token_ids in token_ids_batch)
    min_prompt_len = min(len(token_ids) for token_ids in token_ids_batch)
    LOG(
        f"Batch prompt len max:{max_prompt_len} - min:{min_prompt_len} = "
        f"{max_prompt_len - min_prompt_len}"
    )

    padded_prompts: list[list[int]] = []
    attention_masks: list[list[int]] = []
    for token_ids in token_ids_batch:
        pad_len = max_prompt_len - len(token_ids)
        padded_prompts.append([pad_token_id] * pad_len + token_ids)
        attention_masks.append([0] * pad_len + [1] * len(token_ids))

    input_ids = torch.tensor(padded_prompts, device=device, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, device=device, dtype=torch.long)
    return input_ids, attention_mask


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_histories: list[list[int]],
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply a simple repetition penalty to already-seen tokens.

    This matches the usual sign-aware repetition penalty formulation used by
    transformers generation utilities.
    """

    if repetition_penalty == 1.0:
        return logits

    adjusted = logits.clone()
    for row, history in enumerate(token_histories):
        if not history:
            continue
        unique_token_ids = torch.tensor(
            sorted(set(history)),
            device=adjusted.device,
            dtype=torch.long,
        )
        token_logits = adjusted[row, unique_token_ids]
        token_logits = torch.where(
            token_logits < 0,
            token_logits * repetition_penalty,
            token_logits / repetition_penalty,
        )
        adjusted[row, unique_token_ids] = token_logits
    return adjusted


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Mask logits using standard top-k and top-p sampling filters."""

    filtered = logits

    if top_k > 0 and top_k < filtered.size(-1):
        topk_values = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < topk_values, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        # Keep at least one token per row.
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        scatter_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        scatter_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        filtered = filtered.masked_fill(scatter_mask, float("-inf"))

    return filtered


def _sample_next_tokens(
    logits: torch.Tensor,
    *,
    token_histories: list[list[int]],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    sampling_mode: Literal["project_default", "greedy_fast", "flashinfer_topk_topp"] = "project_default",
    compute_logprobs: bool = True,
) -> tuple[list[int], list[float]]:
    """Sample one next token per batch row and return token ids + logprobs."""

    if sampling_mode == "greedy_fast":
        next_tokens = torch.argmax(logits, dim=-1)
        if not compute_logprobs:
            return next_tokens.tolist(), []
        token_logprobs = torch.log_softmax(logits, dim=-1)
        next_token_logprobs = token_logprobs.gather(1, next_tokens[:, None]).squeeze(-1)
        return next_tokens.tolist(), next_token_logprobs.tolist()

    do_sample = temperature > 0
    processed = _apply_repetition_penalty(logits, token_histories, repetition_penalty)
    if do_sample:
        processed = processed / max(temperature, 1e-5)
        if sampling_mode != "flashinfer_topk_topp":
            processed = _apply_top_k_top_p(processed, top_k=top_k, top_p=top_p)

    if sampling_mode == "flashinfer_topk_topp" and do_sample:
        try:
            from flashinfer.sampling import top_k_top_p_sampling_from_logits
        except Exception as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError("flashinfer sampling is unavailable") from exc
        next_tokens = top_k_top_p_sampling_from_logits(
            processed,
            top_k=top_k,
            top_p=top_p,
        )
    elif do_sample:
        next_tokens = torch.multinomial(torch.softmax(processed, dim=-1), num_samples=1).squeeze(-1)
    else:
        next_tokens = torch.argmax(processed, dim=-1)

    if not compute_logprobs:
        return next_tokens.tolist(), []

    token_logprobs = torch.log_softmax(processed, dim=-1)
    next_token_logprobs = token_logprobs.gather(1, next_tokens[:, None]).squeeze(-1)
    return next_tokens.tolist(), next_token_logprobs.tolist()


def generate_with_logprobs(
    manager: ModelManager,
    prompt_ids: list[int],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = 1.05,
    max_time: float | None = None,
    *,
    adapter_name: str | None = None,
    ignore_eos: bool = False,
    sampling_mode: Literal["project_default", "greedy_fast", "flashinfer_topk_topp"] = "project_default",
    compute_logprobs: bool = True,
) -> GenerationOutput:
    """Generate a response and compute token log-probabilities."""

    cache_mode = getattr(manager.config, "kvcache_mode", "hf")
    if cache_mode == "hf" and getattr(manager.config, "kvcache_decompose", False):
        cache_mode = "br"

    if cache_mode == "br":
        from model.br_kvcache_generation import generate_with_br_kvcache

        resolved = adapter_name or manager.active_logical_adapter_name
        with manager.use_adapter(adapter_name=resolved, role=AdapterRole.SERVE.value):
            with manager.inference_session(enable_cache=False):  # cache managed externally
                with torch.no_grad():
                    return generate_with_br_kvcache(
                        manager,
                        prompt_ids,
                        adapter_name=resolved,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        max_time=max_time,
                        ignore_eos=ignore_eos,
                        sampling_mode=sampling_mode,
                        compute_logprobs=compute_logprobs,
                    )

    if cache_mode == "full":
        from model.full_kvcache_generation import generate_with_full_kvcache

        resolved = adapter_name or manager.active_logical_adapter_name
        with manager.use_adapter(adapter_name=resolved, role=AdapterRole.SERVE.value):
            with manager.inference_session(enable_cache=False):  # cache managed externally
                with torch.no_grad():
                    return generate_with_full_kvcache(
                        manager,
                        prompt_ids,
                        adapter_name=resolved,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        max_time=max_time,
                        ignore_eos=ignore_eos,
                        sampling_mode=sampling_mode,
                        compute_logprobs=compute_logprobs,
                    )

    if cache_mode == "flashinfer":
        from model.flashinfer_kvcache_generation import generate_with_flashinfer_kvcache

        resolved = adapter_name or manager.active_logical_adapter_name
        with manager.use_adapter(adapter_name=resolved, role=AdapterRole.SERVE.value):
            with manager.inference_session(enable_cache=False):  # cache managed externally
                with torch.no_grad():
                    return generate_with_flashinfer_kvcache(
                        manager,
                        prompt_ids,
                        adapter_name=resolved,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        max_time=max_time,
                        ignore_eos=ignore_eos,
                        sampling_mode=sampling_mode,
                        compute_logprobs=compute_logprobs,
                    )

    if sampling_mode == "flashinfer_topk_topp":
        raise ValueError("sampling_mode=flashinfer_topk_topp is only supported by custom generation paths")

    model = manager.model
    tokenizer = manager.tokenizer
    device = manager.device

    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    do_sample = temperature > 0 and sampling_mode != "greedy_fast"
    generation_temperature = max(temperature, 1e-5)
    hf_repetition_penalty = 1.0 if sampling_mode == "greedy_fast" else repetition_penalty

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": generation_temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": hf_repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": None if ignore_eos else tokenizer.eos_token_id,
        "forced_eos_token_id": None,
        "return_dict_in_generate": True,
        "output_scores": compute_logprobs,
    }
    if max_time is not None:
        generation_kwargs["max_time"] = max_time

    if manager.config.cpu_offload:
        model.to(manager.device)

    with manager.use_adapter(adapter_name=adapter_name, role=AdapterRole.SERVE.value):
        with manager.inference_session(enable_cache=True):
            with torch.no_grad():
                outputs = model.generate(**generation_kwargs)

    if manager.config.cpu_offload:
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generated = outputs.sequences[0].tolist()
    response_ids = generated[len(prompt_ids):]

    eos_token_id = tokenizer.eos_token_id
    LOG(
        "DEBUG gen: "
        f"response_ids={response_ids[:10]} len={len(response_ids)} eos={eos_token_id} "
        f"is_eos={response_ids[-1] == eos_token_id if response_ids else False}"
    )

    logprobs = []
    if compute_logprobs:
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
    *,
    adapter_name: str | None = None,
) -> BatchGenerationOutput:
    """Generate responses for a batch of prompts."""

    if not batch_prompt_ids:
        return BatchGenerationOutput(response_texts=[], response_ids=[])

    model = manager.model
    tokenizer = manager.tokenizer
    device = manager.device

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be set for batched generation")

    input_ids, attention_mask = _build_left_padded_batch(
        batch_prompt_ids,
        pad_token_id=pad_token_id,
        device=device,
    )
    do_sample = temperature > 0
    generation_temperature = max(temperature, 1e-5)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": generation_temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if max_time is not None:
        generation_kwargs["max_time"] = max_time

    if manager.config.cpu_offload:
        model.to(manager.device)

    with manager.use_adapter(adapter_name=adapter_name, role=AdapterRole.SERVE.value):
        with manager.inference_session(enable_cache=True):
            with torch.no_grad():
                outputs = model.generate(**generation_kwargs)

    if manager.config.cpu_offload:
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    response_ids_batch: list[list[int]] = []
    response_texts: list[str] = []
    prompt_width = input_ids.size(1)
    for sequence in outputs.tolist():
        response_ids = sequence[prompt_width:]
        response_ids_batch.append(response_ids)
        response_texts.append(tokenizer.decode(response_ids, skip_special_tokens=True))

    return BatchGenerationOutput(response_texts=response_texts, response_ids=response_ids_batch)
