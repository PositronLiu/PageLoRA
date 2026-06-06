"""FlashInfer paged-KV generation loop."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from core.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from core.utils import LOG
from multilora.runtime import AdapterRole
from model.flashinfer.attention_backend import FlashInferAttentionBackend
from model.flashinfer.cuda_graph import (
    FlashInferBatchDecodeCudaGraphRunner,
    FlashInferDecodeCudaGraphRunner,
    can_use_flashinfer_cuda_graph_batch_decode,
    can_use_flashinfer_cuda_graph_decode,
)
from model.flashinfer.layer_runner import _run_flashinfer_attention
from model.flashinfer.layer_runner import _run_flashinfer_attention_batch
from model.flashinfer.norm import apply_rmsnorm
from model.flashinfer.perf import FlashInferPerfTrace
from model.flashinfer.packed_ops import project_linear
from model.flashinfer.sampling import greedy_argmax_token
from model.kvcache.flashinfer_page_table import FlashInferPageTable
from model.kvcache.page_admission import admit_paged_kv_request
from model.kvcache.paged_kvcache_manager import FlashInferPagedKVCacheManager
from model.kvcache.paged_kvcache_request import FlashInferPagedKVRequest
from model.kvcache.radix_types import KVBlockSpan

if TYPE_CHECKING:
    from model.generation import BatchGenerationOutput
    from model.generation import GenerationOutput
    from model.model_manager import ModelManager


def _run_flashinfer_prefill_segment(
    manager: "ModelManager",
    token_ids: list[int],
    *,
    start_pos: int,
    request_cache: FlashInferPagedKVRequest,
    backend: FlashInferAttentionBackend,
    perf: FlashInferPerfTrace | None = None,
) -> torch.Tensor:
    """Run prompt prefill with FlashInfer paged attention."""

    if not token_ids:
        raise ValueError("prefill segment must not be empty")

    total_t0 = perf.tic() if perf is not None else time.perf_counter()
    device = manager.device
    input_t0 = perf.tic() if perf is not None else time.perf_counter()
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    seqlen = input_ids.size(1)
    hidden = manager.model.get_input_embeddings()(input_ids)
    position_ids = torch.arange(start_pos, start_pos + seqlen, device=device).unsqueeze(0)
    if perf is not None:
        perf.add("prefill_input_s", input_t0)

    reserve_t0 = perf.tic() if perf is not None else time.perf_counter()
    manager.kvcache_manager.reserve_chunk(request_cache, seqlen)
    if perf is not None:
        perf.add("prefill_reserve_s", reserve_t0)

    plan_t0 = perf.tic() if perf is not None else time.perf_counter()
    _plan_prefill(
        backend,
        manager.kvcache_manager,
        request_cache,
        qo_len=seqlen,
        device=device,
        dtype=hidden.dtype,
    )
    if perf is not None:
        perf.add("prefill_plan_s", plan_t0)

    layers_t0 = perf.tic() if perf is not None else time.perf_counter()
    for layer_idx, layer in enumerate(manager.lora_runtime.layers):
        hidden = _run_flashinfer_attention(
            layer,
            manager.lora_runtime.layer_plans[layer_idx],
            layer_idx,
            hidden,
            manager.kvcache_manager,
            position_ids,
            request_cache,
            mode="prefill",
            backend=backend,
            perf=perf,
        )
    if perf is not None:
        perf.add("prefill_layers_s", layers_t0)
        perf.add("prefill_total_s", total_t0)
    return hidden


def _make_backend(manager: "ModelManager") -> FlashInferAttentionBackend:
    """Create the FlashInfer backend for paged KV attention."""

    if not manager.lora_runtime.layers:
        raise RuntimeError("FlashInfer generation requires at least one decoder layer")
    layer = manager.lora_runtime.layers[0]
    attn_config = layer.self_attn.config
    backend = FlashInferAttentionBackend(
        device=manager.device,
        workspace_mb=getattr(manager.config, "flashinfer_workspace_mb", 128),
        num_qo_heads=attn_config.num_attention_heads,
        num_kv_heads=attn_config.num_key_value_heads,
        head_dim=layer.self_attn.head_dim,
        page_size=getattr(manager.config, "flashinfer_page_size", 32),
    )
    backend.use_flashinfer_qk_norm = bool(getattr(manager.config, "flashinfer_qk_norm", False))
    return backend


def _backend_cache_key(manager: "ModelManager") -> tuple[object, ...]:
    """Return the static backend properties used for cache reuse."""

    if not manager.lora_runtime.layers:
        raise RuntimeError("FlashInfer generation requires at least one decoder layer")
    layer = manager.lora_runtime.layers[0]
    attn_config = layer.self_attn.config
    return (
        manager.device,
        getattr(manager.config, "flashinfer_workspace_mb", 128),
        int(attn_config.num_attention_heads),
        int(attn_config.num_key_value_heads),
        int(layer.self_attn.head_dim),
        getattr(manager.config, "flashinfer_page_size", 32),
        bool(getattr(manager.config, "flashinfer_qk_norm", False)),
    )


def _get_backend(manager: "ModelManager") -> FlashInferAttentionBackend:
    """Return a manager-scoped FlashInfer backend with stable wrapper state."""

    key = _backend_cache_key(manager)
    cached = getattr(manager, "_flashinfer_backend_entry", None)
    if cached is not None:
        cached_key, cached_backend = cached
        if cached_key == key:
            return cached_backend
    backend = _make_backend(manager)
    setattr(manager, "_flashinfer_backend_entry", (key, backend))
    setattr(manager, "_flashinfer_decode_graph_runner", None)
    setattr(manager, "_flashinfer_batch_decode_graph_runner", None)
    return backend


def _get_decode_graph_runner(
    manager: "ModelManager",
    backend: FlashInferAttentionBackend,
    cache_mgr: FlashInferPagedKVCacheManager,
) -> FlashInferDecodeCudaGraphRunner:
    """Return a persistent decode CUDA graph runner for the current backend."""

    runner = getattr(manager, "_flashinfer_decode_graph_runner", None)
    if isinstance(runner, FlashInferDecodeCudaGraphRunner) and runner.is_compatible(backend, cache_mgr):
        return runner
    runner = FlashInferDecodeCudaGraphRunner(manager, backend, cache_mgr)
    setattr(manager, "_flashinfer_decode_graph_runner", runner)
    return runner


def _get_batch_decode_graph_runner(
    manager: "ModelManager",
    backend: FlashInferAttentionBackend,
    cache_mgr: FlashInferPagedKVCacheManager,
) -> FlashInferBatchDecodeCudaGraphRunner:
    """Return a persistent batched decode CUDA graph runner."""

    runner = getattr(manager, "_flashinfer_batch_decode_graph_runner", None)
    if isinstance(runner, FlashInferBatchDecodeCudaGraphRunner) and runner.is_compatible(backend, cache_mgr):
        return runner
    runner = FlashInferBatchDecodeCudaGraphRunner(manager, backend, cache_mgr)
    setattr(manager, "_flashinfer_batch_decode_graph_runner", runner)
    return runner


def _plan_decode(
    backend: FlashInferAttentionBackend,
    cache_mgr: FlashInferPagedKVCacheManager,
    request_cache: FlashInferPagedKVRequest,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FlashInferPageTable:
    """Build page metadata and plan one FlashInfer decode step."""

    page_table = cache_mgr.build_page_table(
        [*request_cache.prefix_spans, *request_cache.new_spans],
        device=device,
        reuse_buffer=True,
    )
    backend.plan_decode(
        paged_kv_indptr=page_table.paged_kv_indptr,
        paged_kv_indices=page_table.paged_kv_indices,
        paged_kv_last_page_len=page_table.paged_kv_last_page_len,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return page_table


def _plan_decode_batch(
    backend: FlashInferAttentionBackend,
    cache_mgr: FlashInferPagedKVCacheManager,
    request_caches: list[FlashInferPagedKVRequest],
    *,
    device: torch.device,
    dtype: torch.dtype,
    reuse_buffer: bool = False,
) -> FlashInferPageTable:
    """Build batched page metadata and plan one FlashInfer decode step."""

    page_table = cache_mgr.build_batch_page_table(
        [
            [*request_cache.prefix_spans, *request_cache.new_spans]
            for request_cache in request_caches
        ],
        device=device,
        reuse_buffer=reuse_buffer,
    )
    backend.plan_decode(
        paged_kv_indptr=page_table.paged_kv_indptr,
        paged_kv_indices=page_table.paged_kv_indices,
        paged_kv_last_page_len=page_table.paged_kv_last_page_len,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return page_table


def _plan_prefill(
    backend: FlashInferAttentionBackend,
    cache_mgr: FlashInferPagedKVCacheManager,
    request_cache: FlashInferPagedKVRequest,
    *,
    qo_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> FlashInferPageTable:
    """Build page metadata and plan one FlashInfer prefill segment."""

    page_table = cache_mgr.build_page_table(
        [*request_cache.prefix_spans, *request_cache.new_spans],
        device=device,
    )
    qo_indptr = torch.tensor([0, qo_len], device=device, dtype=torch.int32)
    backend.plan_prefill(
        qo_indptr=qo_indptr,
        paged_kv_indptr=page_table.paged_kv_indptr,
        paged_kv_indices=page_table.paged_kv_indices,
        paged_kv_last_page_len=page_table.paged_kv_last_page_len,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return page_table


def _plan_prefill_batch(
    backend: FlashInferAttentionBackend,
    cache_mgr: FlashInferPagedKVCacheManager,
    request_caches: list[FlashInferPagedKVRequest],
    *,
    token_lengths: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> FlashInferPageTable:
    """Build batched page metadata and plan one packed FlashInfer prefill."""

    qo_indptr_values = [0]
    for token_count in token_lengths:
        qo_indptr_values.append(qo_indptr_values[-1] + token_count)
    qo_indptr = torch.tensor(qo_indptr_values, device=device, dtype=torch.int32)
    page_table = cache_mgr.build_batch_page_table(
        [
            [*request_cache.prefix_spans, *request_cache.new_spans]
            for request_cache in request_caches
        ],
        device=device,
    )
    backend.plan_prefill(
        qo_indptr=qo_indptr,
        paged_kv_indptr=page_table.paged_kv_indptr,
        paged_kv_indices=page_table.paged_kv_indices,
        paged_kv_last_page_len=page_table.paged_kv_last_page_len,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return page_table


def _run_flashinfer_prefill_batch(
    manager: "ModelManager",
    batch_token_ids: list[list[int]],
    *,
    start_positions: list[int],
    request_caches: list[FlashInferPagedKVRequest],
    backend: FlashInferAttentionBackend,
    perf: FlashInferPerfTrace | None = None,
) -> torch.Tensor:
    """Run ragged prompt prefill as one packed FlashInfer batch."""

    if not batch_token_ids:
        raise ValueError("prefill batch must not be empty")
    if len(batch_token_ids) != len(start_positions) or len(batch_token_ids) != len(request_caches):
        raise ValueError("prefill batch metadata sizes differ")

    token_lengths = [len(token_ids) for token_ids in batch_token_ids]
    if any(token_count <= 0 for token_count in token_lengths):
        raise ValueError("prefill batch rows must not be empty")

    total_t0 = perf.tic() if perf is not None else time.perf_counter()
    device = manager.device
    input_t0 = perf.tic() if perf is not None else time.perf_counter()
    flat_token_ids = [token_id for token_ids in batch_token_ids for token_id in token_ids]
    flat_position_ids: list[int] = []
    for start_pos, token_count in zip(start_positions, token_lengths, strict=True):
        flat_position_ids.extend(range(start_pos, start_pos + token_count))
    input_ids = torch.tensor([flat_token_ids], device=device, dtype=torch.long)
    position_ids = torch.tensor([flat_position_ids], device=device, dtype=torch.long)
    hidden = manager.model.get_input_embeddings()(input_ids)
    if perf is not None:
        perf.add("prefill_input_s", input_t0)

    reserve_t0 = perf.tic() if perf is not None else time.perf_counter()
    for request_cache, token_count in zip(request_caches, token_lengths, strict=True):
        manager.kvcache_manager.reserve_chunk(request_cache, token_count)
    if perf is not None:
        perf.add("prefill_reserve_s", reserve_t0)

    plan_t0 = perf.tic() if perf is not None else time.perf_counter()
    _plan_prefill_batch(
        backend,
        manager.kvcache_manager,
        request_caches,
        token_lengths=token_lengths,
        device=device,
        dtype=hidden.dtype,
    )
    if perf is not None:
        perf.add("prefill_plan_s", plan_t0)

    layers_t0 = perf.tic() if perf is not None else time.perf_counter()
    for layer_idx, layer in enumerate(manager.lora_runtime.layers):
        hidden = _run_flashinfer_attention_batch(
            layer,
            manager.lora_runtime.layer_plans[layer_idx],
            layer_idx,
            hidden,
            manager.kvcache_manager,
            position_ids,
            request_caches,
            token_lengths,
            mode="prefill",
            backend=backend,
            perf=perf,
        )
    if perf is not None:
        perf.add("prefill_layers_s", layers_t0)
        perf.add("prefill_total_s", total_t0)
    return hidden


def _run_flashinfer_decode_batch(
    manager: "ModelManager",
    token_ids: list[int],
    *,
    positions: list[int],
    request_caches: list[FlashInferPagedKVRequest],
    backend: FlashInferAttentionBackend,
    perf: FlashInferPerfTrace | None = None,
    reserve_chunks: bool = True,
    plan_decode: bool = True,
) -> torch.Tensor:
    """Run one batched FlashInfer decode step for active requests."""

    if not token_ids:
        raise ValueError("decode batch must not be empty")
    if len(token_ids) != len(positions) or len(token_ids) != len(request_caches):
        raise ValueError("decode batch metadata sizes differ")

    device = manager.device
    embed_t0 = perf.tic() if perf is not None else time.perf_counter()
    input_ids = torch.tensor([[token_id] for token_id in token_ids], device=device, dtype=torch.long)
    position_ids = torch.tensor([[position] for position in positions], device=device, dtype=torch.long)
    hidden = manager.model.get_input_embeddings()(input_ids)
    if perf is not None:
        perf.add("decode_embed_s", embed_t0)

    if reserve_chunks:
        reserve_t0 = perf.tic() if perf is not None else time.perf_counter()
        for request_cache in request_caches:
            manager.kvcache_manager.reserve_chunk(request_cache, 1)
        if perf is not None:
            perf.add("decode_reserve_s", reserve_t0)

    if plan_decode:
        plan_t0 = perf.tic() if perf is not None else time.perf_counter()
        _plan_decode_batch(
            backend,
            manager.kvcache_manager,
            request_caches,
            device=device,
            dtype=hidden.dtype,
        )
        if perf is not None:
            perf.add("decode_plan_s", plan_t0)

    layers_t0 = perf.tic() if perf is not None else time.perf_counter()
    token_lengths = [1] * len(token_ids)
    for layer_idx, layer in enumerate(manager.lora_runtime.layers):
        hidden = _run_flashinfer_attention_batch(
            layer,
            manager.lora_runtime.layer_plans[layer_idx],
            layer_idx,
            hidden,
            manager.kvcache_manager,
            position_ids,
            request_caches,
            token_lengths,
            mode="decode",
            backend=backend,
            perf=perf,
        )
    if perf is not None:
        perf.add("decode_layers_s", layers_t0)
    return hidden


def _batch_requests_fit(
    cache_mgr: FlashInferPagedKVCacheManager,
    requests: list[FlashInferPagedKVRequest],
    new_ids_batch: list[list[int]],
    *,
    max_new_tokens: int,
    reserve_pages: int,
) -> bool:
    needed_pages = 0
    for request_cache, new_ids in zip(requests, new_ids_batch, strict=True):
        needed_pages += cache_mgr.estimate_request_new_pages(
            request_cache,
            prefill_tokens=len(new_ids),
            max_decode_tokens=max_new_tokens,
        )
    return needed_pages + max(0, int(reserve_pages)) <= cache_mgr.available_pages


def _admit_paged_kv_batch(
    cache_mgr: FlashInferPagedKVCacheManager,
    *,
    batch_prompt_ids: list[list[int]],
    adapter_name: str,
    requests: list[FlashInferPagedKVRequest],
    max_new_tokens: int,
    reserve_pages: int = 4,
) -> tuple[list[FlashInferPagedKVRequest], list[list[int]], int]:
    """Admit one FlashInfer batch before it reserves physical pages."""

    new_ids_batch = [
        prompt_ids[request_cache.prefill_from:]
        for prompt_ids, request_cache in zip(batch_prompt_ids, requests, strict=True)
    ]
    batch_reserve_pages = max(reserve_pages, len(requests))
    if _batch_requests_fit(
        cache_mgr,
        requests,
        new_ids_batch,
        max_new_tokens=max_new_tokens,
        reserve_pages=batch_reserve_pages,
    ):
        return requests, new_ids_batch, max_new_tokens

    cache_mgr.abort_batch_requests(requests)
    LOG(
        "FlashInfer batch page admission miss; releasing adapter cache "
        f"adapter={adapter_name} batch={len(batch_prompt_ids)} requested_decode={max_new_tokens}"
    )
    cache_mgr.release_adapter_cache_for_generation(adapter_name)
    requests = cache_mgr.begin_batch_requests(batch_prompt_ids, adapter_name)
    new_ids_batch = [
        prompt_ids[request_cache.prefill_from:]
        for prompt_ids, request_cache in zip(batch_prompt_ids, requests, strict=True)
    ]
    if _batch_requests_fit(
        cache_mgr,
        requests,
        new_ids_batch,
        max_new_tokens=max_new_tokens,
        reserve_pages=batch_reserve_pages,
    ):
        return requests, new_ids_batch, max_new_tokens

    reduced_max_new_tokens = max_new_tokens
    while reduced_max_new_tokens > 0 and not _batch_requests_fit(
        cache_mgr,
        requests,
        new_ids_batch,
        max_new_tokens=reduced_max_new_tokens,
        reserve_pages=batch_reserve_pages,
    ):
        reduced_max_new_tokens -= 1
    if reduced_max_new_tokens <= 0:
        cache_mgr.abort_batch_requests(requests)
        raise RuntimeError(
            "FlashInfer batch page admission rejected request: "
            f"batch={len(batch_prompt_ids)} page_size={cache_mgr.page_size} "
            f"available_pages={cache_mgr.available_pages}"
        )

    LOG(
        "FlashInfer batch page admission reduced decode budget "
        f"from={max_new_tokens} to={reduced_max_new_tokens} "
        f"batch={len(batch_prompt_ids)} available_pages={cache_mgr.available_pages}"
    )
    return requests, new_ids_batch, reduced_max_new_tokens


def _token_tensors_to_list(tokens: list[torch.Tensor]) -> list[int]:
    if not tokens:
        return []
    return torch.cat(tokens, dim=0).tolist()


def _model_final_norm_and_head(
    model,
    hidden: torch.Tensor,
    *,
    perf: FlashInferPerfTrace | None = None,
    prefix: str | None = None,
) -> torch.Tensor:
    """Apply final norm and run lm_head through the FlashInfer linear helper."""

    norm_t0 = perf.tic() if perf is not None else time.perf_counter()
    normed = apply_rmsnorm(model.model.norm, hidden[:, -1:, :])
    if perf is not None and prefix is not None:
        perf.add(f"{prefix}_final_norm_s", norm_t0)
    head_t0 = perf.tic() if perf is not None else time.perf_counter()
    logits = project_linear(model.lm_head, normed)
    if perf is not None and prefix is not None:
        perf.add(f"{prefix}_lm_head_project_s", head_t0)
    return logits


def generate_with_flashinfer_kvcache(
    manager: "ModelManager",
    prompt_ids: list[int],
    *,
    adapter_name: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = 1.05,
    max_time: float | None = None,
    ignore_eos: bool = False,
    sampling_mode: str = "project_default",
    compute_logprobs: bool = True,
) -> "GenerationOutput":
    """Generate using paged KV cache and FlashInfer attention."""

    from model.generation import GenerationOutput, _sample_next_tokens

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    tokenizer = manager.tokenizer
    device = manager.device
    model = manager.model

    cache_mgr = manager.kvcache_manager
    if not isinstance(cache_mgr, FlashInferPagedKVCacheManager):
        raise RuntimeError("FlashInferPagedKVCacheManager not initialized")

    cache_mgr.reset_perf_counters()
    perf = FlashInferPerfTrace(device)
    generation_t0 = perf.tic()
    backend_t0 = perf.tic()
    backend = _get_backend(manager)
    perf.add("backend_create_s", backend_t0)
    begin_t0 = perf.tic()
    request_cache = cache_mgr.begin_request(prompt_ids, adapter_name)
    perf.add("begin_request_s", begin_t0)
    committed = False
    prefill_from = request_cache.prefill_from
    if request_cache.match_len:
        LOG(
            "FlashInfer paged KV cache match: "
            f"adapter={adapter_name} match={request_cache.match_len} "
            f"prefill_from={prefill_from}/{len(prompt_ids)}"
        )

    new_ids = prompt_ids[prefill_from:]
    admission = admit_paged_kv_request(
        cache_mgr,
        prompt_ids=prompt_ids,
        adapter_name=adapter_name,
        request_cache=request_cache,
        max_new_tokens=max_new_tokens,
        log_fn=LOG,
    )
    request_cache = admission.request
    new_ids = admission.new_ids
    max_new_tokens = admission.max_new_tokens
    prefill_from = request_cache.prefill_from

    try:
        LOG(f"FlashInfer paged prefill: {len(new_ids)} tokens (cache_start={prefill_from})")
        hidden = _run_flashinfer_prefill_segment(
            manager,
            new_ids,
            start_pos=prefill_from,
            request_cache=request_cache,
            backend=backend,
            perf=perf,
        )
        lm_t0 = perf.tic()
        logits = _model_final_norm_and_head(model, hidden, perf=perf, prefix="prefill")
        perf.add("prefill_lm_head_s", lm_t0)

        generated_ids: list[int] = []
        cached_generated_ids: list[int] = []
        generated_token_tensors: list[torch.Tensor] = []
        cached_generated_token_tensors: list[torch.Tensor] = []
        logprobs: list[float] = []
        use_gpu_greedy = (
            sampling_mode == "greedy_fast"
            and not compute_logprobs
            and repetition_penalty == 1.0
        )
        decode_graph: FlashInferDecodeCudaGraphRunner | None = None
        if can_use_flashinfer_cuda_graph_decode(
            manager,
            use_gpu_greedy=use_gpu_greedy,
            max_new_tokens=max_new_tokens,
        ):
            decode_graph = _get_decode_graph_runner(manager, backend, cache_mgr)
            LOG(f"FlashInfer CUDA graph decode enabled cached={decode_graph.graph_count}")
        token_histories: list[list[int]] | None = None if use_gpu_greedy else [prompt_ids + generated_ids]
        current_pos = len(prompt_ids)
        decode_position_ids = torch.empty((1, 1), device=device, dtype=torch.long)
        t0 = time.perf_counter()
        decode_t0 = perf.tic()

        for _step in range(max_new_tokens):
            if max_time is not None and time.perf_counter() - t0 >= max_time:
                break

            sample_t0 = perf.tic()
            if use_gpu_greedy:
                token_tensor = greedy_argmax_token(logits)
                next_logprobs = []
            else:
                if token_histories is None:
                    raise RuntimeError("token histories must be initialized outside GPU greedy mode")
                next_tokens, next_logprobs = _sample_next_tokens(
                    logits[:, -1, :],
                    token_histories=token_histories,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    sampling_mode=sampling_mode,
                    compute_logprobs=compute_logprobs,
                )
                token_id = next_tokens[0]
                token_tensor = None
            perf.add("decode_sample_s", sample_t0)

            token_id: int | None
            if use_gpu_greedy:
                generated_token_tensors.append(token_tensor.squeeze(1).detach())
                if ignore_eos:
                    token_id = None
                else:
                    token_id = int(token_tensor[0, 0].item())
                    generated_ids.append(token_id)
            else:
                generated_ids.append(token_id)
                if compute_logprobs and next_logprobs:
                    logprobs.append(next_logprobs[0])
                if token_histories is not None:
                    token_histories[0].append(token_id)

            if token_id is not None and not ignore_eos and token_id == tokenizer.eos_token_id:
                break

            reserved_spans = None
            page_table: FlashInferPageTable | None = None
            chunk_index = len(request_cache.page_span_chunks)
            if decode_graph is not None and token_tensor is not None:
                reserve_t0 = perf.tic()
                reserved_spans = cache_mgr.reserve_chunk(request_cache, 1)
                perf.add("decode_reserve_s", reserve_t0)
                plan_t0 = perf.tic()
                page_table = _plan_decode(
                    backend,
                    cache_mgr,
                    request_cache,
                    device=device,
                    dtype=model.get_input_embeddings().weight.dtype,
                )
                perf.add("decode_plan_s", plan_t0)
                graph_t0 = perf.tic()
                graph_logits = decode_graph.run(
                    token_tensor=token_tensor,
                    position=current_pos,
                    chunk_spans=reserved_spans,
                    page_table_pages=page_table.paged_kv_indices.numel(),
                )
                perf.add("decode_cuda_graph_s", graph_t0)
                if graph_logits is not None:
                    cache_mgr.mark_graph_decode_chunk_written(
                        request_cache,
                        chunk_index=chunk_index,
                        num_layers=len(manager.lora_runtime.layers),
                        device=device,
                    )
                    cached_generated_token_tensors.append(token_tensor.squeeze(1).detach())
                    logits = graph_logits
                    current_pos += 1
                    perf.log_decode_progress(
                        generated_tokens=len(generated_token_tensors),
                        cached_tokens=len(cached_generated_token_tensors),
                        current_pos=current_pos,
                        started=t0,
                        cache_snapshot=cache_mgr.perf_snapshot(),
                    )
                    continue
                if decode_graph.disabled_reason is not None:
                    LOG(f"FlashInfer CUDA graph decode fallback: {decode_graph.disabled_reason}")
                    decode_graph = None

            embed_t0 = perf.tic()
            if token_tensor is None:
                token_tensor = torch.tensor([[token_id]], device=device, dtype=torch.long)
            hidden = model.get_input_embeddings()(token_tensor)
            decode_position_ids.fill_(current_pos)
            position_ids = decode_position_ids
            perf.add("decode_embed_s", embed_t0)

            if reserved_spans is None:
                reserve_t0 = perf.tic()
                cache_mgr.reserve_chunk(request_cache, 1)
                perf.add("decode_reserve_s", reserve_t0)
            if page_table is None:
                plan_t0 = perf.tic()
                _plan_decode(
                    backend,
                    cache_mgr,
                    request_cache,
                    device=device,
                    dtype=hidden.dtype,
                )
                perf.add("decode_plan_s", plan_t0)
            layers_t0 = perf.tic()
            for layer_idx, layer in enumerate(manager.lora_runtime.layers):
                hidden = _run_flashinfer_attention(
                    layer,
                    manager.lora_runtime.layer_plans[layer_idx],
                    layer_idx,
                    hidden,
                    cache_mgr,
                    position_ids,
                    request_cache=request_cache,
                    mode="decode",
                    backend=backend,
                    perf=perf,
                )
            perf.add("decode_layers_s", layers_t0)

            if use_gpu_greedy:
                cached_generated_token_tensors.append(token_tensor.squeeze(1).detach())
            elif token_id is not None:
                cached_generated_ids.append(token_id)
            lm_t0 = perf.tic()
            logits = _model_final_norm_and_head(model, hidden, perf=perf, prefix="decode")
            perf.add("decode_lm_head_s", lm_t0)
            current_pos += 1
            perf.log_decode_progress(
                generated_tokens=len(generated_token_tensors) if use_gpu_greedy else len(generated_ids),
                cached_tokens=(
                    len(cached_generated_token_tensors) if use_gpu_greedy else len(cached_generated_ids)
                ),
                current_pos=current_pos,
                started=t0,
                cache_snapshot=cache_mgr.perf_snapshot(),
            )

        perf.add("decode_total_s", decode_t0)
        if use_gpu_greedy:
            generated_ids = _token_tensors_to_list(generated_token_tensors)
            cached_generated_ids = _token_tensors_to_list(cached_generated_token_tensors)
        final_cache_ids = prompt_ids + cached_generated_ids
        request_span_count = len(request_cache.new_spans)
        request_page_chunks = len(request_cache.page_span_chunks)
        commit_t0 = perf.tic()
        cache_mgr.commit_request(request_cache, final_cache_ids)
        perf.add("commit_s", commit_t0)
        committed = True

        elapsed = time.perf_counter() - t0
        text_t0 = perf.tic()
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        perf.add("decode_text_s", text_t0)
        perf.add("generation_total_s", generation_t0)
        prefill_s = (
            perf.sections.get("backend_create_s", 0.0)
            + perf.sections.get("begin_request_s", 0.0)
            + perf.sections.get("prefill_total_s", 0.0)
            + perf.sections.get("prefill_lm_head_s", 0.0)
        )
        decode_s = perf.sections.get("decode_total_s", 0.0)
        total_s = perf.sections.get("generation_total_s", 0.0)
        perf.log_summary(
            prompt_tokens=len(prompt_ids),
            prefill_tokens=len(new_ids),
            generated_tokens=len(generated_ids),
            cached_generated_tokens=len(cached_generated_ids),
            cache_snapshot=cache_mgr.perf_snapshot(),
            request_span_count=request_span_count,
            request_page_chunks=request_page_chunks,
        )
        LOG(f"FlashInfer paged generation: {len(generated_ids)} tokens in {elapsed:.2f}s")
        return GenerationOutput(
            response_text=response_text,
            response_ids=generated_ids,
            logprobs=logprobs,
            timings={
                "total_s": total_s,
                "prefill_s": prefill_s,
                "decode_s": decode_s,
                "cache_match_tokens": request_cache.match_len,
            },
        )
    finally:
        if not committed:
            cache_mgr.abort_request(request_cache)


def generate_batch_with_flashinfer_kvcache(
    manager: "ModelManager",
    batch_prompt_ids: list[list[int]],
    *,
    adapter_name: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = 1.05,
    max_time: float | None = None,
) -> "BatchGenerationOutput":
    """Generate a batch with FlashInfer paged KV prefill and decode."""

    from model.generation import BatchGenerationOutput, _sample_next_tokens

    if not batch_prompt_ids:
        return BatchGenerationOutput(response_texts=[], response_ids=[])
    for index, prompt_ids in enumerate(batch_prompt_ids):
        if not prompt_ids:
            raise ValueError(f"batch_prompt_ids[{index}] must not be empty")

    tokenizer = manager.tokenizer
    device = manager.device
    model = manager.model

    cache_mgr = manager.kvcache_manager
    if not isinstance(cache_mgr, FlashInferPagedKVCacheManager):
        raise RuntimeError("FlashInferPagedKVCacheManager not initialized")

    cache_mgr.reset_perf_counters()
    perf = FlashInferPerfTrace(device)
    generation_t0 = perf.tic()
    backend_t0 = perf.tic()
    backend = _get_backend(manager)
    perf.add("backend_create_s", backend_t0)
    begin_t0 = perf.tic()
    request_caches = cache_mgr.begin_batch_requests(batch_prompt_ids, adapter_name)
    perf.add("begin_request_s", begin_t0)
    committed = False

    try:
        request_caches, new_ids_batch, max_new_tokens = _admit_paged_kv_batch(
            cache_mgr,
            batch_prompt_ids=batch_prompt_ids,
            adapter_name=adapter_name,
            requests=request_caches,
            max_new_tokens=max_new_tokens,
        )
        prefill_froms = [request_cache.prefill_from for request_cache in request_caches]
        for row, request_cache in enumerate(request_caches):
            if request_cache.match_len:
                LOG(
                    "FlashInfer paged KV cache match: "
                    f"adapter={adapter_name} row={row} match={request_cache.match_len} "
                    f"prefill_from={request_cache.prefill_from}/{len(batch_prompt_ids[row])}"
                )
        LOG(
            "FlashInfer paged batch prefill: "
            f"batch={len(batch_prompt_ids)} tokens={sum(len(ids) for ids in new_ids_batch)}"
        )
        hidden = _run_flashinfer_prefill_batch(
            manager,
            new_ids_batch,
            start_positions=prefill_froms,
            request_caches=request_caches,
            backend=backend,
            perf=perf,
        )
        last_offsets = []
        cursor = 0
        for token_ids in new_ids_batch:
            cursor += len(token_ids)
            last_offsets.append(cursor - 1)
        last_offset_tensor = torch.tensor(last_offsets, device=device, dtype=torch.long)
        batch_last_hidden = hidden[0, last_offset_tensor, :].unsqueeze(1).contiguous()
        lm_t0 = perf.tic()
        logits = _model_final_norm_and_head(model, batch_last_hidden, perf=perf, prefix="prefill")
        perf.add("prefill_lm_head_s", lm_t0)

        generated_ids_batch: list[list[int]] = [[] for _ in batch_prompt_ids]
        cached_generated_ids_batch: list[list[int]] = [[] for _ in batch_prompt_ids]
        token_histories = [list(prompt_ids) for prompt_ids in batch_prompt_ids]
        current_positions = [len(prompt_ids) for prompt_ids in batch_prompt_ids]
        active_indices = list(range(len(batch_prompt_ids)))
        decode_graph: FlashInferBatchDecodeCudaGraphRunner | None = None
        if can_use_flashinfer_cuda_graph_batch_decode(
            manager,
            max_new_tokens=max_new_tokens,
        ):
            decode_graph = _get_batch_decode_graph_runner(manager, backend, cache_mgr)
            LOG(f"FlashInfer batch CUDA graph decode enabled cached={decode_graph.graph_count}")
        t0 = time.perf_counter()
        decode_t0 = perf.tic()

        for _step in range(max_new_tokens):
            if max_time is not None and time.perf_counter() - t0 >= max_time:
                break
            if not active_indices:
                break

            sample_t0 = perf.tic()
            active_histories = [token_histories[index] for index in active_indices]
            next_tokens, _next_logprobs = _sample_next_tokens(
                logits[:, -1, :],
                token_histories=active_histories,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                sampling_mode="project_default",
                compute_logprobs=False,
            )
            perf.add("decode_sample_s", sample_t0)

            continuing_indices: list[int] = []
            continuing_tokens: list[int] = []
            continuing_positions: list[int] = []
            continuing_requests: list[FlashInferPagedKVRequest] = []
            for active_index, token_id in zip(active_indices, next_tokens, strict=True):
                generated_ids_batch[active_index].append(token_id)
                token_histories[active_index].append(token_id)
                if token_id == tokenizer.eos_token_id:
                    continue
                continuing_indices.append(active_index)
                continuing_tokens.append(token_id)
                continuing_positions.append(current_positions[active_index])
                continuing_requests.append(request_caches[active_index])

            if not continuing_indices:
                active_indices = []
                break

            graph_reserved_spans: list[list[KVBlockSpan]] | None = None
            graph_chunk_indices: list[int] | None = None
            if decode_graph is not None:
                reserve_t0 = perf.tic()
                graph_reserved_spans = []
                graph_chunk_indices = []
                for request_cache in continuing_requests:
                    graph_chunk_indices.append(len(request_cache.page_span_chunks))
                    graph_reserved_spans.append(cache_mgr.reserve_chunk(request_cache, 1))
                perf.add("decode_reserve_s", reserve_t0)

                plan_t0 = perf.tic()
                page_table = _plan_decode_batch(
                    backend,
                    cache_mgr,
                    continuing_requests,
                    device=device,
                    dtype=model.get_input_embeddings().weight.dtype,
                    reuse_buffer=True,
                )
                perf.add("decode_plan_s", plan_t0)

                graph_t0 = perf.tic()
                graph_logits = decode_graph.run(
                    token_ids=continuing_tokens,
                    positions=continuing_positions,
                    chunk_spans_batch=graph_reserved_spans,
                    page_table_pages=page_table.paged_kv_indices.numel(),
                )
                perf.add("decode_cuda_graph_s", graph_t0)
                if graph_logits is not None:
                    for request_cache, chunk_index in zip(continuing_requests, graph_chunk_indices, strict=True):
                        cache_mgr.mark_graph_decode_chunk_written(
                            request_cache,
                            chunk_index=chunk_index,
                            num_layers=len(manager.lora_runtime.layers),
                            device=device,
                        )
                    for active_index, token_id in zip(continuing_indices, continuing_tokens, strict=True):
                        cached_generated_ids_batch[active_index].append(token_id)
                        current_positions[active_index] += 1
                    logits = graph_logits
                    active_indices = continuing_indices
                    perf.log_decode_progress(
                        generated_tokens=sum(len(ids) for ids in generated_ids_batch),
                        cached_tokens=sum(len(ids) for ids in cached_generated_ids_batch),
                        current_pos=max(current_positions, default=0),
                        started=t0,
                        cache_snapshot=cache_mgr.perf_snapshot(),
                    )
                    continue
                if decode_graph.disabled_reason is not None:
                    LOG(f"FlashInfer batch CUDA graph decode fallback: {decode_graph.disabled_reason}")
                    decode_graph = None

            hidden = _run_flashinfer_decode_batch(
                manager,
                continuing_tokens,
                positions=continuing_positions,
                request_caches=continuing_requests,
                backend=backend,
                perf=perf,
                reserve_chunks=graph_reserved_spans is None,
                plan_decode=graph_reserved_spans is None,
            )
            for active_index, token_id in zip(continuing_indices, continuing_tokens, strict=True):
                cached_generated_ids_batch[active_index].append(token_id)
                current_positions[active_index] += 1
            lm_t0 = perf.tic()
            logits = _model_final_norm_and_head(model, hidden, perf=perf, prefix="decode")
            perf.add("decode_lm_head_s", lm_t0)
            active_indices = continuing_indices
            perf.log_decode_progress(
                generated_tokens=sum(len(ids) for ids in generated_ids_batch),
                cached_tokens=sum(len(ids) for ids in cached_generated_ids_batch),
                current_pos=max(current_positions, default=0),
                started=t0,
                cache_snapshot=cache_mgr.perf_snapshot(),
            )

        perf.add("decode_total_s", decode_t0)
        final_cache_ids_batch = [
            list(prompt_ids) + cached_generated_ids
            for prompt_ids, cached_generated_ids in zip(
                batch_prompt_ids,
                cached_generated_ids_batch,
                strict=True,
            )
        ]
        request_span_count = sum(len(request_cache.new_spans) for request_cache in request_caches)
        request_page_chunks = sum(len(request_cache.page_span_chunks) for request_cache in request_caches)
        commit_t0 = perf.tic()
        cache_mgr.commit_batch_requests(request_caches, final_cache_ids_batch)
        perf.add("commit_s", commit_t0)
        committed = True

        elapsed = time.perf_counter() - t0
        text_t0 = perf.tic()
        response_texts = [
            tokenizer.decode(generated_ids, skip_special_tokens=True)
            for generated_ids in generated_ids_batch
        ]
        perf.add("decode_text_s", text_t0)
        perf.add("generation_total_s", generation_t0)
        perf.log_summary(
            prompt_tokens=sum(len(prompt_ids) for prompt_ids in batch_prompt_ids),
            prefill_tokens=sum(len(new_ids) for new_ids in new_ids_batch),
            generated_tokens=sum(len(generated_ids) for generated_ids in generated_ids_batch),
            cached_generated_tokens=sum(len(ids) for ids in cached_generated_ids_batch),
            cache_snapshot=cache_mgr.perf_snapshot(),
            request_span_count=request_span_count,
            request_page_chunks=request_page_chunks,
        )
        LOG(
            "FlashInfer paged batch generation: "
            f"batch={len(batch_prompt_ids)} tokens={sum(len(ids) for ids in generated_ids_batch)} "
            f"in {elapsed:.2f}s"
        )
        return BatchGenerationOutput(
            response_texts=response_texts,
            response_ids=generated_ids_batch,
        )
    finally:
        if not committed:
            cache_mgr.abort_batch_requests(request_caches)


def prefill_flashinfer_paged_kv_chunk(
    manager: "ModelManager",
    token_ids: list[int],
    *,
    adapter_name: str,
    target_len: int,
    weight_adapter_name: str | None = None,
    cache_adapter_name: str | None = None,
):
    """Populate paged KV cache for one bounded pre-prefill chunk."""

    from model.fullcache.prefill import FullKVPrePrefillResult

    cache_mgr = manager.kvcache_manager
    if not isinstance(cache_mgr, FlashInferPagedKVCacheManager):
        raise RuntimeError("paged pre-prefill requires FlashInferPagedKVCacheManager")

    weight_adapter = weight_adapter_name or adapter_name
    cache_adapter = cache_adapter_name or adapter_name
    request = cache_mgr.begin_preprefill_request(
        token_ids,
        cache_adapter,
        target_len=target_len,
    )
    if request is None:
        completed_to = min(max(0, target_len), len(token_ids))
        return FullKVPrePrefillResult(
            adapter_name=cache_adapter,
            target_len=completed_to,
            filled_from=completed_to,
            filled_to=completed_to,
            filled_tokens=0,
            elapsed_s=0.0,
            already_complete=True,
        )

    committed = False
    fill_from = request.prefill_from
    fill_to = request.materialized_to
    segment = request.token_ids[fill_from:fill_to]
    t0 = time.perf_counter()
    try:
        with manager.use_adapter(weight_adapter, role=AdapterRole.SERVE.value):
            with manager.inference_session(enable_cache=False):
                with torch.no_grad():
                    _run_flashinfer_prefill_segment(
                        manager,
                        segment,
                        start_pos=fill_from,
                        request_cache=request,
                        backend=_make_backend(manager),
                    )
        cache_mgr.commit_request(request, request.token_ids)
        committed = True
        elapsed = time.perf_counter() - t0
        return FullKVPrePrefillResult(
            adapter_name=cache_adapter,
            target_len=fill_to,
            filled_from=fill_from,
            filled_to=fill_to,
            filled_tokens=fill_to - fill_from,
            elapsed_s=elapsed,
        )
    finally:
        if not committed:
            cache_mgr.abort_request(request)
