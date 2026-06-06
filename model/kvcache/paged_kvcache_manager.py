"""Paged full-KV cache manager with radix prefix metadata.

This manager is the project-owned part needed before wiring FlashInfer kernels:
it stores adapter-specific full KV in fixed-size pages while reusing the same
radix prefix cache interface as ``FullKVCacheManager``.
"""

from __future__ import annotations

import time
from itertools import count

import torch

from model.kvcache.flashinfer_page_table import (
    FlashInferBatchPageTableBuffer,
    FlashInferPageTable,
    FlashInferPageTableBuffer,
    _coalesce_block_spans,
    build_flashinfer_batch_page_table,
    build_flashinfer_page_table,
)
from model.kvcache.paged_kv_store import PagedKVBlockStore
from model.kvcache.paged_kvcache_request import FlashInferPagedKVRequest
from model.kvcache.radix_cache import RadixPrefixCache
from model.kvcache.radix_cache import RadixHotPrefix
from model.kvcache.radix_priority import RadixPriorityIndex
from model.kvcache.radix_storage import slice_block_spans
from model.kvcache.radix_storage import span_total_length
from model.kvcache.radix_types import (
    FULL_KEY,
    KVBlockSpan,
)


class FlashInferPagedKVCacheManager:
    """Manages adapter-specific full KV caches in fixed-size pages."""

    def __init__(
        self,
        num_layers: int,
        *,
        page_size: int = 32,
        max_pages: int = 0,
        workspace_mb: int = 128,
        kv_append_mode: str = "manual",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        model_config: object | None = None,
        memory_coordinator: object | None = None,
    ) -> None:
        """Create a paged cache manager compatible with full-KV APIs."""

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self._num_layers = int(num_layers)
        self._page_size = int(page_size)
        self._workspace_mb = int(workspace_mb)
        self._device = device
        self._dtype = dtype
        self._model_config = model_config
        self._memory_coordinator = memory_coordinator
        self._prefix_cache = RadixPrefixCache()
        self._block_store = PagedKVBlockStore(
            page_size=page_size,
            max_pages=max_pages,
            num_layers=num_layers,
            kv_append_mode=kv_append_mode,
            device=device,
            dtype=dtype,
        )
        self._request_ids = count(1)
        self._active_request: FlashInferPagedKVRequest | None = None
        self._active_batch_requests: list[FlashInferPagedKVRequest] | None = None
        self._max_len_by_adapter: dict[str, int] = {}
        self._latest_token_ids_by_adapter: dict[str, list[int]] = {}
        self._perf_counters: dict[str, float] = {}
        self._page_table_buffer: FlashInferPageTableBuffer | None = None
        self._batch_page_table_buffer: FlashInferBatchPageTableBuffer | None = None

    @property
    def paged_kv_cache(self) -> torch.Tensor | None:
        """Return dense FlashInfer KV cache tensor, if allocated."""

        return self._block_store.paged_kv_cache

    @property
    def can_use_flashinfer(self) -> bool:
        """Return whether the manager has a dense paged cache tensor."""

        return self._block_store.can_use_dense_cache

    @property
    def page_size(self) -> int:
        """Return the fixed token capacity of each physical page."""

        return self._page_size

    @property
    def priority_index(self) -> RadixPriorityIndex:
        """Expose the radix priority index used by pre-prefill/offload policy."""

        return self._prefix_cache.priority_index

    @property
    def allocated_pages(self) -> int:
        """Return live physical page count."""

        return self._block_store.allocated_pages

    @property
    def dense_capacity_pages(self) -> int:
        """Return the current dense FlashInfer cache capacity."""

        return self._block_store.dense_capacity_pages

    @property
    def free_pages(self) -> int:
        """Return reusable physical page count."""

        return self._block_store.free_pages

    @property
    def max_pages(self) -> int:
        """Return the configured physical page cap, or 0 when uncapped."""

        return self._block_store.max_pages

    @property
    def available_pages(self) -> int:
        """Return pages that can still be allocated before hitting the cap."""

        return self._block_store.available_pages

    def estimate_request_new_pages(
        self,
        request: FlashInferPagedKVRequest,
        *,
        prefill_tokens: int,
        max_decode_tokens: int,
    ) -> int:
        """Conservatively estimate new pages needed by one request."""

        del request
        tokens = max(0, int(prefill_tokens)) + max(0, int(max_decode_tokens))
        if tokens == 0:
            return 0
        return (tokens + self._page_size - 1) // self._page_size

    def can_fit_request(
        self,
        request: FlashInferPagedKVRequest,
        *,
        prefill_tokens: int,
        max_decode_tokens: int,
        reserve_pages: int = 4,
    ) -> bool:
        needed_pages = self.estimate_request_new_pages(
            request,
            prefill_tokens=prefill_tokens,
            max_decode_tokens=max_decode_tokens,
        )
        return needed_pages + max(0, int(reserve_pages)) <= self.available_pages

    def max_decode_tokens_that_fit(
        self,
        request: FlashInferPagedKVRequest,
        *,
        prefill_tokens: int,
        requested_decode_tokens: int,
        reserve_pages: int = 4,
    ) -> int:
        del request
        token_budget = max(0, self.available_pages - max(0, int(reserve_pages))) * self._page_size
        return max(0, min(int(requested_decode_tokens), token_budget - max(0, int(prefill_tokens))))

    def reset_perf_counters(self) -> None:
        """Clear manager and store performance counters."""

        self._perf_counters.clear()
        self._block_store.reset_perf_counters()

    def perf_snapshot(self) -> dict[str, float]:
        """Return accumulated manager and page-store performance counters."""

        snapshot = dict(self._perf_counters)
        snapshot.update(self._block_store.perf_snapshot())
        snapshot["manager_allocated_pages"] = float(self.allocated_pages)
        snapshot["manager_free_pages"] = float(self.free_pages)
        return snapshot

    def _perf_add(self, name: str, value: float) -> None:
        self._perf_counters[name] = self._perf_counters.get(name, 0.0) + float(value)

    def reserve_chunk(
        self,
        request: FlashInferPagedKVRequest,
        token_count: int,
    ) -> list[KVBlockSpan]:
        """Reserve page spans for the next request-local token chunk."""

        t0 = time.perf_counter()
        spans = self._block_store.reserve_spans(
            adapter=request.adapter_name,
            request_spans=request.new_spans,
            token_count=token_count,
        )
        request.page_span_chunks.append(spans)
        self._perf_add("manager_reserve_chunk_s", time.perf_counter() - t0)
        self._perf_add("manager_reserve_chunk_calls", 1.0)
        self._perf_add("manager_reserved_tokens", float(token_count))
        self._perf_add("manager_reserved_spans", float(len(spans)))
        return spans

    def layer_cache(self, layer_idx: int) -> torch.Tensor | None:
        """Return one layer's dense paged cache tensor for FlashInfer."""

        return self._block_store.layer_cache(layer_idx)

    def ensure_dense_cache_shape(
        self,
        *,
        layer_idx: int,
        required_pages: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Ensure the dense paged cache has enough capacity for graph writes."""

        self._block_store.ensure_dense_cache_shape(
            layer_idx=layer_idx,
            required_pages=required_pages,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    def mark_graph_decode_chunk_written(
        self,
        request: FlashInferPagedKVRequest,
        *,
        chunk_index: int,
        num_layers: int,
        device: torch.device,
    ) -> None:
        """Advance request metadata after CUDA graph writes a decode KV chunk."""

        if chunk_index < 0 or chunk_index >= len(request.page_span_chunks):
            raise RuntimeError("missing reserved page spans for graph KV chunk")
        spans = request.page_span_chunks[chunk_index]
        self._block_store.touch_spans(spans, device)
        for layer_idx in range(num_layers):
            current = request.layer_chunk_counts.get(layer_idx, 0)
            if current != chunk_index:
                raise RuntimeError("graph decode chunk metadata is out of sync")
            request.layer_chunk_counts[layer_idx] = chunk_index + 1
        self._perf_add("manager_append_full_calls", float(num_layers))
        self._perf_add("manager_append_full_nhd_calls", float(num_layers))
        self._perf_add("manager_append_full_tokens", float(num_layers * sum(span.length for span in spans)))

    def begin_request(
        self,
        token_ids: list[int],
        adapter_name: str,
    ) -> FlashInferPagedKVRequest:
        """Open a paged full-KV cache request for one foreground generation."""

        if self._active_request is not None or self._active_batch_requests is not None:
            raise RuntimeError("KV cache request already active")

        total_t0 = time.perf_counter()
        request = self._begin_request_unchecked(token_ids, adapter_name)
        self._active_request = request
        self._perf_add("manager_begin_request_s", time.perf_counter() - total_t0)
        self._perf_add("manager_begin_request_calls", 1.0)
        return request

    def begin_batch_requests(
        self,
        batch_token_ids: list[list[int]],
        adapter_name: str,
    ) -> list[FlashInferPagedKVRequest]:
        """Open multiple foreground requests for one FlashInfer batch."""

        if self._active_request is not None or self._active_batch_requests is not None:
            raise RuntimeError("KV cache request already active")
        if not batch_token_ids:
            return []

        total_t0 = time.perf_counter()
        requests = [
            self._begin_request_unchecked(token_ids, adapter_name)
            for token_ids in batch_token_ids
        ]
        self._active_batch_requests = requests
        self._perf_add("manager_begin_batch_request_s", time.perf_counter() - total_t0)
        self._perf_add("manager_begin_batch_request_calls", 1.0)
        self._perf_add("manager_begin_batch_size", float(len(requests)))
        return requests

    def _begin_request_unchecked(
        self,
        token_ids: list[int],
        adapter_name: str,
    ) -> FlashInferPagedKVRequest:
        """Build request state without changing the active-request guard."""

        key = FULL_KEY(adapter_name)
        lookup_t0 = time.perf_counter()
        match = self._prefix_cache.lookup(token_ids, keys=(key,))
        self._perf_add("manager_radix_lookup_s", time.perf_counter() - lookup_t0)
        self._perf_add("manager_radix_lookup_calls", 1.0)
        prompt_len = len(token_ids)
        match_len = min(prompt_len, match.key_match_len.get(key, 0))

        prefill_from = match_len
        if prompt_len > 0 and match_len == prompt_len:
            # A full prompt hit still replays the final prompt token so the
            # custom generation loop can produce next-token logits.
            prefill_from = prompt_len - 1
        prefix_source_spans = match.key_spans.get(key, [])
        usable_t0 = time.perf_counter()
        prefill_from = self._flashinfer_usable_prefix_len(
            prefix_source_spans,
            desired_len=prefill_from,
        )
        self._perf_add("manager_usable_prefix_s", time.perf_counter() - usable_t0)

        slice_t0 = time.perf_counter()
        prefix_spans = _coalesce_block_spans(slice_block_spans(prefix_source_spans, 0, prefill_from))
        self._perf_add("manager_prefix_span_slice_s", time.perf_counter() - slice_t0)

        request = FlashInferPagedKVRequest(
            request_id=next(self._request_ids),
            token_ids=list(token_ids),
            adapter_name=adapter_name,
            match_len=match_len,
            prefill_from=prefill_from,
            materialized_from=prefill_from,
            materialized_to=prompt_len,
            prefix_spans=prefix_spans,
            current_kv_len=prefill_from,
        )
        self._perf_add("manager_begin_prompt_tokens", float(prompt_len))
        self._perf_add("manager_begin_match_len", float(match_len))
        self._perf_add("manager_begin_prefill_from", float(prefill_from))
        self._perf_add("manager_begin_prefix_spans", float(len(prefix_spans)))
        return request

    def begin_preprefill_request(
        self,
        token_ids: list[int],
        adapter_name: str,
        *,
        target_len: int,
    ) -> FlashInferPagedKVRequest | None:
        """Open a bounded background pre-prefill request.

        Unlike foreground generation, a complete prefix hit returns ``None``
        instead of replaying the final token, because no logits are needed.
        """

        if self._active_request is not None or self._active_batch_requests is not None:
            raise RuntimeError("KV cache request already active")

        target_len = min(max(0, target_len), len(token_ids))
        if target_len == 0:
            return None

        total_t0 = time.perf_counter()
        prefix_ids = list(token_ids[:target_len])
        key = FULL_KEY(adapter_name)
        lookup_t0 = time.perf_counter()
        match = self._prefix_cache.lookup(prefix_ids, keys=(key,))
        self._perf_add("manager_preprefill_radix_lookup_s", time.perf_counter() - lookup_t0)
        self._perf_add("manager_preprefill_radix_lookup_calls", 1.0)
        match_len = min(target_len, match.key_match_len.get(key, 0))
        if match_len >= target_len:
            self._perf_add("manager_begin_preprefill_complete_s", time.perf_counter() - total_t0)
            self._perf_add("manager_begin_preprefill_complete_calls", 1.0)
            return None
        prefix_source_spans = match.key_spans.get(key, [])
        usable_t0 = time.perf_counter()
        fill_from = self._flashinfer_usable_prefix_len(
            prefix_source_spans,
            desired_len=match_len,
        )
        self._perf_add("manager_preprefill_usable_prefix_s", time.perf_counter() - usable_t0)

        slice_t0 = time.perf_counter()
        prefix_spans = _coalesce_block_spans(slice_block_spans(prefix_source_spans, 0, fill_from))
        self._perf_add("manager_preprefill_prefix_span_slice_s", time.perf_counter() - slice_t0)

        request = FlashInferPagedKVRequest(
            request_id=next(self._request_ids),
            token_ids=prefix_ids,
            adapter_name=adapter_name,
            match_len=match_len,
            prefill_from=fill_from,
            materialized_from=fill_from,
            materialized_to=target_len,
            prefix_spans=prefix_spans,
            current_kv_len=fill_from,
        )
        self._active_request = request
        self._perf_add("manager_begin_preprefill_request_s", time.perf_counter() - total_t0)
        self._perf_add("manager_begin_preprefill_request_calls", 1.0)
        self._perf_add("manager_begin_preprefill_target_len", float(target_len))
        self._perf_add("manager_begin_preprefill_match_len", float(match_len))
        self._perf_add("manager_begin_preprefill_fill_from", float(fill_from))
        return request

    def commit_request(
        self,
        request: FlashInferPagedKVRequest,
        final_token_ids: list[int],
    ) -> None:
        """Commit materialized KV chunks into pages and radix coverage."""

        if self._active_request is not request or self._active_batch_requests is not None:  # protection
            raise RuntimeError("cannot commit inactive KV cache request")

        self._commit_request_unchecked(request, final_token_ids)
        self._active_request = None

    def commit_batch_requests(
        self,
        requests: list[FlashInferPagedKVRequest],
        final_token_ids_batch: list[list[int]],
    ) -> None:
        """Commit all requests from the currently active FlashInfer batch."""

        if self._active_request is not None or self._active_batch_requests is not requests:
            raise RuntimeError("cannot commit inactive KV cache batch")
        if len(requests) != len(final_token_ids_batch):
            raise ValueError("request and final-token batch sizes differ")
        try:
            for request, final_token_ids in zip(requests, final_token_ids_batch, strict=True):
                self._commit_request_unchecked(request, final_token_ids)
        finally:
            self._active_batch_requests = None

    def _commit_request_unchecked(
        self,
        request: FlashInferPagedKVRequest,
        final_token_ids: list[int],
    ) -> None:
        """Commit one request without checking or clearing the active guard."""

        total_t0 = time.perf_counter()
        final_len = len(final_token_ids)
        start = min(request.match_len, final_len)
        skip = max(0, start - request.materialized_from)
        # append_full writes request-local chunks into pages immediately.
        # Commit only inserts the non-prefix suffix into radix metadata.
        coalesce_t0 = time.perf_counter()
        commit_len = max(0, final_len - start)
        commit_spans = _coalesce_block_spans(
            slice_block_spans(request.new_spans, skip, commit_len)
        )
        self._perf_add("manager_commit_coalesce_s", time.perf_counter() - coalesce_t0)
        if commit_spans:
            try:
                # Radix coverage insertion is kept separate from page writes so
                # the tree remains backend-neutral. Roll back pages if insertion
                # detects non-contiguous coverage.
                insert_t0 = time.perf_counter()
                self._prefix_cache.insert_coverage(
                    final_token_ids,
                    key=FULL_KEY(request.adapter_name),
                    start=start,
                    spans=commit_spans,
                )
                self._perf_add("manager_radix_insert_coverage_s", time.perf_counter() - insert_t0)
                self._perf_add("manager_radix_insert_coverage_calls", 1.0)
            except Exception:
                self._block_store.drop_spans(request.new_spans)
                request.new_spans = []
                raise
        if request.new_spans:
            committed_page_ids = {span.block_id for span in commit_spans}
            uncommitted_spans = [
                span for span in request.new_spans if span.block_id not in committed_page_ids
            ]
            self._block_store.drop_spans(uncommitted_spans)

        cached_len = start + span_total_length(commit_spans)
        self._max_len_by_adapter[request.adapter_name] = max(
            self._max_len_by_adapter.get(request.adapter_name, 0),
            cached_len,
        )
        self._latest_token_ids_by_adapter[request.adapter_name] = list(final_token_ids)
        self._perf_add("manager_commit_request_s", time.perf_counter() - total_t0)
        self._perf_add("manager_commit_request_calls", 1.0)
        self._perf_add("manager_commit_final_len", float(final_len))
        self._perf_add("manager_commit_start", float(start))
        self._perf_add("manager_commit_spans", float(len(commit_spans)))

        if self._memory_coordinator is not None:
            set_bytes = getattr(self._memory_coordinator, "set_kvcache_bytes", None)
            if callable(set_bytes):
                set_bytes(request.adapter_name, self._block_store.adapter_bytes(request.adapter_name))
            self._memory_coordinator.notify_kvcache_access(request.adapter_name)
            self._memory_coordinator.maybe_offload()

    def abort_request(self, request: FlashInferPagedKVRequest) -> None:
        """Close an active request without committing request-local chunks."""

        if self._active_request is request:
            self._block_store.drop_spans(request.new_spans)
            request.new_spans = []
            self._active_request = None

    def abort_batch_requests(self, requests: list[FlashInferPagedKVRequest]) -> None:
        """Close an active batch without committing request-local chunks."""

        if self._active_batch_requests is not requests:
            return
        for request in requests:
            self._block_store.drop_spans(request.new_spans)
            request.new_spans = []
        self._active_batch_requests = None

    def append_full(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        request: FlashInferPagedKVRequest | None = None,
    ) -> None:
        """Append one layer's request-local K/V chunk before commit."""

        request = request or self._active_request
        if request is None:
            raise RuntimeError("append_full requires an active request")
        t0 = time.perf_counter()
        chunk_index = request.layer_chunk_counts.get(layer_idx, 0)
        if chunk_index == len(request.page_span_chunks):
            self.reserve_chunk(request, k.size(2))
        if chunk_index >= len(request.page_span_chunks):
            raise RuntimeError("missing reserved page spans for KV chunk")
        self._block_store.write_layer(
            request.page_span_chunks[chunk_index],
            layer_idx=layer_idx,
            k=k,
            v=v,
        )
        request.layer_chunk_counts[layer_idx] = chunk_index + 1
        self._perf_add("manager_append_full_s", time.perf_counter() - t0)
        self._perf_add("manager_append_full_calls", 1.0)
        self._perf_add("manager_append_full_tokens", float(k.size(2)))

    def append_full_nhd(
        self,
        layer_idx: int,
        k_nhd: torch.Tensor,
        v_nhd: torch.Tensor,
        *,
        request: FlashInferPagedKVRequest | None = None,
    ) -> None:
        """Append one layer's request-local K/V chunk in NHD layout."""

        request = request or self._active_request
        if request is None:
            raise RuntimeError("append_full_nhd requires an active request")
        t0 = time.perf_counter()
        chunk_index = request.layer_chunk_counts.get(layer_idx, 0)
        if chunk_index == len(request.page_span_chunks):
            self.reserve_chunk(request, k_nhd.size(0))
        if chunk_index >= len(request.page_span_chunks):
            raise RuntimeError("missing reserved page spans for KV chunk")
        self._block_store.write_layer_nhd(
            request.page_span_chunks[chunk_index],
            layer_idx=layer_idx,
            k_nhd=k_nhd,
            v_nhd=v_nhd,
        )
        request.layer_chunk_counts[layer_idx] = chunk_index + 1
        self._perf_add("manager_append_full_s", time.perf_counter() - t0)
        self._perf_add("manager_append_full_calls", 1.0)
        self._perf_add("manager_append_full_nhd_calls", 1.0)
        self._perf_add("manager_append_full_tokens", float(k_nhd.size(0)))

    def get_full(
        self,
        layer_idx: int,
        *,
        request: FlashInferPagedKVRequest | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Paged FlashInfer cache no longer supports contiguous K/V reads."""

        raise RuntimeError(
            "FlashInfer paged KV manager no longer materializes contiguous KV; "
            "use layer_cache() and build_page_table() instead"
        )

    def ensure_prefix_node(self, token_ids: list[int]) -> int:
        """Ensure the token path exists in the radix tree and return node id."""

        return self._prefix_cache.ensure_token_path(token_ids)

    def lookup_match_len(self, token_ids: list[int], adapter_name: str) -> int:
        """Return adapter-specific cached prefix length for ``token_ids``."""

        key = FULL_KEY(adapter_name)
        match = self._prefix_cache.lookup(token_ids, keys=(key,))
        return min(len(token_ids), match.key_match_len.get(key, 0))

    def top_hot_prefixes(self, adapter_name: str, limit: int) -> list[RadixHotPrefix]:
        """Return hottest full-KV prefixes for one adapter."""

        return self._prefix_cache.top_hot_prefixes(FULL_KEY(adapter_name), limit)

    def score_prefix_candidate(
        self,
        adapter_name: str,
        token_ids: list[int],
        *,
        target_len: int,
        current_match_len: int = 0,
        limit: int = 16,
    ) -> float:
        """Score a background prefill candidate against hot radix prefixes."""

        return self._prefix_cache.score_prefix_candidate(
            FULL_KEY(adapter_name),
            token_ids,
            target_len=target_len,
            current_match_len=current_match_len,
            limit=limit,
        )

    def build_page_table(
        self,
        spans: list[KVBlockSpan],
        *,
        device: torch.device | None = None,
        reuse_buffer: bool = False,
    ) -> FlashInferPageTable:
        """Build FlashInfer page-table tensors for the given page spans."""

        table_device = device or self._device or torch.device("cpu")
        t0 = time.perf_counter()
        if reuse_buffer:
            if self._page_table_buffer is None or self._page_table_buffer.device != table_device:
                self._page_table_buffer = FlashInferPageTableBuffer(
                    page_size=self._page_size,
                    device=table_device,
                )
            table = self._page_table_buffer.update_from_spans(spans)
        else:
            table = build_flashinfer_page_table(spans, page_size=self._page_size, device=table_device)
        self._perf_add("manager_build_page_table_s", time.perf_counter() - t0)
        self._perf_add("manager_build_page_table_calls", 1.0)
        self._perf_add("manager_build_page_table_input_spans", float(len(spans)))
        self._perf_add("manager_build_page_table_pages", float(table.paged_kv_indices.numel()))
        return table

    def build_batch_page_table(
        self,
        batch_spans: list[list[KVBlockSpan]],
        *,
        device: torch.device | None = None,
        reuse_buffer: bool = False,
    ) -> FlashInferPageTable:
        """Build FlashInfer page-table tensors for multiple requests."""

        table_device = device or self._device or torch.device("cpu")
        t0 = time.perf_counter()
        if reuse_buffer:
            if self._batch_page_table_buffer is None or self._batch_page_table_buffer.device != table_device:
                self._batch_page_table_buffer = FlashInferBatchPageTableBuffer(
                    page_size=self._page_size,
                    device=table_device,
                )
            table = self._batch_page_table_buffer.update_from_batch_spans(batch_spans)
        else:
            table = build_flashinfer_batch_page_table(
                batch_spans,
                page_size=self._page_size,
                device=table_device,
            )
        self._perf_add("manager_build_batch_page_table_s", time.perf_counter() - t0)
        self._perf_add("manager_build_batch_page_table_calls", 1.0)
        self._perf_add("manager_build_batch_page_table_rows", float(len(batch_spans)))
        self._perf_add(
            "manager_build_batch_page_table_input_spans",
            float(sum(len(spans) for spans in batch_spans)),
        )
        self._perf_add("manager_build_batch_page_table_pages", float(table.paged_kv_indices.numel()))
        return table

    def page_table_buffer_signature(self) -> tuple[int, int, int]:
        """Return reusable decode page-table buffer pointers for graph keys."""

        buffer = self._page_table_buffer
        if buffer is None:
            return (0, 0, 0)
        return (
            int(buffer.paged_kv_indptr.data_ptr()),
            int(buffer.paged_kv_indices.data_ptr()),
            int(buffer.paged_kv_last_page_len.data_ptr()),
        )

    def batch_page_table_buffer_signature(self) -> tuple[int, int, int]:
        """Return reusable batched decode page-table buffer pointers for graph keys."""

        buffer = self._batch_page_table_buffer
        if buffer is None:
            return (0, 0, 0)
        return (
            int(buffer.paged_kv_indptr.data_ptr()),
            int(buffer.paged_kv_indices.data_ptr()),
            int(buffer.paged_kv_last_page_len.data_ptr()),
        )

    def _flashinfer_usable_prefix_len(
        self,
        spans: list[KVBlockSpan],
        *,
        desired_len: int,
    ) -> int:
        """Return the longest page-table-compatible prefix length."""

        if desired_len <= 0:
            return 0
        usable = 0
        for span in _coalesce_block_spans(slice_block_spans(spans, 0, desired_len)):
            if span.start != 0 or span.length != self._page_size:
                break
            usable += span.length
        return min(usable, desired_len)

    def invalidate_adapter(self, adapter: str) -> None:
        """Remove radix coverage and free all pages for ``adapter``."""

        self._prefix_cache.remove_key(FULL_KEY(adapter))
        self._block_store.drop_adapter(adapter)
        self._max_len_by_adapter.pop(adapter, None)
        self._latest_token_ids_by_adapter.pop(adapter, None)
        self._set_kvcache_bytes(adapter, 0)

    def replace_adapter_cache(self, from_adapter: str, to_adapter: str) -> None:
        """Replace one adapter's paged cache with a staged shadow cache."""

        if from_adapter == to_adapter:
            return
        self.invalidate_adapter(to_adapter)
        self._prefix_cache.rename_key(FULL_KEY(from_adapter), FULL_KEY(to_adapter))
        self._block_store.rename_adapter(from_adapter, to_adapter)
        if from_adapter in self._max_len_by_adapter:
            self._max_len_by_adapter[to_adapter] = self._max_len_by_adapter.pop(from_adapter)
        if from_adapter in self._latest_token_ids_by_adapter:
            self._latest_token_ids_by_adapter[to_adapter] = self._latest_token_ids_by_adapter.pop(from_adapter)

    def offload_adapter_cache(self, adapter: str) -> None:
        """Move cached pages for ``adapter`` to CPU."""

        self._block_store.offload_adapter_cache(adapter)

    def release_adapter_cache_for_train(self, adapter: str) -> None:
        """Free paged KV pages before train; page-level CPU offload is not implemented."""

        self.invalidate_adapter(adapter)

    def release_adapter_cache_for_generation(self, adapter: str) -> None:
        """Free paged KV pages before foreground generation admission retry."""

        self.invalidate_adapter(adapter)

    def release_non_current_cache_for_generation(
        self,
        adapter: str,
        request: FlashInferPagedKVRequest,
    ) -> int:
        """Free KV pages that are not needed by the active foreground request."""

        protected_spans = _coalesce_block_spans(list(request.prefix_spans))
        protected_page_ids = {span.block_id for span in protected_spans}
        protected_page_ids.update(span.block_id for span in request.new_spans)
        protected_len = span_total_length(protected_spans)
        freed_pages = 0

        for other_adapter in self._block_store.adapters():
            if other_adapter == adapter:
                continue
            before = self.allocated_pages
            self.invalidate_adapter(other_adapter)
            freed_pages += max(0, before - self.allocated_pages)

        key = FULL_KEY(adapter)
        self._prefix_cache.remove_key(key)
        freed_pages += self._block_store.drop_adapter_except(adapter, protected_page_ids)

        request.match_len = min(request.match_len, protected_len)
        request.prefill_from = min(request.prefill_from, protected_len)
        request.materialized_from = min(request.materialized_from, protected_len)
        request.current_kv_len = min(request.current_kv_len, protected_len)
        request.prefix_spans = protected_spans

        if protected_len > 0:
            protected_tokens = list(request.token_ids[:protected_len])
            self._prefix_cache.insert_coverage(
                protected_tokens,
                key=key,
                start=0,
                spans=protected_spans,
            )
            self._prefix_cache.lookup(protected_tokens, keys=(key,))
            self._max_len_by_adapter[adapter] = protected_len
            self._latest_token_ids_by_adapter[adapter] = protected_tokens
        else:
            self._max_len_by_adapter.pop(adapter, None)
            self._latest_token_ids_by_adapter.pop(adapter, None)

        self._set_kvcache_bytes(adapter, self._block_store.adapter_bytes(adapter))
        return freed_pages

    def ensure_adapter_cache_on_device(self, adapter: str, device: torch.device) -> None:
        """Move cached pages for ``adapter`` to ``device``."""

        self._block_store.ensure_adapter_cache_on_device(adapter, device)

    def offload_residual(self, adapter: str) -> None:
        """Compatibility alias used by adapter-offload code."""

        self.offload_adapter_cache(adapter)

    def ensure_residual_on_device(self, adapter: str, device: torch.device) -> None:
        """Compatibility alias used by adapter-offload code."""

        self.ensure_adapter_cache_on_device(adapter, device)

    def residual_length(self, adapter: str) -> int:
        """Return the longest cached/request-local prefix length for adapter."""

        if self._active_request is not None and self._active_request.adapter_name == adapter:
            return self._active_request.prefill_from + self._request_chunk_len(self._active_request)
        return self._max_len_by_adapter.get(adapter, 0)

    @property
    def base_length(self) -> int:
        """Return the maximum cached prefix length across adapters."""

        return max(self._max_len_by_adapter.values(), default=0)

    @staticmethod
    def _request_chunk_len(request: FlashInferPagedKVRequest) -> int:
        """Return request-local token length from reserved page chunks."""

        return sum(sum(span.length for span in spans) for spans in request.page_span_chunks)

    def _set_kvcache_bytes(self, adapter: str, num_bytes: int) -> None:
        if self._memory_coordinator is None:
            return
        set_bytes = getattr(self._memory_coordinator, "set_kvcache_bytes", None)
        if callable(set_bytes):
            set_bytes(adapter, num_bytes)
