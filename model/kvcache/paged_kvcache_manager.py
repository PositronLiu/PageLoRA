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
    FlashInferPageTable,
    FlashInferPageTableBuffer,
    _coalesce_block_spans,
    build_flashinfer_page_table,
)
from model.kvcache.paged_kv_store import PagedKVBlockStore
from model.kvcache.paged_kvcache_request import FlashInferPagedKVRequest
from model.kvcache.radix_cache import RadixPrefixCache
from model.kvcache.radix_priority import RadixPriorityIndex
from model.kvcache.radix_storage import slice_block_spans
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
        self._max_len_by_adapter: dict[str, int] = {}
        self._latest_token_ids_by_adapter: dict[str, list[int]] = {}
        self._perf_counters: dict[str, float] = {}
        self._page_table_buffer: FlashInferPageTableBuffer | None = None

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
    def free_pages(self) -> int:
        """Return reusable physical page count."""

        return self._block_store.free_pages

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

    def begin_request(
        self,
        token_ids: list[int],
        adapter_name: str,
    ) -> FlashInferPagedKVRequest:
        """Open a paged full-KV cache request for one foreground generation."""

        if self._active_request is not None:
            raise RuntimeError("KV cache request already active")

        total_t0 = time.perf_counter()
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
        self._active_request = request
        self._perf_add("manager_begin_request_s", time.perf_counter() - total_t0)
        self._perf_add("manager_begin_request_calls", 1.0)
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

        if self._active_request is not None:
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

        if self._active_request is not request:  # protection
            raise RuntimeError("cannot commit inactive KV cache request")

        total_t0 = time.perf_counter()
        final_len = len(final_token_ids)
        start = min(request.match_len, final_len)
        skip = max(0, start - request.materialized_from)
        # append_full writes request-local chunks into pages immediately.
        # Commit only inserts the non-prefix suffix into radix metadata.
        coalesce_t0 = time.perf_counter()
        commit_spans = [] if skip else _coalesce_block_spans(request.new_spans)
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
        elif request.new_spans:
            # If a request had to replay already-cached tokens to keep the
            # FlashInfer page table aligned, its request-local pages overlap
            # existing radix coverage and should not be committed as fragments.
            self._block_store.drop_spans(request.new_spans)

        cached_len = final_len if commit_spans else start
        self._max_len_by_adapter[request.adapter_name] = max(
            self._max_len_by_adapter.get(request.adapter_name, 0),
            cached_len,
        )
        self._latest_token_ids_by_adapter[request.adapter_name] = list(final_token_ids)
        self._active_request = None
        self._perf_add("manager_commit_request_s", time.perf_counter() - total_t0)
        self._perf_add("manager_commit_request_calls", 1.0)
        self._perf_add("manager_commit_final_len", float(final_len))
        self._perf_add("manager_commit_start", float(start))
        self._perf_add("manager_commit_spans", float(len(commit_spans)))

        if self._memory_coordinator is not None:
            self._memory_coordinator.notify_kvcache_access(request.adapter_name)
            self._memory_coordinator.maybe_offload()

    def abort_request(self, request: FlashInferPagedKVRequest) -> None:
        """Close an active request without committing request-local chunks."""

        if self._active_request is request:
            self._block_store.drop_spans(request.new_spans)
            request.new_spans = []
            self._active_request = None

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

    def offload_adapter_cache(self, adapter: str) -> None:
        """Move cached pages for ``adapter`` to CPU."""

        self._block_store.offload_adapter_cache(adapter)

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
