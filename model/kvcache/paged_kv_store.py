"""Physical paged-KV tensor storage primitives.

This module owns lazy physical page allocation and the FlashInfer-compatible
dense paged KV tensor. Radix/request lifecycle code lives in
``model.kvcache.paged_kvcache_manager``.
"""

from __future__ import annotations

import time

import torch

from model.kvcache.paged_kv_allocator import PagedKVAllocator, PagedKVPage
from model.kvcache.paged_kv_append import write_layer_flashinfer, write_layer_manual
from model.kvcache.radix_types import KVBlockSpan

PageSpan = KVBlockSpan
# Page spans reuse KVBlockSpan so radix metadata can stay backend-neutral.


class PagedKVBlockStore:
    """Own FlashInfer paged K/V tensors while radix nodes keep page spans."""

    def __init__(
        self,
        *,
        page_size: int,
        max_pages: int = 0,
        num_layers: int | None = None,
        kv_append_mode: str = "manual",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create a lazy page store backed by per-page tensors."""

        if kv_append_mode not in {"manual", "flashinfer"}:
            raise ValueError("kv_append_mode must be 'manual' or 'flashinfer'")
        self._page_size = int(page_size)
        self._allocator = PagedKVAllocator(page_size=page_size, max_pages=max_pages)
        self._pages: dict[int, PagedKVPage] = {}
        self._adapter_pages: dict[str, set[int]] = {}
        self._clock = 0
        self._num_layers = num_layers
        self._device = device
        self._dtype = dtype
        self._max_pages = int(max_pages)
        self._kv_append_mode = kv_append_mode
        self.paged_kv_cache: torch.Tensor | None = None
        self._dense_capacity = 0
        self._perf_counters: dict[str, float] = {}

    @property
    def allocated_pages(self) -> int:
        """Return live physical page count."""

        return self._allocator.allocated_pages

    @property
    def free_pages(self) -> int:
        """Return reusable physical page count."""

        return self._allocator.free_pages

    @property
    def can_use_dense_cache(self) -> bool:
        """Return whether a FlashInfer-compatible dense cache is available."""

        return self.paged_kv_cache is not None

    def reset_perf_counters(self) -> None:
        """Clear page-store performance counters."""

        self._perf_counters.clear()

    def perf_snapshot(self) -> dict[str, float]:
        """Return accumulated page-store performance counters."""

        snapshot = dict(self._perf_counters)
        snapshot["store_allocated_pages"] = float(self.allocated_pages)
        snapshot["store_free_pages"] = float(self.free_pages)
        snapshot["store_dense_capacity_pages"] = float(self._dense_capacity)
        return snapshot

    def _perf_add(self, name: str, value: float) -> None:
        self._perf_counters[name] = self._perf_counters.get(name, 0.0) + float(value)

    def reserve_spans(
        self,
        *,
        adapter: str,
        request_spans: list[KVBlockSpan],
        token_count: int,
    ) -> list[KVBlockSpan]:
        """Reserve page spans for the next request-local token chunk.

        ``request_spans`` is updated in place to represent the complete
        request suffix. Returned spans represent only the newly reserved chunk
        and may start inside the previous partial page.
        """

        remaining = int(token_count)
        if remaining < 0:
            raise ValueError("token_count must be non-negative")
        if remaining == 0:
            return []

        t0 = time.perf_counter()
        requested_tokens = remaining
        original_len = len(request_spans)
        original_last = request_spans[-1] if request_spans else None
        chunk_spans: list[KVBlockSpan] = []
        created_page_ids: list[int] = []
        reused_partial_tokens = 0
        try:
            if request_spans and request_spans[-1].length < self._page_size:
                last = request_spans[-1]
                take = min(self._page_size - last.length, remaining)
                updated = KVBlockSpan(last.block_id, last.start, last.length + take)
                request_spans[-1] = updated
                page = self._pages[last.block_id]
                page.length = updated.length
                chunk_spans.append(KVBlockSpan(last.block_id, last.length, take))
                remaining -= take
                reused_partial_tokens += take

            while remaining > 0:
                length = min(self._page_size, remaining)
                page_id = self._allocator.allocate_page()
                created_page_ids.append(page_id)
                self._pages[page_id] = PagedKVPage(
                    page_id=page_id,
                    adapter=adapter,
                    length=length,
                    device=self._device or torch.device("cpu"),
                )
                self._adapter_pages.setdefault(adapter, set()).add(page_id)
                span = KVBlockSpan(block_id=page_id, start=0, length=length)
                request_spans.append(span)
                chunk_spans.append(span)
                remaining -= length
        except Exception:
            for page_id in created_page_ids:
                self._pages.pop(page_id, None)
                adapter_pages = self._adapter_pages.get(adapter)
                if adapter_pages is not None:
                    adapter_pages.discard(page_id)
                    if not adapter_pages:
                        self._adapter_pages.pop(adapter, None)
                self._allocator.free_page(page_id)
            del request_spans[original_len:]
            if original_last is not None and request_spans:
                request_spans[-1] = original_last
                page = self._pages.get(original_last.block_id)
                if page is not None:
                    page.length = original_last.length
            raise

        self._perf_add("store_reserve_spans_s", time.perf_counter() - t0)
        self._perf_add("store_reserve_spans_calls", 1.0)
        self._perf_add("store_reserved_tokens", float(requested_tokens))
        self._perf_add("store_reserved_chunk_spans", float(len(chunk_spans)))
        self._perf_add("store_created_pages", float(len(created_page_ids)))
        self._perf_add("store_reused_partial_tokens", float(reused_partial_tokens))
        return chunk_spans

    def write_layer(
        self,
        spans: list[KVBlockSpan],
        *,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write one layer's K/V chunk into previously reserved page spans."""

        t0 = time.perf_counter()
        total_len = sum(span.length for span in spans)
        if total_len != k.size(2) or total_len != v.size(2):
            raise ValueError("reserved span length does not match KV chunk length")
        required_pages = max((span.block_id for span in spans), default=-1) + 1
        self._ensure_dense_cache(k, layer_idx=layer_idx, required_pages=required_pages)

        if self._kv_append_mode == "flashinfer":
            write_layer_flashinfer(
                spans,
                paged_kv_cache=self.paged_kv_cache,
                page_size=self._page_size,
                layer_idx=layer_idx,
                k=k,
                v=v,
                touch_page=self._touch_page,
                perf_add=self._perf_add,
            )
        else:
            write_layer_manual(
                spans,
                paged_kv_cache=self.paged_kv_cache,
                layer_idx=layer_idx,
                k=k,
                v=v,
                touch_page=self._touch_page,
                perf_add=self._perf_add,
            )

        self._perf_add("store_write_layer_s", time.perf_counter() - t0)
        self._perf_add("store_write_layer_calls", 1.0)
        self._perf_add("store_written_tokens", float(total_len))
        self._perf_add("store_written_spans", float(len(spans)))

    def _touch_page(self, span: KVBlockSpan, device: torch.device) -> None:
        page = self._pages.get(span.block_id)
        if page is None:
            raise KeyError(f"unknown KV page id: {span.block_id}")
        if span.start < 0 or span.length < 0 or span.start + span.length > page.length:
            raise ValueError("page span is out of range")

        page_t0 = time.perf_counter()
        page.device = device
        self._clock += 1
        page.last_access = self._clock
        self._perf_add("store_page_metadata_s", time.perf_counter() - page_t0)

    def layer_cache(self, layer_idx: int) -> torch.Tensor | None:
        """Return one layer's dense paged cache for FlashInfer, if available."""

        if self.paged_kv_cache is None:
            return None
        return self.paged_kv_cache[layer_idx]

    def put(
        self,
        *,
        adapter: str,
        k_chunks: dict[int, list[torch.Tensor]],
        v_chunks: dict[int, list[torch.Tensor]],
        start_offset: int = 0,
    ) -> list[KVBlockSpan]:
        """Persist request-local KV chunks into fixed-size pages.

        ``start_offset`` skips tokens already covered by a radix prefix hit.
        Returned spans reference the physical pages and are inserted into the
        radix tree by the caller.
        """

        if not k_chunks:
            return []

        # Merge per-layer request chunks first so every layer is sliced into
        # identical token page boundaries.
        layer_tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        total_len: int | None = None
        for layer_idx, k_list in k_chunks.items():
            v_list = v_chunks.get(layer_idx)
            if not v_list:
                raise ValueError(f"missing V chunks for layer {layer_idx}")
            k = torch.cat(k_list, dim=2) if len(k_list) > 1 else k_list[0]
            v = torch.cat(v_list, dim=2) if len(v_list) > 1 else v_list[0]
            if k.size(2) != v.size(2):
                raise ValueError("K and V chunk lengths differ")
            if total_len is None:
                total_len = k.size(2)
            elif total_len != k.size(2):
                raise ValueError("all KV layers must have the same token length")
            layer_tensors[layer_idx] = (k, v)

        if total_len is None:
            return []
        if start_offset < 0 or start_offset > total_len:
            raise ValueError("paged KV start offset out of range")
        if start_offset == total_len:
            return []

        spans: list[KVBlockSpan] = []
        try:
            spans = self.reserve_spans(
                adapter=adapter,
                request_spans=[],
                token_count=total_len - start_offset,
            )
            for layer_idx, (k, v) in layer_tensors.items():
                self.write_layer(
                    spans,
                    layer_idx=layer_idx,
                    k=k[:, :, start_offset:, :],
                    v=v[:, :, start_offset:, :],
                )
        except Exception:
            # Page allocation is transactional at the block-store level: if
            # any page write fails, no partially-created pages remain visible.
            self.drop_spans(spans)
            raise
        return spans

    def materialize(
        self,
        spans: list[KVBlockSpan],
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Paged FlashInfer storage no longer materializes contiguous K/V."""

        raise RuntimeError(
            "FlashInfer paged KV store no longer materializes contiguous KV; "
            "use layer_cache() and page tables instead"
        )

    def drop_adapter(self, adapter: str) -> None:
        """Free every page owned by ``adapter``."""

        for page_id in list(self._adapter_pages.get(adapter, ())):
            self._pages.pop(page_id, None)
            self._allocator.free_page(page_id)
        self._adapter_pages.pop(adapter, None)

    def drop_spans(self, spans: list[KVBlockSpan]) -> None:
        """Free the physical pages referenced by ``spans``.

        Used to roll back radix insertion failures after page writes succeed.
        """

        for page_id in {span.block_id for span in spans}:
            page = self._pages.pop(page_id, None)
            if page is None:
                continue
            adapter_pages = self._adapter_pages.get(page.adapter)
            if adapter_pages is not None:
                adapter_pages.discard(page_id)
                if not adapter_pages:
                    self._adapter_pages.pop(page.adapter, None)
            self._allocator.free_page(page_id)

    def offload_adapter_cache(self, adapter: str) -> None:
        """Mark adapter pages as CPU-resident.

        The FlashInfer path stores physical K/V in one dense paged tensor, so
        per-adapter page offload needs a separate page-level copy design.
        """

        for page_id in self._adapter_pages.get(adapter, ()):
            page = self._pages.get(page_id)
            if page is not None:
                page.device = torch.device("cpu")

    def ensure_adapter_cache_on_device(self, adapter: str, device: torch.device) -> None:
        """Mark adapter pages as resident on ``device``."""

        for page_id in self._adapter_pages.get(adapter, ()):
            page = self._pages.get(page_id)
            if page is not None:
                page.device = device

    def _ensure_dense_cache(
        self,
        k: torch.Tensor,
        *,
        layer_idx: int,
        required_pages: int,
    ) -> None:
        if required_pages <= self._dense_capacity:
            return
        if self._num_layers is None:
            return
        if layer_idx < 0 or layer_idx >= self._num_layers:
            raise ValueError("layer_idx out of range")

        t0 = time.perf_counter()
        _, num_kv_heads, _, head_dim = k.shape
        dtype = self._dtype or k.dtype
        device = self._device or k.device
        if self._max_pages > 0:
            if required_pages > self._max_pages:
                raise RuntimeError("paged KV dense cache capacity exceeded")
            new_capacity = self._max_pages
        else:
            new_capacity = max(16, self._dense_capacity * 2, required_pages)
        new_cache = torch.empty(
            (
                self._num_layers,
                new_capacity,
                2,
                self._page_size,
                num_kv_heads,
                head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        if self.paged_kv_cache is not None:
            new_cache[:, : self._dense_capacity] = self.paged_kv_cache
        old_capacity = self._dense_capacity
        self.paged_kv_cache = new_cache
        self._dense_capacity = new_capacity
        self._perf_add("store_dense_realloc_s", time.perf_counter() - t0)
        self._perf_add("store_dense_realloc_count", 1.0)
        self._perf_add("store_dense_realloc_old_pages", float(old_capacity))
        self._perf_add("store_dense_realloc_new_pages", float(new_capacity))
