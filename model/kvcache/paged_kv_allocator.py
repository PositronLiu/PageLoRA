"""Physical page allocation primitives for paged KV storage."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import torch


@dataclass
class PagedKVPage:
    """One physical KV page owned by a logical adapter."""

    page_id: int
    adapter: str
    length: int
    device: torch.device
    last_access: int = 0


class PagedKVAllocator:
    """Small free-list allocator for physical KV page ids."""

    def __init__(self, *, page_size: int, max_pages: int = 0) -> None:
        """Create an allocator for fixed-size KV pages.

        ``max_pages=0`` means no explicit allocator cap. The prototype stores
        page tensors lazily, so this avoids preallocating a large GPU buffer.
        """

        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if max_pages < 0:
            raise ValueError("max_pages must be non-negative")
        self.page_size = int(page_size)
        self.max_pages = int(max_pages)
        self._next_page_id = count(0)
        self._allocated: set[int] = set()
        self._free: list[int] = []

    def allocate_page(self) -> int:
        """Return one free physical page id or raise on allocator OOM."""

        if self._free:
            page_id = self._free.pop()
            self._allocated.add(page_id)
            return page_id
        if self.max_pages and len(self._allocated) >= self.max_pages:
            raise RuntimeError("paged KV cache is out of free pages")
        page_id = next(self._next_page_id)
        self._allocated.add(page_id)
        return page_id

    def free_page(self, page_id: int) -> None:
        """Return a physical page id to the free list if it is allocated."""

        if page_id not in self._allocated:
            return
        self._allocated.remove(page_id)
        self._free.append(page_id)

    @property
    def allocated_pages(self) -> int:
        """Return the number of page ids currently owned by live pages."""

        return len(self._allocated)

    @property
    def free_pages(self) -> int:
        """Return the number of reusable page ids in the free list."""

        return len(self._free)
