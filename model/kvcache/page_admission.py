"""Request-level admission helpers for paged KV cache generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar


class PagedKVAdmissionManager(Protocol):
    page_size: int
    available_pages: int

    def begin_request(self, prompt_ids: list[int], adapter_name: str): ...
    def abort_request(self, request) -> None: ...
    def can_fit_request(
        self,
        request,
        *,
        prefill_tokens: int,
        max_decode_tokens: int,
        reserve_pages: int,
    ) -> bool: ...
    def max_decode_tokens_that_fit(
        self,
        request,
        *,
        prefill_tokens: int,
        requested_decode_tokens: int,
        reserve_pages: int,
    ) -> int: ...


RequestT = TypeVar("RequestT")


@dataclass(frozen=True)
class PagedKVAdmissionResult:
    request: object
    new_ids: list[int]
    max_new_tokens: int
    cache_released: bool = False
    decode_budget_reduced: bool = False


def admit_paged_kv_request(
    cache_mgr: PagedKVAdmissionManager,
    *,
    prompt_ids: list[int],
    adapter_name: str,
    request_cache: RequestT,
    max_new_tokens: int,
    reserve_pages: int = 4,
    log_fn: Callable[[str], None] | None = None,
) -> PagedKVAdmissionResult:
    """Admit one paged-KV request before it starts reserving physical pages."""

    prefill_from = int(getattr(request_cache, "prefill_from"))
    new_ids = prompt_ids[prefill_from:]
    if cache_mgr.can_fit_request(
        request_cache,
        prefill_tokens=len(new_ids),
        max_decode_tokens=max_new_tokens,
        reserve_pages=reserve_pages,
    ):
        return PagedKVAdmissionResult(
            request=request_cache,
            new_ids=new_ids,
            max_new_tokens=max_new_tokens,
        )

    release_non_current_fn = getattr(cache_mgr, "release_non_current_cache_for_generation", None)
    if callable(release_non_current_fn):
        _log(
            log_fn,
            "FlashInfer page admission miss; releasing non-current KV cache "
            f"adapter={adapter_name} prompt_tokens={len(prompt_ids)} "
            f"requested_decode={max_new_tokens}",
        )
        freed_pages = int(release_non_current_fn(adapter_name, request_cache) or 0)
        prefill_from = int(getattr(request_cache, "prefill_from"))
        new_ids = prompt_ids[prefill_from:]
        if cache_mgr.can_fit_request(
            request_cache,
            prefill_tokens=len(new_ids),
            max_decode_tokens=max_new_tokens,
            reserve_pages=reserve_pages,
        ):
            return PagedKVAdmissionResult(
                request=request_cache,
                new_ids=new_ids,
                max_new_tokens=max_new_tokens,
                cache_released=freed_pages > 0,
            )

    cache_mgr.abort_request(request_cache)
    _log(
        log_fn,
        "FlashInfer page admission miss; releasing adapter cache "
        f"adapter={adapter_name} prompt_tokens={len(prompt_ids)} "
        f"requested_decode={max_new_tokens}",
    )
    release_fn = getattr(cache_mgr, "release_adapter_cache_for_generation", None)
    if not callable(release_fn):
        release_fn = getattr(cache_mgr, "release_adapter_cache_for_train", None)
    if callable(release_fn):
        release_fn(adapter_name)

    request_cache = cache_mgr.begin_request(prompt_ids, adapter_name)
    prefill_from = int(getattr(request_cache, "prefill_from"))
    new_ids = prompt_ids[prefill_from:]
    fit_decode_tokens = cache_mgr.max_decode_tokens_that_fit(
        request_cache,
        prefill_tokens=len(new_ids),
        requested_decode_tokens=max_new_tokens,
        reserve_pages=reserve_pages,
    )
    if fit_decode_tokens <= 0:
        cache_mgr.abort_request(request_cache)
        raise RuntimeError(
            "FlashInfer page admission rejected request: "
            f"prompt_tokens={len(prompt_ids)} page_size={cache_mgr.page_size} "
            f"available_pages={cache_mgr.available_pages}"
        )

    reduced = fit_decode_tokens < max_new_tokens
    if reduced:
        _log(
            log_fn,
            "FlashInfer page admission reduced decode budget "
            f"from={max_new_tokens} to={fit_decode_tokens} "
            f"prompt_tokens={len(prompt_ids)} available_pages={cache_mgr.available_pages}",
        )
    return PagedKVAdmissionResult(
        request=request_cache,
        new_ids=new_ids,
        max_new_tokens=fit_decode_tokens,
        cache_released=True,
        decode_budget_reduced=reduced,
    )


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)
