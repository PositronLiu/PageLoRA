from __future__ import annotations

from dataclasses import dataclass

import pytest

from model.kvcache.page_admission import admit_paged_kv_request


@dataclass
class _Request:
    adapter_name: str
    prefill_from: int
    match_len: int = 0


class _FakeCacheManager:
    page_size = 32
    available_pages = 100

    def __init__(self, *, first_fit: bool, second_decode_fit: int) -> None:
        self.first_fit = first_fit
        self.second_decode_fit = second_decode_fit
        self.begin_calls = 0
        self.abort_calls = 0
        self.release_calls: list[str] = []
        self.can_fit_calls: list[tuple[int, int]] = []

    def begin_request(self, prompt_ids: list[int], adapter_name: str) -> _Request:
        self.begin_calls += 1
        if self.begin_calls == 1:
            return _Request(adapter_name=adapter_name, prefill_from=8, match_len=8)
        return _Request(adapter_name=adapter_name, prefill_from=0, match_len=0)

    def abort_request(self, request: _Request) -> None:
        self.abort_calls += 1

    def release_adapter_cache_for_generation(self, adapter_name: str) -> None:
        self.release_calls.append(adapter_name)

    def can_fit_request(
        self,
        request: _Request,
        *,
        prefill_tokens: int,
        max_decode_tokens: int,
        reserve_pages: int,
    ) -> bool:
        self.can_fit_calls.append((prefill_tokens, max_decode_tokens))
        return self.first_fit if len(self.can_fit_calls) == 1 else True

    def max_decode_tokens_that_fit(
        self,
        request: _Request,
        *,
        prefill_tokens: int,
        requested_decode_tokens: int,
        reserve_pages: int,
    ) -> int:
        return self.second_decode_fit


def test_page_admission_flushes_cached_request_that_would_not_fit() -> None:
    mgr = _FakeCacheManager(first_fit=False, second_decode_fit=16)
    request = mgr.begin_request(list(range(10)), "general")

    result = admit_paged_kv_request(
        mgr,
        prompt_ids=list(range(10)),
        adapter_name="general",
        request_cache=request,
        max_new_tokens=16,
    )

    assert mgr.abort_calls == 1
    assert mgr.release_calls == ["general"]
    assert mgr.begin_calls == 2
    assert result.request.prefill_from == 0
    assert result.new_ids == list(range(10))
    assert result.max_new_tokens == 16
    assert result.cache_released


def test_page_admission_reduces_decode_budget_after_flush() -> None:
    mgr = _FakeCacheManager(first_fit=False, second_decode_fit=5)
    request = mgr.begin_request(list(range(10)), "general")

    result = admit_paged_kv_request(
        mgr,
        prompt_ids=list(range(10)),
        adapter_name="general",
        request_cache=request,
        max_new_tokens=16,
    )

    assert result.max_new_tokens == 5
    assert result.decode_budget_reduced


def test_page_admission_rejects_prompt_that_cannot_fit() -> None:
    mgr = _FakeCacheManager(first_fit=False, second_decode_fit=0)
    request = mgr.begin_request(list(range(10)), "general")

    with pytest.raises(RuntimeError, match="page admission rejected"):
        admit_paged_kv_request(
            mgr,
            prompt_ids=list(range(10)),
            adapter_name="general",
            request_cache=request,
            max_new_tokens=16,
        )
