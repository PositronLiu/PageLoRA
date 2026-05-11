import pytest

from model.radix_cache import FULL_KEY, KVBlockSpan, RadixPrefixCache, span_total_length


def test_radix_lookup_partial_inside_edge():
    cache = RadixPrefixCache()
    cache.insert_base([1, 2, 3, 4], 0, [KVBlockSpan(10, 0, 4)])

    match = cache.lookup([1, 2, 3, 9])

    assert match.matched_len == 3
    assert match.base_match_len == 3
    assert not match.complete_node
    assert span_total_length(match.base_spans) == 3


def test_radix_split_keeps_shared_prefix_spans():
    cache = RadixPrefixCache()
    cache.insert_base([1, 2, 3, 4], 0, [KVBlockSpan(10, 0, 4)])
    cache.insert_base([1, 2, 9], 2, [KVBlockSpan(11, 0, 1)])

    left = cache.lookup([1, 2, 3, 4])
    right = cache.lookup([1, 2, 9])

    assert left.base_match_len == 4
    assert right.base_match_len == 3
    assert [span.block_id for span in right.base_spans] == [10, 11]


def test_radix_residual_is_adapter_specific():
    cache = RadixPrefixCache()
    cache.insert_base([5, 6, 7], 0, [KVBlockSpan(20, 0, 3)])
    cache.insert_residual([5, 6, 7], "agent_a", 0, [KVBlockSpan(30, 0, 3)])

    match_a = cache.lookup([5, 6, 7], adapter="agent_a")
    match_b = cache.lookup([5, 6, 7], adapter="agent_b")

    assert match_a.base_match_len == 3
    assert match_a.residual_match_len == 3
    assert match_b.base_match_len == 3
    assert match_b.residual_match_len == 0


def test_partial_key_coverage_does_not_split_edge():
    cache = RadixPrefixCache()
    general_key = FULL_KEY("general")
    coding_key = FULL_KEY("coding")

    cache.insert_coverage(
        [1, 2, 3, 4],
        key=general_key,
        start=0,
        spans=[KVBlockSpan(10, 0, 4)],
    )
    node_count_before = cache.node_count()

    cache.insert_coverage(
        [1, 2, 3, 4],
        key=coding_key,
        start=0,
        spans=[KVBlockSpan(20, 0, 2)],
    )

    assert cache.node_count() == node_count_before
    match = cache.lookup([1, 2, 3, 4], keys=(general_key, coding_key))
    assert match.matched_len == 4
    assert match.key_match_len[general_key] == 4
    assert match.key_match_len[coding_key] == 2


def test_partial_key_coverage_can_continue_inside_same_edge():
    cache = RadixPrefixCache()
    key = FULL_KEY("coding")
    cache.ensure_token_path([1, 2, 3, 4])

    cache.insert_coverage(
        [1, 2, 3, 4],
        key=key,
        start=0,
        spans=[KVBlockSpan(20, 0, 2)],
    )
    node_count_before = cache.node_count()

    cache.insert_coverage(
        [1, 2, 3, 4],
        key=key,
        start=2,
        spans=[KVBlockSpan(21, 0, 2)],
    )

    match = cache.lookup([1, 2, 3, 4], keys=(key,))
    assert cache.node_count() == node_count_before
    assert match.key_match_len[key] == 4
    assert [span.block_id for span in match.key_spans[key]] == [20, 21]


def test_non_contiguous_key_coverage_insert_fails():
    cache = RadixPrefixCache()
    key = FULL_KEY("coding")
    cache.ensure_token_path([1, 2, 3, 4])
    cache.insert_coverage(
        [1, 2, 3, 4],
        key=key,
        start=0,
        spans=[KVBlockSpan(20, 0, 2)],
    )

    with pytest.raises(ValueError, match="non-contiguous"):
        cache.insert_coverage(
            [1, 2, 3, 4],
            key=key,
            start=3,
            spans=[KVBlockSpan(21, 0, 1)],
        )


def test_priority_index_peeks_best_and_worst_entries():
    cache = RadixPrefixCache()
    key_a = FULL_KEY("a")
    key_b = FULL_KEY("b")
    node_a = cache.ensure_token_path([1, 2])
    node_b = cache.ensure_token_path([3, 4])

    cache.priority_index.upsert(
        cache_key=key_a,
        radix_node_id=node_a,
        priority=10.0,
        target_len=2,
        token_ids=[1, 2],
    )
    cache.priority_index.upsert(
        cache_key=key_b,
        radix_node_id=node_b,
        priority=1.0,
        target_len=2,
        token_ids=[3, 4],
    )

    assert cache.priority_index.peek_max().key == (key_a, node_a)
    assert cache.priority_index.peek_min().key == (key_b, node_b)


def test_priority_index_ignores_stale_heap_versions():
    cache = RadixPrefixCache()
    key = FULL_KEY("a")
    node_id = cache.ensure_token_path([1, 2])

    cache.priority_index.upsert(
        cache_key=key,
        radix_node_id=node_id,
        priority=1.0,
    )
    cache.priority_index.upsert(
        cache_key=key,
        radix_node_id=node_id,
        priority=9.0,
    )

    assert cache.priority_index.peek_max().priority == 9.0
    assert cache.priority_index.peek_min().priority == 9.0
    assert len(cache.priority_index) == 1


def test_priority_index_reuses_same_entry_for_partial_progress():
    cache = RadixPrefixCache()
    key = FULL_KEY("coding")
    node_id = cache.ensure_token_path([1, 2, 3, 4])
    cache.priority_index.upsert(
        cache_key=key,
        radix_node_id=node_id,
        priority=5.0,
        target_len=4,
        current_match_len=0,
        token_ids=[1, 2, 3, 4],
    )

    cache.insert_coverage(
        [1, 2, 3, 4],
        key=key,
        start=0,
        spans=[KVBlockSpan(30, 0, 2)],
    )
    entry = cache.priority_index.get((key, node_id))
    assert entry is not None
    assert entry.active
    assert entry.current_match_len == 2

    cache.priority_index.upsert(
        cache_key=key,
        radix_node_id=node_id,
        priority=3.0,
        target_len=4,
        current_match_len=2,
    )
    assert len(cache.priority_index) == 1

    cache.insert_coverage(
        [1, 2, 3, 4],
        key=key,
        start=2,
        spans=[KVBlockSpan(31, 0, 2)],
    )
    assert entry.current_match_len == 4
    assert not entry.active
    assert cache.priority_index.peek_max() is None


def test_priority_index_remove_key_and_clear_are_synchronized():
    cache = RadixPrefixCache()
    key = FULL_KEY("coding")
    node_id = cache.ensure_token_path([1, 2])
    cache.priority_index.upsert(
        cache_key=key,
        radix_node_id=node_id,
        priority=2.0,
    )

    cache.remove_key(key)
    assert cache.priority_index.get((key, node_id)) is None
    assert len(cache.priority_index) == 0

    cache.priority_index.upsert(
        cache_key=key,
        radix_node_id=node_id,
        priority=2.0,
    )
    cache.clear()
    assert len(cache.priority_index) == 0
