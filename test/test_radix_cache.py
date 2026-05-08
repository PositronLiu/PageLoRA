from model.radix_cache import KVBlockSpan, RadixPrefixCache, span_total_length


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
