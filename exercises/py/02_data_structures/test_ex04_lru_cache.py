import time

import pytest

from study.loader import load

ex = load(__file__, "ex04_lru_cache")


@pytest.mark.topic("py.data_structures.ordereddict")
def test_put_get_and_eviction_order():
    c = ex.LRUCache(2)
    assert c.put("a", [0.1]) is None
    assert c.put("b", [0.2]) is None
    assert c.get("a") == [0.1]  # a becomes most recent
    assert c.put("c", [0.3]) == ("b", [0.2])  # b was least recent
    assert c.keys() == ["a", "c"]
    assert c.get("b") is None
    assert len(c) == 2


@pytest.mark.topic("py.data_structures.ordereddict")
def test_update_refreshes_recency_without_growing():
    c = ex.LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.put("a", 10) is None  # update, no eviction
    assert len(c) == 2
    assert c.keys() == ["b", "a"]
    assert c.put("c", 3) == ("b", 2)
    assert c.get("a") == 10


@pytest.mark.topic("py.data_structures.dict_lookup")
def test_contains_does_not_touch_recency_or_stats():
    c = ex.LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert "a" in c and "zzz" not in c
    assert c.stats() == (0, 0, 2)
    c.put("c", 3)  # "a" is still the LRU entry -> evicted
    assert "a" not in c
    assert c.keys() == ["b", "c"]


@pytest.mark.topic("py.data_structures.dict_lookup", "py.data_structures.namedtuple")
def test_stats_and_none_values():
    c = ex.LRUCache(3)
    c.put("q", None)  # e.g. "no embedding available" cached on purpose
    assert c.get("q", default="MISS") is None
    assert c.get("x", default="MISS") == "MISS"
    assert c.get("x") is None
    stats = c.stats()
    assert stats == ex.CacheStats(hits=1, misses=2, size=1)
    assert stats.hits == 1 and stats.misses == 2 and stats.size == 1


@pytest.mark.topic("py.data_structures.ordereddict")
def test_capacity_one_and_validation():
    c = ex.LRUCache(1)
    assert c.put(1, "x") is None
    assert c.put(2, "y") == (1, "x")
    assert c.keys() == [2]
    for bad in (0, -5):
        with pytest.raises(ValueError):
            ex.LRUCache(bad)


@pytest.mark.topic("py.data_structures.complexity", "py.data_structures.ordereddict")
def test_operations_are_constant_time():
    c = ex.LRUCache(20_000)
    start = time.perf_counter()
    for i in range(60_000):
        c.put(i, i)
        c.get(i - 10_000)
    elapsed = time.perf_counter() - start
    assert len(c) == 20_000
    assert elapsed < 1.0, "some operation is O(n) (list.remove / index / sorting?)"
