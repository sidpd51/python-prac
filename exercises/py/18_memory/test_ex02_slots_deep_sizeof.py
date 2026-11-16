import sys
import tracemalloc

import pytest

from study.loader import load

ex = load(__file__, "ex02_slots_deep_sizeof")


@pytest.mark.topic("py.memory.object_model")
def test_slots_record_has_no_dict():
    rec = ex.EmbeddingRecord("d1", [0.1, 0.2], 0.5)
    assert (rec.doc_id, rec.vector, rec.score) == ("d1", [0.1, 0.2], 0.5)
    assert ex.EmbeddingRecord("d2", None).score == 0.0
    assert not hasattr(rec, "__dict__")
    with pytest.raises(AttributeError):
        rec.extra = 1
    assert repr(rec) == "EmbeddingRecord(doc_id='d1', score=0.5)"


@pytest.mark.topic("py.memory.allocation", "py.memory.object_model")
def test_slots_use_less_memory_than_plain_instances():
    vec = (0.0,) * 4  # shared value so only the instances themselves are measured

    def measure(cls):
        tracemalloc.start()
        try:
            objs = [cls("doc", vec, 0.0) for _ in range(20_000)]
            current, _ = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert len(objs) == 20_000
        return current

    slotted, plain = measure(ex.EmbeddingRecord), measure(ex.PlainRecord)
    assert slotted < 0.8 * plain, (slotted, plain)


@pytest.mark.topic("py.memory.allocation")
def test_deep_sizeof_scalars_and_containers():
    assert ex.deep_sizeof(12345) == sys.getsizeof(12345)
    s = "x" * 500
    lst = [s]
    assert ex.deep_sizeof(lst) == sys.getsizeof(lst) + sys.getsizeof(s)
    assert ex.deep_sizeof(lst) > sys.getsizeof(lst) + 500

    key, val = "chunk_text", "y" * 300
    d = {key: val}
    assert ex.deep_sizeof(d) == sys.getsizeof(d) + sys.getsizeof(key) + sys.getsizeof(val)


@pytest.mark.topic("py.memory.references")
def test_deep_sizeof_counts_shared_objects_once():
    s = "z" * 1000
    shared = [s, s, s]
    assert ex.deep_sizeof(shared) == sys.getsizeof(shared) + sys.getsizeof(s)
    t = (s, [s], {s})
    inner_list, inner_set = t[1], t[2]
    expected = sum(map(sys.getsizeof, (t, s, inner_list, inner_set)))
    assert ex.deep_sizeof(t) == expected


@pytest.mark.topic("py.memory.references", "py.memory.cycles")
def test_deep_sizeof_handles_cycles_and_depth():
    a: list = []
    a.append(a)
    assert ex.deep_sizeof(a) == sys.getsizeof(a)

    lists: list[list] = [[]]
    for _ in range(100_000):
        nxt: list = []
        lists[-1].append(nxt)
        lists.append(nxt)
    expected = sum(map(sys.getsizeof, lists))
    assert ex.deep_sizeof(lists[0]) == expected  # and no RecursionError


@pytest.mark.topic("py.memory.object_model")
def test_deep_sizeof_instances():
    big = "v" * 2000
    plain = ex.PlainRecord("d1", big, 0.25)
    slotted = ex.EmbeddingRecord("d1", big, 0.25)
    assert ex.deep_sizeof(plain) > sys.getsizeof(plain) + 2000, "must walk the instance __dict__"
    assert ex.deep_sizeof(slotted) > sys.getsizeof(slotted) + 2000, "must walk __slots__"
    assert ex.deep_sizeof(slotted) < ex.deep_sizeof(plain)

    unset = ex.EmbeddingRecord.__new__(ex.EmbeddingRecord)  # slots never assigned
    assert ex.deep_sizeof(unset) == sys.getsizeof(unset)


@pytest.mark.topic("py.memory.object_model")
def test_deep_sizeof_skips_classes_and_functions():
    def fn():
        return 1

    holder = [fn, ex.PlainRecord, sys]
    assert ex.deep_sizeof(holder) == sys.getsizeof(holder)
