import copy
import json

import pytest

from study.loader import load

ex = load(__file__, "ex03_aliasing_bugs")


@pytest.mark.topic("py.fundamentals.mutability", "py.fundamentals.nested_structures")
def test_init_scores_rows_are_independent():
    grid = ex.init_scores(3, 2)
    assert grid == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    grid[0][1] = 0.9
    assert grid == [[0.0, 0.9], [0.0, 0.0], [0.0, 0.0]]
    assert len({id(row) for row in grid}) == 3
    assert ex.init_scores(0, 5) == []


@pytest.mark.topic("py.fundamentals.mutability", "py.fundamentals.nested_structures")
def test_with_overrides_does_not_touch_base():
    base = {"llm": {"model": "small", "temperature": 0.7}, "retrieval": {"k": 4}}
    snapshot = copy.deepcopy(base)
    cfg = ex.with_overrides(base, llm__temperature=0.2, retrieval__k=8, cache__ttl=60)
    assert cfg == {
        "llm": {"model": "small", "temperature": 0.2},
        "retrieval": {"k": 8},
        "cache": {"ttl": 60},
    }
    assert base == snapshot, "base (or one of its nested dicts) was mutated"
    cfg["llm"]["model"] = "big"
    assert base["llm"]["model"] == "small", "result shares nested dicts with base"


@pytest.mark.topic("py.fundamentals.mutability")
def test_with_overrides_no_overrides_is_still_a_copy():
    base = {"llm": {"model": "small"}}
    cfg = ex.with_overrides(base)
    assert cfg == base
    assert cfg is not base and cfg["llm"] is not base["llm"]


@pytest.mark.topic("py.fundamentals.identity_equality")
def test_find_doc_uses_equality_not_identity():
    docs = json.loads('[{"id": "doc-1000", "t": "a"}, {"id": "doc-2000", "t": "b"}]')
    runtime_id = "doc-" + str(2000)  # equal, but (almost certainly) not the same object
    found = ex.find_doc(docs, runtime_id)
    assert found is docs[1]
    assert ex.find_doc(docs, "".join(["doc-", "1000"])) is docs[0]
    assert ex.find_doc(docs, "doc-3000") is None
    assert ex.find_doc([], "x") is None


@pytest.mark.topic("py.fundamentals.mutability", "py.fundamentals.operators")
def test_add_tag_never_mutates_caller_list():
    tags = ["rag"]
    out = ex.add_tag(tags, "eval")
    assert out == ["rag", "eval"]
    assert tags == ["rag"]
    same = ex.add_tag(tags, "rag")
    assert same == ["rag"]
    assert same is not tags


@pytest.mark.topic("py.fundamentals.none", "py.fundamentals.identity_equality")
def test_is_missing_cannot_be_fooled_by_eq():
    class EqualsEverything:
        def __eq__(self, other):
            return True

        __hash__ = object.__hash__

    assert ex.is_missing(None) is True
    assert ex.is_missing(EqualsEverything()) is False
    for falsy in (0, "", [], {}, False):
        assert ex.is_missing(falsy) is False
