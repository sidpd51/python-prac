from collections.abc import Mapping

import pytest

from study.loader import load

ex = load(__file__, "ex05_optimization")


class Tok:
    """Hashable token that counts equality comparisons (shared counter)."""

    eq_calls = 0

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        Tok.eq_calls += 1
        return isinstance(other, Tok) and self.value == other.value

    def __repr__(self):
        return f"Tok({self.value!r})"


class CountingDoc(Mapping):
    """Read-only dict that counts field reads."""

    reads = 0

    def __init__(self, **fields):
        self._fields = fields

    def __getitem__(self, key):
        CountingDoc.reads += 1
        return self._fields[key]

    def __iter__(self):
        return iter(self._fields)

    def __len__(self):
        return len(self._fields)


@pytest.mark.topic("py.internals.optimization")
def test_dedupe_preserves_first_occurrence():
    assert ex.dedupe_preserve_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert ex.dedupe_preserve_order([]) == []
    docs = [{"id": 1, "v": "x"}, {"id": 2, "v": "y"}, {"id": 1, "v": "z"}]
    assert ex.dedupe_preserve_order(docs, key=lambda d: d["id"]) == docs[:2]


@pytest.mark.topic("py.internals.optimization")
def test_dedupe_is_linear():
    n = 3000
    items = [Tok(i % (n // 2)) for i in range(n)]  # every value appears twice
    Tok.eq_calls = 0
    out = ex.dedupe_preserve_order(items)
    assert [t.value for t in out] == list(range(n // 2))
    assert out[0] is items[0], "must keep the FIRST occurrence"
    assert Tok.eq_calls <= 3 * n, f"{Tok.eq_calls} comparisons - is this O(n^2)?"


def slow_join(queries, docs):
    out = []
    for q in queries:
        for d in docs:
            if d["id"] == q["doc_id"]:
                out.append((q["qid"], d["title"]))
    return out


@pytest.mark.topic("py.internals.optimization")
def test_join_results_same_output_as_original():
    docs = [
        {"id": "d1", "title": "Intro"},
        {"id": "d2", "title": "Setup"},
        {"id": "d1", "title": "Intro (v2)"},  # duplicate id -> both match, docs order
    ]
    queries = [
        {"qid": "q1", "doc_id": "d2"},
        {"qid": "q2", "doc_id": "missing"},
        {"qid": "q3", "doc_id": "d1"},
        {"qid": "q4", "doc_id": "d2"},
    ]
    expected = slow_join(queries, docs)
    assert expected == [("q1", "Setup"), ("q3", "Intro"), ("q3", "Intro (v2)"), ("q4", "Setup")]
    assert ex.join_results(queries, docs) == expected
    assert ex.join_results([], docs) == []
    assert ex.join_results(queries, []) == []


@pytest.mark.topic("py.internals.optimization", "py.internals.cpython")
def test_join_results_is_linear():
    docs = [CountingDoc(id=f"d{i}", title=f"T{i}") for i in range(400)]
    queries = [{"qid": f"q{i}", "doc_id": f"d{(i * 7) % 500}"} for i in range(400)]
    expected = slow_join(queries, docs)
    CountingDoc.reads = 0
    assert ex.join_results(queries, docs) == expected
    assert CountingDoc.reads <= 3 * len(docs), f"{CountingDoc.reads} doc reads - O(q*d)?"


@pytest.mark.topic("py.internals.optimization")
def test_top_k_frequent():
    tokens = ["the", "cat", "and", "the", "dog", "and", "the", "bird", "cat"]
    assert ex.top_k_frequent(tokens, 2) == [("the", 3), ("cat", 2)]  # cat before 'and' (tie)
    assert ex.top_k_frequent(tokens, 3) == [("the", 3), ("cat", 2), ("and", 2)]
    assert ex.top_k_frequent(tokens, 0) == []
    assert ex.top_k_frequent([], 3) == []
    assert len(ex.top_k_frequent(tokens, 100)) == 5
