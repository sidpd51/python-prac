import copy

import pytest

from study.loader import load

ex = load(__file__, "ex01_rank_results")

RESULTS = [
    {"doc_id": "d3", "score": 0.50, "source": "wiki"},
    {"doc_id": "d1", "score": 0.90, "source": "blog"},
    {"doc_id": "d2", "score": 0.50, "source": "docs"},
    {"doc_id": "d4", "score": 0.10, "source": "wiki"},
]


@pytest.mark.topic("py.functional.lambda")
def test_sort_results():
    snapshot = copy.deepcopy(RESULTS)
    out = ex.sort_results(RESULTS)
    assert [r["doc_id"] for r in out] == ["d1", "d2", "d3", "d4"]
    assert snapshot == RESULTS, "input order must not change"
    assert ex.sort_results([]) == []
    assert ex.sort_results(iter(RESULTS))[0]["doc_id"] == "d1"


@pytest.mark.topic("py.functional.filter", "py.functional.map")
def test_top_ids():
    assert ex.top_ids(RESULTS, 0.5) == ["d1", "d2", "d3"]
    assert ex.top_ids(RESULTS, 0.95) == []
    assert ex.top_ids(RESULTS, 0.0) == ["d1", "d2", "d3", "d4"]
    assert isinstance(ex.top_ids(RESULTS, 0.5), list)


@pytest.mark.topic("py.functional.lambda", "py.functional.filter")
def test_score_at_least_predicate():
    pred = ex.score_at_least(0.5)
    assert callable(pred)
    assert pred({"score": 0.5}) is True
    assert pred({"score": 0.49}) is False
    assert [r["doc_id"] for r in filter(ex.score_at_least(0.4), RESULTS)] == ["d3", "d1", "d2"]
    p1, p2 = ex.score_at_least(0.1), ex.score_at_least(0.9)
    assert (p1({"score": 0.5}), p2({"score": 0.5})) == (True, False)  # independent thresholds


@pytest.mark.topic("py.functional.map", "py.functional.immutability")
def test_apply_boosts_returns_new_dicts():
    snapshot = copy.deepcopy(RESULTS)
    out = ex.apply_boosts(RESULTS, {"docs": 2.0, "wiki": 0.5})
    assert [r["doc_id"] for r in out] == ["d3", "d1", "d2", "d4"]  # order unchanged
    assert [r["score"] for r in out] == pytest.approx([0.25, 0.90, 1.0, 0.05])
    assert snapshot == RESULTS, "input dicts were mutated"
    assert all(a is not b for a, b in zip(out, RESULTS, strict=True))
    assert out[0]["source"] == "wiki"
    assert ex.apply_boosts([], {"x": 2}) == []
