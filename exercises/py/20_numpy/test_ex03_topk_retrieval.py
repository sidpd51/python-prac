import numpy as np
import pytest

from study.loader import load

ex = load(__file__, "ex03_topk_retrieval")

rng = np.random.default_rng(42)
SCORES = rng.random((5, 50))  # continuous -> no ties


@pytest.mark.topic("py.numpy.indexing", "py.numpy.aggregation")
def test_top_k_basic():
    scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    idx = ex.top_k(scores, 3)
    np.testing.assert_array_equal(idx, [1, 3, 4])
    assert np.issubdtype(idx.dtype, np.integer)
    np.testing.assert_array_equal(ex.top_k(scores, 10), [1, 3, 4, 2, 0])
    assert ex.top_k(scores, 0).shape == (0,)
    assert ex.top_k(np.array([]), 3).shape == (0,)


@pytest.mark.topic("py.numpy.indexing", "py.numpy.vectorization")
def test_top_k_matches_full_sort():
    row = SCORES[0]
    for k in (1, 5, 49, 50):
        np.testing.assert_array_equal(ex.top_k(row, k), np.argsort(-row)[:k])


@pytest.mark.topic("py.numpy.axis", "py.numpy.indexing")
def test_top_k_rows():
    idx, vals = ex.top_k_rows(SCORES, 4)
    assert idx.shape == vals.shape == (5, 4)
    np.testing.assert_array_equal(idx, np.argsort(-SCORES, axis=1)[:, :4])
    np.testing.assert_allclose(vals, -np.sort(-SCORES, axis=1)[:, :4])
    assert np.all(np.diff(vals, axis=1) < 0), "each row sorted descending"

    idx_all, _ = ex.top_k_rows(SCORES[:, :3], 10)
    assert idx_all.shape == (5, 3)
    np.testing.assert_array_equal(idx_all, np.argsort(-SCORES[:, :3], axis=1))


@pytest.mark.topic("py.numpy.boolean_indexing")
def test_filter_hits():
    ids = np.array([101, 102, 103, 104, 105])
    scores = np.array([0.55, 0.91, 0.20, 0.78, 0.60])
    kept_ids, kept_scores = ex.filter_hits(ids, scores, min_score=0.5)
    np.testing.assert_array_equal(kept_ids, [102, 104, 105, 101])
    np.testing.assert_allclose(kept_scores, [0.91, 0.78, 0.60, 0.55])

    kept_ids, _ = ex.filter_hits(ids, scores, min_score=0.5, exclude={104, 999})
    np.testing.assert_array_equal(kept_ids, [102, 105, 101])

    kept_ids, kept_scores = ex.filter_hits(ids, scores, min_score=0.95)
    assert kept_ids.shape == kept_scores.shape == (0,)
    assert scores[1] == 0.91, "input must not be modified"


@pytest.mark.topic("py.numpy.aggregation", "py.numpy.broadcasting")
def test_recall_and_mrr():
    retrieved = np.array([
        [3, 1, 2],   # gold 1 at position 1 -> 1/2
        [0, 4, 5],   # gold 0 at position 0 -> 1
        [7, 8, 9],   # gold 2 missing -> 0
        [6, 2, 5],   # gold 5 at position 2 -> 1/3
    ])
    relevant = np.array([1, 0, 2, 5])
    assert ex.recall_at_k(retrieved, relevant) == pytest.approx(0.75)
    assert ex.mrr(retrieved, relevant) == pytest.approx((0.5 + 1 + 0 + 1 / 3) / 4)
    assert isinstance(ex.mrr(retrieved, relevant), float)
    assert ex.mrr(retrieved[2:3], relevant[2:3]) == 0.0  # argmax of all-False row is 0!
