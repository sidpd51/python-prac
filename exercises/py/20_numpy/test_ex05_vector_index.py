import numpy as np
import pytest

from study.loader import load

ex = load(__file__, "ex05_vector_index")

rng = np.random.default_rng(2024)
DIM = 16
CORPUS = rng.standard_normal((200, DIM)).astype(np.float32)
IDS = [f"chunk-{i}" for i in range(200)]


def brute_cosine(query, corpus, k):
    c = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    s = c @ (query / np.linalg.norm(query))
    order = np.argsort(-s)[:k]
    return [IDS[i] for i in order], s[order]


@pytest.fixture
def index():
    idx = ex.VectorIndex(DIM)
    idx.add(IDS, CORPUS)
    return idx


@pytest.mark.topic("py.numpy.matmul", "py.numpy.linalg")
def test_cosine_search_matches_brute_force(index):
    query = rng.standard_normal(DIM).astype(np.float32)
    hits = index.search(query, k=5)
    expected_ids, expected_scores = brute_cosine(query, CORPUS, 5)
    assert [h[0] for h in hits] == expected_ids
    np.testing.assert_allclose([h[1] for h in hits], expected_scores, rtol=1e-4, atol=1e-5)
    assert all(type(score) is float for _, score in hits)
    assert len(index) == 200


@pytest.mark.topic("py.numpy.linalg")
def test_cosine_is_scale_invariant_and_self_match(index):
    best_id, best_score = index.search(CORPUS[17] * 42.0, k=1)[0]
    assert best_id == "chunk-17"
    assert best_score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.topic("py.numpy.matmul")
def test_dot_and_l2_metrics():
    vecs = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
    ids = ["a", "b", "c", "d"]
    dot = ex.VectorIndex(2, metric="dot")
    dot.add(ids, vecs)
    assert [i for i, _ in dot.search([1.0, 0.0], k=2)] == ["b", "a"]  # magnitude matters
    l2 = ex.VectorIndex(2, metric="l2")
    l2.add(ids, vecs)
    hits = l2.search([1.0, 0.5], k=3)
    assert [i for i, _ in hits] == ["a", "c", "b"]  # nearest first
    np.testing.assert_allclose([s for _, s in hits], [0.5, np.hypot(1, 1.5), np.hypot(2, 0.5)],
                               rtol=1e-5)
    with pytest.raises(ValueError):
        ex.VectorIndex(2, metric="manhattan")


@pytest.mark.topic("py.numpy.creation")
def test_add_validation_is_atomic():
    idx = ex.VectorIndex(4)
    idx.add(["x"], np.ones((1, 4)))
    with pytest.raises(ValueError):
        idx.add(["y", "z"], np.ones((2, 3)))  # wrong dim
    with pytest.raises(ValueError):
        idx.add(["y", "z"], np.ones((3, 4)))  # ids/vectors length mismatch
    with pytest.raises(ValueError):
        idx.add(["y", "y"], np.ones((2, 4)))  # duplicate in batch
    with pytest.raises(ValueError):
        idx.add(["z", "x"], np.ones((2, 4)))  # already exists
    assert len(idx) == 1
    assert [i for i, _ in idx.search(np.ones(4), k=10)] == ["x"]
    with pytest.raises(ValueError):
        idx.search(np.ones(5))


@pytest.mark.topic("py.numpy.creation", "py.numpy.vs_lists")
def test_many_small_adds_and_zero_vectors():
    idx = ex.VectorIndex(DIM)
    for i in range(200):  # one row at a time: must stay correct across buffer growth
        idx.add([IDS[i]], CORPUS[i : i + 1])
    query = CORPUS[3] + 0.01
    assert [h[0] for h in idx.search(query, k=5)] == brute_cosine(query, CORPUS, 5)[0]

    idx.add(["empty"], np.zeros((1, DIM)))
    hits = dict(idx.search(np.zeros(DIM), k=3))
    assert all(np.isfinite(s) for s in hits.values()), "zero vectors must not produce NaN"


@pytest.mark.topic("py.numpy.boolean_indexing", "py.numpy.indexing")
def test_allowed_ids_prefilter_and_remove(index):
    query = CORPUS[50]
    allowed = {"chunk-1", "chunk-2", "chunk-50", "not-in-index"}
    hits = index.search(query, k=10, allowed_ids=allowed)
    assert len(hits) == 3, "pre-filter: return every allowed match, not top-10 then filter"
    assert hits[0][0] == "chunk-50"
    assert {h[0] for h in hits} == {"chunk-1", "chunk-2", "chunk-50"}

    index.remove("chunk-50")
    assert len(index) == 199
    assert "chunk-50" not in [h[0] for h in index.search(query, k=200)]
    assert len(index.search(query, k=500)) == 199
    with pytest.raises(KeyError):
        index.remove("chunk-50")
    assert index.search(query, k=5, allowed_ids=set()) == []
    assert ex.VectorIndex(DIM).search(query) == []


@pytest.mark.topic("py.numpy.matmul", "py.numpy.vs_lists")
def test_search_batch_equals_single(index):
    queries = rng.standard_normal((7, DIM)).astype(np.float32)
    batch = index.search_batch(queries, k=4)
    assert len(batch) == 7
    for q, hits in zip(queries, batch, strict=True):
        single = index.search(q, k=4)
        assert [h[0] for h in hits] == [h[0] for h in single]
        np.testing.assert_allclose([h[1] for h in hits], [h[1] for h in single], rtol=1e-5)
    with pytest.raises(ValueError):
        index.search_batch(np.ones(DIM))
