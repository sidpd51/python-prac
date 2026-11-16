import warnings

import numpy as np
import pytest

from study.loader import load

ex = load(__file__, "ex02_cosine_similarity")

rng = np.random.default_rng(0)
A = rng.standard_normal((6, 16)).astype(np.float32)
B = rng.standard_normal((4, 16)).astype(np.float32)


@pytest.mark.topic("py.numpy.broadcasting", "py.numpy.linalg")
def test_l2_normalize_rows():
    np.testing.assert_allclose(ex.l2_normalize(np.array([3.0, 4.0])), [0.6, 0.8])
    x = np.array([[3.0, 4.0], [0.0, 5.0], [1.0, 1.0]])
    out = ex.l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)
    np.testing.assert_allclose(out[2], [2**-0.5, 2**-0.5])


@pytest.mark.topic("py.numpy.math", "py.numpy.vectorization")
def test_l2_normalize_zero_vectors_are_safe():
    x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]], dtype=np.float32)
    before = x.copy()
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        out = ex.l2_normalize(x)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [1 / 3, 2 / 3, 2 / 3], rtol=1e-6)
    assert out.dtype == np.float32, "keep float32 (half the memory of float64)"
    np.testing.assert_array_equal(x, before)  # input untouched


@pytest.mark.topic("py.numpy.axis")
def test_l2_normalize_axis_and_int_input():
    x = np.array([[3, 0], [4, 0]])  # int input -> float output
    out = ex.l2_normalize(x, axis=0)
    assert np.issubdtype(out.dtype, np.floating)
    np.testing.assert_allclose(out, [[0.6, 0.0], [0.8, 0.0]])


def loop_cosine(a, b):
    out = np.zeros((len(a), len(b)))
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            out[i, j] = 0.0 if nu == 0 or nv == 0 else u @ v / (nu * nv)
    return out


@pytest.mark.topic("py.numpy.matmul", "py.numpy.vectorization")
def test_cosine_similarity_matches_reference():
    sim = ex.cosine_similarity(A, B)
    assert sim.shape == (6, 4)
    np.testing.assert_allclose(sim, loop_cosine(A, B), rtol=1e-5, atol=1e-6)
    self_sim = ex.cosine_similarity(A, A)
    np.testing.assert_allclose(np.diag(self_sim), 1.0, rtol=1e-5)
    np.testing.assert_allclose(self_sim, self_sim.T, atol=1e-6)
    assert np.all(np.abs(sim) <= 1.0 + 1e-6)


@pytest.mark.topic("py.numpy.broadcasting", "py.numpy.matmul")
def test_cosine_similarity_query_vector_and_zero_rows():
    query = A[0]
    sim = ex.cosine_similarity(query, B)
    assert sim.shape == (1, 4)
    docs = np.vstack([B, np.zeros((1, 16), dtype=np.float32)])
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        sim = ex.cosine_similarity(query, docs)
    assert sim[0, -1] == 0.0
    # scale invariance: cosine ignores vector length
    np.testing.assert_allclose(ex.cosine_similarity(query * 10, B), ex.cosine_similarity(query, B),
                               rtol=1e-5)
    with pytest.raises(ValueError):
        ex.cosine_similarity(A, np.ones((2, 8)))


@pytest.mark.topic("py.numpy.broadcasting", "py.numpy.linalg", "py.numpy.matmul")
def test_pairwise_l2():
    a = A.astype(np.float64)
    b = B.astype(np.float64)
    expected = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    np.testing.assert_allclose(ex.pairwise_l2(a, b), expected, rtol=1e-9, atol=1e-9)

    same = ex.pairwise_l2(A, A)
    assert not np.isnan(same).any(), "clip negative rounding error before sqrt"
    # float32 + the expansion = cancellation error near 0 (fine for ranking, not for exact 0)
    np.testing.assert_allclose(np.diag(same), 0.0, atol=2e-2)
    np.testing.assert_allclose(ex.pairwise_l2(np.array([0.0, 0.0]), np.array([[3.0, 4.0]])), [[5.0]])
    with pytest.raises(ValueError):
        ex.pairwise_l2(A, np.ones((2, 3)))
