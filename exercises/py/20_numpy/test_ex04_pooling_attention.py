import warnings

import numpy as np
import pytest

from study.loader import load

ex = load(__file__, "ex04_pooling_attention")

rng = np.random.default_rng(7)


@pytest.mark.topic("py.numpy.axis", "py.numpy.broadcasting")
def test_mean_pool_ignores_padding():
    embs = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],  # last token is padding
            [[2.0, 2.0], [-50.0, 9.0], [-50.0, 9.0]],  # only first token real
        ]
    )
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(ex.mean_pool(embs, mask), [[2.0, 3.0], [2.0, 2.0]])
    np.testing.assert_allclose(ex.mean_pool(embs, mask.astype(bool)), [[2.0, 3.0], [2.0, 2.0]])


@pytest.mark.topic("py.numpy.axis", "py.numpy.aggregation")
def test_mean_pool_matches_loop_and_empty_rows():
    embs = rng.standard_normal((4, 6, 8)).astype(np.float32)
    mask = np.array([[1] * 6, [1, 1, 1, 0, 0, 0], [0] * 6, [1, 0, 1, 0, 1, 0]])
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        pooled = ex.mean_pool(embs, mask)
    assert pooled.shape == (4, 8)
    for i in range(4):
        real = embs[i][mask[i] == 1]
        expected = real.mean(axis=0) if len(real) else np.zeros(8)
        np.testing.assert_allclose(pooled[i], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.topic("py.numpy.math")
def test_softmax_stable():
    np.testing.assert_allclose(ex.softmax(np.array([1000.0, 1000.0])), [0.5, 0.5])
    x = np.array([[1.0, 2.0, 3.0], [-1000.0, 0.0, 1000.0]])
    with np.errstate(over="raise"):
        out = ex.softmax(x)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    e = np.exp([1.0, 2.0, 3.0])
    np.testing.assert_allclose(out[0], e / e.sum())
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1.0], atol=1e-12)
    cols = ex.softmax(x, axis=0)
    np.testing.assert_allclose(cols.sum(axis=0), 1.0)


def loop_attention(q, k, v):
    out = np.zeros((q.shape[0], q.shape[1], v.shape[2]))
    for b in range(q.shape[0]):
        s = q[b] @ k[b].T / np.sqrt(q.shape[2])
        w = np.exp(s - s.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        out[b] = w @ v[b]
    return out


@pytest.mark.topic("py.numpy.matmul", "py.numpy.broadcasting")
def test_attention_matches_reference():
    q = rng.standard_normal((2, 3, 4))
    k = rng.standard_normal((2, 5, 4))
    v = rng.standard_normal((2, 5, 6))
    out, weights = ex.attention(q, k, v)
    assert out.shape == (2, 3, 6) and weights.shape == (2, 3, 5)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    np.testing.assert_allclose(out, loop_attention(q, k, v), rtol=1e-10)


@pytest.mark.topic("py.numpy.matmul", "py.numpy.boolean_indexing")
def test_attention_key_mask():
    q = rng.standard_normal((2, 2, 4))
    k = rng.standard_normal((2, 3, 4))
    v = rng.standard_normal((2, 3, 2))
    mask = np.array([[1, 1, 0], [0, 0, 0]])  # 2nd sequence: everything padded
    with np.errstate(invalid="raise", divide="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        out, weights = ex.attention(q, k, v, key_mask=mask)
    assert np.all(weights[0, :, 2] == 0.0), "masked keys get exactly zero weight"
    np.testing.assert_allclose(weights[0].sum(axis=-1), 1.0)
    np.testing.assert_allclose(out[0], loop_attention(q[:1], k[:1, :2], v[:1, :2])[0], rtol=1e-10)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(weights[1], 0.0)
    np.testing.assert_array_equal(out[1], 0.0)


@pytest.mark.topic("py.numpy.reshaping")
def test_split_and_merge_heads():
    x = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
    heads = ex.split_heads(x, 4)
    assert heads.shape == (2, 4, 3, 2)
    # head h of token t holds hidden dims [h*2, h*2+1] of that token
    np.testing.assert_array_equal(heads[1, 2, 0], x[1, 0, 4:6])
    np.testing.assert_array_equal(heads[0, 3, 2], x[0, 2, 6:8])
    np.testing.assert_array_equal(ex.merge_heads(heads), x)
    with pytest.raises(ValueError):
        ex.split_heads(x, 3)
