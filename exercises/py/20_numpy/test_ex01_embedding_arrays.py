import numpy as np
import pytest

from study.loader import load

ex = load(__file__, "ex01_embedding_arrays")


@pytest.mark.topic("py.numpy.creation", "py.numpy.arrays", "py.numpy.vs_lists")
def test_to_matrix():
    m = ex.to_matrix([[1, 2, 3], [4.5, 5, 6]])
    assert isinstance(m, np.ndarray)
    assert m.dtype == np.float32 and m.shape == (2, 3)
    np.testing.assert_array_equal(m, [[1, 2, 3], [4.5, 5, 6]])


@pytest.mark.topic("py.numpy.creation")
@pytest.mark.parametrize("bad", [[], [[1.0, 2.0], [3.0]], [1.0, 2.0, 3.0], [[[1.0]]]])
def test_to_matrix_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        ex.to_matrix(bad)


@pytest.mark.topic("py.numpy.attributes", "py.numpy.ndarray")
def test_describe():
    arr = np.zeros((4, 384), dtype=np.float32)
    assert ex.describe(arr) == {
        "shape": (4, 384),
        "ndim": 2,
        "dtype": "float32",
        "size": 1536,
        "nbytes": 6144,
        "itemsize": 4,
    }
    assert ex.describe(np.arange(5))["dtype"] == "int64"
    assert ex.describe(np.float64(1.0).reshape(()))["shape"] == ()


@pytest.mark.topic("py.numpy.reshaping", "py.numpy.ndarray")
def test_as_batches_is_a_view():
    flat = np.arange(12, dtype=np.float32)
    batches = ex.as_batches(flat, 4)
    assert batches.shape == (3, 4)
    np.testing.assert_array_equal(batches[1], [4, 5, 6, 7])
    assert np.shares_memory(batches, flat), "reshape should not copy"
    for bad_flat, dim in [(np.arange(10.0), 4), (np.zeros((2, 6)), 3)]:
        with pytest.raises(ValueError):
            ex.as_batches(bad_flat, dim)


@pytest.mark.topic("py.numpy.random")
def test_random_embeddings_reproducible_and_isolated():
    np.random.seed(123)
    expected_global = np.random.rand()
    np.random.seed(123)

    a = ex.random_embeddings(5, 8, seed=42)
    b = ex.random_embeddings(5, 8, seed=42)
    c = ex.random_embeddings(5, 8, seed=7)
    assert a.shape == (5, 8) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.random.rand() == expected_global, "must not use/consume the global RNG"

    big = ex.random_embeddings(2000, 50, seed=0)
    assert abs(float(big.mean())) < 0.02 and abs(float(big.std()) - 1.0) < 0.02


@pytest.mark.topic("py.numpy.indexing")
def test_every_other_dim_is_view():
    arr = np.arange(12.0).reshape(2, 6)
    out = ex.every_other_dim(arr)
    np.testing.assert_array_equal(out, [[0, 2, 4], [6, 8, 10]])
    assert np.shares_memory(out, arr)
    out[0, 0] = -1.0
    assert arr[0, 0] == -1.0, "a view writes through to the original"


@pytest.mark.topic("py.numpy.indexing")
def test_last_rows():
    arr = np.arange(10.0).reshape(5, 2)
    np.testing.assert_array_equal(ex.last_rows(arr, 2), [[6, 7], [8, 9]])
    assert ex.last_rows(arr, 0).shape == (0, 2), "arr[-0:] is the whole array - classic bug"
    assert ex.last_rows(arr, 10).shape == (5, 2)
