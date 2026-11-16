"""
Exercise 20.1 - Embedding arrays: creation, dtype, shape, views     level: ★☆☆
topics: numpy.arrays, numpy.ndarray, numpy.creation, numpy.attributes, numpy.reshaping,
        numpy.indexing, numpy.random, numpy.vs_lists

Problem
-------
Embedding APIs give you Python lists of floats; everything downstream wants a
(n_vectors, dim) float32 ndarray.

1. to_matrix(rows) -> np.ndarray
   Convert a list of equal-length lists of numbers into a 2-D float32 array.
   ValueError if rows is empty, ragged (different lengths), or not 2-D (e.g. a flat list
   of numbers).

2. describe(arr) -> dict with keys "shape" (tuple), "ndim", "dtype" (str, e.g. "float32"),
   "size" (number of elements), "nbytes", "itemsize".

3. as_batches(flat, dim) -> np.ndarray
   A 1-D buffer of n*dim numbers (e.g. read from a binary file) reshaped to (n, dim)
   WITHOUT copying (the result must share memory with `flat`).
   ValueError if flat isn't 1-D or its length isn't a multiple of dim.

4. random_embeddings(n, dim, seed) -> np.ndarray
   (n, dim) float32 standard-normal values from np.random.default_rng(seed). The same seed
   gives identical arrays; the legacy global state (np.random.seed / np.random.rand) must not
   be touched.

5. every_other_dim(arr) -> np.ndarray
   Columns 0, 2, 4, ... of a 2-D array, as a VIEW (basic slicing, not fancy indexing).

6. last_rows(arr, k) -> np.ndarray
   The last k rows of a 2-D array (k=0 -> shape (0, dim); k > n -> all rows).

Interviewers look for
---------------------
* An ndarray is one contiguous typed buffer + shape + strides; a list is an array of
  pointers to boxed Python objects (8-byte pointer + ~24-byte float each).
* float32 halves memory vs float64 and is what vector DBs / GPUs use.
* Basic slicing returns views (no copy); fancy/boolean indexing returns copies;
  reshape returns a view when strides allow.
* np.random.default_rng(seed) (Generator API) instead of the legacy global np.random.seed.
"""

import numpy as np
import numpy.typing as npt


def to_matrix(rows: list[list[float]]) -> np.ndarray:
    raise NotImplementedError


def describe(arr: np.ndarray) -> dict[str, object]:
    raise NotImplementedError


def as_batches(flat: npt.NDArray[np.floating], dim: int) -> np.ndarray:
    raise NotImplementedError


def random_embeddings(n: int, dim: int, seed: int) -> np.ndarray:
    raise NotImplementedError


def every_other_dim(arr: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def last_rows(arr: np.ndarray, k: int) -> np.ndarray:
    raise NotImplementedError
