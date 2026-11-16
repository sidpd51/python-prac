"""
Exercise 20.2 - L2 normalisation and cosine similarity, vectorised   level: ★★☆
topics: numpy.broadcasting, numpy.vectorization, numpy.math, numpy.linalg, numpy.matmul,
        numpy.axis

Problem
-------
The core math of semantic search. NO Python loops over rows - whole-array operations only.

1. l2_normalize(x, axis=-1) -> np.ndarray
   Scale vectors to unit L2 norm along `axis` (works for 1-D and 2-D input).
   Zero vectors must stay all-zero: no division-by-zero warnings, no NaN.
   Don't modify the input; keep the input's floating dtype (float32 in -> float32 out).

2. cosine_similarity(a, b) -> np.ndarray
   a: (n, d), b: (m, d) -> (n, m) matrix with sim[i, j] = cos(a[i], b[j]).
   A 1-D a or b is treated as a single row (shape (d,) -> (1, d)).
   Similarity involving a zero vector is 0.0. Mismatched d -> ValueError.

3. pairwise_l2(a, b) -> np.ndarray
   (n, m) Euclidean distances via the expansion ||a||^2 + ||b||^2 - 2 a.b (matrix product
   plus broadcasting), NOT by materialising an (n, m, d) difference array. Clip tiny negative
   values (floating-point error) to 0 before the square root.

Interviewers look for
---------------------
* keepdims=True (or [:, None]) so (n, 1) broadcasts against (n, d).
* Normalise once, then cosine = dot product -> one matmul gives all n*m similarities.
* Guarding zero norms with np.where / np.maximum(norm, eps) instead of letting NaN leak
  into your index.
* The (n, m, d) broadcasting version is elegant but O(n*m*d) memory - a classic OOM.
  The expansion trick is memory-cheap but suffers cancellation error for near-identical
  vectors in float32 (distance ~1e-3 instead of 0) - fine for ranking.
"""

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    # divide only where norm > 0; zero vectors stay zero (and no warning is raised)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: {a.shape[1]} vs {b.shape[1]}")
    return l2_normalize(a) @ l2_normalize(b).T


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: {a.shape[1]} vs {b.shape[1]}")
    sq = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.sqrt(np.maximum(sq, 0.0))
