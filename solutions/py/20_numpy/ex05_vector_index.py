"""
Exercise 20.5 - A brute-force vector index                           level: ★★★
topics: numpy.matmul, numpy.boolean_indexing, numpy.creation, numpy.linalg, numpy.indexing,
        numpy.vs_lists

Problem
-------
Before reaching for FAISS/pgvector, build the exact (brute-force) version every ANN index
is evaluated against.

VectorIndex(dim, metric="cosine")      metric in {"cosine", "dot", "l2"}, else ValueError

  add(ids, vectors)
    * ids: sequence of str; vectors: array-like of shape (len(ids), dim)
    * ValueError on wrong shape, on an id that already exists, or duplicate ids within
      the batch. Atomic: on error nothing is added.
    * stored as float32; for "cosine" store the L2-normalised vectors (normalise once at
      insert, not on every search). Zero vectors stay zero.
    * many small add() calls must not be O(n^2) overall: don't np.vstack the whole matrix
      on every call - keep a preallocated buffer and grow its capacity geometrically
      (like list does).

  __len__() -> number of live (not removed) vectors

  remove(id) -> None    KeyError if the id isn't live. Removed vectors never appear in results.

  search(query, k=5, allowed_ids=None) -> list[(id, score)]
    * query: shape (dim,); ValueError on wrong shape
    * "cosine": cosine similarity, "dot": raw dot product - highest first
      "l2": Euclidean distance - LOWEST first (score is the distance)
    * allowed_ids: optional collection of ids - only those may be returned (metadata
      pre-filtering). Unknown ids in it are ignored.
    * at most k results (fewer if fewer candidates); scores are Python floats
    * one matrix-vector product for the whole index, top-k via np.argpartition

  search_batch(queries, k=5) -> list[list[(id, score)]]
    queries (q, dim); result i equals search(queries[i], k). One matrix-matrix product
    for all queries (no Python loop over the index rows).

Interviewers look for
---------------------
* Brute force is O(N*d) per query and exact; it's the ground truth for measuring ANN
  recall (HNSW/IVF trade recall for speed).
* Normalise at write time so cosine == dot at read time.
* Pre-filtering with a boolean mask vs post-filtering top-k (which can return < k hits).
* Amortised growth: geometric capacity -> O(1) amortised append, the same trick as list.
"""

from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt

_METRICS = ("cosine", "dot", "l2")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def _grow(vecs: np.ndarray, alive: np.ndarray, needed: int) -> tuple[np.ndarray, np.ndarray]:
    capacity = vecs.shape[0]
    while capacity < needed:
        capacity *= 2  # geometric growth -> amortised O(1) per added row
    new_vecs = np.empty((capacity, vecs.shape[1]), dtype=vecs.dtype)
    new_vecs[: vecs.shape[0]] = vecs
    new_alive = np.zeros(capacity, dtype=bool)
    new_alive[: alive.shape[0]] = alive
    return new_vecs, new_alive


def _scores(mat: np.ndarray, qs: np.ndarray, metric: str) -> np.ndarray:
    """(q, n) scores where higher is better for every metric (l2 -> -squared distance)."""
    if metric == "cosine":
        qs = _normalize_rows(qs)
    dots = qs @ mat.T
    if metric != "l2":
        return dots
    sq = (qs * qs).sum(axis=1)[:, None] + (mat * mat).sum(axis=1)[None, :] - 2 * dots
    return -np.maximum(sq, 0.0)


def _rank(
    scores: np.ndarray, mask: np.ndarray, k: int, row_ids: list[str], metric: str
) -> list[tuple[str, float]]:
    candidates = np.flatnonzero(mask)
    k = min(k, candidates.size)
    if k <= 0:
        return []
    cand_scores = scores[candidates]
    top = np.argpartition(-cand_scores, k - 1)[:k]
    top = top[np.argsort(-cand_scores[top], kind="stable")]
    results = []
    for i in top:
        score = float(cand_scores[i])
        if metric == "l2":
            score = float(np.sqrt(-score))
        results.append((row_ids[candidates[i]], score))
    return results


class VectorIndex:
    def __init__(self, dim: int, metric: str = "cosine") -> None:
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.metric = metric
        self._vecs = np.empty((16, dim), dtype=np.float32)
        self._alive = np.zeros(16, dtype=bool)
        self._n = 0  # rows used (live + removed)
        self._row_ids: list[str] = []
        self._rows: dict[str, int] = {}  # live id -> row

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, ids: Sequence[str], vectors: npt.ArrayLike) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        ids = list(ids)
        if vecs.ndim != 2 or vecs.shape != (len(ids), self.dim):
            raise ValueError(f"expected shape ({len(ids)}, {self.dim}), got {vecs.shape}")
        if len(set(ids)) != len(ids) or any(i in self._rows for i in ids):
            raise ValueError("duplicate ids")
        if self.metric == "cosine":
            vecs = _normalize_rows(vecs)  # once at write time: cosine == dot at read time
        if self._n + len(ids) > self._vecs.shape[0]:
            self._vecs, self._alive = _grow(self._vecs, self._alive, self._n + len(ids))
        start, stop = self._n, self._n + len(ids)
        self._vecs[start:stop] = vecs
        self._alive[start:stop] = True
        self._row_ids.extend(ids)
        self._rows.update(zip(ids, range(start, stop), strict=True))
        self._n = stop

    def remove(self, id_: str) -> None:
        row = self._rows.pop(id_)  # KeyError if not live
        self._alive[row] = False

    def search(
        self, query: npt.ArrayLike, k: int = 5, allowed_ids: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query must have shape ({self.dim},), got {q.shape}")
        mask = self._alive[: self._n].copy()
        if allowed_ids is not None:
            allowed = np.zeros(self._n, dtype=bool)
            allowed[[self._rows[i] for i in allowed_ids if i in self._rows]] = True
            mask &= allowed
        scores = _scores(self._vecs[: self._n], q[None, :], self.metric)[0]
        return _rank(scores, mask, k, self._row_ids, self.metric)

    def search_batch(self, queries: npt.ArrayLike, k: int = 5) -> list[list[tuple[str, float]]]:
        qs = np.asarray(queries, dtype=np.float32)
        if qs.ndim != 2 or qs.shape[1] != self.dim:
            raise ValueError(f"queries must have shape (q, {self.dim}), got {qs.shape}")
        scores = _scores(self._vecs[: self._n], qs, self.metric)  # (q, n): one matmul
        mask = self._alive[: self._n]
        return [_rank(row, mask, k, self._row_ids, self.metric) for row in scores]
