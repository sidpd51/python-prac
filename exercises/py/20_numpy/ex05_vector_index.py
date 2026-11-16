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

import numpy.typing as npt


class VectorIndex:
    def __init__(self, dim: int, metric: str = "cosine") -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def add(self, ids: Sequence[str], vectors: npt.ArrayLike) -> None:
        raise NotImplementedError

    def remove(self, id_: str) -> None:
        raise NotImplementedError

    def search(
        self, query: npt.ArrayLike, k: int = 5, allowed_ids: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        raise NotImplementedError

    def search_batch(self, queries: npt.ArrayLike, k: int = 5) -> list[list[tuple[str, float]]]:
        raise NotImplementedError
