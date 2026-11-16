"""
Exercise 20.3 - Top-k retrieval and retrieval metrics                level: ★★☆
topics: numpy.indexing, numpy.boolean_indexing, numpy.aggregation, numpy.axis,
        numpy.vectorization

Problem
-------
After computing similarity scores you need the best k documents per query, filtering,
and offline metrics. No Python loops over queries or documents.

1. top_k(scores, k) -> np.ndarray of int indices
   scores: 1-D. Indices of the k highest scores, highest first. Use np.argpartition
   (O(n)) to select, then sort only those k. k > len(scores) -> all indices;
   k <= 0 -> empty int array. (Inputs in tests have no ties.)

2. top_k_rows(scores, k) -> (indices, values)
   scores: (n_queries, n_docs). Per row, the top-k indices and their scores, both of shape
   (n_queries, min(k, n_docs)), highest first in each row. Hint: argpartition along axis=1
   and np.take_along_axis.

3. filter_hits(ids, scores, min_score, exclude=None) -> (ids, scores)
   ids, scores: 1-D arrays of equal length. Keep hits with score >= min_score whose id is
   NOT in `exclude` (an iterable of ids already shown to the user, may be None), sorted by
   score descending. Use boolean masks (np.isin for exclusion).

4. recall_at_k(retrieved, relevant) -> float
   retrieved: (n_queries, k) int doc indices; relevant: (n_queries,) the gold doc index per
   query. Fraction of queries whose gold doc appears anywhere in its row.

5. mrr(retrieved, relevant) -> float
   Mean reciprocal rank: per query 1/(position+1) of the gold doc in its row (position 0 =
   first), or 0 if absent; averaged over queries.

Interviewers look for
---------------------
* argpartition vs argsort: O(n) selection + O(k log k) sort instead of O(n log n) for a
  million-doc corpus.
* Broadcasting a comparison (retrieved == relevant[:, None]) to build a boolean matrix,
  then reducing along an axis with any()/argmax().
* Boolean masks return copies; combining masks with & | ~ (not `and`/`or`).
"""

from collections.abc import Iterable

import numpy as np


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    raise NotImplementedError


def top_k_rows(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def filter_hits(
    ids: np.ndarray, scores: np.ndarray, min_score: float, exclude: Iterable | None = None
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def recall_at_k(retrieved: np.ndarray, relevant: np.ndarray) -> float:
    raise NotImplementedError


def mrr(retrieved: np.ndarray, relevant: np.ndarray) -> float:
    raise NotImplementedError
