"""
Milestone M0.4a - offline hashing embedder (satisfies interfaces.Embedder)
topics: py.numpy.vectorization, py.numpy.broadcasting, py.genai.tokenization, py.async.coroutine_functions

Why: lets the whole pipeline run and be tested with NO API key, deterministically.
Later phases swap in real embedders behind the same Protocol.

Build
-----
HashingEmbedder(dim=256)
  * tokens = lowercase words matching r"\\w+"
  * for each token: h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
      bucket = h % dim ; sign = +1 if (h >> 63) & 1 == 0 else -1 ; vec[bucket] += sign
    (Python's built-in hash() is randomised per process - why is that a problem here?)
  * L2-normalise each row; an all-zero row (no tokens) stays all-zero (no division by zero)
  * `async def embed(texts) -> np.ndarray` shape (len(texts), dim), dtype float32
  * dim <= 0 -> ValueError
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"\w+")


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 == 0 else -1.0
        return vec

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.stack([self._vector(t) for t in texts])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return np.divide(mat, norms, out=np.zeros_like(mat), where=norms > 0)
