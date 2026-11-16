"""
Exercise 14.4 - Structural typing: an Embedder Protocol          level: ★★☆
topics: type_hints.protocol, type_hints.collections, type_hints.callable,
        type_hints.any_vs_object, type_hints.static_checking

Problem
-------
Your RAG code should work with OpenAI, a local sentence-transformers model, or a fake in
tests - without them sharing a base class. `Embedder` (a runtime_checkable Protocol) is
given.

1. class HashingEmbedder(dim=8) - a deterministic, dependency-free embedder
     * does NOT inherit from Embedder - it satisfies it structurally
     * dim < 1 -> ValueError; attribute .dim
     * embed(texts) -> one vector per text: for each lower-cased whitespace token,
       index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dim, add 1.0 there;
       then L2-normalize (an all-zero vector, e.g. for "", stays all zeros)
       (Python's hash() is salted per process - why can't you use it here?)

2. class CachingEmbedder(inner: Embedder)
     * .dim is inner's dim; .calls counts calls to inner.embed
     * embed(texts): look up each text in a cache; call inner.embed AT MOST ONCE per
       embed() call, with only the uncached texts, de-duplicated, in first-seen order;
       return vectors in the order of `texts`. No call at all if everything is cached.

3. embed_in_batches(embedder: Embedder, texts, batch_size=32) -> list[list[float]]
     * isinstance(embedder, Embedder) must hold, else TypeError (runtime_checkable)
     * batch_size < 1 -> ValueError
     * calls embedder.embed on consecutive slices of at most batch_size texts
     * every returned vector must have length embedder.dim -> ValueError otherwise
       (a runtime Protocol check only sees that the members EXIST)

4. cosine(a, b) -> float
     cosine similarity; 0.0 if either vector is all zeros; different lengths -> ValueError

Interviewers look for
---------------------
* Protocol (structural, "duck typing the type checker understands") vs ABC (nominal,
  must inherit/register). Protocols fit third-party classes you can't modify.
* @runtime_checkable isinstance() checks only that the attributes exist - not
  signatures or return types.
* Sequence[str] for read-only inputs (accepts list, tuple) vs list[str] for outputs.
"""

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__stub_keep__ = ["Embedder"]


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 8) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            out.append([x / norm for x in vec] if norm else vec)
        return out


class CachingEmbedder:
    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.calls = 0
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            self.calls += 1
            for text, vec in zip(missing, self.inner.embed(missing), strict=True):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]


def embed_in_batches(embedder: Embedder, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    if not isinstance(embedder, Embedder):
        raise TypeError(f"{type(embedder).__name__} does not implement Embedder")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = embedder.embed(texts[start : start + batch_size])
        if any(len(v) != embedder.dim for v in batch):
            raise ValueError(f"embedder returned vectors that are not of dim {embedder.dim}")
        vectors.extend(batch)
    return vectors


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)
