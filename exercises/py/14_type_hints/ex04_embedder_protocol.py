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

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 8) -> None:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class CachingEmbedder:
    def __init__(self, inner: Embedder) -> None:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


def embed_in_batches(embedder: Embedder, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    raise NotImplementedError


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    raise NotImplementedError
