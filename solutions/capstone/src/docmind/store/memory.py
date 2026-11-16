"""
Milestone M0.5 - in-memory vector store with persistence (satisfies interfaces.VectorStore)
topics: py.numpy.matmul, py.numpy.boolean_indexing, py.numpy.vectorization, py.oop.dunder,
        py.files.json, py.files.read_write

Build
-----
InMemoryVectorStore(dim)
  * upsert(chunks, vectors): vectors shape (len(chunks), dim) else ValueError. Rows are
    L2-normalised on insert. Upserting an existing chunk.id REPLACES it (no duplicates).
  * search(vector, k=5, where=None) -> list[ScoredChunk], best first, score = cosine similarity.
    `where` = exact-match filter on chunk.metadata keys, plus the special key "doc_id".
    Fully vectorised: one matrix-vector product for all scores - no Python loop over rows.
    Use np.argpartition for top-k when k < number of candidates. Empty store -> [].
  * delete(doc_id) -> number of chunks removed
  * __len__ -> number of chunks;  __contains__(chunk_id) -> bool
  * save(directory): writes vectors.npy + chunks.jsonl (one Chunk JSON per line);
    classmethod load(directory) -> InMemoryVectorStore restoring the exact same state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..models import Chunk, ScoredChunk


class InMemoryVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._ids: dict[str, int] = {}
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._ids

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(f"vectors must have shape ({len(chunks)}, {self.dim}), got {vectors.shape}")
        vectors = self._normalise(vectors)
        new_rows = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            if (pos := self._ids.get(chunk.id)) is not None:
                self._chunks[pos] = chunk
                self._vectors[pos] = vec
            else:
                self._ids[chunk.id] = len(self._chunks) + len(new_rows)
                new_rows.append((chunk, vec))
        if new_rows:
            self._chunks.extend(c for c, _ in new_rows)
            self._vectors = np.vstack([self._vectors, np.stack([v for _, v in new_rows])])

    def search(
        self, vector: np.ndarray, k: int = 5, where: Mapping[str, Any] | None = None
    ) -> list[ScoredChunk]:
        if not self._chunks or k <= 0:
            return []
        q = self._normalise(np.asarray(vector, dtype=np.float32).reshape(-1))
        scores = self._vectors @ q
        candidates = np.arange(len(self._chunks))
        if where:
            mask = np.fromiter(
                (
                    all(
                        (c.doc_id if key == "doc_id" else c.metadata.get(key)) == val
                        for key, val in where.items()
                    )
                    for c in self._chunks
                ),
                dtype=bool,
                count=len(self._chunks),
            )
            candidates = candidates[mask]
        if candidates.size == 0:
            return []
        cand_scores = scores[candidates]
        if k < candidates.size:
            top = np.argpartition(-cand_scores, k)[:k]
        else:
            top = np.arange(candidates.size)
        top = top[np.argsort(-cand_scores[top], kind="stable")]
        return [ScoredChunk(chunk=self._chunks[candidates[i]], score=float(cand_scores[i])) for i in top]

    def delete(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = self._vectors[keep]
            self._ids = {c.id: i for i, c in enumerate(self._chunks)}
        return removed

    def save(self, directory: Path | str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self._vectors)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(c.model_dump_json() + "\n")

    @classmethod
    def load(cls, directory: Path | str) -> InMemoryVectorStore:
        d = Path(directory)
        vectors = np.load(d / "vectors.npy")
        store = cls(vectors.shape[1])
        with (d / "chunks.jsonl").open(encoding="utf-8") as f:
            store._chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        store._vectors = vectors.astype(np.float32)
        store._ids = {c.id: i for i, c in enumerate(store._chunks)}
        return store
