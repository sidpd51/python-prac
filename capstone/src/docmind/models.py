"""Core data models shared by every phase. (GIVEN - you don't need to write this file.)

Later phases ADD models here (Answer, Citation, EvalCase, AgentStep...) - they never change
the existing ones in breaking ways, so earlier milestones keep passing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A source document. `id` is stable (e.g. relative path) so re-ingesting replaces it."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit. `start`/`end` are character offsets into the parent document."""

    model_config = ConfigDict(frozen=True)

    id: str  # f"{doc_id}#{index}"
    doc_id: str
    index: int = Field(ge=0)
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class IngestReport(BaseModel):
    documents: int = 0
    chunks: int = 0
    skipped: int = 0
    seconds: float = 0.0
