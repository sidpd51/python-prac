import numpy as np
import pytest
from docmind.embeddings.hashing import HashingEmbedder
from docmind.ingest.chunking import CharChunker
from docmind.models import Document
from docmind.pipeline import ingest, search
from docmind.store.memory import InMemoryVectorStore


class SpyEmbedder:
    """Fake embedder that records batch sizes (a test double satisfying the Protocol)."""

    dim = 4

    def __init__(self):
        self.batches: list[int] = []

    async def embed(self, texts):
        self.batches.append(len(texts))
        return np.ones((len(texts), self.dim), dtype=np.float32)


@pytest.mark.topic("py.generators.pipelines", "py.genai.batching", "py.testing.mocking")
async def test_ingest_batches_across_documents():
    docs = [Document(id=f"d{i}", text="x" * 10) for i in range(3)]  # 3 chunks each (size 4)
    emb, store = SpyEmbedder(), InMemoryVectorStore(4)
    report = await ingest(docs, CharChunker(4), emb, store, batch_size=4)
    assert emb.batches == [4, 4, 1]
    assert (report.documents, report.chunks, report.skipped) == (3, 9, 0)
    assert len(store) == 9 and report.seconds >= 0


@pytest.mark.topic("py.genai.pipelines", "py.generators.memory_efficiency")
async def test_ingest_is_streaming():
    pulled = []

    def docs():
        for i in range(100):
            pulled.append(i)
            yield Document(id=f"d{i}", text="abcd")

    emb = SpyEmbedder()

    class Stop(Exception):
        pass

    async def embed_then_stop(texts):
        # by the first embedding call, only ~batch_size docs may have been read
        assert len(pulled) <= 5
        raise Stop

    emb.embed = embed_then_stop
    with pytest.raises(Stop):
        await ingest(docs(), CharChunker(4), emb, InMemoryVectorStore(4), batch_size=5)


@pytest.mark.topic("py.genai.document_processing")
async def test_reingest_replaces_and_counts_skipped():
    emb, store = SpyEmbedder(), InMemoryVectorStore(4)
    await ingest([Document(id="a", text="x" * 12)], CharChunker(4), emb, store)
    assert len(store) == 3
    report = await ingest(
        [Document(id="a", text="x" * 4), Document(id="empty", text="")], CharChunker(4), emb, store
    )
    assert len(store) == 1, "re-ingesting a document must replace its old chunks"
    assert report.skipped == 1


@pytest.mark.topic("py.async.await", "py.type_hints.protocol")
async def test_end_to_end_search():
    docs = [
        Document(
            id="vec",
            text="Vector databases store embeddings for similarity search.",
            metadata={"source": "vec"},
        ),
        Document(id="bake", text="Banana bread needs ripe bananas and flour.", metadata={"source": "bake"}),
    ]
    emb, store = HashingEmbedder(512), InMemoryVectorStore(512)
    await ingest(docs, CharChunker(200), emb, store)
    hits = await search("similarity search in a vector database", emb, store, k=1)
    assert hits[0].chunk.doc_id == "vec"
    hits = await search("similarity search in a vector database", emb, store, k=1, where={"source": "bake"})
    assert hits[0].chunk.doc_id == "bake"
