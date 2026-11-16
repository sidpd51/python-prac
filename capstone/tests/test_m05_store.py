import numpy as np
import pytest
from docmind.interfaces import VectorStore
from docmind.models import Chunk
from docmind.store.memory import InMemoryVectorStore


def chunk(doc, i, **meta):
    return Chunk(id=f"{doc}#{i}", doc_id=doc, index=i, text=f"{doc} {i}", start=0, end=1, metadata=meta)


@pytest.fixture
def store():
    s = InMemoryVectorStore(3)
    s.upsert(
        [chunk("a", 0, lang="en"), chunk("a", 1, lang="de"), chunk("b", 0, lang="en")],
        np.array([[1, 0, 0], [0, 2, 0], [1, 1, 0]], dtype=np.float32),
    )
    return s


@pytest.mark.topic("py.numpy.matmul", "py.type_hints.protocol")
def test_search_cosine_order(store):
    assert isinstance(store, VectorStore)
    hits = store.search(np.array([1, 0.1, 0]), k=2)
    assert [h.chunk.id for h in hits] == ["a#0", "b#0"]
    assert hits[0].score == pytest.approx(1 / np.sqrt(1.01), rel=1e-4)
    assert len(store.search(np.array([1, 0, 0]), k=10)) == 3


@pytest.mark.topic("py.numpy.boolean_indexing")
def test_search_filters(store):
    assert [h.chunk.id for h in store.search(np.array([0, 1, 0]), k=5, where={"lang": "en"})] == [
        "b#0",
        "a#0",
    ]
    assert [h.chunk.id for h in store.search(np.array([0, 1, 0]), where={"doc_id": "a"})] == ["a#1", "a#0"]
    assert store.search(np.array([0, 1, 0]), where={"lang": "fr"}) == []
    assert InMemoryVectorStore(3).search(np.array([1, 0, 0])) == []


@pytest.mark.topic("py.oop.dunder")
def test_upsert_replaces_and_validates(store):
    store.upsert([chunk("a", 0, lang="fr")], np.array([[0, 0, 1]]))
    assert len(store) == 3 and "a#0" in store and "zzz" not in store
    assert store.search(np.array([0, 0, 1]), k=1)[0].chunk.metadata == {"lang": "fr"}
    with pytest.raises(ValueError):
        store.upsert([chunk("c", 0)], np.zeros((1, 4)))


@pytest.mark.topic("py.data_structures.dicts")
def test_delete(store):
    assert store.delete("a") == 2
    assert len(store) == 1 and "a#0" not in store
    assert store.delete("nope") == 0
    store.upsert([chunk("c", 0)], np.array([[1, 0, 0]]))
    assert store.search(np.array([1, 0, 0]), k=1)[0].chunk.id == "c#0"


@pytest.mark.topic("py.files.json", "py.files.read_write", "py.oop.class_methods")
def test_save_load_roundtrip(store, tmp_path):
    store.save(tmp_path / "idx")
    loaded = InMemoryVectorStore.load(tmp_path / "idx")
    assert len(loaded) == 3
    q = np.array([0.3, 0.7, 0.1])
    assert [(h.chunk, round(h.score, 5)) for h in loaded.search(q)] == [
        (h.chunk, round(h.score, 5)) for h in store.search(q)
    ]


@pytest.mark.topic("py.numpy.vectorization")
def test_large_topk():
    rng = np.random.default_rng(0)
    s = InMemoryVectorStore(16)
    vecs = rng.normal(size=(2000, 16)).astype(np.float32)
    s.upsert([chunk("d", i) for i in range(2000)], vecs)
    q = rng.normal(size=16)
    hits = s.search(q, k=5)
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    expected = np.argsort(-(norm @ (q / np.linalg.norm(q))))[:5]
    assert [h.chunk.index for h in hits] == list(expected)
