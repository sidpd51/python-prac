import pytest

from study.loader import load

ex = load(__file__, "ex02_rag_chunker")


@pytest.mark.topic("py.generators.rag_chunking")
def test_basic_no_overlap():
    chunks = list(ex.chunk_text("abcdefghij", size=4))
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (4, 8), (8, 10)]
    assert [c.index for c in chunks] == [0, 1, 2]


@pytest.mark.topic("py.generators.rag_chunking")
def test_overlap():
    chunks = list(ex.chunk_text("abcdefghij", size=4, overlap=2))
    assert [c.text for c in chunks] == ["abcd", "cdef", "efgh", "ghij"]
    # reconstructable from offsets
    assert all("abcdefghij"[c.start : c.end] == c.text for c in chunks)


@pytest.mark.topic("py.generators.rag_chunking")
@pytest.mark.parametrize(
    ("text", "size", "overlap", "expected"),
    [
        ("", 5, 0, []),
        ("abc", 5, 0, ["abc"]),
        ("abcde", 5, 2, ["abcde"]),  # no tiny trailing chunk
        ("abcdef", 3, 1, ["abc", "cde", "ef"]),
    ],
)
def test_edges(text, size, overlap, expected):
    assert [c.text for c in ex.chunk_text(text, size, overlap)] == expected


@pytest.mark.topic("py.generators.suspension", "py.generators.functions")
@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (4, 4), (4, 5), (4, -1)])
def test_validation_is_eager(size, overlap):
    with pytest.raises(ValueError):
        ex.chunk_text("abc", size, overlap)  # note: no list()/next() here


@pytest.mark.topic("py.generators.chunk_metadata", "py.generators.document_processing")
def test_chunk_documents_metadata():
    docs = [
        {"id": "a", "text": "hello world", "source": "a.md", "page": 1},
        {"id": "b", "text": "xyz", "source": "b.md"},
    ]
    chunks = list(ex.chunk_documents(docs, size=6, overlap=1))
    assert [c.text for c in chunks] == ["hello ", " world", "xyz"]
    assert chunks[0].metadata == {"doc_id": "a", "source": "a.md", "page": 1}
    assert chunks[2].metadata == {"doc_id": "b", "source": "b.md"}
    assert [c.index for c in chunks] == [0, 1, 0], "index restarts per document"
    assert all("text" not in c.metadata for c in chunks)


@pytest.mark.topic("py.generators.chunk_metadata")
def test_metadata_not_shared_between_chunks():
    chunks = list(ex.chunk_documents([{"id": "a", "text": "abcdefgh"}], size=4))
    chunks[0].metadata["poisoned"] = True
    assert "poisoned" not in chunks[1].metadata


@pytest.mark.topic("py.generators.yield_from", "py.generators.lazy_evaluation")
def test_chunk_documents_is_lazy():
    pulled = []

    def docs():
        for i in range(1000):
            pulled.append(i)
            yield {"id": str(i), "text": "abcdefgh"}

    it = ex.chunk_documents(docs(), size=4)
    assert pulled == []
    next(it), next(it), next(it)
    assert pulled == [0, 1]
