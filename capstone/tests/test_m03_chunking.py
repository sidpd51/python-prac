import pytest
from docmind.ingest.chunking import CharChunker, RecursiveChunker
from docmind.interfaces import Chunker
from docmind.models import Document

DOC = Document(id="d1", text="abcdefghij", metadata={"source": "d1.md"})

PROSE = (
    "RAG systems retrieve context. They then generate answers.\n\n"
    "Chunking matters a lot. Bad chunks mean bad retrieval.\n"
    "Overlap helps at boundaries.\n\n"
    "Supercalifragilisticexpialidocious is long."
)


@pytest.mark.topic("py.genai.chunking", "py.type_hints.protocol")
def test_char_chunker_basic():
    ch = CharChunker(4, overlap=1)
    assert isinstance(ch, Chunker)
    chunks = list(ch.chunk(DOC))
    assert [c.text for c in chunks] == ["abcd", "defg", "ghij"]
    assert [c.id for c in chunks] == ["d1#0", "d1#1", "d1#2"]
    assert all(c.doc_id == "d1" and c.metadata == {"source": "d1.md"} for c in chunks)
    assert all(DOC.text[c.start : c.end] == c.text for c in chunks)


@pytest.mark.topic("py.genai.chunking")
@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1)])
def test_char_chunker_validation(size, overlap):
    with pytest.raises(ValueError):
        CharChunker(size, overlap)


@pytest.mark.topic("py.generators.chunk_metadata")
def test_char_chunker_empty_and_metadata_copy():
    assert list(CharChunker(5).chunk(Document(id="e", text=""))) == []
    chunks = list(CharChunker(4).chunk(DOC))
    chunks[0].metadata["x"] = 1
    assert "x" not in chunks[1].metadata and "x" not in DOC.metadata


@pytest.mark.topic("py.genai.chunking", "py.generators.rag_chunking")
@pytest.mark.parametrize("size", [12, 25, 40, 60, 200])
def test_recursive_invariants(size):
    doc = Document(id="p", text=PROSE)
    chunks = list(RecursiveChunker(size).chunk(doc))
    assert all(len(c.text) <= size for c in chunks)
    assert all(doc.text[c.start : c.end] == c.text for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # gaps between chunks may only contain whitespace
    pos = 0
    for c in chunks:
        assert doc.text[pos : c.start].strip() == ""
        pos = c.end
    assert doc.text[pos:].strip() == ""


@pytest.mark.topic("py.genai.chunking")
def test_recursive_respects_structure():
    doc = Document(id="p", text=PROSE)
    chunks = [c.text for c in RecursiveChunker(80).chunk(doc)]
    assert chunks[0] == "RAG systems retrieve context. They then generate answers.\n\n"
    # paragraph 2 doesn't fit in 80 chars -> split on lines, not mid-sentence
    assert chunks[1] == "Chunking matters a lot. Bad chunks mean bad retrieval.\n"
    # a word longer than size is hard-split
    tiny = [c.text for c in RecursiveChunker(10).chunk(Document(id="w", text="Supercalifragilistic"))]
    assert tiny == ["Supercalif", "ragilistic"]
