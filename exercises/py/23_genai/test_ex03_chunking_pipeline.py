import json
import logging

import pytest

from study.loader import load

ex = load(__file__, "ex03_chunking_pipeline")

TEXT = (
    "Intro para one two three.\n\n"
    "Second para has five words. And then six more words here.\nA new line with words.\n\n"
    "Short end."
)


@pytest.mark.topic("py.genai.chunking")
def test_split_fits_whole():
    assert ex.split_recursive(TEXT, 100) == [TEXT]
    assert ex.split_recursive("  padded text \n", 5) == ["padded text"]
    assert ex.split_recursive("", 3) == []
    assert ex.split_recursive(" \n\n ", 3) == []


@pytest.mark.topic("py.genai.chunking")
def test_split_prefers_paragraphs_then_lines():
    assert ex.split_recursive(TEXT, 12) == [
        "Intro para one two three.",
        "Second para has five words. And then six more words here.",
        "A new line with words.",
        "Short end.",
    ]


@pytest.mark.topic("py.genai.chunking")
def test_split_falls_back_to_sentences_then_words():
    assert ex.split_recursive(TEXT, 8) == [
        "Intro para one two three.",
        "Second para has five words.",
        "And then six more words here.",
        "A new line with words.",
        "Short end.",
    ]
    assert ex.split_recursive(TEXT, 5) == [
        "Intro para one two three.",
        "Second para has five words.",
        "And then six more words",
        "here.",
        "A new line with words.",
        "Short end.",
    ]


@pytest.mark.topic("py.genai.chunking")
def test_split_greedy_merge_and_budget():
    assert ex.split_recursive("a b c d e f g", 3) == ["a b c", "d e f", "g"]
    for max_tokens in (1, 2, 3, 4, 7, 9):
        pieces = ex.split_recursive(TEXT, max_tokens)
        assert all(ex.count_tokens(p) <= max_tokens for p in pieces)
        assert " ".join(pieces).split() == TEXT.split(), "no words lost or duplicated"


@pytest.mark.topic("py.genai.chunking")
def test_split_unsplittable_and_custom_separators():
    assert ex.split_recursive("supercalifragilistic", 0) == ["supercalifragilistic"]
    # "a;b;" is ONE whitespace token, so it merges; "c d" is over budget with no separator left
    assert ex.split_recursive("a;b;c d", 1, separators=(";",)) == ["a;b;", "c d"]


DOC = {"id": "d1", "text": "one two three four five six seven eight nine ten", "source": "a.md", "page": 2}


@pytest.mark.topic("py.genai.chunking", "py.genai.document_processing")
def test_chunk_document_no_overlap():
    chunks = ex.chunk_document(DOC, max_tokens=4)
    assert [c.text for c in chunks] == ["one two three four", "five six seven eight", "nine ten"]
    assert [c.id for c in chunks] == ["d1:0", "d1:1", "d1:2"]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.n_tokens for c in chunks] == [4, 4, 2]
    assert all(c.doc_id == "d1" for c in chunks)


@pytest.mark.topic("py.genai.chunking")
def test_chunk_document_overlap():
    chunks = ex.chunk_document(DOC, max_tokens=4, overlap=1)
    assert [c.text for c in chunks] == [
        "one two three",
        "three four five six",
        "six seven eight nine",
        "nine ten",
    ]
    assert max(c.n_tokens for c in chunks) <= 4
    two = ex.chunk_document(DOC, max_tokens=5, overlap=2)
    assert [c.text for c in two] == [
        "one two three",
        "two three four five six",
        "five six seven eight nine",
        "eight nine ten",
    ]


@pytest.mark.topic("py.genai.document_processing")
def test_chunk_metadata_copied_per_chunk():
    chunks = ex.chunk_document(DOC, max_tokens=4)
    assert chunks[0].metadata == {"source": "a.md", "page": 2}
    chunks[0].metadata["poison"] = 1
    assert "poison" not in chunks[1].metadata
    assert "text" not in chunks[1].metadata and "id" not in chunks[1].metadata


@pytest.mark.topic("py.genai.chunking")
@pytest.mark.parametrize(("max_tokens", "overlap"), [(0, 0), (4, 4), (4, 5), (4, -1)])
def test_chunk_document_validation(max_tokens, overlap):
    with pytest.raises(ValueError):
        ex.chunk_document(DOC, max_tokens=max_tokens, overlap=overlap)


def _line(**doc):
    return json.dumps(doc)


@pytest.mark.topic("py.genai.pipelines", "py.genai.document_processing")
def test_ingest_end_to_end(caplog):
    lines = [
        _line(id="a", text="  alpha beta gamma  ", lang="en"),
        "",
        "{broken json",
        _line(id="b", text="alpha beta gamma"),  # duplicate after normalisation
        json.dumps(["not", "an", "object"]),
        _line(id="c", text="   "),  # empty after normalisation
        _line(id="d", body="no text"),
        _line(id="e", text="one two three four five"),
    ]
    with caplog.at_level(logging.WARNING, logger="genai.ingest"):
        chunks = list(ex.ingest(lines, max_tokens=3))
    assert [(c.id, c.text) for c in chunks] == [
        ("a:0", "alpha beta gamma"),
        ("e:0", "one two three"),
        ("e:1", "four five"),
    ]
    assert chunks[0].metadata == {"lang": "en"}
    warned = [r for r in caplog.records if r.name == "genai.ingest"]
    assert [r.args[0] for r in warned] == [3, 5, 7], "line numbers of the skipped lines (1-based)"
    assert all(r.levelno == logging.WARNING for r in warned)


@pytest.mark.topic("py.genai.pipelines", "py.genai.normalization")
def test_ingest_uses_injected_normalize_for_text_and_dedupe():
    lines = [_line(id="a", text="Hello World"), _line(id="b", text="hello   world")]
    chunks = list(ex.ingest(lines, max_tokens=5, normalize=lambda t: " ".join(t.split()).lower()))
    assert [(c.id, c.text) for c in chunks] == [("a:0", "hello world")]


@pytest.mark.topic("py.genai.pipelines")
def test_ingest_validates_eagerly():
    with pytest.raises(ValueError):
        ex.ingest([], max_tokens=3, overlap=3)  # no next() here


@pytest.mark.topic("py.genai.pipelines")
def test_ingest_is_lazy():
    pulled = []

    def source():
        for i in range(10_000):
            pulled.append(i)
            yield _line(id=str(i), text=f"doc {i} has five words")

    it = ex.ingest(source(), max_tokens=3)
    assert pulled == []
    first, second, third = next(it), next(it), next(it)
    assert (first.id, second.id, third.id) == ("0:0", "0:1", "1:0")
    assert pulled == [0, 1]
