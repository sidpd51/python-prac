import hashlib
import json
import logging

import pytest
from docmind.ingest import loaders


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.md").write_text("# Title\nHello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.TXT").write_text("bee", encoding="utf-8")
    (tmp_path / "sub" / "skip.pdf").write_bytes(b"%PDF")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "c.md").write_text("ignored", encoding="utf-8")
    (tmp_path / "bom.txt").write_bytes("﻿with bom".encode())
    (tmp_path / "latin.txt").write_bytes("café".encode("latin-1"))
    rows = [{"id": 1, "text": "one", "lang": "en"}, "not json", {"text": "no id"}, {"id": 2, "text": "two"}]
    (tmp_path / "rows.jsonl").write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n\n", encoding="utf-8"
    )
    return tmp_path


@pytest.mark.topic("py.files.pathlib", "py.files.dir_ops")
def test_iter_files(corpus):
    names = [p.relative_to(corpus).as_posix() for p in loaders.iter_files(corpus)]
    assert names == ["a.md", "bom.txt", "latin.txt", "rows.jsonl", "sub/b.TXT"]


@pytest.mark.topic("py.generators.suspension", "py.exceptions.raising")
def test_iter_files_validates_eagerly(tmp_path):
    with pytest.raises(FileNotFoundError):
        loaders.iter_files(tmp_path / "missing")


@pytest.mark.topic("py.files.encoding")
def test_read_text_encodings(corpus, caplog):
    assert loaders.read_text(corpus / "bom.txt") == "with bom"
    with caplog.at_level(logging.WARNING, logger="docmind.ingest"):
        assert loaders.read_text(corpus / "latin.txt") == "café"
    assert "latin.txt" in caplog.text


@pytest.mark.topic("py.generators.document_processing", "py.files.json", "py.exceptions.logging")
def test_load_documents(corpus, caplog):
    with caplog.at_level(logging.WARNING, logger="docmind.ingest"):
        docs = {d.id: d for d in loaders.load_documents(corpus)}
    assert set(docs) == {"a.md", "bom.txt", "latin.txt", "sub/b.TXT", "rows.jsonl:1", "rows.jsonl:2"}
    a = docs["a.md"]
    assert a.text == "# Title\nHello"
    assert a.metadata == {
        "source": "a.md",
        "suffix": ".md",
        "sha256": hashlib.sha256(b"# Title\nHello").hexdigest(),
    }
    assert docs["sub/b.TXT"].metadata["suffix"] == ".txt"
    assert docs["rows.jsonl:1"].metadata == {"lang": "en", "source": "rows.jsonl"}
    assert docs["rows.jsonl:2"].text == "two"
    assert caplog.text.count("rows.jsonl") >= 2, "invalid lines should be logged"


@pytest.mark.topic("py.files.large_files", "py.generators.lazy_evaluation")
def test_load_documents_is_lazy(corpus):
    it = loaders.load_documents(corpus)
    first = next(it)
    assert first.id == "a.md"
