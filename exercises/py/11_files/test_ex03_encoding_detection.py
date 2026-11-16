import codecs

import pytest

from study.loader import load

ex = load(__file__, "ex03_encoding_detection")

TEXT = "Café – naïve résumé\r\nline 2\n"


@pytest.mark.topic("py.files.encoding")
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (codecs.BOM_UTF8 + "café".encode(), "utf-8-sig"),
        ("café".encode("utf-16"), "utf-16"),  # Python's utf-16 codec writes a BOM
        ("café ✓".encode(), "utf-8"),
        (b"plain ascii", "utf-8"),
        (b"", "utf-8"),
        ("café – €".encode("cp1252"), "cp1252"),
        (b"caf\xe9 \x81", "latin-1"),  # 0x81 is undefined in cp1252
    ],
)
def test_detect_encoding(data, expected):
    assert ex.detect_encoding(data) == expected


@pytest.mark.topic("py.files.encoding", "py.files.read_write")
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1252"])
def test_read_text_any(tmp_path, encoding):
    p = tmp_path / f"doc-{encoding}.txt"
    p.write_bytes(TEXT.encode(encoding))
    text, enc = ex.read_text_any(p)
    assert enc == encoding
    assert text == "Café – naïve résumé\nline 2\n"
    assert not text.startswith("﻿"), "the BOM must not leak into the text"


@pytest.mark.topic("py.files.encoding")
def test_read_text_any_lone_cr_and_latin1(tmp_path):
    p = tmp_path / "old-mac.txt"
    p.write_bytes(b"a\rb\r\nc \x81")
    text, enc = ex.read_text_any(p)
    assert enc == "latin-1"
    assert text == "a\nb\nc \x81"


@pytest.mark.topic("py.files.text_vs_binary", "py.files.modes")
def test_is_binary(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    (tmp_path / "c.txt").write_bytes("hi there".encode("utf-16"))  # NULs, but text
    (tmp_path / "empty.txt").write_bytes(b"")
    assert ex.is_binary(tmp_path / "a.txt") is False
    assert ex.is_binary(tmp_path / "b.bin") is True
    assert ex.is_binary(tmp_path / "c.txt") is False
    assert ex.is_binary(tmp_path / "empty.txt") is False


@pytest.mark.topic("py.files.text_vs_binary", "py.files.large_files")
def test_is_binary_reads_only_a_sample(tmp_path):
    p = tmp_path / "big.txt"
    p.write_bytes(b"a" * 10_000 + b"\x00")
    assert ex.is_binary(p) is False, "NUL is beyond the 8 KiB sample"
    assert ex.is_binary(p, sample_size=20_000) is True


@pytest.mark.topic("py.files.encoding", "py.files.read_write")
def test_convert_to_utf8(tmp_path):
    src, dst = tmp_path / "src.txt", tmp_path / "dst.txt"
    src.write_bytes(TEXT.encode("utf-8-sig"))
    assert ex.convert_to_utf8(src, dst) == "utf-8-sig"
    assert dst.read_bytes() == "Café – naïve résumé\nline 2\n".encode()
    src.write_bytes(TEXT.encode("utf-16"))
    assert ex.convert_to_utf8(src, dst) == "utf-16"
    assert dst.read_bytes() == "Café – naïve résumé\nline 2\n".encode()
