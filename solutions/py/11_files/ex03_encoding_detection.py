"""
Exercise 11.3 - Ingesting text files of unknown encoding         level: ★★☆
topics: files.encoding, files.text_vs_binary, files.read_write, files.modes

Problem
-------
A RAG corpus scraped from a shared drive has UTF-8, UTF-8-with-BOM (Excel/Notepad),
UTF-16 and legacy Windows-1252 files - plus some binaries named .txt.

1. detect_encoding(data: bytes) -> str        (check in this order)
     * starts with the UTF-8 BOM  b"\\xef\\xbb\\xbf"           -> "utf-8-sig"
     * starts with a UTF-16 BOM  (b"\\xff\\xfe" / b"\\xfe\\xff")  -> "utf-16"
     * decodes as strict UTF-8                                -> "utf-8"
     * decodes as strict cp1252                               -> "cp1252"
     * otherwise                                              -> "latin-1"
       (latin-1 maps every byte to a char, so it never fails - which is also why it
        can't *detect* anything; it must be the last resort)

2. is_binary(path, sample_size=8192) -> bool
     * read at most sample_size bytes (never the whole file)
     * a file with a UTF-16 BOM is text; otherwise a NUL byte in the sample -> binary
     * empty file -> False

3. read_text_any(path) -> tuple[str, str]
     * returns (text, encoding) using detect_encoding on the raw bytes
     * the BOM is not part of the text; "\\r\\n" and lone "\\r" become "\\n"

4. convert_to_utf8(src, dst) -> str
     * write dst as UTF-8 *without* BOM and with "\\n" line endings (also on Windows);
       return the detected source encoding

Interviewers look for
---------------------
* Text vs bytes: a file is bytes on disk; decoding is an explicit step with a codec.
  Text mode = bytes + decoding + newline translation.
* Why "utf-8-sig" exists (a stray \\ufeff at the start of chunk 0 breaks exact-match,
  JSON parsing and tokenization).
* errors="strict" for detection; know errors="replace"/"surrogateescape" and when
  silent replacement corrupts data.
* Detect binaries before feeding them to an embedding model.
"""

import codecs
from pathlib import Path

_UTF16_BOMS = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)


def detect_encoding(data: bytes) -> str:
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if data.startswith(_UTF16_BOMS):
        return "utf-16"
    for enc in ("utf-8", "cp1252"):
        try:
            data.decode(enc)
        except UnicodeDecodeError:
            continue
        return enc
    return "latin-1"


def is_binary(path: str | Path, sample_size: int = 8192) -> bool:
    with open(path, "rb") as f:
        sample = f.read(sample_size)
    if sample.startswith(_UTF16_BOMS):
        return False
    return b"\x00" in sample


def read_text_any(path: str | Path) -> tuple[str, str]:
    data = Path(path).read_bytes()
    enc = detect_encoding(data)
    text = data.decode(enc)  # "utf-8-sig" / "utf-16" codecs drop the BOM
    return text.replace("\r\n", "\n").replace("\r", "\n"), enc


def convert_to_utf8(src: str | Path, dst: str | Path) -> str:
    text, enc = read_text_any(src)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return enc
