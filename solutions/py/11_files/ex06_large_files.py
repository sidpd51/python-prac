"""
Exercise 11.6 - Large logs: count, tail, resume                  level: ★★★
topics: files.large_files, files.text_vs_binary, files.open, files.read_write

Problem
-------
LLM request logs grow to many GB. Memory use must stay O(chunk), not O(file).
All three functions work in BINARY mode.

1. count_lines(path, chunk_size=1 << 20) -> int
     * number of lines = number of b"\\n", plus 1 if the file is non-empty and does not
       end with b"\\n"; read fixed-size chunks (no readlines(), no read() of everything)

2. tail(path, n=10, chunk_size=8192) -> list[str]
     * the last n lines, decoded as UTF-8, without the "\\n", oldest first
     * seek to the end and read chunks BACKWARDS until you have enough newlines;
       reading the whole file is not allowed (the test checks memory usage)
     * a trailing newline does not create an empty last line; n <= 0 -> [];
       fewer than n lines -> all of them
     * a multi-byte UTF-8 character may straddle a chunk boundary - collect bytes,
       decode once at the end

3. read_new_lines(path, offset=0) -> tuple[list[str], int]
     * resumable ingestion of a file another process is still appending to:
       start at byte `offset`, return the COMPLETE lines found (decoded, without "\\n")
       and the new offset = the byte position just after the last complete line
     * a final partial line (no "\\n" yet) is NOT returned and not consumed - the next
       call (with the returned offset) picks it up once it is complete
     * offset beyond the end of the file -> ValueError (file was truncated/rotated)

Interviewers look for
---------------------
* Why binary mode: byte offsets from tell()/seek() are only meaningful for binary files
  (text-mode tell() returns an opaque cookie), and you avoid decoding cost.
* Chunked reading with a bounded buffer; how `for line in f` is already buffered and
  lazy, and when it isn't enough (one 2 GB line).
* Checkpointing a byte offset to resume after a crash - how log shippers work.
"""

import os
from pathlib import Path


def count_lines(path: str | Path, chunk_size: int = 1 << 20) -> int:
    lines = 0
    last = b""
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        lines += 1
    return lines


def tail(path: str | Path, n: int = 10, chunk_size: int = 8192) -> list[str]:
    if n <= 0:
        return []
    with open(path, "rb") as f:
        end = f.seek(0, os.SEEK_END)
        pos = end
        buf = b""
        # n lines need n+1 newlines unless we reach the start of the file
        while pos > 0 and buf.count(b"\n") <= n:
            step = min(chunk_size, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
    lines = buf.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()  # trailing newline
    return [ln.decode("utf-8") for ln in lines[-n:]]


def read_new_lines(path: str | Path, offset: int = 0) -> tuple[list[str], int]:
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        if offset > size:
            raise ValueError(f"offset {offset} beyond end of file ({size} bytes)")
        f.seek(offset)
        lines: list[str] = []
        for raw in f:  # binary iteration still splits on b"\n", lazily
            if not raw.endswith(b"\n"):
                break  # partial line: leave it for the next call
            lines.append(raw[:-1].decode("utf-8"))
            offset += len(raw)
    return lines, offset
