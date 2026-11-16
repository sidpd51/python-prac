"""
Exercise 22.7 - Integration test with real temp files                  level: ★★☆
topics: testing.integration, testing.fixtures, testing.coverage

Problem
-------
Unit tests with mocks can't tell you that the file you wrote is actually readable, UTF-8,
newline-terminated, or that a crash leaves no half-written output behind. An integration
test runs the real code against a real (temporary) filesystem.

The code under test is GIVEN: dedupe_jsonl(src, dst) -> dict[str, int]
  * reads `src` as UTF-8 JSONL; blank lines are ignored (not counted anywhere)
  * a line that isn't a JSON object with a str "text" field counts as "invalid"
  * duplicates: same text after collapsing whitespace and casefolding
    ("Hello  World" == "hello world"); the FIRST occurrence is kept
  * writes kept records - the whole original object, all fields - to `dst` as JSONL: one
    json.dumps(record, ensure_ascii=False) per line, every line ends with "\\n", parent
    directories are created
  * atomic write: writes to "<dst>.tmp" then os.replace() -> no .tmp file is left behind
  * returns {"read": non-blank lines, "written": n, "duplicates": n, "invalid": n}

1. check_dedupe(impl, workdir: Path) -> None
   A mutation-tested suite (as in ex01): create input files under `workdir` (our tests
   pass pytest's tmp_path), run `impl(src, dst)`, then assert on the RETURNED STATS and on
   the OUTPUT FILE: read it back raw (bytes/text) and parsed, and check the directory
   contents. dst should be in a sub-directory that doesn't exist yet.
   Must pass for the real implementation and fail (any exception) for each mutant.

Interviewers look for
---------------------
* tmp_path (per-test unique dir, auto-cleaned) instead of writing into the repo or /tmp.
* Assert on observable outputs (file bytes, return value), not on internals - the test
  survives a rewrite of the implementation.
* Integration tests are slower and fewer (test pyramid), but they catch encoding,
  newline and atomicity bugs that mocks hide. 100% line coverage of dedupe_jsonl with
  mocks could still miss every one of the mutants here - coverage != correctness.
"""

import json
import os
from pathlib import Path


def dedupe_jsonl(src: Path, dst: Path) -> dict[str, int]:
    """Code under test - correct implementation (given)."""
    stats = {"read": 0, "written": 0, "duplicates": 0, "invalid": 0}
    seen: set[str] = set()
    kept: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["read"] += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["invalid"] += 1
            continue
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            stats["invalid"] += 1
            continue
        key = " ".join(record["text"].split()).casefold()
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        kept.append(json.dumps(record, ensure_ascii=False))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text("".join([line + "\n" for line in kept]), encoding="utf-8")
    os.replace(tmp, dst)
    stats["written"] = len(kept)
    return stats


def check_dedupe(impl, workdir: Path) -> None:
    raise NotImplementedError
