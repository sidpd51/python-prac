"""
Exercise 11.2 - Eval results CSV with type coercion              level: ★☆☆
topics: files.csv, files.read_write, files.text_vs_binary, files.with

Problem
-------
An eval harness exports one row per test case. CSV has no types - everything is a string.

    id,model,score,passed,latency_ms,tags
    q1,gpt-4o,0.92,true,812,rag;faq
    q2,claude,0.40,no,,

1. read_eval_csv(path) -> list[EvalRow]      (EvalRow dataclass is given)
     * header required columns: id, model, score, passed -> otherwise
       ValueError naming the missing column(s). latency_ms and tags are optional columns
       (absent column -> None / []). Extra columns are ignored.
     * coercion: score -> float; passed -> bool from "true/false/1/0/yes/no"
       (case-insensitive, surrounding whitespace ignored); latency_ms "" -> None else int;
       tags "a; b" -> ["a", "b"], "" -> []
     * a value that can't be coerced -> ValueError mentioning f"line {n}" where n is the
       file line number the row ends on (csv.reader's line_num), chained from the original
       error if there is one
     * quoted fields may contain commas, quotes and newlines (including "\\r\\n" inside
       model output) - they must come back exactly

2. write_eval_csv(path, rows: Iterable[EvalRow]) -> int
     * header id,model,score,passed,latency_ms,tags ; passed written as "true"/"false",
       None as "", tags joined with ";" ; returns the number of rows
     * read_eval_csv(write_eval_csv(rows)) must give back equal rows

Interviewers look for
---------------------
* open(..., newline="") for csv in BOTH directions: the csv module does its own newline
  handling; without it embedded "\\r\\n" gets mangled (and on Windows you get blank rows).
* csv.DictReader/DictWriter instead of split(",") (quoting!).
* Validating at the boundary with a helpful line number instead of a KeyError later.
"""

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED = ("id", "model", "score", "passed")
COLUMNS = ("id", "model", "score", "passed", "latency_ms", "tags")
_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


@dataclass
class EvalRow:
    id: str
    model: str
    score: float
    passed: bool
    latency_ms: int | None = None
    tags: list[str] = field(default_factory=list)


def _to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _row(rec: dict[str, str]) -> EvalRow:
    latency = (rec.get("latency_ms") or "").strip()
    tags = rec.get("tags") or ""
    return EvalRow(
        id=rec["id"],
        model=rec["model"],
        score=float(rec["score"]),
        passed=_to_bool(rec["passed"]),
        latency_ms=int(latency) if latency else None,
        tags=[t.strip() for t in tags.split(";") if t.strip()],
    )


def read_eval_csv(path: str | Path) -> list[EvalRow]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(missing)}")
        rows: list[EvalRow] = []
        for rec in reader:
            try:
                rows.append(_row(rec))
            except ValueError as e:
                raise ValueError(f"line {reader.line_num}: {e}") from e
        return rows


def write_eval_csv(path: str | Path, rows: Iterable[EvalRow]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "id": r.id,
                    "model": r.model,
                    "score": repr(r.score),
                    "passed": "true" if r.passed else "false",
                    "latency_ms": "" if r.latency_ms is None else r.latency_ms,
                    "tags": ";".join(r.tags),
                }
            )
            n += 1
    return n
