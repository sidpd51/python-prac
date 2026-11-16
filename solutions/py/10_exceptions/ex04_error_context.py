"""
Exercise 10.4 - Enriching and reporting errors in an ingest job   level: ★★☆
topics: exceptions.contextual, exceptions.logging, exceptions.best_practices,
        exceptions.chaining

Problem
-------
A document-ingest job must tell you *which* document broke, without losing the original
exception or its traceback.

1. error_context(**context) - a context manager
     * any Exception leaving the block gets ONE note (BaseException.add_note, 3.11+):
           "context: doc_id=a1, stage=embed"      (keys in the order given)
       and is re-raised: same object, same type, traceback intact
     * BaseExceptions that are not Exceptions (KeyboardInterrupt, SystemExit) pass
       through untouched; nothing happens on success

2. ingest(docs, process, *, logger, strict=False) -> IngestReport
     * docs: iterable of dicts with an "id"; calls process(doc) for each
     * success -> id appended to report.ok
     * an Exception in process():
         - strict=False: logger.exception("failed to ingest %s", doc_id,
                                          extra={"doc_id": doc_id})
           (logs at ERROR *with* the traceback), record
           report.failed[doc_id] = f"{type(e).__name__}: {e}", continue with the next doc
         - strict=True: add the note f"doc_id={doc_id}" and re-raise the same exception
           with a bare `raise` (don't log it AND raise it - pick one)
     * KeyboardInterrupt etc. are never caught

3. root_cause(exc) -> BaseException
     follow __cause__, or __context__ when __suppress_context__ is False, to the end.

4. error_chain(exc) -> str
     "TypeName: message" for exc and each exception below it (same walk as root_cause),
     joined with " <- ", e.g.
     "IngestError: doc a1 failed <- KeyError: 'text'"

Interviewers look for
---------------------
* Adding context without changing the exception type (add_note) vs wrapping it in a new
  type (raise X from e) - both valid, different contracts for callers.
* logger.exception (= error + exc_info) inside an except block, structured `extra`
  fields for log search; never "log and re-raise" at every layer (duplicate logs).
* Partial-failure reporting for batch jobs instead of crashing hour 3 of a run.
* Knowing how __cause__ / __context__ / __suppress_context__ form the chain.
"""

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IngestReport:
    ok: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


@contextmanager
def error_context(**context: Any) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        e.add_note("context: " + ", ".join(f"{k}={v}" for k, v in context.items()))
        raise


def ingest(
    docs: Iterable[dict[str, Any]],
    process: Callable[[dict[str, Any]], object],
    *,
    logger: logging.Logger,
    strict: bool = False,
) -> IngestReport:
    report = IngestReport()
    for doc in docs:
        doc_id = doc["id"]
        try:
            process(doc)
        except Exception as e:
            if strict:
                e.add_note(f"doc_id={doc_id}")
                raise
            logger.exception("failed to ingest %s", doc_id, extra={"doc_id": doc_id})
            report.failed[doc_id] = f"{type(e).__name__}: {e}"
        else:
            report.ok.append(doc_id)
    return report


def _next_in_chain(exc: BaseException) -> BaseException | None:
    if exc.__cause__ is not None:
        return exc.__cause__
    if not exc.__suppress_context__:
        return exc.__context__
    return None


def root_cause(exc: BaseException) -> BaseException:
    seen = {id(exc)}
    while (nxt := _next_in_chain(exc)) is not None and id(nxt) not in seen:
        seen.add(id(nxt))
        exc = nxt
    return exc


def error_chain(exc: BaseException) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    seen = {id(exc)}
    cur: BaseException | None = _next_in_chain(exc)
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = _next_in_chain(cur)
    return " <- ".join(parts)
