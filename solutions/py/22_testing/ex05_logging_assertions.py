"""
Exercise 22.5 - Asserting on logs (and never logging secrets)          level: ★★☆
topics: testing.logging, testing.logging_module, testing.unit

Problem
-------
Logs are part of a service's contract: alerts, dashboards and error trackers depend on
levels, logger names and message templates. Test them.

1. capture_logs(logger_name="", level=logging.DEBUG) - context manager yielding a list
   that fills with the logging.LogRecord objects emitted to that logger (and its
   children, via propagation) while the block runs. This is what pytest's `caplog` and
   unittest's `assertLogs` do under the hood:
   * attach your own logging.Handler subclass/instance to the logger, set the logger's
     level to `level` for the duration of the block
   * on exit (also via exception): remove the handler and restore the logger's previous
     level - leave no trace

2. check_ingest_logging(impl) -> None - a mutation-tested suite (as in ex01/ex03) for the
   GIVEN `ingest` below. Spec of ingest(docs, api_key) -> int:
   * logs only via the logger named "app.ingest" (LOGGER below)
   * DEBUG  "auth configured (key ending %s)" with the last 4 chars of the key - the
     full key must never appear in any rendered message (record.getMessage())
   * doc whose "text" is empty/whitespace -> WARNING with the TEMPLATE
     "skipping %s: empty text" and args (doc_id,) - not a pre-formatted f-string,
     so error trackers can group all such warnings into one issue
   * doc without a "text" key -> ERROR "bad document %s" WITH exception info
     (logger.exception inside the except block -> record.exc_info is set)
   * at the end INFO "ingested %d docs, skipped %d" (skipped = empty + bad docs)
   * returns the number of ingested docs

Interviewers look for
---------------------
* Assert on structured fields (record.levelno, record.name, record.msg, record.args,
  record.exc_info), not on a formatted string that changes when someone edits the format.
* logger.info("x %s", v) formats lazily (only if the record is emitted) and keeps a stable
  template; f-strings format eagerly and defeat grouping in Sentry & co.
* Use module-level named loggers (getLogger(__name__)), not the root logger, so apps can
  route/silence per component.
* Test isolation: a capturing handler left attached leaks into every later test.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

__stub_keep__ = ["LOGGER", "ingest"]

LOGGER = logging.getLogger("app.ingest")


def ingest(docs: list[dict], api_key: str) -> int:
    """Code under test - correct implementation (given)."""
    LOGGER.debug("auth configured (key ending %s)", api_key[-4:])
    ok = skipped = 0
    for doc in docs:
        try:
            text = doc["text"]
        except KeyError:
            LOGGER.exception("bad document %s", doc.get("id"))
            skipped += 1
            continue
        if not text.strip():
            LOGGER.warning("skipping %s: empty text", doc["id"])
            skipped += 1
            continue
        ok += 1
    LOGGER.info("ingested %d docs, skipped %d", ok, skipped)
    return ok


@contextmanager
def capture_logs(logger_name: str = "", level: int = logging.DEBUG) -> Iterator[list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(logger_name)
    handler = ListHandler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def check_ingest_logging(impl) -> None:
    key = "sk-test-SUPERSECRET-9f3a"
    docs = [
        {"id": "d1", "text": "hello"},
        {"id": "d2", "text": "   "},
        {"id": "d3"},
        {"id": "d4", "text": "world"},
    ]
    with capture_logs("app.ingest") as records:
        assert impl(docs, key) == 2

    assert records, "nothing was logged to 'app.ingest'"
    assert all(r.name == "app.ingest" for r in records)
    assert all(key not in r.getMessage() for r in records), "secret leaked into logs"

    debug = [r for r in records if r.levelno == logging.DEBUG]
    assert len(debug) == 1 and debug[0].getMessage().endswith("(key ending 9f3a)")

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].msg == "skipping %s: empty text"
    assert warnings[0].args == ("d2",)

    errors = [r for r in records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].getMessage() == "bad document d3"
    assert errors[0].exc_info is not None, "use logger.exception so the traceback is kept"

    infos = [r for r in records if r.levelno == logging.INFO]
    assert [r.getMessage() for r in infos] == ["ingested 2 docs, skipped 2"]
