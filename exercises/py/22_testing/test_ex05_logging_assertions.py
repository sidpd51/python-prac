import logging

import pytest

from study.loader import load

ex = load(__file__, "ex05_logging_assertions")


def assert_suite_fails(run_suite, *args):
    """Like a red pytest run: ANY exception counts as "the suite caught the bug".
    NotImplementedError is re-raised so an unsolved stub shows up as TODO, not as a pass."""
    try:
        run_suite(*args)
    except NotImplementedError:
        raise
    except (Exception, pytest.fail.Exception):
        return
    raise AssertionError("your suite PASSED on a buggy implementation - add a check for it")


def _make(
    *,
    skip_level=logging.WARNING,
    skip_via_root=False,
    leak_key=False,
    keep_exc_info=True,
    count_bad_as_skipped=True,
    eager_format=False,
):
    log = logging.getLogger("app.ingest")

    def impl(docs, api_key):
        log.debug("auth configured (key ending %s)", api_key if leak_key else api_key[-4:])
        ok = skipped = 0
        for doc in docs:
            try:
                text = doc["text"]
            except KeyError:
                if keep_exc_info:
                    log.exception("bad document %s", doc.get("id"))
                else:
                    log.error("bad document %s", doc.get("id"))
                skipped += count_bad_as_skipped
                continue
            if not text.strip():
                target = logging.getLogger() if skip_via_root else log
                if eager_format:
                    target.log(skip_level, f"skipping {doc['id']}: empty text")
                else:
                    target.log(skip_level, "skipping %s: empty text", doc["id"])
                skipped += 1
                continue
            ok += 1
        log.info("ingested %d docs, skipped %d", ok, skipped)
        return ok

    return impl


MUTANTS = {
    "skip_logged_at_info": _make(skip_level=logging.INFO),
    "skip_logged_on_root_logger": _make(skip_via_root=True),
    "full_api_key_in_debug_log": _make(leak_key=True),
    "error_without_traceback": _make(keep_exc_info=False),
    "bad_docs_not_counted_as_skipped": _make(count_bad_as_skipped=False),
    "f_string_message": _make(eager_format=True),
}


@pytest.mark.topic("py.testing.logging", "py.testing.logging_module")
def test_capture_logs_collects_records_and_children():
    log = logging.getLogger("t22.capture")
    with ex.capture_logs("t22.capture") as records:
        log.debug("d %s", 1)
        logging.getLogger("t22.capture.child").warning("w")
        logging.getLogger("t22.other").error("not mine")
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.DEBUG, "d 1"),
        (logging.WARNING, "w"),
    ]
    assert isinstance(records[0], logging.LogRecord)


@pytest.mark.topic("py.testing.logging_module")
def test_capture_logs_respects_level():
    log = logging.getLogger("t22.level")
    with ex.capture_logs("t22.level", level=logging.WARNING) as records:
        log.info("dropped")
        log.warning("kept")
    assert [r.getMessage() for r in records] == ["kept"]


@pytest.mark.topic("py.testing.logging_module", "py.testing.fixtures")
def test_capture_logs_leaves_no_trace_even_on_error():
    log = logging.getLogger("t22.cleanup")
    log.setLevel(logging.ERROR)
    handlers_before = list(log.handlers)
    with pytest.raises(KeyError), ex.capture_logs("t22.cleanup", level=logging.DEBUG):
        raise KeyError("boom")
    assert log.handlers == handlers_before
    assert log.level == logging.ERROR
    log.setLevel(logging.NOTSET)


@pytest.mark.topic("py.testing.logging", "py.testing.unit")
def test_suite_passes_on_correct_impl():
    ex.check_ingest_logging(ex.ingest)


@pytest.mark.topic("py.testing.logging", "py.testing.logging_module")
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_suite_catches_mutant(name):
    assert_suite_fails(ex.check_ingest_logging, MUTANTS[name])
