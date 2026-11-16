import logging
import traceback

import pytest

from study.loader import load

ex = load(__file__, "ex04_error_context")

LOGGER = logging.getLogger("study.tests.ingest")


@pytest.mark.topic("py.exceptions.contextual")
def test_error_context_adds_note_and_reraises_same_object():
    original = KeyError("text")
    with pytest.raises(KeyError) as info, ex.error_context(doc_id="a1", stage="embed"):
        raise original
    assert info.value is original
    assert original.__notes__ == ["context: doc_id=a1, stage=embed"]
    # traceback still points at the raise inside the block
    frames = traceback.extract_tb(original.__traceback__)
    assert any("raise original" in (f.line or "") for f in frames)


@pytest.mark.topic("py.exceptions.contextual")
def test_error_context_nested_and_success():
    with ex.error_context(doc_id="x"):
        value = 1
    assert value == 1
    with (
        pytest.raises(ValueError) as info,
        ex.error_context(job="nightly"),
        ex.error_context(doc_id="d7"),
    ):
        raise ValueError("bad chunk")
    assert info.value.__notes__ == ["context: doc_id=d7", "context: job=nightly"]


@pytest.mark.topic("py.exceptions.contextual", "py.exceptions.hierarchy")
def test_error_context_ignores_base_exceptions():
    err = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt), ex.error_context(doc_id="x"):
        raise err
    assert not hasattr(err, "__notes__")


def _process(doc):
    if doc["id"] == "bad":
        raise KeyError("text")
    if doc["id"] == "worse":
        raise ValueError("empty document")


DOCS = [{"id": "a"}, {"id": "bad"}, {"id": "b"}, {"id": "worse"}]


@pytest.mark.topic("py.exceptions.logging", "py.exceptions.best_practices")
def test_ingest_collects_and_logs_with_traceback(caplog):
    caplog.set_level(logging.ERROR, logger=LOGGER.name)
    report = ex.ingest(DOCS, _process, logger=LOGGER)
    assert report.ok == ["a", "b"]
    assert report.failed == {"bad": "KeyError: 'text'", "worse": "ValueError: empty document"}
    recs = [r for r in caplog.records if r.name == LOGGER.name]
    assert len(recs) == 2
    assert all(r.levelno == logging.ERROR for r in recs)
    assert all(r.exc_info is not None for r in recs), "use logger.exception (traceback!)"
    assert [r.doc_id for r in recs] == ["bad", "worse"], "pass extra={'doc_id': ...}"
    assert "bad" in recs[0].getMessage()


@pytest.mark.topic("py.exceptions.contextual", "py.exceptions.best_practices")
def test_ingest_strict_reraises_with_note_and_does_not_log(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER.name)
    with pytest.raises(KeyError) as info:
        ex.ingest(DOCS, _process, logger=LOGGER, strict=True)
    assert info.value.__notes__ == ["doc_id=bad"]
    assert [r for r in caplog.records if r.name == LOGGER.name] == []
    frames = traceback.extract_tb(info.value.__traceback__)
    assert frames[-1].name == "_process", "bare `raise` keeps the original traceback"


@pytest.mark.topic("py.exceptions.best_practices")
def test_ingest_never_catches_keyboard_interrupt():
    def process(doc):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ex.ingest(DOCS, process, logger=LOGGER)


def _chain():
    try:
        try:
            {}["text"]
        except KeyError as e:
            raise ValueError("bad doc") from e
    except ValueError:
        raise RuntimeError("ingest failed")  # noqa: B904 - implicit context on purpose


@pytest.mark.topic("py.exceptions.chaining")
def test_root_cause_and_chain():
    with pytest.raises(RuntimeError) as info:
        _chain()
    err = info.value
    assert isinstance(ex.root_cause(err), KeyError)
    assert ex.error_chain(err) == ("RuntimeError: ingest failed <- ValueError: bad doc <- KeyError: 'text'")


@pytest.mark.topic("py.exceptions.chaining")
def test_chain_respects_from_none_and_single():
    try:
        try:
            int("x")
        except ValueError:
            raise LookupError("clean") from None
    except LookupError as e:
        err = e
    assert ex.root_cause(err) is err
    assert ex.error_chain(err) == "LookupError: clean"
    lone = ValueError("v")
    assert ex.root_cause(lone) is lone
