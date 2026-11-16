"""pytest plugin: every test run on exercises automatically updates your progress.

Tag tests with the curriculum topics they exercise::

    @pytest.mark.topic("py.generators.send", "py.generators.lazy_evaluation")
    def test_something(): ...

Outcome mapping:
  * passed                        -> "passed"
  * raised NotImplementedError    -> "todo"   (you haven't attempted it; not counted as weak)
  * anything else failing         -> "failed"
"""

from __future__ import annotations

import os
from collections import defaultdict

import pytest

from .curriculum import load_curriculum
from .loader import using_solutions
from .progress import Progress

_RESULTS_KEY = pytest.StashKey[dict[str, tuple[str, list[str]]]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--no-record", action="store_true", help="study: don't write results to progress.json")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "topic(*ids): curriculum topic ids exercised by this test")
    config.stash[_RESULTS_KEY] = {}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    cur = load_curriculum()
    bad = sorted(
        {
            f"{item.nodeid}: {tid}"
            for item in items
            for m in item.iter_markers("topic")
            for tid in m.args
            if tid not in cur.topics
        }
    )
    if bad:
        raise pytest.UsageError("unknown topic ids in @pytest.mark.topic:\n  " + "\n  ".join(bad))


def _recording(config: pytest.Config) -> bool:
    return not (config.getoption("--no-record") or using_solutions() or os.environ.get("STUDY_NO_RECORD"))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    rep: pytest.TestReport = yield
    if rep.when not in ("call", "setup") or (rep.when == "setup" and not rep.failed):
        return rep
    todo = (
        rep.failed
        and call.excinfo is not None
        and call.excinfo.errisinstance(NotImplementedError)
        and not using_solutions()
    )
    if todo:
        # Show unattempted exercises as skipped "TODO" instead of noisy failures.
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, "TODO: not implemented yet")
    topics = [tid for m in item.iter_markers("topic") for tid in m.args]
    if not topics or (rep.skipped and not todo):
        return rep
    status = "todo" if todo else "passed" if rep.passed else "failed"
    item.config.stash[_RESULTS_KEY][item.nodeid] = (status, topics)
    return rep


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session) -> None:
    results = session.config.stash[_RESULTS_KEY]
    if not results or not _recording(session.config):
        return
    prog = Progress.load()
    per_topic: dict[str, list[str]] = defaultdict(list)
    for nodeid, (status, topics) in results.items():
        prog.record_test(nodeid, status, topics)
        for t in topics:
            per_topic[t].append(status)
    for topic_id, statuses in per_topic.items():
        attempted = [s for s in statuses if s != "todo"]
        if attempted:
            grade = 4 if all(s == "passed" for s in attempted) else 2
            prog.record_review(topic_id, grade, "test")
    prog.save()


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config) -> None:  # type: ignore[no-untyped-def]
    results = config.stash[_RESULTS_KEY]
    if not results:
        return
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0, "todo": 0})
    for status, topics in results.values():
        for t in topics:
            counts[t][status] += 1
    tr = terminalreporter
    tr.section("study: topic results")
    for t in sorted(counts):
        c = counts[t]
        mark = "✅" if c["failed"] == 0 and c["passed"] else ("❌" if c["failed"] else "⬜")
        tr.write_line(f"{mark} {t:<55} pass {c['passed']:>2}  fail {c['failed']:>2}  todo {c['todo']:>2}")
    if _recording(config):
        tr.write_line("(recorded to progress/progress.json — run `study report` to see mastery)")
    else:
        tr.write_line("(not recorded)")
