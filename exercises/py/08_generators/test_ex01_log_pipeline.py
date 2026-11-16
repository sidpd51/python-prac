import types

import pytest

from study.loader import load

ex = load(__file__, "ex01_log_pipeline")

LINES = [
    "# header",
    "INFO  service started",
    "",
    "error disk full",
    "WARN  slow request",
    "ERROR   oom killed  ",
    "INFO bye",
]


class CountingSource:
    """Iterator that records how many items were pulled from it."""

    def __init__(self, items):
        self._it = iter(items)
        self.pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._it)
        self.pulled += 1
        return item


@pytest.mark.topic("py.generators.functions", "py.generators.yield")
def test_parse_basic():
    out = list(ex.parse(LINES))
    assert out[0] == {"level": "INFO", "msg": "service started"}
    assert {"level": "ERROR", "msg": "disk full"} in out
    assert {"level": "ERROR", "msg": "oom killed"} in out
    assert len(out) == 5


@pytest.mark.topic("py.generators.lazy_evaluation")
def test_parse_is_lazy_generator():
    src = CountingSource(LINES)
    gen = ex.parse(src)
    assert src.pulled == 0, "nothing should be read before the first next()"
    assert isinstance(gen, types.GeneratorType) or hasattr(gen, "__next__")
    next(gen)
    assert src.pulled == 2  # skipped the comment, then yielded INFO


@pytest.mark.topic("py.generators.pipelines", "py.generators.chaining")
def test_pipeline():
    errors = list(ex.take(5, ex.only(ex.parse(LINES), "error")))
    assert [e["msg"] for e in errors] == ["disk full", "oom killed"]


@pytest.mark.topic("py.generators.lazy_evaluation", "py.generators.memory_efficiency")
def test_pipeline_pulls_minimum():
    src = CountingSource(LINES)
    first_error = list(ex.take(1, ex.only(ex.parse(src), "ERROR")))
    assert first_error == [{"level": "ERROR", "msg": "disk full"}]
    assert src.pulled == 4, "pipeline read more lines than needed"


@pytest.mark.topic("py.generators.exhaustion")
def test_take_does_not_overconsume_shared_iterator():
    it = iter(range(10))
    assert list(ex.take(3, it)) == [0, 1, 2]
    assert next(it) == 3, "take() pulled one item too many"
    assert list(ex.take(0, it)) == []
    assert next(it) == 4


@pytest.mark.topic("py.generators.memory_efficiency")
def test_works_on_infinite_input():
    def forever():
        while True:
            yield "ERROR boom"

    assert len(list(ex.take(3, ex.only(ex.parse(forever()), "ERROR")))) == 3
