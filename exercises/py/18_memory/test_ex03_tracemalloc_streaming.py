import json
import tracemalloc

import pytest

from study.loader import load

ex = load(__file__, "ex03_tracemalloc_streaming")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "docs.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(20_000):
            words = " ".join(f"tok{j}" for j in range(i % 25 + 1))
            f.write(json.dumps({"id": f"doc-{i}", "text": f"{words} padding " * 3}) + "\n")
            if i % 1000 == 0:
                f.write("\n")  # blank lines must be skipped
    return path


@pytest.mark.topic("py.memory.profiling")
def test_measure_peak_returns_result_and_peak():
    result, peak = ex.measure_peak(lambda n: [0] * n, 500_000)
    assert len(result) == 500_000
    assert peak >= 500_000 * 8  # a list of 500k pointers
    assert not tracemalloc.is_tracing(), "must stop tracing it started"

    _, idle_peak = ex.measure_peak(lambda: None)
    assert idle_peak < 10_000


@pytest.mark.topic("py.memory.profiling")
def test_measure_peak_is_relative_and_counts_temporaries():
    def temp_spike():
        big = bytearray(2_000_000)
        return len(big)  # freed before returning: only the PEAK sees it

    keep = bytearray(5_000_000)  # allocated before the call: must not be counted
    result, peak = ex.measure_peak(temp_spike)
    assert result == 2_000_000
    assert 2_000_000 <= peak < 3_000_000
    del keep


@pytest.mark.topic("py.memory.profiling")
def test_measure_peak_respects_existing_session_and_errors():
    tracemalloc.start()
    try:
        ex.measure_peak(lambda: None)
        assert tracemalloc.is_tracing(), "must not stop someone else's tracing session"
    finally:
        tracemalloc.stop()

    def boom():
        raise KeyError("x")

    with pytest.raises(KeyError):
        ex.measure_peak(boom)
    assert not tracemalloc.is_tracing()


@pytest.mark.topic("py.memory.generator_memory", "py.generators.lazy_evaluation")
def test_iter_docs_is_lazy(corpus):
    it = ex.iter_docs(corpus)
    first = next(it)
    assert first["id"] == "doc-0"
    assert next(it)["id"] == "doc-1"
    it.close()


@pytest.mark.topic("py.memory.efficient_processing")
def test_corpus_stats_matches_eager(corpus):
    expected = ex.corpus_stats_eager(corpus)
    assert expected["docs"] == 20_000
    assert ex.corpus_stats(ex.iter_docs(corpus)) == expected
    assert ex.corpus_stats(iter([])) == {"docs": 0, "tokens": 0, "longest": 0}


@pytest.mark.topic("py.memory.generator_memory", "py.memory.efficient_processing",
                   "py.memory.profiling")
def test_streaming_peak_is_far_lower(corpus):
    eager_result, eager_peak = ex.measure_peak(ex.corpus_stats_eager, corpus)
    lazy_result, lazy_peak = ex.measure_peak(lambda p: ex.corpus_stats(ex.iter_docs(p)), corpus)
    assert lazy_result == eager_result
    assert eager_peak > 2_000_000
    assert lazy_peak * 10 < eager_peak, (lazy_peak, eager_peak)
    assert lazy_peak < 1_000_000


@pytest.mark.topic("py.memory.profiling", "py.memory.allocation")
def test_biggest_allocation_site():
    def allocate():
        small = [1, 2, 3]
        big = [str(i) * 3 for i in range(50_000)]  # BIG LINE
        return small, big

    filename, lineno, size = ex.biggest_allocation_site(allocate)
    assert filename == __file__
    with open(__file__, encoding="utf-8") as f:
        src = f.read().splitlines()
    assert "BIG LINE" in src[lineno - 1]
    assert size > 1_000_000
    assert not tracemalloc.is_tracing()
