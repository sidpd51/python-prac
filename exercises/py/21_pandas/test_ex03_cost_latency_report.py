import numpy as np
import pandas as pd
import pytest

from study.loader import load

ex = load(__file__, "ex03_cost_latency_report")

PRICING = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6), "claude-sonnet": (3.0, 15.0)}

LOG = pd.DataFrame(
    {
        "request_id": [f"r{i}" for i in range(8)],
        "model": ["gpt-4o", "gpt-4o", "gpt-4o-mini", "claude-sonnet", "gpt-4o", "gpt-4o-mini",
                  "claude-sonnet", "gpt-4o-mini"],
        "prompt_tokens": [1000, 2000, 4000, 1000, 500, 1000, 3000, 2000],
        "completion_tokens": [200, 100, 1000, 500, 0, 0, 1000, 500],
        "latency_ms": [800.0, 1200.0, 300.0, 2500.0, np.nan, 250.0, 4000.0, 499.9],
        "status": ["ok", "ok", "ok", "error", "error", "ok", "ok", "ok"],
    }
)


@pytest.fixture(scope="module")
def costed():
    return ex.add_cost(LOG, PRICING)


@pytest.mark.topic("py.pandas.apply_map")
def test_add_cost_vectorised(costed):
    expected = [
        1000 / 1000 * 2.5 + 200 / 1000 * 10.0,
        2000 / 1000 * 2.5 + 100 / 1000 * 10.0,
        4000 / 1000 * 0.15 + 1000 / 1000 * 0.6,
        1000 / 1000 * 3.0 + 500 / 1000 * 15.0,
        500 / 1000 * 2.5,
        1000 / 1000 * 0.15,
        3000 / 1000 * 3.0 + 1000 / 1000 * 15.0,
        2000 / 1000 * 0.15 + 500 / 1000 * 0.6,
    ]
    np.testing.assert_allclose(costed["cost_usd"], expected)
    assert costed["cost_usd"].dtype == np.float64
    assert "cost_usd" not in LOG.columns, "return a copy, don't mutate the input"


@pytest.mark.topic("py.pandas.apply_map")
def test_add_cost_unknown_models():
    bad = pd.concat([LOG, LOG.iloc[[0]].assign(model="llama-70b"),
                     LOG.iloc[[0]].assign(model="aaa-model")])
    with pytest.raises(KeyError) as info:
        ex.add_cost(bad, PRICING)
    msg = str(info.value)
    assert "llama-70b" in msg and "aaa-model" in msg
    assert msg.index("aaa-model") < msg.index("llama-70b"), "list unknown models sorted"


@pytest.mark.topic("py.pandas.groupby", "py.pandas.aggregations", "py.pandas.sorting")
def test_model_report(costed):
    report = ex.model_report(costed)
    assert report.index.name == "model"
    assert list(report.columns) == ["requests", "error_rate", "total_cost", "avg_prompt",
                                    "p50_latency_ms", "p95_latency_ms"]
    assert list(report.index) == ["claude-sonnet", "gpt-4o", "gpt-4o-mini"]  # by total_cost desc
    assert report["requests"].tolist() == [2, 3, 3]
    np.testing.assert_allclose(report["error_rate"], [0.5, 1 / 3, 0.0])
    np.testing.assert_allclose(report.loc["gpt-4o", "total_cost"], 4.5 + 6.0 + 1.25)
    np.testing.assert_allclose(report.loc["gpt-4o-mini", "avg_prompt"], 7000 / 3)


@pytest.mark.topic("py.pandas.aggregations", "py.pandas.analysis")
def test_latency_percentiles_ignore_nan(costed):
    report = ex.model_report(costed)
    gpt = report.loc["gpt-4o"]  # latencies 800, 1200, NaN
    assert gpt["p50_latency_ms"] == pytest.approx(1000.0)
    assert gpt["p95_latency_ms"] == pytest.approx(np.percentile([800, 1200], 95))
    mini = report.loc["gpt-4o-mini"]
    assert mini["p95_latency_ms"] == pytest.approx(np.percentile([300, 250, 499.9], 95))


@pytest.mark.topic("py.pandas.groupby", "py.pandas.analysis")
def test_most_expensive_request(costed):
    top = ex.most_expensive_request(costed)
    assert list(top.index) == ["claude-sonnet", "gpt-4o", "gpt-4o-mini"]
    assert top["request_id"].tolist() == ["r6", "r1", "r2"]
    assert "model" not in top.columns and "cost_usd" in top.columns


@pytest.mark.topic("py.pandas.apply_map", "py.pandas.analysis")
def test_latency_buckets():
    counts = ex.latency_buckets(LOG)
    assert [str(b) for b in counts.index] == ["<500ms", "500ms-2s", ">=2s"]
    assert counts.tolist() == [3, 2, 2]  # 499.9 is <500, NaN not counted
    fast_only = ex.latency_buckets(LOG[LOG["latency_ms"] < 400])
    assert fast_only.tolist() == [2, 0, 0], "empty buckets must still appear"
