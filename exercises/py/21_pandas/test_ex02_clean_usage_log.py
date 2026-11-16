import numpy as np
import pandas as pd
import pytest

from study.loader import load

ex = load(__file__, "ex02_clean_usage_log")

RAW = pd.DataFrame(
    {
        "request_id": ["r1", "r2", "r3", "r2", "r4", "r5", "r6", "r7"],
        "ts": [
            "2026-11-01T10:00:00Z",
            "2026-11-01T09:00:00Z",
            "2026-11-01 15:30:00+05:30",  # = 10:00 UTC
            "2026-11-01T09:05:00Z",  # retry of r2 (later) -> wins
            "not a timestamp",  # dropped
            "2026-11-01T08:00:00Z",
            "2026-11-01T11:00:00Z",
            None,  # dropped
        ],
        "model": [" GPT_4o ", "Claude  Sonnet", "gpt4o", "claude-sonnet", "gpt-4o",
                  "SONNET", "Mistral_Large", "gpt-4o"],
        "prompt_tokens": ["1,200", "800", " 300 ", "810", "5", "", "40", "1"],
        "completion_tokens": ["300", None, "15", "120", "1", "9", "", "1"],
        "latency": ["850ms", "1.2s", "2 s", "900", "10ms", "fast", None, "1ms"],
        "status": ["OK", "error", " success ", "200", "ok", "ok", None, "ok"],
    },
    dtype="str",
)


@pytest.fixture(scope="module")
def clean():
    return ex.clean_usage_log(RAW)


@pytest.mark.topic("py.pandas.cleaning", "py.pandas.duplicates", "py.pandas.datetime")
def test_rows_kept_and_sorted(clean):
    # r4 (bad ts), r7 (no ts) and r5 (no prompt_tokens) dropped; r2 deduplicated
    assert list(clean.columns) == ["request_id", "ts", "model", "prompt_tokens",
                                   "completion_tokens", "latency_ms", "status"]
    assert clean["request_id"].tolist() == ["r2", "r1", "r3", "r6"]
    assert list(clean.index) == [0, 1, 2, 3]
    assert clean["ts"].is_monotonic_increasing


@pytest.mark.topic("py.pandas.duplicates")
def test_latest_retry_wins(clean):
    r2 = clean[clean["request_id"] == "r2"].iloc[0]
    assert r2["prompt_tokens"] == 810
    assert r2["completion_tokens"] == 120
    assert r2["ts"] == pd.Timestamp("2026-11-01T09:05:00Z")


@pytest.mark.topic("py.pandas.datetime", "py.pandas.dtypes")
def test_timestamps_are_utc(clean):
    assert pd.api.types.is_datetime64_any_dtype(clean["ts"])
    assert str(clean["ts"].dt.tz) == "UTC"
    r3 = clean.loc[clean["request_id"] == "r3", "ts"].iloc[0]
    assert r3 == pd.Timestamp("2026-11-01 10:00:00", tz="UTC")


@pytest.mark.topic("py.pandas.string_ops", "py.pandas.cleaning")
def test_model_names_normalised(clean):
    assert clean["model"].tolist() == ["claude-sonnet", "gpt-4o", "gpt-4o", "mistral-large"]


@pytest.mark.topic("py.pandas.missing", "py.pandas.dtypes")
def test_token_columns(clean):
    assert clean["prompt_tokens"].tolist() == [810, 1200, 300, 40]
    assert clean["completion_tokens"].tolist() == [120, 300, 15, 0]  # missing -> 0
    assert clean["prompt_tokens"].dtype == np.int64
    assert clean["completion_tokens"].dtype == np.int64


@pytest.mark.topic("py.pandas.string_ops", "py.pandas.missing")
def test_latency_ms(clean):
    assert clean["latency_ms"].dtype == np.float64
    np.testing.assert_allclose(clean["latency_ms"].iloc[:3], [900.0, 850.0, 2000.0])
    assert np.isnan(clean["latency_ms"].iloc[3]), "missing latency stays NaN (row kept)"
    one = RAW.iloc[[1]]  # "1.2s"
    assert ex.clean_usage_log(one)["latency_ms"].iloc[0] == pytest.approx(1200.0)


@pytest.mark.topic("py.pandas.dtypes", "py.pandas.cleaning")
def test_status_categorical(clean):
    assert isinstance(clean["status"].dtype, pd.CategoricalDtype)
    assert list(clean["status"].cat.categories) == ["ok", "error"]
    assert clean["status"].tolist() == ["ok", "ok", "ok", "error"]  # missing -> error


@pytest.mark.topic("py.pandas.cleaning")
def test_input_not_mutated():
    raw = RAW.copy()
    ex.clean_usage_log(raw)
    pd.testing.assert_frame_equal(raw, RAW)
