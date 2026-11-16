import json

import numpy as np
import pandas as pd
import pytest

from study.loader import load

ex = load(__file__, "ex05_io_roundtrip")

CSV = """run_id,created_at,model,question_id,correct,latency_ms
run1,2026-11-01T10:15:00Z,gpt-4o,0042,true,812.5
run1,2026-11-01T10:05:00Z,gpt-4o,0007,false,640
run3,2026-11-01T12:30:00Z,claude-sonnet,0042,TRUE,1200
"""


def epoch_ms(ts: str) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


JSONL_ROWS = [
    {"run_id": "run2", "created_at": epoch_ms("2026-11-01T10:05:00Z"), "model": "claude-sonnet",
     "question_id": "0007", "correct": True, "latency_ms": 900},
    # duplicate of a CSV row (run1, 0042) -> the CSV version wins
    {"run_id": "run1", "created_at": epoch_ms("2026-11-01T10:15:00Z"), "model": "gpt-4o",
     "question_id": "0042", "correct": False, "latency_ms": 1.0},
    {"run_id": "run2", "created_at": epoch_ms("2026-11-01T12:59:59Z"), "model": "gpt-4o",
     "question_id": "0100", "correct": True, "latency_ms": 450.25},
]


@pytest.fixture
def files(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    jsonl_path = tmp_path / "results.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in JSONL_ROWS) + "\n", encoding="utf-8")
    return csv_path, jsonl_path


@pytest.fixture
def results(files):
    return ex.load_results(*files)


@pytest.mark.topic("py.pandas.io", "py.pandas.concat", "py.pandas.duplicates")
def test_load_results_rows_and_order(results):
    assert list(results.columns) == ["run_id", "created_at", "model", "question_id", "correct",
                                     "latency_ms"]
    assert len(results) == 5, "duplicate (run_id, question_id) must be dropped"
    assert list(results.index) == list(range(5))
    # sorted by created_at, then question_id (10:05 tie: 0007/run1 and 0007/run2)
    assert results["question_id"].tolist() == ["0007", "0007", "0042", "0042", "0100"]
    assert results["run_id"].tolist() == ["run1", "run2", "run1", "run3", "run2"]
    dup = results[(results["run_id"] == "run1") & (results["question_id"] == "0042")]
    assert dup["latency_ms"].tolist() == [812.5], "the CSV row (first occurrence) wins"


@pytest.mark.topic("py.pandas.dtypes", "py.pandas.datetime", "py.pandas.io")
def test_load_results_dtypes(results):
    assert str(results["created_at"].dt.tz) == "UTC"
    assert results["created_at"].iloc[-1] == pd.Timestamp("2026-11-01T12:59:59Z")
    assert results["correct"].dtype == bool
    assert results["correct"].tolist() == [False, True, True, True, True]
    assert results["latency_ms"].dtype == np.float64
    assert pd.api.types.is_string_dtype(results["question_id"])
    assert results["question_id"].str.len().eq(4).all(), "leading zeros must survive"


@pytest.mark.topic("py.pandas.export", "py.pandas.io")
def test_export_roundtrip(results, tmp_path):
    out = tmp_path / "out" / "nested"
    paths = ex.export_results(results, out)
    assert set(paths) == {"csv", "jsonl", "summary"}
    assert all(p.exists() and p.parent == out for p in paths.values())

    back = pd.read_csv(paths["csv"], dtype={"question_id": "str"})
    assert list(back.columns) == list(results.columns), "no index column in the CSV"
    assert back["question_id"].tolist() == results["question_id"].tolist()

    lines = paths["jsonl"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["question_id"] == "0007"
    assert pd.Timestamp(first["created_at"]) == results["created_at"].iloc[0]


@pytest.mark.topic("py.pandas.export", "py.pandas.aggregations")
def test_export_summary(results, tmp_path):
    paths = ex.export_results(results, tmp_path)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["rows"] == 5
    assert summary["accuracy"] == pytest.approx({"claude-sonnet": 1.0, "gpt-4o": 2 / 3})


@pytest.mark.topic("py.pandas.datetime")
def test_hourly_volume(results):
    vol = ex.hourly_volume(results)
    assert list(vol.index) == [
        pd.Timestamp("2026-11-01 10:00", tz="UTC"),
        pd.Timestamp("2026-11-01 11:00", tz="UTC"),
        pd.Timestamp("2026-11-01 12:00", tz="UTC"),
    ]
    assert vol.tolist() == [3, 0, 2], "empty hours must be included as 0"
