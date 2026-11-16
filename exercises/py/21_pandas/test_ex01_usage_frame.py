import pandas as pd
import pytest

from study.loader import load

ex = load(__file__, "ex01_usage_frame")

RECORDS = [
    {"request_id": "r1", "model": "gpt-4o", "prompt_tokens": 1200, "completion_tokens": 300},
    {"request_id": "r2", "model": "claude-sonnet", "prompt_tokens": 800, "completion_tokens": 700},
    {"request_id": "r3", "model": "gpt-4o-mini", "prompt_tokens": 50, "completion_tokens": 20},
    {"request_id": "r4", "model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": 500},
]


@pytest.fixture
def df():
    return ex.usage_frame(RECORDS)


@pytest.mark.topic("py.pandas.creation", "py.pandas.series_dataframe")
def test_usage_frame_shape_and_index(df):
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["model", "prompt_tokens", "completion_tokens", "total_tokens"]
    assert df.index.name == "request_id"
    assert list(df.index) == ["r1", "r2", "r3", "r4"]
    assert df["total_tokens"].tolist() == [1500, 1500, 70, 1500]
    assert isinstance(df["total_tokens"], pd.Series)

    empty = ex.usage_frame([])
    assert len(empty) == 0
    assert list(empty.columns) == list(df.columns) and empty.index.name == "request_id"


@pytest.mark.topic("py.pandas.inspection")
def test_summarize():
    frame = ex.usage_frame(RECORDS)
    frame.loc["r3", "model"] = None
    info = ex.summarize(frame)
    assert info["rows"] == 4
    assert info["columns"] == ["model", "prompt_tokens", "completion_tokens", "total_tokens"]
    assert info["nulls"] == {"model": 1, "prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0}
    assert info["numeric_columns"] == ["prompt_tokens", "completion_tokens", "total_tokens"]
    assert all(type(v) is int for v in info["nulls"].values())


@pytest.mark.topic("py.pandas.selection", "py.pandas.sorting")
def test_heavy_requests(df):
    heavy = ex.heavy_requests(df, 1000)
    assert list(heavy.columns) == ["model", "total_tokens"]
    assert list(heavy.index) == ["r1", "r2", "r4"], "stable: ties keep original order"
    assert ex.heavy_requests(df, 10**9).empty

    more = ex.usage_frame(RECORDS + [{"request_id": "r5", "model": "x", "prompt_tokens": 2000,
                                      "completion_tokens": 1}])
    assert list(ex.heavy_requests(more, 1000).index) == ["r5", "r1", "r2", "r4"]


@pytest.mark.topic("py.pandas.loc_iloc")
def test_row_at_vs_row_for_with_integer_index():
    frame = pd.DataFrame({"model": ["a", "b", "c"], "total_tokens": [1, 2, 3]}, index=[10, 3, 7])
    assert ex.row_at(frame, 0)["model"] == "a"
    assert ex.row_for(frame, 3)["model"] == "b"
    assert ex.row_at(frame, 1)["model"] == "b"
    assert ex.row_for(frame, 7)["model"] == "c"
    assert ex.row_at(frame, -1)["model"] == "c"
    with pytest.raises(KeyError):
        ex.row_for(frame, 0)  # there is no LABEL 0
    with pytest.raises(IndexError):
        ex.row_at(frame, 3)


@pytest.mark.topic("py.pandas.loc_iloc", "py.pandas.selection")
def test_flag_requests_returns_new_frame(df):
    before = df.copy()
    flagged = ex.flag_requests(df, ["r2", "r4"])
    assert flagged["flagged"].tolist() == [False, True, False, True]
    assert flagged["flagged"].dtype == bool
    pd.testing.assert_frame_equal(df, before)  # input untouched
    assert "flagged" not in df.columns
    assert ex.flag_requests(df, [], column="pii")["pii"].sum() == 0
