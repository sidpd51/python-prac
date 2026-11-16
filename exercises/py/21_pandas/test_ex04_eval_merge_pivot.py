import numpy as np
import pandas as pd
import pytest

from study.loader import load

ex = load(__file__, "ex04_eval_merge_pivot")

RUN_A = pd.DataFrame(
    {
        "experiment_id": ["e1", "e1", "e2", "e2", "e3"],
        "metric": ["faithfulness", "relevance", "faithfulness", "relevance", "faithfulness"],
        "score": [0.9, 0.7, 0.6, 0.8, 0.5],
    }
)
RUN_B = pd.DataFrame(
    {
        "experiment_id": ["e1", "e2", "e3"],
        "metric": ["faithfulness", "faithfulness", "relevance"],
        "score": [0.7, 0.8, 0.9],
        "judge": ["gpt-4o", "gpt-4o", "gpt-4o"],  # only run B has this column
    }
)
EXPERIMENTS = pd.DataFrame(
    {
        "experiment_id": ["e1", "e2", "e3"],
        "model": ["gpt-4o", "claude-sonnet", "llama-70b"],
        "prompt_version": ["v1", "v1", "v2"],
        "temperature": [0.0, 0.2, 0.0],
    }
)


@pytest.mark.topic("py.pandas.concat")
def test_combine_runs():
    combined = ex.combine_runs({"run-a": RUN_A, "run-b": RUN_B})
    assert list(combined.columns) == ["run", "experiment_id", "metric", "score", "judge"]
    assert combined["run"].tolist() == ["run-a"] * 5 + ["run-b"] * 3
    assert list(combined.index) == list(range(8))
    assert combined["judge"].isna().sum() == 5
    with pytest.raises(ValueError):
        ex.combine_runs({})


@pytest.mark.topic("py.pandas.merge_join")
def test_attach_metadata_left_join_keeps_order():
    combined = ex.combine_runs({"run-a": RUN_A, "run-b": RUN_B})
    merged = ex.attach_metadata(combined, EXPERIMENTS)
    assert len(merged) == len(combined), "a left join on a unique key must not add rows"
    assert merged["experiment_id"].tolist() == combined["experiment_id"].tolist()
    assert merged.loc[merged["experiment_id"] == "e2", "model"].unique().tolist() == [
        "claude-sonnet"
    ]
    assert "_merge" not in merged.columns
    assert {"model", "prompt_version", "temperature", "run", "score"} <= set(merged.columns)


@pytest.mark.topic("py.pandas.merge_join")
def test_attach_metadata_validates_cardinality_and_orphans():
    dup = pd.concat([EXPERIMENTS, EXPERIMENTS.iloc[[0]].assign(model="gpt-4o-mini")])
    with pytest.raises(pd.errors.MergeError):
        ex.attach_metadata(RUN_A, dup)

    orphans = pd.concat([RUN_A, pd.DataFrame({"experiment_id": ["e9", "e7"],
                                               "metric": ["relevance"] * 2, "score": [1.0, 0.0]})])
    with pytest.raises(ValueError) as info:
        ex.attach_metadata(orphans, EXPERIMENTS)
    assert "e7" in str(info.value) and "e9" in str(info.value)


@pytest.fixture(scope="module")
def table():
    combined = ex.combine_runs({"run-a": RUN_A, "run-b": RUN_B})
    return ex.metric_table(ex.attach_metadata(combined, EXPERIMENTS))


@pytest.mark.topic("py.pandas.aggregations", "py.pandas.analysis")
def test_metric_table(table):
    expected = pd.DataFrame(
        {"faithfulness": [0.7, 0.8, 0.5], "relevance": [0.8, 0.7, 0.9]},
        index=pd.Index(["claude-sonnet", "gpt-4o", "llama-70b"], name="model"),
    )
    expected.columns.name = "metric"
    pd.testing.assert_frame_equal(table, expected, atol=1e-12)


@pytest.mark.topic("py.pandas.analysis", "py.pandas.sorting")
def test_leaderboard(table):
    board = ex.leaderboard(table, {"faithfulness": 2.0, "relevance": 1.0})
    assert list(board.columns) == ["score", "rank"]
    assert list(board.index) == ["gpt-4o", "claude-sonnet", "llama-70b"]
    np.testing.assert_allclose(board["score"], [(1.6 + 0.7) / 3, (1.4 + 0.8) / 3,
                                                (1.0 + 0.9) / 3])
    assert board["rank"].tolist() == [1, 2, 3]
    assert pd.api.types.is_integer_dtype(board["rank"])


@pytest.mark.topic("py.pandas.analysis", "py.pandas.missing")
def test_leaderboard_ties_and_missing_metrics():
    t = pd.DataFrame(
        {"acc": [0.8, 0.8, 0.9, 0.95], "cost": [1.0, 1.0, 1.0, np.nan]},
        index=pd.Index(["b-model", "a-model", "c-model", "d-model"], name="model"),
    )
    board = ex.leaderboard(t, {"acc": 1.0, "cost": 0.0})
    assert "d-model" not in board.index, "missing metric -> excluded, not scored as 0"
    assert list(board.index) == ["c-model", "a-model", "b-model"]  # tie broken by name
    assert board["rank"].tolist() == [1, 2, 2]
    only_acc = ex.leaderboard(t, {"acc": 1.0})
    assert only_acc.index[0] == "d-model"
