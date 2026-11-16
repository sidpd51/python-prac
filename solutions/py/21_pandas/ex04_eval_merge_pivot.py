"""
Exercise 21.4 - Eval results: concat, merge, pivot, leaderboard      level: ★★★
topics: pandas.concat, pandas.merge_join, pandas.analysis, pandas.aggregations, pandas.sorting

Problem
-------
You ran an LLM eval suite several times. Each run produced a long-format frame with columns
    experiment_id, metric, score          (one row per experiment x metric x example)
and there is an experiments table
    experiment_id, model, prompt_version, temperature

1. combine_runs(runs) -> DataFrame
   runs: {"run name": DataFrame}. Stack them into one frame with a leading "run" column
   (the dict key), then the union of all columns (a column missing in one run -> NaN).
   Index reset to 0..n-1. Use pd.concat once (not in a loop). Empty dict -> ValueError.

2. attach_metadata(results, experiments) -> DataFrame
   Left-join experiment metadata onto results on experiment_id.
     * experiments must have one row per experiment_id: enforce it with
       merge(validate="many_to_one") -> pandas raises pandas.errors.MergeError otherwise
     * results whose experiment_id has no metadata -> ValueError listing those ids
       (hint: indicator=True)
     * keep the results' row order; the "_merge" helper column must not be in the output

3. metric_table(df) -> DataFrame
   Mean score per (model, metric) as a wide table: index "model", one column per metric
   (sorted alphabetically), columns axis named "metric". Missing combos -> NaN.
   (pivot_table, or groupby + unstack)

4. leaderboard(table, weights) -> DataFrame
   weights: {metric: weight}. For each model the weighted average of those metrics
   (sum(w * score) / sum(w)). Models missing ANY weighted metric are excluded (don't
   silently treat missing as 0). Columns: "score" and "rank" (1 = best, ties share the
   lower rank: method="min", int dtype), sorted by rank then model name.
   Vectorised: multiply the table by a weights Series (aligns on column labels).

Interviewers look for
---------------------
* merge cardinality bugs (a duplicated key on the "one" side silently multiplies rows) ->
  validate=, and indicator= to find orphans; how="left" vs "inner".
* concat(keys=...) vs a loop of appends (quadratic, and DataFrame.append is gone).
* Long vs wide format: pivot_table aggregates duplicates, pivot raises on them.
* Index alignment: arithmetic between a DataFrame and a Series aligns on labels.
"""

import pandas as pd


def combine_runs(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not runs:
        raise ValueError("no runs given")
    combined = pd.concat(runs, names=["run", None])
    return combined.reset_index(level="run").reset_index(drop=True)


def attach_metadata(results: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(
        experiments, on="experiment_id", how="left", validate="many_to_one", indicator=True
    )
    orphans = merged.loc[merged["_merge"] == "left_only", "experiment_id"].unique()
    if len(orphans):
        raise ValueError(f"results reference unknown experiments: {sorted(orphans)}")
    return merged.drop(columns="_merge")


def metric_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(index="model", columns="metric", values="score", aggfunc="mean")
    return table.sort_index(axis=1)


def leaderboard(table: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    w = pd.Series(weights, dtype="float64")
    sub = table[list(w.index)].dropna()
    score = sub.mul(w, axis=1).sum(axis=1) / w.sum()
    board = pd.DataFrame({"score": score})
    board["rank"] = board["score"].rank(ascending=False, method="min").astype(int)
    return board.rename_axis("model").sort_values(["rank", "model"])
