"""
Exercise 21.3 - Per-model cost & latency report                     level: ★★☆
topics: pandas.groupby, pandas.aggregations, pandas.apply_map, pandas.analysis, pandas.sorting

Problem
-------
Input: a cleaned usage log with columns
    request_id, model, prompt_tokens, completion_tokens, latency_ms (float, may be NaN),
    status ("ok" / "error")

1. add_cost(df, pricing) -> DataFrame
   pricing: {model: (usd_per_1k_input_tokens, usd_per_1k_output_tokens)}
   Return a copy with a float column cost_usd = prompt/1000*in_price + completion/1000*out_price.
   Vectorised: map the model column to prices (Series.map) - no apply(axis=1), no loops
   over rows. Models missing from pricing -> KeyError whose message lists them (sorted);
   silently producing NaN cost is how you under-report spend.

2. model_report(df) -> DataFrame indexed by model (index name "model"), columns in order:
     requests        number of rows
     error_rate      fraction of rows with status == "error"
     total_cost      sum of cost_usd
     avg_prompt      mean prompt_tokens
     p50_latency_ms  median latency (NaN latencies ignored)
     p95_latency_ms  95th percentile latency (pandas default linear interpolation)
   sorted by total_cost descending. Use groupby + named aggregation.

3. most_expensive_request(df) -> DataFrame
   For each model the single row with the highest cost_usd: the "model" column becomes the
   index, all other columns are kept; models in alphabetical order. (groupby + idxmax,
   then .loc.)

4. latency_buckets(df) -> Series
   Count requests per latency bucket: "<500ms" [0, 500), "500ms-2s" [500, 2000),
   ">=2s" [2000, inf). Index: exactly those three labels in that order (zero counts
   included); NaN latencies are not counted. Use pd.cut (right=False) - not apply.

Interviewers look for
---------------------
* Named aggregation: df.groupby("model").agg(requests=("request_id", "size"), ...).
* Boolean mean = rate: (status == "error").mean().
* p95 not mean for latency SLOs (tail latency).
* apply(axis=1) runs a Python function per row (slow); map/merge/arithmetic are vectorised.
"""

import numpy as np
import pandas as pd

BUCKETS = ["<500ms", "500ms-2s", ">=2s"]


def add_cost(df: pd.DataFrame, pricing: dict[str, tuple[float, float]]) -> pd.DataFrame:
    unknown = sorted(set(df["model"]) - set(pricing))
    if unknown:
        raise KeyError(f"no pricing for models: {unknown}")
    in_price = df["model"].map({m: p[0] for m, p in pricing.items()})
    out_price = df["model"].map({m: p[1] for m, p in pricing.items()})
    out = df.copy()
    cost = df["prompt_tokens"] / 1000 * in_price + df["completion_tokens"] / 1000 * out_price
    out["cost_usd"] = cost.astype("float64")
    return out


def model_report(df: pd.DataFrame) -> pd.DataFrame:
    report = (
        df.assign(is_error=df["status"] == "error")
        .groupby("model")
        .agg(
            requests=("request_id", "size"),
            error_rate=("is_error", "mean"),
            total_cost=("cost_usd", "sum"),
            avg_prompt=("prompt_tokens", "mean"),
            p50_latency_ms=("latency_ms", "median"),
            p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
        )
    )
    return report.sort_values("total_cost", ascending=False)


def most_expensive_request(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.groupby("model")["cost_usd"].idxmax()
    return df.loc[rows.values].set_index("model").sort_index()


def latency_buckets(df: pd.DataFrame) -> pd.Series:
    buckets = pd.cut(df["latency_ms"], bins=[0, 500, 2000, np.inf], right=False, labels=BUCKETS)
    return buckets.value_counts(sort=False).reindex(BUCKETS, fill_value=0)
