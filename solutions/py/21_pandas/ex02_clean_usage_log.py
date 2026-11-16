"""
Exercise 21.2 - Clean a messy LLM-gateway log                       level: ★★☆
topics: pandas.cleaning, pandas.missing, pandas.duplicates, pandas.dtypes, pandas.string_ops,
        pandas.datetime

Problem
-------
The raw log comes from several services, as strings (think pd.read_csv(dtype=str)).
Columns: request_id, ts, model, prompt_tokens, completion_tokens, latency, status.

clean_usage_log(raw) -> DataFrame, WITHOUT mutating `raw`, applying these rules:

  model     strip, lower-case, turn runs of spaces/underscores into a single "-", then map
            through MODEL_ALIASES (names not in it stay as they are).
            " GPT_4o " -> "gpt-4o";  "Claude  Sonnet" -> "claude-sonnet"
  ts        ISO-8601 strings with mixed offsets ("2026-11-01T10:00:00Z",
            "2026-11-01 15:30:00+05:30") -> tz-aware UTC datetimes.
            Unparseable/missing -> drop the row.
  tokens    strings like "1,200", " 300 ", "" or missing. Strip commas/spaces, convert to
            numbers. Missing prompt_tokens -> drop the row; missing completion_tokens -> 0
            (request failed before generating). Final dtype: int64.
  latency   "850ms", "1.2s", "2 s", "900" (bare number = ms) -> float milliseconds in a
            column named latency_ms. Unparseable/missing -> NaN (keep the row).
  status    strip + lower-case; "ok", "success" and "200" -> "ok", anything else
            (including missing) -> "error". dtype: categorical with categories ["ok", "error"].
  dupes     the gateway logs retries with the same request_id: keep only the row with the
            LATEST ts for each request_id.

Output: columns exactly
    ["request_id", "ts", "model", "prompt_tokens", "completion_tokens", "latency_ms", "status"]
sorted by ts ascending (stable: ties keep input order), index reset to 0..n-1.
Use vectorised .str / pd.to_datetime / pd.to_numeric operations - no row-wise apply/iterrows.

Interviewers look for
---------------------
* Decide a policy per column (drop / fill / flag) instead of a blanket dropna().
* errors="coerce" turns bad values into NaN/NaT so you can count and handle them.
* Missing completion_tokens means 0, missing latency means "unknown" - different fills.
* Dedup needs a rule (latest wins) - drop_duplicates after sort_values, or idxmax per group.
* Everything in UTC; convert to local time only for display.
"""

import pandas as pd

MODEL_ALIASES = {
    "gpt4o": "gpt-4o",
    "gpt-4-o": "gpt-4o",
    "claude-3-5-sonnet": "claude-sonnet",
    "sonnet": "claude-sonnet",
}

OUTPUT_COLUMNS = [
    "request_id", "ts", "model", "prompt_tokens", "completion_tokens", "latency_ms", "status",
]


def _parse_tokens(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _parse_latency_ms(s: pd.Series) -> pd.Series:
    parts = s.str.strip().str.lower().str.extract(r"^([\d.]+)\s*(ms|s)?$")
    value = pd.to_numeric(parts[0], errors="coerce")
    return value.where(parts[1] != "s", value * 1000.0).astype("float64")


def clean_usage_log(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    df["model"] = (
        df["model"].str.strip().str.lower().str.replace(r"[\s_]+", "-", regex=True)
        .replace(MODEL_ALIASES)
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce", format="ISO8601")
    df["prompt_tokens"] = _parse_tokens(df["prompt_tokens"])
    df["completion_tokens"] = _parse_tokens(df["completion_tokens"]).fillna(0)
    df["latency_ms"] = _parse_latency_ms(df["latency"])
    ok = df["status"].str.strip().str.lower().isin(["ok", "success", "200"])
    df["status"] = pd.Categorical(ok.map({True: "ok", False: "error"}), categories=["ok", "error"])

    df = df.dropna(subset=["ts", "prompt_tokens"])
    df = df.astype({"prompt_tokens": "int64", "completion_tokens": "int64"})

    # latest attempt per request wins
    df = df.sort_values("ts", kind="stable").drop_duplicates("request_id", keep="last")
    return df[OUTPUT_COLUMNS].reset_index(drop=True)
