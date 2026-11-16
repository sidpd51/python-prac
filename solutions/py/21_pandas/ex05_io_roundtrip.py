"""
Exercise 21.5 - Load, combine and export eval results               level: ★★☆
topics: pandas.io, pandas.export, pandas.dtypes, pandas.datetime, pandas.concat,
        pandas.duplicates

Problem
-------
Two systems write eval results for the same question set:
  * a CSV  with header run_id,created_at,model,question_id,correct,latency_ms
      - created_at: ISO-8601 strings with "Z", correct: "true"/"false",
        question_id: zero-padded strings like "0042" (must NOT become the int 42!)
  * a JSONL file (one JSON object per line) with the same keys, where created_at is an
    epoch timestamp in MILLISECONDS and correct is a JSON boolean.

1. load_results(csv_path, jsonl_path) -> DataFrame
   Read both (pd.read_csv / pd.read_json(lines=True)), and return one frame with:
     columns       run_id, created_at, model, question_id, correct, latency_ms (this order)
     dtypes        created_at tz-aware UTC datetime; correct bool; latency_ms float64;
                   question_id keeps its leading zeros (string)
     duplicates    the same (run_id, question_id) may appear in both files: keep the first
                   occurrence (CSV rows come first)
     order         sorted by created_at then question_id, index 0..n-1

2. export_results(df, out_dir) -> dict[str, Path]
   Write into out_dir (create it if needed):
     "csv"     results.csv     - no index column
     "jsonl"   results.jsonl   - JSON lines, one record per row, ISO-8601 dates
     "summary" summary.json    - {"accuracy": {model: fraction correct}, "rows": n}
                                 written with the json module (plain Python floats/ints)
   Return {"csv": path, "jsonl": path, "summary": path}.

3. hourly_volume(df) -> Series
   Number of results per UTC hour (from created_at), as a Series indexed by the hour start
   (tz-aware), covering every hour from the first to the last - hours with no results
   are 0. (resample("h") - lower-case: "H" raises in pandas 3.)

Interviewers look for
---------------------
* Control parsing explicitly: dtype=/converters for ids, parse dates yourself, unit="ms"
  for epochs. Silent dtype inference (ids -> int, "true" -> object) is a classic data bug.
* CSV loses dtypes (everything is text again); Parquet/Arrow keeps them. index=False on export.
* JSON lines streams well and appends easily; orient="records".
* resample needs a DatetimeIndex (or on=); empty periods appear as 0 for counts.
"""

import json
from pathlib import Path

import pandas as pd

COLUMNS = ["run_id", "created_at", "model", "question_id", "correct", "latency_ms"]


def load_results(csv_path: Path, jsonl_path: Path) -> pd.DataFrame:
    csv = pd.read_csv(csv_path, dtype={"question_id": "str", "run_id": "str"})
    csv["created_at"] = pd.to_datetime(csv["created_at"], utc=True, format="ISO8601")
    csv["correct"] = csv["correct"].astype("str").str.strip().str.lower().map(
        {"true": True, "false": False}
    )

    # dtype=False: no inference at all ("0042" stays a string, the epoch stays an int)
    jl = pd.read_json(jsonl_path, lines=True, dtype=False)
    jl["created_at"] = pd.to_datetime(jl["created_at"], unit="ms", utc=True)

    df = pd.concat([csv[COLUMNS], jl[COLUMNS]], ignore_index=True)
    df = df.astype({"correct": "bool", "latency_ms": "float64"})
    df = df.drop_duplicates(subset=["run_id", "question_id"], keep="first")
    return df.sort_values(["created_at", "question_id"], kind="stable").reset_index(drop=True)


def export_results(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / "results.csv",
        "jsonl": out_dir / "results.jsonl",
        "summary": out_dir / "summary.json",
    }
    df.to_csv(paths["csv"], index=False)
    df.to_json(paths["jsonl"], orient="records", lines=True, date_format="iso")
    accuracy = df.groupby("model")["correct"].mean()
    summary = {
        "accuracy": {str(m): float(a) for m, a in accuracy.items()},
        "rows": int(len(df)),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths


def hourly_volume(df: pd.DataFrame) -> pd.Series:
    return df.set_index("created_at").resample("h").size()
