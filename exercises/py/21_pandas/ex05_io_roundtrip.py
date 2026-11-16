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

from pathlib import Path

import pandas as pd

COLUMNS = ["run_id", "created_at", "model", "question_id", "correct", "latency_ms"]


def load_results(csv_path: Path, jsonl_path: Path) -> pd.DataFrame:
    raise NotImplementedError


def export_results(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    raise NotImplementedError


def hourly_volume(df: pd.DataFrame) -> pd.Series:
    raise NotImplementedError
