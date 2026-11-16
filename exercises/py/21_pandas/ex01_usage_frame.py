"""
Exercise 21.1 - LLM usage DataFrame: build, inspect, select         level: ★☆☆
topics: pandas.series_dataframe, pandas.creation, pandas.inspection, pandas.selection,
        pandas.loc_iloc, pandas.sorting

Problem
-------
You get per-request usage records from an LLM gateway:
    {"request_id": "r1", "model": "gpt-4o", "prompt_tokens": 1200, "completion_tokens": 300}

1. usage_frame(records) -> DataFrame
   * index: request_id (named "request_id"); columns exactly, in order:
     ["model", "prompt_tokens", "completion_tokens", "total_tokens"]
   * total_tokens = prompt_tokens + completion_tokens (vectorised column arithmetic)
   * an empty list -> empty frame with those columns and index name

2. summarize(df) -> dict
   {"rows": int, "columns": list of column names,
    "nulls": {column: number of missing values},
    "numeric_columns": names of numeric columns, in column order}

3. heavy_requests(df, min_total) -> DataFrame
   Rows with total_tokens >= min_total, only columns ["model", "total_tokens"], sorted by
   total_tokens descending; ties keep the original row order (stable sort).

4. row_at(df, position) -> Series    the row at integer POSITION (0 = first row)
   row_for(df, label)    -> Series    the row whose index LABEL is `label`
   (With an integer index like [10, 3, 7] these differ - that's the point.)

5. flag_requests(df, labels, column="flagged") -> DataFrame
   Return a NEW frame with a boolean column that is True for the given index labels and
   False elsewhere. The input frame must not change. Use .loc[row_labels, column] = ...
   (chained assignment like df[column][labels] = True does nothing under pandas 3's
   Copy-on-Write).

Interviewers look for
---------------------
* A DataFrame is a dict of column Series sharing one index; a Series is values + index.
* .loc is label-based (slices INCLUDE the end label), .iloc is position-based
  (slices exclude the end) - and plain df[...] means columns, or boolean row masks.
* Boolean masks combine with & | ~ and parentheses.
* Copy-on-Write (default in pandas 3): every indexing result behaves like a copy; modify
  with a single .loc call on the object you own.
"""

from collections.abc import Hashable, Iterable
from typing import Any

import pandas as pd

COLUMNS = ["model", "prompt_tokens", "completion_tokens", "total_tokens"]


def usage_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    raise NotImplementedError


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    raise NotImplementedError


def heavy_requests(df: pd.DataFrame, min_total: int) -> pd.DataFrame:
    raise NotImplementedError


def row_at(df: pd.DataFrame, position: int) -> pd.Series:
    raise NotImplementedError


def row_for(df: pd.DataFrame, label: Hashable) -> pd.Series:
    raise NotImplementedError


def flag_requests(df: pd.DataFrame, labels: Iterable[Hashable], column: str = "flagged") -> pd.DataFrame:
    raise NotImplementedError
