# Pandas

Pandas is the tool for tabular data in Python: LLM usage logs, eval result tables, labelled datasets, feature
tables. The interview questions are rarely exotic. They check whether you think in **columns, not rows**,
know what `loc` vs `iloc` and `groupby` actually do, can reason about missing data and merges without losing
rows, and avoid `apply` loops where a vectorised expression exists. Everything here was run on pandas 3.1; where
pandas 3 differs from older tutorials (copy-on-write, `str` dtype, datetime resolution, frequency aliases) it
is called out.

## Series and DataFrame
<!-- topics: series_dataframe, creation -->

A **Series** is a 1-D labelled array: values (a NumPy array or an extension array) plus an **index**. A
**DataFrame** is a table: an index plus named columns, where **each column is a Series with its own dtype**. Think
"dict of aligned columns", not "list of rows".

```text
            columns ->   model     pt     lat           each column = one Series, one dtype
index  rid
  r1                      a        10     1.5           df["pt"]            -> Series  (int64)
  r2                      b        20     NaN           df[["pt", "lat"]]   -> DataFrame
```

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(
    {"model": ["a", "b", "a", "b", "c"], "pt": [10, 20, 30, 40, 50], "lat": [1.5, np.nan, 2.5, 3.0, 0.5]},
    index=pd.Index(["r1", "r2", "r3", "r4", "r5"], name="rid"),
)
print(df)
#     model  pt  lat
# rid
# r1      a  10  1.5
# r2      b  20  NaN
# r3      a  30  2.5
# r4      b  40  3.0
# r5      c  50  0.5
print(type(df["pt"]).__name__, type(df[["pt"]]).__name__)   # prints: Series DataFrame
```

Ways to build one: from a dict of columns (above), from a **list of dicts** (`pd.DataFrame(records)`, the natural shape
for API logs), from a NumPy array with `columns=`, or from files. Operations **align on the index** by label, not by position:

```python
import pandas as pd

a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["z", "y", "w"])
print((a + b).to_dict())   # prints: {'w': nan, 'x': nan, 'y': 22.0, 'z': 13.0}
```

> **Remember** The index is not a row number: it is a label that drives alignment, `loc`, joins and `groupby` results.

## Reading and writing: CSV, JSON, export
<!-- topics: io, export -->

`read_csv` / `read_json` parse text into a DataFrame, and **inference is the hazard**: IDs like `0042` become the
int `42`, true/false become bool, empty cells become NaN. State the types you care about:

```python
import io
import pandas as pd

csv = ("run_id,created_at,question_id,correct,latency_ms\n"
       "r1,2025-01-01T10:00:00Z,0042,true,120\n"
       "r1,2025-01-01T10:00:05Z,0043,false,\n")

naive = pd.read_csv(io.StringIO(csv))
print(naive.question_id.tolist())    # prints: [42, 43]      (leading zeros lost!)

d = pd.read_csv(io.StringIO(csv), dtype={"question_id": "string"}, parse_dates=["created_at"])
print(d.question_id.tolist())        # prints: ['0042', '0043']
print(d.dtypes["created_at"], d.dtypes["correct"], d.dtypes["latency_ms"])
# prints: datetime64[us, UTC] bool float64
```

Useful arguments: `dtype=`, `usecols=` (load only needed columns), `parse_dates=`, `na_values=`, `nrows=`, and
`chunksize=` (an iterator of DataFrames for files bigger than RAM). `read_csv(dtype=str)` then converting explicitly is the
safest pattern for messy data.

JSON: `read_json(..., lines=True)` reads JSON Lines; nested objects stay as dict cells and `pd.json_normalize(list_of_dicts)` flattens them into columns like `u.p`.

Exporting is the mirror image. Know that the **round trip must preserve what you care about**:

```python
import io
import pandas as pd

d = pd.DataFrame({"qid": ["0042"], "ok": [True], "lat": [1.5]})
buf = io.StringIO()
d.to_csv(buf, index=False)                  # index=False: do not write the row numbers
print(buf.getvalue().splitlines()[1])       # prints: 0042,True,1.5
print(d.to_json(orient="records"))          # prints: [{"qid":"0042","ok":true,"lat":1.5}]
```

`to_parquet` (typed, columnar, needs `pyarrow`) beats CSV for data you reload. For datetimes in JSON pass `date_format="iso"`; the epoch default is being changed and pandas 3 warns.

> **Gotcha** CSV loses types: reading back gives strings for IDs and no timezone. Parquet or explicit `dtype=` /
> `parse_dates=` fixes that. Always `index=False` unless the index carries data.

## Inspecting data
<!-- topics: inspection -->

First five minutes with any frame:

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"model": list("abab") + ["c"], "pt": range(10, 60, 10),
                   "lat": [1.5, np.nan, 2.5, 3.0, 0.5]}, index=["r1", "r2", "r3", "r4", "r5"])
print(df.shape)            # prints: (5, 3)
print({c: str(t) for c, t in df.dtypes.items()})   # prints: {'model': 'str', 'pt': 'int64', 'lat': 'float64'}
```

(`df` is the frame from the first section, repeated in the later examples. In pandas 3, text columns default to the dedicated `str` dtype; `df.info()` shows it as `str`.)

| Call | Tells you |
|---|---|
| `df.head()`, `df.sample(5, random_state=0)` | what the rows look like |
| `df.info()` | dtypes, non-null counts, memory |
| `df.describe()` | count/mean/std/quartiles of numeric columns (`include="all"` for text) |
| `df.isna().sum()` | missing values per column |
| `df["model"].value_counts()` | category frequencies |

`describe` counts exclude NaN, so they already reveal missingness. `info()` prints and returns `None`.

## Selecting and filtering: [], loc, iloc, masks
<!-- topics: selection, loc_iloc -->

Three ways in, with different meanings:

| Form | Selects by | Slice end |
|---|---|---|
| `df["col"]`, `df[["a", "b"]]` | column label(s) | n/a |
| `df.loc[rows, cols]` | **labels** (index values, column names), or boolean mask | **inclusive** |
| `df.iloc[rows, cols]` | **integer positions** | exclusive (like Python) |

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"model": list("abab") + ["c"], "pt": range(10, 60, 10),
                   "lat": [1.5, np.nan, 2.5, 3.0, 0.5]}, index=["r1", "r2", "r3", "r4", "r5"])
print(df.loc["r2":"r4", "pt"].tolist())     # prints: [20, 30, 40]     (label slice includes r4)
print(df.iloc[1:3, 0:2].values.tolist())    # prints: [['b', 20], ['a', 30]]   (positions 1,2 and columns 0,1)
print(df.loc[df.pt > 25, ["model", "pt"]].index.tolist())   # prints: ['r3', 'r4', 'r5']
```

**Boolean masks** are how you filter. Combine with `&`, `|`, `~` and **parentheses** (not `and`/`or`):

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"model": list("abab") + ["c"], "pt": range(10, 60, 10),
                   "lat": [1.5, np.nan, 2.5, 3.0, 0.5]}, index=["r1", "r2", "r3", "r4", "r5"])
mask = (df.pt > 15) & (df.model == "b")
print(df.loc[mask].index.tolist())          # prints: ['r2', 'r4']
```

> **Gotcha** `iloc` on a frame with an integer index vs `loc`: after filtering, the index labels are no longer `0..n-1`,
> so `df.loc[0]` and `df.iloc[0]` mean different things. Use `iloc` for "the first row", `loc` for "the row with this id".

**Copy-on-write (pandas 3 default).** Any result derived from a frame behaves as an independent copy; writing to it
never changes the parent, and chained assignment such as `df[mask]["col"] = 0` **does nothing** (it only emits a
`ChainedAssignmentError` warning):

```python
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
sub = df[df.a > 1]
sub.loc[1, "b"] = 99        # modifies only sub
print(df.b.tolist())        # prints: [4, 5, 6]

df.loc[df.a > 1, "b"] = 0   # the correct way: one .loc call with the mask and the column
print(df.b.tolist())        # prints: [4, 0, 0]
```

The rule: **one `.loc[mask, col] = value`** for conditional updates, or `df.assign(col=...)` to build a new frame. (Before pandas 3 this was the `SettingWithCopyWarning` mess.)

## Missing values
<!-- topics: missing -->

Missing data is `NaN` (float) or `pd.NA` (nullable dtypes); `None` in a numeric column becomes NaN. NaN is
**contagious in comparison and skipped in aggregations**:

```python
import numpy as np
import pandas as pd

s = pd.Series([1.0, np.nan, 3.0])
print(s.mean(), s.sum(skipna=False))   # prints: 2.0 nan      (mean skips NaN by default)
print(s.isna().tolist())               # prints: [False, True, False]
print((s == np.nan).tolist())          # prints: [False, False, False]   (NaN != NaN: use isna)
```

Handling options, from honest to risky:

```python
import numpy as np
import pandas as pd

d = pd.DataFrame({"model": ["a", "b", "a"], "lat": [1.0, np.nan, 3.0]})
print(d.dropna(subset=["lat"]).shape)                          # prints: (2, 2)  drop rows missing latency
print(d.lat.fillna(d.lat.mean()).tolist())                     # prints: [1.0, 2.0, 3.0]
print(d.assign(lat_missing=d.lat.isna()).lat_missing.tolist()) # prints: [False, True, False]
```

Choose by meaning: drop when
missing means "unusable", fill with a constant when there is a natural default (0 tokens), with a statistic when you
need a complete numeric matrix, and keep an **indicator column** when the fact of being missing is informative
(a timed-out request is not a 0 ms request!).

**Nullable dtypes** (`"Int64"`, `"boolean"`, `"string"`) hold `pd.NA` without turning integers into floats: `pd.Series([1, 2, None], dtype="Int64")` keeps `1, 2, <NA>`, whereas the default gives `1.0, 2.0, NaN`.

> **Gotcha** `fillna(0)` on latency makes failed calls look instant and drags the mean down. Decide per column,
> and say so in the code.

## Duplicates and data cleaning
<!-- topics: duplicates, cleaning -->

```python
import pandas as pd

d = pd.DataFrame({"id": ["a", "a", "b", "a"], "v": [1, 1, 2, 9]})
print(d.duplicated().tolist())                    # prints: [False, True, False, False]  (full-row duplicates)
print(d.duplicated(subset=["id"]).tolist())       # prints: [False, True, False, True]   (key duplicates)
print(d.drop_duplicates(subset=["id"], keep="first").index.tolist())   # prints: [0, 2]
print(d.drop_duplicates(subset=["id"], keep="last").index.tolist())    # prints: [2, 3]
```

`duplicated` marks every repeat **after** the first by default (`keep="first"`); `keep=False` marks all copies. After
dedup the index has gaps, so `reset_index(drop=True)` if you rely on `0..n-1`. Decide the **key** first: what makes two rows "the same
request"? Then decide which copy wins (latest by timestamp: `sort_values("ts").drop_duplicates("id", keep="last")`).

A cleaning pass is a sequence of small, named, non-mutating steps. The typical order for a raw log:

1. Copy the input (`df = raw.copy()`) so the caller's data is untouched.
2. Normalise text: `str.strip()`, `str.lower()`, map blanks to NA.
3. Convert types with errors turned into NaN (`to_numeric(errors="coerce")`, `to_datetime(errors="coerce")`).
4. Validate and drop impossible rows (negative tokens, missing id), logging how many were dropped.
5. Deduplicate on the business key, then fix the index.

## Data types, conversion and string operations
<!-- topics: dtypes, string_ops -->

Wrong dtypes cause silent errors (string `"9"` sorts after `"10"`) and wasted memory. Convert with:

```python
import pandas as pd

s = pd.Series(["10", "20", "x", None])
print(pd.to_numeric(s, errors="coerce").tolist())    # prints: [10.0, 20.0, nan, nan]
c = pd.Series(["a", "b", "a", "a"]).astype("category")
print(c.cat.categories.tolist(), c.cat.codes.tolist())  # prints: ['a', 'b'] [0, 1, 0, 0]
```

`astype` raises on bad values (good: loud), `to_numeric(errors="coerce")` turns them into NaN (convenient: *then count them*).
`category` stores each distinct string once plus small integer codes, a big memory win for low-cardinality columns such as `model`
or `status`. In pandas 3 the default text dtype is `str` (backed by a real string array, missing = NaN); older versions used `object`.

The `.str` accessor applies string methods to a whole column, propagating missing values:

```python
import pandas as pd

m = pd.Series([" GPT-4o ", "gpt-4o", "Claude", None])
clean = m.str.strip().str.lower()
print(clean.tolist())                                   # prints: ['gpt-4o', 'gpt-4o', 'claude', nan]
print(clean.str.replace("-", "_", regex=False).tolist())   # prints: ['gpt_4o', 'gpt_4o', 'claude', nan]
print(clean.str.split("-").str[0].tolist())             # prints: ['gpt', 'gpt', 'claude', nan]
print(clean.str.contains("gpt").tolist())               # prints: [True, True, False, False]
```

Prefer `.str` methods to `apply(lambda s: s.strip())`: they handle missing values and are faster. Set `regex=False`
for literal replacement (default patterns are regexes for `contains`, `extract`).

> **Gotcha** On old object-dtype columns `.str.contains` returned NaN for missing values and broke `df[mask]`; pandas 3's `str` dtype returns `False`. Pass `na=False` to be explicit.

## Dates and times
<!-- topics: datetime -->

Parse text to real datetimes early; then the `.dt` accessor, arithmetic and resampling work.

```python
import pandas as pd

raw = pd.Series(["2025-01-01T10:00:00Z", "2025-01-01T12:30:00Z", "bad"])
t = pd.to_datetime(raw, errors="coerce", utc=True)
print(str(t.dtype))                       # prints: datetime64[us, UTC]
print(t.dt.hour.tolist())                 # prints: [10.0, 12.0, nan]
print(t.iloc[1] - t.iloc[0])              # prints: 0 days 02:30:00
print(t.dt.tz_convert("Asia/Kolkata").iloc[0])   # prints: 2025-01-01 15:30:00+05:30
```

* Always attach a **timezone** (`utc=True`) for logs from several systems; naive and aware timestamps cannot be mixed.
* `errors="coerce"` gives `NaT` (the datetime NaN) for unparseable values.
* pandas 3 infers microsecond (`us`) resolution for parsed strings; older versions used nanoseconds (`ns`).
**Time series bucketing** is `resample` on a datetime index (the temporal `groupby`), e.g. hourly request counts or daily cost:

```python
import pandas as pd

idx = pd.to_datetime(["2025-01-01 00:10", "2025-01-01 00:50", "2025-01-01 01:20", "2025-01-01 02:05"])
s = pd.Series([1, 2, 3, 4], index=idx)
print(s.resample("h").sum().tolist())   # prints: [3, 3, 4]
```

> **Gotcha** Frequency aliases were renamed: `"H"` is now `"h"`, month-end `"M"` is `"ME"`, `"T"` is `"min"`. In
> pandas 3 the old ones raise `ValueError: Invalid frequency: H ... Did you mean h?`. Tutorials written before 2024 use the old
> names.

## Sorting
<!-- topics: sorting -->

```python
import pandas as pd

r = pd.DataFrame({"model": ["b", "a", "b", "a"], "score": [0.6, 0.8, 0.9, 0.7]})
print(r.sort_values(["model", "score"], ascending=[True, False]).index.tolist())   # prints: [1, 3, 2, 0]
print(r.score.rank(ascending=False).tolist())         # prints: [4.0, 2.0, 1.0, 3.0]
```

`sort_values` takes several keys with a per-key `ascending` list; NaN sorts last. Pass `kind="stable"` when you sort twice to break ties. `nlargest(n, col)` beats a full sort for top-n; `rank` gives positions (ties average unless `method="min"`). Sorting returns a **new** frame with a permuted index: `reset_index(drop=True)` to renumber.

## GroupBy and aggregations
<!-- topics: groupby, aggregations -->

`groupby` is **split-apply-combine**: split the rows into groups by key, apply a function per group, combine the results.

```text
model  pt                  groups              aggregate         combined
  a    10        split      a: [10, 30]    ->   sum = 40
  b    20      ---------->  b: [20, 40]    ->   sum = 60    ---> a 40 / b 60 / c 50
  a    30                   c: [50]        ->   sum = 50
  b    40
  c    50
```

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"model": list("abab") + ["c"], "pt": range(10, 60, 10),
                   "lat": [1.5, np.nan, 2.5, 3.0, 0.5]}, index=["r1", "r2", "r3", "r4", "r5"])
out = df.groupby("model").agg(n=("pt", "size"), total=("pt", "sum"), avg_lat=("lat", "mean"))
print(out)
#        n  total  avg_lat
# model
# a      2     40      2.0
# b      2     60      3.0
# c      1     50      0.5
print(df.groupby("model")["pt"].transform("sum").tolist())    # prints: [40, 60, 40, 60, 50]
print(df.groupby("model").filter(lambda g: len(g) > 1).index.tolist())   # prints: ['r1', 'r2', 'r3', 'r4']
```

**Named aggregation** (`new_name=("column", "func")`) is the clearest form. The key becomes the **index** of the result; use
`as_index=False` or `reset_index()` to get it back as a column. Three behaviours to keep distinct:

| Method | Output shape | Example |
|---|---|---|
| `agg` / `sum` / `mean` | one row per group | totals per model |
| `transform` | **same length as input**, aligned to original rows | `groupby("model").pt.transform("sum")` -> share of group total |
| `apply` / `filter` | custom: any frame / keep or drop whole groups | `filter(lambda g: len(g) > 1)` |

Group by several keys for a MultiIndex; `.size()` counts rows, `.count()` non-null values; NaN keys are **dropped by default** (`dropna=False` keeps them).

> **Interview** "Difference between `agg` and `transform`?" `agg` reduces each group to one row; `transform` returns a
> result the same shape as the input, so you can assign it back as a column (z-scores per group, group share, group mean fill).

> **AI use** Per-model cost and p95 latency: `usage.groupby("model").agg(requests=("id", "size"), cost=("cost", "sum"),
> p95=("latency_ms", lambda s: s.quantile(0.95)))`. Error rate per model is the mean of a boolean column (`status == "error"`).

## Merging, joining and concatenating
<!-- topics: merge_join, concat -->

`merge` is a SQL join on column values; `join` is the same keyed on the index; `concat` stacks frames (rows or columns) without matching keys.

```python
import pandas as pd

left = pd.DataFrame({"k": [1, 2, 3], "x": ["a", "b", "c"]})
right = pd.DataFrame({"k": [2, 3, 4], "y": [20, 30, 40]})
print(left.merge(right, on="k", how="inner").k.tolist())   # prints: [2, 3]
print(left.merge(right, on="k", how="left"))
#    k  x     y
# 0  1  a   NaN
# 1  2  b  20.0
# 2  3  c  30.0
print(left.merge(right, on="k", how="outer", indicator=True)["_merge"].tolist())
# prints: ['left_only', 'both', 'both', 'right_only']
```

| `how` | Keeps |
|---|---|
| `inner` | keys in both (default) |
| `left` | all left rows, NaN where no match (use for "enrich my main table") |
| `right` | all right rows |
| `outer` | everything; `indicator=True` says where each row came from |

The trap is **duplicate keys**: a many-to-many match multiplies rows. Guard with `validate=`:

```python
import pandas as pd

left = pd.DataFrame({"k": [1, 2, 3], "x": ["a", "b", "c"]})
dup = pd.DataFrame({"k": [1, 1], "y": [1, 2]})
print(len(left.merge(dup, on="k", how="left")))   # prints: 4   (key 1 matched twice: 3 rows became 4)
try:
    left.merge(dup, on="k", validate="one_to_one")
except pd.errors.MergeError as e:
    print("MergeError")                           # prints: MergeError
```

Also: `left_on=`/`right_on=` for different key names, `suffixes=` for overlapping columns, and a **dtype mismatch** (`"1"` vs `1`) matches nothing or raises. A left join that introduces NaN turns integer columns into floats (see `y`).

`concat` combines **by position/label**, not by key:

```python
import pandas as pd

a = pd.DataFrame({"k": [1, 2]})
print(pd.concat([a, a]).index.tolist())                      # prints: [0, 1, 0, 1]   (duplicate labels!)
print(pd.concat([a, a], ignore_index=True).index.tolist())   # prints: [0, 1, 2, 3]
print(pd.concat([a, a], keys=["run1", "run2"]).index.tolist())
# prints: [('run1', 0), ('run1', 1), ('run2', 0), ('run2', 1)]
```

`axis=0` stacks rows (columns align by name, gaps become NaN); `axis=1` sets frames side by side on the index. Call `concat` **once** on a list of frames; appending in a loop is O(n^2) and `DataFrame.append` is gone.

> **AI use** Eval tables: `concat` the per-run result frames (with `keys=` or a `run_id` column), `merge` with the experiments
> table on `experiment_id` (many results to one experiment: `validate="many_to_one"`), then `pivot_table` to a leaderboard.

## Apply, map and vectorised alternatives
<!-- topics: apply_map -->

`map` transforms a **Series** elementwise (with a function or a dict); `apply` runs a function over each column or each row (`axis=1`);
`DataFrame.map` is elementwise over a whole frame (it replaced `applymap`).

```python
import pandas as pd

s = pd.Series([1, 2, 3])
print(s.map({1: "a", 2: "b"}).tolist())       # prints: ['a', 'b', nan]    (unmapped keys become NaN)

d = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
print(d.apply(lambda row: row.a + row.b, axis=1).tolist())   # prints: [4, 6]   (a Python call per row)
print((d.a + d.b).tolist())                                   # prints: [4, 6]   (vectorised, same result)
```

**Prefer vectorised operations**: `df.a + df.b`, `np.where`, `.str`/`.dt` accessors, `groupby.transform`. `apply(axis=1)` calls Python once per row and is typically 10-100x slower. Use `apply` only with no vectorised form, and consider `map` with a dict for lookups (e.g. price per model).

> **Interview** "Why is `apply(axis=1)` slow?" It is a Python-level loop that materialises a Series per row. Vectorise with
> column arithmetic, `map` with a dict, `np.where`, or `groupby`. Mention that for truly custom row logic you drop to a list comprehension over
> `zip(df.a, df.b)`, which still beats `apply(axis=1)`.

## Basic analysis: pivots, counts and leaderboards
<!-- topics: analysis -->

The analysis toolkit answers "which is best, how many, how is it distributed":

```python
import pandas as pd

r = pd.DataFrame({
    "model": ["a", "a", "b", "b"],
    "metric": ["acc", "f1", "acc", "f1"],
    "score": [0.8, 0.7, 0.9, 0.6],
})
print(r.pivot(index="model", columns="metric", values="score"))
# metric  acc   f1
# model
# a       0.8  0.7
# b       0.9  0.6
```

* `pivot` reshapes long to wide (**errors on duplicate index/column pairs**); `pivot_table` does the same with an aggregation
  (`aggfunc="mean"`) for duplicates; `melt` is the reverse (wide to long). Keep data **long** for storage and groupby, **wide** for display.
* A leaderboard is `groupby(...).agg(...)` then `sort_values(..., ascending=False)` then `reset_index()`.

```python
import pandas as pd

runs = pd.DataFrame({"model": ["a", "a", "b", "b"], "ok": [True, False, True, True],
                     "latency_ms": [100.0, 300.0, 200.0, 220.0]})
board = (runs.groupby("model")
             .agg(accuracy=("ok", "mean"), p50=("latency_ms", "median"))
             .sort_values("accuracy", ascending=False)
             .reset_index())
print(board.values.tolist())   # prints: [['b', 1.0, 210.0], ['a', 0.5, 200.0]]
```

> **Gotcha** Averages of averages mislead: the mean of two group means is not the overall mean when groups differ in size. Aggregate
> from the row level, and report counts next to percentages. Also `std` here is the sample std (`ddof=1`), unlike NumPy's default.

## Quick recap
- A DataFrame is aligned, typed columns plus an index; operations align on labels; think columns, not rows.
- `loc` = labels (end inclusive) or a mask; `iloc` = positions (end exclusive); combine masks with `&`, `|`, `~` and parentheses.
- pandas 3 is copy-on-write: derived frames are independent; chained assignment does nothing. Use `df.loc[mask, "col"] = v`.
- Missing: `isna`, `dropna(subset=)`, `fillna`, indicator columns; NaN != NaN, aggregations skip it; fill choices change meaning.
- Duplicates: define the key, choose `keep`, reset the index; cleaning is a sequence of small non-mutating steps.
- Convert dtypes explicitly (`to_numeric`/`to_datetime(errors="coerce")`, `category`, `Int64`); `.str` and `.dt` are vectorised.
- Datetimes: parse once, use UTC, `resample` for time buckets; `"H"` is `"h"`, `"M"` is `"ME"`; pandas 3 parses to `us` resolution.
- `groupby` = split, apply, combine; `agg` reduces, `transform` keeps shape, `filter` drops groups; key becomes the index.
- `merge` how: inner/left/right/outer; duplicate keys multiply rows, so use `validate=` and `indicator=`; `concat` once, `ignore_index=True`.
- Prefer vectorised code over `apply(axis=1)`; `map` with a dict for lookups; `pivot` long to wide.
- Read CSV with explicit `dtype`/`parse_dates` (IDs with leading zeros!); export with `index=False`; Parquet preserves types.
