'''
Exercise 01.3 - Fix the aliasing / identity bugs                    level: ★★☆
topics: fundamentals.mutability, fundamentals.identity_equality, fundamentals.none,
        fundamentals.operators, fundamentals.nested_structures

Problem
-------
An eval harness has four helpers that "work on my machine". Each has a bug rooted in how
Python names, objects and mutability work. Write the FIXED versions (same names/behaviour
as described, minus the bug).

```python
def init_scores(n_models, n_tasks):
    return [[0.0] * n_tasks] * n_models                 # (1)

def with_overrides(base, **overrides):                  # with_overrides(cfg, llm__temperature=0.2)
    cfg = base.copy()                                   # (2)
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        cfg[section][key] = value
    return cfg

def find_doc(docs, doc_id):
    for doc in docs:
        if doc["id"] is doc_id:                         # (3)
            return doc
    return None

def add_tag(tags, tag):
    if tag not in tags:
        tags += [tag]                                   # (4)
    return tags

def is_missing(value):
    return value == None                                # (5)
```

Required behaviour
1. init_scores(n_models, n_tasks) -> n_models x n_tasks grid of 0.0 whose rows are
   independent lists (writing grid[0][1] must not change any other row).
2. with_overrides(base, **overrides) -> new config dict; `base` (including its nested
   dicts) must be left completely unchanged. A missing section is created.
3. find_doc(docs, doc_id) -> the first doc whose "id" equals doc_id, else None - also when
   the id string was built at runtime (read from a file, concatenated, json-decoded...).
4. add_tag(tags, tag) -> a NEW list with tag appended if not already present; the caller's
   list is never mutated.
5. is_missing(value) -> True only for None, even for objects whose __eq__ claims to equal
   everything (mocks, NumPy arrays, ORM columns).

Interviewers look for
---------------------
* `[row] * n` repeats a *reference* to one list; use a comprehension to build n lists.
* dict.copy() / list.copy() / slicing are SHALLOW: nested containers are shared.
  copy.deepcopy, or rebuild the nested level you change.
* `is` compares identity; equal strings are not guaranteed to be the same object
  (interning of literals is a CPython implementation detail). Use `==` for values,
  `is` only for singletons (None, True/False, sentinels).
* `a += b` on a list mutates in place (list.__iadd__ == extend) - every alias sees it;
  `a = a + b` rebinds to a new list. On a tuple, += always creates a new object.
* `x == None` dispatches to x.__eq__ which can return anything; `x is None` can't be fooled.
'''

import copy
from typing import Any


def init_scores(n_models: int, n_tasks: int) -> list[list[float]]:
    return [[0.0] * n_tasks for _ in range(n_models)]


def with_overrides(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    for dotted, value in overrides.items():
        section, key = dotted.split("__", 1)
        cfg.setdefault(section, {})[key] = value
    return cfg


def find_doc(docs: list[dict[str, Any]], doc_id: str) -> dict[str, Any] | None:
    for doc in docs:
        if doc["id"] == doc_id:
            return doc
    return None


def add_tag(tags: list[str], tag: str) -> list[str]:
    return list(tags) if tag in tags else [*tags, tag]


def is_missing(value: object) -> bool:
    return value is None
