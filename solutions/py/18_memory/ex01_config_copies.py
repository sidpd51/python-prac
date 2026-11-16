"""
Exercise 18.1 - Aliasing bugs in nested configs                      level: ★☆☆
topics: memory.references, memory.copy, memory.object_model

Problem
-------
Python variables are *names bound to objects*; assignment never copies. Three small
functions about identity and copying:

1. identity_groups(items) -> list[list[int]]
   Group the INDICES of `items` that refer to the very same object (`is`, not `==`).
   Groups are ordered by first appearance; indices inside a group ascend.
       a = [1]; b = [1]
       identity_groups([a, b, a]) == [[0, 2], [1]]      # a == b, but a is not b
   Must work for unhashable items (lists, dicts).

2. make_run_configs(base, seeds) -> list[dict]
   A teammate wrote this to fan one experiment config out into one config per seed:

       def make_run_configs(base, seeds):
           configs = []
           for seed in seeds:
               cfg = base.copy()
               cfg["training"]["seed"] = seed
               configs.append(cfg)
           return configs

   Every config ends up with the LAST seed, and `base` gets mutated too. Write the
   fixed version: each returned config is fully independent (mutating any nested
   value in one never affects another, nor `base`), and `base` is left untouched.
   Assume base["training"] exists and is a dict.

3. merge_config(base, override) -> dict
   Recursive merge for layered config (defaults <- env file <- CLI flags):
     * keys only in one side are kept
     * if BOTH values are dicts -> merge them recursively
     * otherwise the override value wins (lists are replaced, not concatenated)
     * neither input is mutated, and the result shares NO mutable containers with
       either input (mutating result["a"]["b"] must not change base or override)

Interviewers look for
---------------------
* `dict.copy()` / `list(x)` / `copy.copy` are SHALLOW: the outer container is new, the
  nested objects are shared. `copy.deepcopy` recursively copies (and handles cycles via
  its memo dict).
* `is` compares identity (same object, same id() while both alive); `==` compares value.
* Deep copies cost time and memory - copy at the boundary where ownership changes, not
  everywhere "just in case".
"""

import copy
from typing import Any


def identity_groups(items: list[Any]) -> list[list[int]]:
    # key by id(): valid here because every item stays alive (referenced by `items`)
    # for the whole call, so ids cannot be reused.
    groups: dict[int, list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(id(item), []).append(i)
    return list(groups.values())


def make_run_configs(base: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    configs = []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        cfg["training"]["seed"] = seed
        configs.append(cfg)
    return configs


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
