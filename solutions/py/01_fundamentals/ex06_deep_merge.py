"""
Exercise 01.6 - Layered config: deep merge and diff                 level: ★★★
topics: fundamentals.nested_structures, fundamentals.mutability, fundamentals.dict_comprehensions,
        fundamentals.none, fundamentals.identity_equality

Problem
-------
An LLM service loads config in layers: defaults.yaml <- env/prod.yaml <- CLI overrides.

1. deep_merge(base: dict, override: dict) -> dict
     * returns a NEW dict; neither input is mutated
     * keys only in one side are kept; when both sides hold a dict -> merge recursively
     * any other value in `override` replaces the base value (lists are REPLACED, not
       concatenated; a dict replaces a scalar and vice versa)
     * an override value of None DELETES the key from the result (Helm-style "unset")
     * the result must share no mutable objects with the inputs: mutating anything inside
       the result (nested dicts, lists) must not affect base or override

2. merge_all(*configs: dict) -> dict
     * deep_merge left-to-right (later layers win); no configs -> {}

3. config_diff(old: dict, new: dict) -> dict[str, tuple[Any, Any]]
     * dotted path -> (old_value, new_value) for every leaf that differs
     * recurse only where BOTH sides hold a dict; everything else is compared with ==
     * a key only in `new` -> (MISSING, value); only in `old` -> (value, MISSING)
     * MISSING is the sentinel defined below (compare with `is`)
     * paths sorted alphabetically in the returned dict
     * config_diff(cfg, cfg) == {}

Interviewers look for
---------------------
* Shallow `{**a, **b}` / `a | b` only merges the top level: nested sections get REPLACED.
* Mutating inputs during a merge is the classic bug (a "defaults" dict shared by the
  whole process silently changes after the first request).
* Recursion on nested dicts; base case = "not both dicts".
* A dedicated sentinel distinguishes "absent" from "present with value None".
"""

import copy
from enum import Enum
from typing import Any


class _Sentinel(Enum):
    MISSING = "MISSING"


MISSING = _Sentinel.MISSING


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    # deepcopy up front: the result must not alias anything inside `base`
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)  # explicit "unset"
        elif isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_all(*configs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for cfg in configs:
        merged = deep_merge(merged, cfg)
    return merged


def config_diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    changes: dict[str, tuple[Any, Any]] = {}
    _diff(old, new, "", changes)
    return {path: changes[path] for path in sorted(changes)}


def _diff(old: dict[str, Any], new: dict[str, Any], prefix: str, out: dict[str, Any]) -> None:
    for key in old.keys() | new.keys():
        path = f"{prefix}.{key}" if prefix else str(key)
        a, b = old.get(key, MISSING), new.get(key, MISSING)
        if isinstance(a, dict) and isinstance(b, dict):
            _diff(a, b, path, out)
        elif a is MISSING or b is MISSING or a != b:
            out[path] = (a, b)
