"""
Exercise 01.1 - Parse a .env file into typed settings              level: ★☆☆
topics: fundamentals.type_conversion, fundamentals.strings, fundamentals.truthiness,
        fundamentals.none, fundamentals.loops, fundamentals.loop_control,
        fundamentals.conditionals, fundamentals.variables

Problem
-------
LLM apps are configured through env vars / .env files (model name, temperature, max tokens,
feature flags...). Everything arrives as a *string* - converting it is your job.

1. coerce(raw: str) -> bool | int | float | str | None
     * strip surrounding whitespace first
     * value wrapped in matching quotes ('...' or "...") -> the inner string, NOT converted
       ('"42"' -> "42", "'true'" -> "true")
     * "true" / "yes" / "on"  -> True;  "false" / "no" / "off" -> False   (case-insensitive)
     * "" / "none" / "null"   -> None                                     (case-insensitive)
     * anything int() accepts   -> int    ("42", "-3", "+7", "1_000")
     * else anything float() accepts -> float  ("0.7", "1e-4")
     * otherwise -> the stripped string unchanged ("gpt-4o-mini")

2. parse_env(text: str) -> dict
     * one KEY=VALUE per line; split on the FIRST "=" only (values may contain "=",
       e.g. "DSN=postgres://h/db?sslmode=require")
     * skip blank lines and lines whose first non-space character is "#"
     * an optional leading "export " is ignored
     * key is stripped and must satisfy str.isidentifier(); value goes through coerce()
     * for an UNQUOTED value, " #" starts an inline comment: "MODEL=small # cheap" -> "small"
     * a non-blank, non-comment line without "=" or with an invalid key ->
       ValueError whose message contains "line <n>" (1-based line number)
     * later duplicates override earlier ones

3. first_set(*values, default=None)
     * return the first value that is not None, else `default`
     * 0, 0.0, False and "" are legitimate settings and must be returned
       (precedence chain: CLI flag -> env var -> config file -> default)

Interviewers look for
---------------------
* bool("False") is True - bool() of a non-empty string is always True; parse explicitly.
* int("3.0") raises ValueError; float("3") works - order of attempts matters.
* `cli or env or default` is a bug: it skips 0 / False / "" (truthiness != "is set").
  Use `is not None`.
* str.partition / split("=", 1) instead of split("=") for values containing "=".
* `continue` to skip lines keeps the happy path un-nested.
"""

from typing import Any

Scalar = bool | int | float | str | None


def coerce(raw: str) -> Scalar:
    raise NotImplementedError


def parse_env(text: str) -> dict[str, Scalar]:
    raise NotImplementedError


def first_set(*values: Any, default: Any = None) -> Any:
    raise NotImplementedError
