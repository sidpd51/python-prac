"""
Exercise 22.2 - Fixtures: factories, setup/teardown, isolation       level: ★☆☆
topics: testing.fixtures, testing.pytest, testing.clean_code

Problem
-------
Tests for a RAG app need lots of chunk dicts and need to tweak environment variables.
Write the helpers a good conftest.py would provide.

1. make_chunk_factory() -> Callable[..., dict]
   Returns a `make_chunk(**overrides)` function (the "factory as fixture" pattern).
   Defaults:
       {"id": "chunk-<n>", "doc_id": "doc-1", "text": "lorem ipsum", "score": 0.0, "metadata": {}}
   * n starts at 1 and increments on every call - PER FACTORY (two factories don't
     share a counter); an explicit id=... override still advances the counter
   * overrides replace defaults; an unknown key -> TypeError (catch typos in tests early)
   * every call returns a NEW dict with a NEW metadata dict (no shared mutable state)

2. temp_env(**changes) - a context manager (what monkeypatch.setenv/delenv do for you)
   * value str  -> os.environ[key] = value
   * value None -> key removed from os.environ (no error if absent)
   * on exit - normal OR via exception - restore the exact previous state: keys that
     didn't exist are removed again, changed keys get their old values back
   * yields nothing; exceptions from the block propagate

3. chunk_factory - a pytest fixture (decorate with @pytest.fixture) that returns a fresh
   make_chunk_factory() for each test.

Interviewers look for
---------------------
* Fixtures = setup + teardown + dependency injection for tests. Code after `yield` (or in
  `finally`) is the teardown, and it must run even when the test fails.
* Function scope by default -> no state leaks between tests. Wider scopes (module/session)
  are for expensive, effectively immutable resources.
* Factories beat giant shared fixture objects: each test states only what it cares about.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest


def make_chunk_factory() -> Callable[..., dict[str, Any]]:
    raise NotImplementedError


@contextmanager
def temp_env(**changes: str | None) -> Iterator[None]:
    raise NotImplementedError


@pytest.fixture
def chunk_factory() -> Callable[..., dict[str, Any]]:
    raise NotImplementedError
