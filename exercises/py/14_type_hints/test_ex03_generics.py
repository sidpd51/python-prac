import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import get_args

import pytest

from study.loader import load

ex = load(__file__, "ex03_generics")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.mark.topic("py.type_hints.generics")
def test_ttl_cache_basic_and_expiry():
    clock = Clock()
    cache = ex.TTLCache[str, int](10, clock=clock)
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None
    clock.now = 9.9
    assert cache.get("a") == 1
    assert len(cache) == 1
    clock.now = 10.0
    assert cache.get("a") is None
    assert len(cache) == 0


@pytest.mark.topic("py.type_hints.generics")
def test_ttl_cache_refresh_and_eviction():
    clock = Clock()
    cache = ex.TTLCache(5, max_size=2, clock=clock)
    cache.set("a", 1)
    clock.now = 1
    cache.set("b", 2)
    clock.now = 2
    cache.set("a", 10)  # refresh: "a" is now the newest
    cache.set("c", 3)  # evicts the oldest-inserted -> "b"
    assert cache.get("b") is None
    assert (cache.get("a"), cache.get("c")) == (10, 3)
    clock.now = 6.5  # "a" (set at 2) expires at 7, "c" too
    cache.set("d", 4)
    assert len(cache) == 2
    with pytest.raises(ValueError):
        ex.TTLCache(0)
    with pytest.raises(ValueError):
        ex.TTLCache(1, max_size=0)


@pytest.mark.topic("py.type_hints.generics", "py.type_hints.callable")
def test_get_or_set_calls_factory_once():
    clock = Clock()
    calls = []
    cache = ex.TTLCache[str, list[float]](60, clock=clock)

    def embed():
        calls.append(1)
        return [0.1, 0.2]

    assert cache.get_or_set("q", embed) == [0.1, 0.2]
    assert cache.get_or_set("q", embed) == [0.1, 0.2]
    assert len(calls) == 1
    clock.now = 61
    cache.get_or_set("q", embed)
    assert len(calls) == 2


@pytest.mark.topic("py.type_hints.generics", "py.type_hints.typevar")
def test_type_parameters_exist_at_runtime_but_are_erased():
    ex.TTLCache(1)
    assert [p.__name__ for p in ex.TTLCache.__type_params__] == ["K", "V"]
    alias = ex.TTLCache[str, int]
    assert get_args(alias) == (str, int)
    assert type(alias(1)) is ex.TTLCache  # no separate class per parametrization


@pytest.mark.topic("py.type_hints.aliases", "py.type_hints.generics")
def test_result_helpers():
    ok = ex.try_call(lambda: 42)
    assert ok == ex.Ok(42)
    err = ex.try_call(lambda: 1 / 0)
    assert isinstance(err, ex.Err) and isinstance(err.error, ZeroDivisionError)
    assert ex.unwrap_or(ok, 0) == 42
    assert ex.unwrap_or(err, 0) == 0
    with pytest.raises(KeyboardInterrupt):
        ex.try_call(lambda: (_ for _ in ()).throw(KeyboardInterrupt))
    assert ex.Result.__value__ == ex.Ok[ex.Result.__type_params__[0]] | ex.Err


@pytest.mark.topic("py.type_hints.generics")
def test_collect_stops_at_first_err():
    pulled = []

    def results():
        for r in [ex.Ok(1), ex.Err(ValueError("a")), ex.Err(KeyError("b")), ex.Ok(4)]:
            pulled.append(r)
            yield r

    first = ex.collect(results())
    assert isinstance(first, ex.Err) and isinstance(first.error, ValueError)
    assert len(pulled) == 2
    assert ex.collect([ex.Ok(1), ex.Ok(2)]) == ex.Ok([1, 2])
    assert ex.collect([]) == ex.Ok([])


@dataclass
class Hit:
    doc_id: str
    score: float


@pytest.mark.topic("py.type_hints.typevar", "py.type_hints.protocol")
def test_top_k():
    hits = [Hit("a", 0.2), Hit("b", 0.9), Hit("c", 0.5), Hit("d", 0.9)]
    assert [h.doc_id for h in ex.top_k(hits, 3)] == ["b", "d", "c"]
    assert ex.top_k(hits, 0) == []
    assert ex.top_k(iter(hits), 10)[-1].doc_id == "a"
    with pytest.raises(ValueError):
        ex.top_k(hits, -1)


SNIPPET = """\
from dataclasses import dataclass
from typed_mod import TTLCache, top_k, try_call, unwrap_or

cache: TTLCache[str, list[float]] = TTLCache(60)
reveal_type(cache.get("q"))  # R1
cache.set("q", "not a vector")  # E1


@dataclass
class Hit:
    doc_id: str
    score: float


reveal_type(top_k([Hit("a", 1.0)], 1))  # R2
top_k([1, 2, 3], 2)  # E2
reveal_type(unwrap_or(try_call(lambda: 42), 0))  # R3
"""


def run_mypy(tmp_path, module_file, snippet):
    pytest.importorskip("mypy")
    shutil.copy(module_file, tmp_path / "typed_mod.py")
    (tmp_path / "snippet.py").write_text(snippet, encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, "-m", "mypy", "snippet.py",
            "--strict", "--follow-imports=silent", "--no-error-summary",
            "--hide-error-context", "--no-color-output", "--no-pretty",
            "--cache-dir", str(tmp_path / ".mypy_cache"),
        ],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    by_marker = {}
    lines = snippet.splitlines()
    for m in re.finditer(r"^snippet\.py:(\d+): (\w+): (.*)$", proc.stdout, re.M):
        marker = re.search(r"# (\w+)$", lines[int(m.group(1)) - 1])
        key = marker.group(1) if marker else f"line {m.group(1)}"
        msg = m.group(3).replace("builtins.", "")  # mypy < 2 prints builtins.int
        by_marker.setdefault(key, []).append((m.group(2), msg))
    return by_marker


@pytest.mark.topic("py.type_hints.static_checking", "py.type_hints.mypy_pyright")
def test_mypy_sees_the_generic_types(tmp_path):
    ex.TTLCache(1).set("warm-up", 1)  # (unsolved stub -> TODO before running mypy)
    out = run_mypy(tmp_path, ex.__file__, SNIPPET)
    assert out["R1"] == [("note", 'Revealed type is "list[float] | None"')]
    assert out["R2"] == [("note", 'Revealed type is "list[snippet.Hit]"')]
    assert out["R3"] == [("note", 'Revealed type is "int"')]
    assert [kind for kind, _ in out.get("E1", [])] == ["error"], "set() must reject a str value"
    assert "error" in [kind for kind, _ in out.get("E2", [])], "int has no .score"
    assert set(out) == {"R1", "R2", "R3", "E1", "E2"}, f"unexpected mypy output: {out}"
