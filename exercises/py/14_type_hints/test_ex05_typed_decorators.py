import inspect
import re
import shutil
import subprocess
import sys

import pytest

from study.loader import load

ex = load(__file__, "ex05_typed_decorators")


@pytest.mark.topic("py.type_hints.callable", "py.type_hints.typevar")
def test_retry_behaviour():
    sleeps, calls = [], []

    @ex.retry(attempts=3, exceptions=(TimeoutError,), base_delay=0.5, sleep=sleeps.append)
    def ask(prompt: str, temperature: float = 0.0) -> str:
        """Ask the model."""
        calls.append(prompt)
        if len(calls) < 3:
            raise TimeoutError("slow")
        return f"answer to {prompt} @ {temperature}"

    assert ask("q", temperature=0.7) == "answer to q @ 0.7"
    assert sleeps == [0.5, 1.0]
    assert ask.__name__ == "ask" and ask.__doc__ == "Ask the model."
    assert inspect.signature(ask) == inspect.signature(ask.__wrapped__)


@pytest.mark.topic("py.type_hints.callable")
def test_retry_gives_up_and_filters():
    sleeps = []

    @ex.retry(attempts=2, exceptions=(TimeoutError,), sleep=sleeps.append)
    def always_slow() -> None:
        raise TimeoutError("still slow")

    with pytest.raises(TimeoutError, match="still slow"):
        always_slow()
    assert len(sleeps) == 1

    @ex.retry(attempts=5, exceptions=(TimeoutError,), sleep=sleeps.append)
    def auth() -> None:
        raise PermissionError("401")

    with pytest.raises(PermissionError):
        auth()
    assert len(sleeps) == 1, "non-matching exceptions are not retried"
    with pytest.raises(ValueError):
        ex.retry(attempts=0)


@pytest.mark.topic("py.type_hints.literal", "py.type_hints.aliases")
def test_get_setting():
    assert ex.get_setting("temperature") == 0.2
    assert ex.get_setting("max_tokens") == 1024
    assert ex.get_setting("model") == "gpt-4o-mini"
    with pytest.raises(KeyError):
        ex.get_setting("top_p")


@pytest.mark.topic("py.type_hints.generics", "py.type_hints.callable")
def test_compose_and_pipeline():
    tokens = ex.compose(str.split, len)
    assert tokens("how many tokens here") == 4
    clean = ex.pipeline(str.strip, str.lower, lambda s: s.replace("  ", " "))
    assert clean("  Hello  World ") == "hello world"
    assert ex.pipeline()(42) == 42


SNIPPET = """\
from typed_mod import compose, get_setting, pipeline, retry


@retry(attempts=2)
def ask(prompt: str, temperature: float = 0.0) -> str:
    return prompt


reveal_type(ask("hi"))  # R_ask
ask(123)  # E_arg
ask("hi", temperature="hot")  # E_kw
reveal_type(get_setting("temperature"))  # R_temp
reveal_type(get_setting("max_tokens"))  # R_max
reveal_type(get_setting("model"))  # R_model
get_setting("top_p")  # E_key
reveal_type(compose(str.split, len))  # R_compose
reveal_type(pipeline(str.strip, str.lower))  # R_pipe
"""


def run_mypy(tmp_path, snippet, *files):
    pytest.importorskip("mypy")
    shutil.copy(ex.__file__, tmp_path / "typed_mod.py")
    (tmp_path / "snippet.py").write_text(snippet, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", *files, "--strict", "--no-error-summary",
         "--no-pretty", "--hide-error-context", "--cache-dir", str(tmp_path / ".mypy_cache")],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    return proc.stdout.replace("builtins.", "")


def by_marker(output, snippet):
    lines = snippet.splitlines()
    found = {}
    for m in re.finditer(r"^snippet\.py:(\d+): (\w+): (.*)$", output, re.M):
        tag = re.search(r"# (\w+)$", lines[int(m.group(1)) - 1])
        found.setdefault(tag.group(1) if tag else m.group(1), []).append((m.group(2), m.group(3)))
    return found


@pytest.mark.topic("py.type_hints.static_checking", "py.type_hints.mypy_pyright")
def test_module_passes_mypy_strict(tmp_path):
    ex.compose(str, len)("warm up")  # unsolved stub -> TODO before running mypy
    out = run_mypy(tmp_path, "", "typed_mod.py")
    assert out.strip() == "", f"mypy --strict found problems in your module:\n{out}"


@pytest.mark.topic("py.type_hints.static_checking", "py.type_hints.callable")
def test_mypy_keeps_signatures_and_literal_overloads(tmp_path):
    ex.compose(str, len)("warm up")
    found = by_marker(run_mypy(tmp_path, SNIPPET, "snippet.py"), SNIPPET)
    assert found.pop("R_ask") == [("note", 'Revealed type is "str"')]
    assert [k for k, _ in found.pop("E_arg")] == ["error"], "ParamSpec lost: ask(123) accepted"
    assert [k for k, _ in found.pop("E_kw")] == ["error"]
    assert found.pop("R_temp") == [("note", 'Revealed type is "float"')]
    assert found.pop("R_max") == [("note", 'Revealed type is "int"')]
    assert found.pop("R_model") == [("note", 'Revealed type is "str"')]
    assert "error" in [k for k, _ in found.pop("E_key")]
    ((kind, compose_msg),) = found.pop("R_compose")
    assert "def (str) -> int" in compose_msg
    ((kind, pipe_msg),) = found.pop("R_pipe")
    assert "def (str) -> str" in pipe_msg
    assert found == {}, f"unexpected mypy output: {found}"
