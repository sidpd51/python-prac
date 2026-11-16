import os
import types
from pathlib import Path

import pytest

from study.loader import load

ex = load(__file__, "ex02_temp_env_and_cwd")

A, B = "STUDY_CM_TEST_A", "STUDY_CM_TEST_B"


@pytest.fixture
def env(monkeypatch):
    """A=orig exists, B doesn't; monkeypatch undoes everything even if the code under test is buggy."""
    monkeypatch.setenv(A, "orig")
    monkeypatch.setenv(B, "x")
    monkeypatch.delenv(B)
    return monkeypatch


@pytest.mark.topic("py.context_managers.contextmanager")
def test_temp_env_sets_and_restores(env):
    with ex.temp_env(**{A: "new", B: "added"}) as value:
        assert value is None
        assert os.environ[A] == "new"
        assert os.environ[B] == "added"
    assert os.environ[A] == "orig"
    assert B not in os.environ, "vars that didn't exist must be removed again"


@pytest.mark.topic("py.context_managers.contextmanager")
def test_temp_env_none_unsets(env):
    with ex.temp_env(**{A: None, B: None}):
        assert A not in os.environ
        assert B not in os.environ
    assert os.environ[A] == "orig"
    assert B not in os.environ


@pytest.mark.topic("py.context_managers.cleanup")
def test_temp_env_restores_on_exception_without_swallowing(env):
    with pytest.raises(LookupError, match="boom"), ex.temp_env(**{A: "", B: "b"}):
        assert os.environ[A] == ""
        raise LookupError("boom")
    assert os.environ[A] == "orig"
    assert B not in os.environ


@pytest.mark.topic("py.context_managers.cleanup")
def test_temp_env_validates_before_changing_anything(env):
    with pytest.raises(TypeError), ex.temp_env(**{B: "set-me", A: 123}):
        pass  # pragma: no cover
    assert os.environ[A] == "orig"
    assert B not in os.environ


@pytest.mark.topic("py.context_managers.contextmanager")
def test_temp_env_nests(env):
    with ex.temp_env(**{A: "1"}), ex.temp_env(**{A: "2"}):
        assert os.environ[A] == "2"
    assert os.environ[A] == "orig"


@pytest.mark.topic("py.context_managers.resources", "py.context_managers.cleanup")
def test_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "work"
    sub.mkdir()
    with ex.working_directory(sub) as cwd:
        assert cwd == sub.resolve()
        assert Path.cwd() == sub.resolve()
        Path("out.txt").write_text("x", encoding="utf-8")
    assert Path.cwd() == tmp_path.resolve()
    assert (sub / "out.txt").exists()
    with pytest.raises(KeyError), ex.working_directory(sub):
        raise KeyError("x")
    assert Path.cwd() == tmp_path.resolve()


@pytest.mark.topic("py.context_managers.resources")
def test_working_directory_missing_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError), ex.working_directory(tmp_path / "nope"):
        pass  # pragma: no cover
    assert Path.cwd() == tmp_path.resolve()


@pytest.mark.topic("py.context_managers.contextmanager", "py.context_managers.cleanup")
def test_patched_existing_and_missing_attr():
    cfg = types.SimpleNamespace(model="gpt-4o")
    with ex.patched(cfg, "model", "fake-model"):
        assert cfg.model == "fake-model"
    assert cfg.model == "gpt-4o"
    with pytest.raises(ValueError), ex.patched(cfg, "temperature", 0.0):
        assert cfg.temperature == 0.0
        raise ValueError
    assert not hasattr(cfg, "temperature")
    assert cfg.model == "gpt-4o"
