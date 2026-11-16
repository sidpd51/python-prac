import dataclasses
import importlib
import importlib.machinery
import sys
import sysconfig

import pytest

from study.loader import load

ex = load(__file__, "ex04_import_machinery")


def _forget(prefix):
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


@pytest.fixture
def on_path():
    """Temporarily put directories on sys.path; forget given module prefixes afterwards."""
    added, prefixes = [], []

    def add(directory, *forget):
        sys.path.insert(0, str(directory))
        added.append(str(directory))
        prefixes.extend(forget)
        importlib.invalidate_caches()

    yield add
    for d in added:
        sys.path.remove(d)
    for p in prefixes:
        _forget(p)


@pytest.mark.topic("py.packaging.imports", "py.packaging.modules")
def test_import_from_path(tmp_path):
    src = tmp_path / "my_tool.py"
    src.write_text(
        "from dataclasses import dataclass\n"
        "import sys\n"
        "@dataclass\n"
        "class Config:\n"
        "    model: str = 'gpt-4o'\n"
        "SELF_REGISTERED = __name__ in sys.modules\n",
        encoding="utf-8",
    )
    try:
        mod = ex.import_from_path(src, "study_user_tool")
        assert mod.__name__ == "study_user_tool"
        assert sys.modules["study_user_tool"] is mod
        assert mod.SELF_REGISTERED is True, "register in sys.modules before exec"
        assert dataclasses.is_dataclass(mod.Config)
        assert mod.Config().model == "gpt-4o"
        assert mod.__file__ == str(src)
    finally:
        sys.modules.pop("study_user_tool", None)


@pytest.mark.topic("py.packaging.imports")
def test_import_from_path_failure_cleans_up(tmp_path):
    src = tmp_path / "broken.py"
    src.write_text("X = 1\nraise ZeroDivisionError('bad plugin')\n", encoding="utf-8")
    with pytest.raises(ZeroDivisionError):
        ex.import_from_path(src, "study_broken_tool")
    assert "study_broken_tool" not in sys.modules
    with pytest.raises(FileNotFoundError):
        ex.import_from_path(tmp_path / "nope.py", "study_nope")


@pytest.mark.topic("py.packaging.search_path")
def test_find_origin_first_entry_wins_and_does_not_import(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for d in (first, second):
        d.mkdir()
        (d / "study_shadowed.py").write_text(
            f"open({str(d / 'IMPORTED')!r}, 'w').close()\n", encoding="utf-8"
        )
    path = [str(tmp_path / "empty"), str(first), str(second)]
    assert ex.find_origin("study_shadowed", path) == str(first)
    assert ex.find_origin("study_shadowed", [str(second), str(first)]) == str(second)
    assert not (first / "IMPORTED").exists(), "locating must not execute the module"
    assert "study_shadowed" not in sys.modules
    assert ex.find_origin("study_nowhere", path) is None


@pytest.mark.topic("py.packaging.search_path")
def test_find_origin_defaults_and_builtins(tmp_path):
    assert ex.find_origin("sys") == "built-in"
    origin = ex.find_origin("pytest")
    assert origin in sys.path
    assert "site-packages" in origin
    pkg = tmp_path / "study_pkg_origin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    assert ex.find_origin("study_pkg_origin", [str(tmp_path)]) == str(tmp_path)


@pytest.mark.topic("py.packaging.search_path", "py.packaging.packages")
def test_shadowing_modules(tmp_path):
    for name in ("random.py", "openai.py", "app.py", "notes.txt", "logging.pyc"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "json").mkdir()
    (tmp_path / "json" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "email").mkdir()  # namespace package: doesn't shadow the stdlib one
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "__init__.py").write_text("", encoding="utf-8")
    assert ex.shadowing_modules(tmp_path) == ["json", "random"]
    assert ex.shadowing_modules(tmp_path, extra={"openai", "tools"}) == [
        "json",
        "openai",
        "random",
        "tools",
    ]


@pytest.mark.topic("py.packaging.search_path")
def test_namespace_dir_really_loses(tmp_path):
    (tmp_path / "email").mkdir()  # no __init__.py
    assert ex.shadowing_modules(tmp_path) == []
    stdlib = sysconfig.get_paths()["stdlib"]
    spec = importlib.machinery.PathFinder.find_spec("email", [str(tmp_path), stdlib])
    assert spec.origin.startswith(stdlib), "the regular package later on the path wins"


def _make_tools_pkg(root, name="study_agent_tools"):
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "search.py").write_text(
        "NAME = 'web_search'\ndef TOOL(q):\n    return f'results for {q}'\n", encoding="utf-8"
    )
    (pkg / "calc.py").write_text("TOOL = lambda expr: str(eval(expr))\n", encoding="utf-8")
    (pkg / "helpers.py").write_text("def util():\n    return 1\n", encoding="utf-8")
    (pkg / "_private.py").write_text("raise ImportError('must not be imported')\n", encoding="utf-8")
    (pkg / "not_callable.py").write_text("TOOL = 'oops'\n", encoding="utf-8")
    return pkg


@pytest.mark.topic("py.packaging.packages", "py.packaging.imports")
def test_discover_tools(tmp_path, on_path):
    _make_tools_pkg(tmp_path)
    on_path(tmp_path, "study_agent_tools")
    tools = ex.discover_tools("study_agent_tools")
    assert set(tools) == {"web_search", "calc"}
    assert tools["web_search"]("rag") == "results for rag"
    assert tools["calc"]("2+3") == "5"
    assert "study_agent_tools._private" not in sys.modules
    assert "study_agent_tools.helpers" in sys.modules


@pytest.mark.topic("py.packaging.packages")
def test_discover_tools_duplicate_name(tmp_path, on_path):
    pkg = _make_tools_pkg(tmp_path, "study_dup_tools")
    (pkg / "search2.py").write_text("NAME = 'web_search'\nTOOL = print\n", encoding="utf-8")
    on_path(tmp_path, "study_dup_tools")
    with pytest.raises(ValueError):
        ex.discover_tools("study_dup_tools")
