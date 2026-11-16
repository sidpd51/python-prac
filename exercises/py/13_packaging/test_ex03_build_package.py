import importlib
import runpy
import shutil
import subprocess
import sys

import pytest

from study.loader import load

ex = load(__file__, "ex03_build_package")


def _forget(prefix):
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


@pytest.fixture
def pkg_root(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    pkg = ex.write_package(root)
    assert pkg == root / "ragkit"
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    _forget("ragkit")
    yield root
    sys.path.remove(str(root))
    _forget("ragkit")


@pytest.mark.topic("py.packaging.packages", "py.packaging.init_py")
def test_layout_on_disk(pkg_root):
    pkg = pkg_root / "ragkit"
    for rel in [
        "__init__.py",
        "_internal.py",
        "embed.py",
        "cli.py",
        "__main__.py",
        "text/__init__.py",
        "text/chunking.py",
    ]:
        assert (pkg / rel).is_file(), rel


@pytest.mark.topic("py.packaging.init_py", "py.packaging.imports")
def test_import_and_reexports(pkg_root):
    ragkit = importlib.import_module("ragkit")
    assert ragkit.__version__ == "0.1.0"
    assert ragkit.embed("abcdefghij") == [4, 4, 2]
    assert ragkit.chunk("abcdef", 4) == ["abcd", "ef"]
    assert ragkit.chunk is importlib.import_module("ragkit.text.chunking").chunk
    assert "ragkit.cli" not in sys.modules, "__init__ must not import the CLI"
    assert importlib.import_module("ragkit._internal").SECRET == 42


@pytest.mark.topic("py.packaging.init_py", "py.packaging.modules")
def test_star_import_respects_all(pkg_root):
    ns = {}
    exec("from ragkit import *", ns)
    assert set(ns) - {"__builtins__"} == {"chunk", "embed", "__version__"}


@pytest.mark.topic("py.packaging.packages")
def test_subpackage(pkg_root):
    text = importlib.import_module("ragkit.text")
    assert text.chunk("abc", 2) == ["ab", "c"]
    with pytest.raises(ValueError):
        text.chunk("abc", 0)
    assert text.__name__ == "ragkit.text"
    assert text.__package__ == "ragkit.text"


@pytest.mark.topic("py.packaging.abs_vs_rel")
def test_relative_imports_survive_a_rename(pkg_root, tmp_path):
    fork_root = tmp_path / "forked"
    shutil.copytree(pkg_root / "ragkit", fork_root / "ragkit_fork")
    sys.path.remove(str(pkg_root))  # the original package is NOT importable now
    sys.path.insert(0, str(fork_root))
    importlib.invalidate_caches()
    try:
        fork = importlib.import_module("ragkit_fork")
        assert fork.embed("abcdefghij") == [4, 4, 2]
        assert importlib.import_module("ragkit_fork.cli").main([]) == 0
        assert not any(m == "ragkit" or m.startswith("ragkit.") for m in sys.modules), (
            "an absolute `import ragkit...` sneaked in"
        )
    finally:
        sys.path.remove(str(fork_root))
        sys.path.insert(0, str(pkg_root))
        _forget("ragkit_fork")


@pytest.mark.topic("py.packaging.main_guard")
def test_import_cli_prints_nothing_but_run_as_main_does(pkg_root, capsys):
    cli = importlib.import_module("ragkit.cli")
    assert capsys.readouterr().out == ""
    assert cli.main([]) == 0
    assert capsys.readouterr().out == "ragkit 0.1.0\n"
    _forget("ragkit.cli")  # else runpy warns: already imported, would run a second copy
    with pytest.raises(SystemExit) as info:
        runpy.run_module("ragkit.cli", run_name="__main__")
    assert info.value.code == 0
    assert capsys.readouterr().out == "ragkit 0.1.0\n"


@pytest.mark.topic("py.packaging.main_guard", "py.packaging.packages")
def test_python_dash_m(pkg_root):
    for target in ("ragkit", "ragkit.cli"):
        proc = subprocess.run(
            [sys.executable, "-m", target],
            cwd=pkg_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == "ragkit 0.1.0\n"


@pytest.mark.topic("py.packaging.abs_vs_rel", "py.packaging.main_guard")
def test_running_module_file_directly_breaks_relative_imports(pkg_root):
    proc = subprocess.run(
        [sys.executable, str(pkg_root / "ragkit" / "cli.py")],
        cwd=pkg_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "ImportError" in proc.stderr  # "attempted relative import with no known parent package"
