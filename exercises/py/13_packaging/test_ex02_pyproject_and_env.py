import importlib.metadata
import sys

import pytest

from study.loader import load

ex = load(__file__, "ex02_pyproject_and_env")

PYPROJECT = """\
[project]
name = "rag-service"
version = "0.3.0"
requires-python = ">=3.12"
dependencies = [
    "openai==1.51.0",
    "Pydantic[email]>=2.7 ; python_version >= '3.12'",
    "httpx>=0.27",
]

[project.optional-dependencies]
gpu = ["torch==2.4.1", "httpx>=0.27"]
eval = ["ragas>=0.1"]

[dependency-groups]
test = ["pytest>=8", "pytest-asyncio"]
lint = ["ruff==0.6.9"]
dev = [{include-group = "test"}, {include-group = "lint"}, "ipython"]

[tool.ruff]
line-length = 100
"""


@pytest.fixture
def pyproject(tmp_path):
    p = tmp_path / "pyproject.toml"
    p.write_text(PYPROJECT, encoding="utf-8")
    return p


@pytest.mark.topic("py.packaging.pyproject")
def test_load_project(pyproject):
    info = ex.load_project(pyproject)
    assert info.name == "rag-service"
    assert info.version == "0.3.0"
    assert info.requires_python == ">=3.12"
    assert info.dependencies[0] == "openai==1.51.0"
    assert info.optional == {"gpu": ["torch==2.4.1", "httpx>=0.27"], "eval": ["ragas>=0.1"]}


@pytest.mark.topic("py.packaging.pyproject", "py.packaging.dependencies")
def test_dependency_groups_include(pyproject):
    info = ex.load_project(pyproject)
    assert info.groups["test"] == ["pytest>=8", "pytest-asyncio"]
    assert info.groups["dev"] == ["pytest>=8", "pytest-asyncio", "ruff==0.6.9", "ipython"]


@pytest.mark.topic("py.packaging.pyproject")
def test_minimal_and_invalid(tmp_path):
    p = tmp_path / "pyproject.toml"
    p.write_text('[project]\nname = "tiny"\ndynamic = ["version"]\n', encoding="utf-8")
    info = ex.load_project(p)
    assert (info.name, info.version, info.requires_python) == ("tiny", None, None)
    assert (info.dependencies, info.optional, info.groups) == ([], {}, {})
    p.write_text("[tool.black]\nline-length = 88\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ex.load_project(p)


@pytest.mark.topic("py.packaging.pyproject")
@pytest.mark.parametrize(
    "groups",
    [
        'a = [{include-group = "b"}]\nb = [{include-group = "a"}]\n',
        'a = [{include-group = "missing"}]\n',
    ],
)
def test_bad_groups(tmp_path, groups):
    p = tmp_path / "pyproject.toml"
    p.write_text(f'[project]\nname = "x"\n\n[dependency-groups]\n{groups}', encoding="utf-8")
    with pytest.raises(ValueError):
        ex.load_project(p)


@pytest.mark.topic("py.packaging.dependencies")
def test_collect_requirements(pyproject):
    info = ex.load_project(pyproject)
    assert ex.collect_requirements(info) == info.dependencies
    reqs = ex.collect_requirements(info, extras=["gpu"], groups=["dev"])
    assert reqs == [
        "openai==1.51.0",
        "Pydantic[email]>=2.7 ; python_version >= '3.12'",
        "httpx>=0.27",
        "torch==2.4.1",
        "pytest>=8",
        "pytest-asyncio",
        "ruff==0.6.9",
        "ipython",
    ]
    with pytest.raises(KeyError):
        ex.collect_requirements(info, extras=["nope"])


@pytest.mark.topic("py.packaging.dependencies")
@pytest.mark.parametrize(
    ("req", "name"),
    [
        ("Pydantic[email]>=2 ; python_version>'3.9'", "pydantic"),
        ("sentence_transformers==3.0", "sentence-transformers"),
        ("ruff", "ruff"),
        ("  mypkg @ https://x/y.whl", "mypkg"),
    ],
)
def test_requirement_name(req, name):
    assert ex.requirement_name(req) == name


@pytest.mark.topic("py.packaging.pip", "py.packaging.dependencies")
def test_check_installed():
    installed = {"OpenAI": "1.40.0", "pydantic": "2.9.2", "HTTPX": "0.27.2", "Torch": "2.4.1"}
    reqs = [
        "openai==1.51.0",
        "Pydantic[email]>=2.7 ; python_version >= '3.12'",
        "httpx>=0.27",
        "torch==2.4.1",
        "tiktoken",
    ]
    assert ex.check_installed(reqs, installed) == {
        "openai": "installed 1.40.0, required ==1.51.0",
        "tiktoken": "missing",
    }


@pytest.mark.topic("py.packaging.pip", "py.packaging.venv")
def test_installed_versions_matches_metadata():
    versions = ex.installed_versions()
    assert versions["pytest"] == importlib.metadata.version("pytest")
    assert all(k == k.lower() and "_" not in k for k in versions)


@pytest.mark.topic("py.packaging.venv")
def test_environment_info(monkeypatch):
    info = ex.environment_info()
    assert set(info) == {"in_venv", "prefix", "base_prefix", "executable", "site_packages"}
    assert info["in_venv"] == (sys.prefix != sys.base_prefix)
    assert info["executable"] == sys.executable
    assert str(info["site_packages"]).startswith(sys.prefix)
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    assert ex.environment_info()["in_venv"] is False
