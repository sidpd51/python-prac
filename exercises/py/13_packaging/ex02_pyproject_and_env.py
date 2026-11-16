"""
Exercise 13.2 - pyproject.toml dependencies & the environment    level: ★★☆
topics: packaging.pyproject, packaging.dependencies, packaging.venv, packaging.pip

Problem
-------
A CI step must answer "does the active environment satisfy this project?".

1. load_project(path) -> ProjectInfo           (ProjectInfo dataclass is given)
     * parse pyproject.toml with tomllib (stdlib 3.11+)
     * name, version (None if missing - e.g. dynamic), requires_python (None if missing),
       dependencies ([project].dependencies, default []),
       optional ([project.optional-dependencies], default {}),
       groups ([dependency-groups] - PEP 735, default {})
     * no [project] table -> ValueError
     * dependency-group entries may be {include-group = "other"} tables: resolve them
       (recursively, in place, keeping order); unknown group or a cycle -> ValueError

2. collect_requirements(info, extras=(), groups=()) -> list[str]
     dependencies + the requested extras + the requested groups, in that order, exact
     duplicates removed (first occurrence wins); unknown extra/group -> KeyError

3. requirement_name(req: str) -> str
     the normalized project name at the start of a requirement string
     ("Pydantic[email]>=2 ; python_version>'3.9'" -> "pydantic")

4. check_installed(requirements, installed: Mapping[str, str]) -> dict[str, str]
     * installed maps project name -> version (names may be un-normalized)
     * returns problems only: {name: "missing"} or, for an exact pin "name==X" whose
       installed version differs, {name: "installed Y, required ==X"}
       (other specifiers: only presence is checked)

5. installed_versions() -> dict[str, str]
     normalized name -> version of every distribution in the running environment
     (importlib.metadata)

6. environment_info() -> dict[str, object]
     keys: "in_venv" (bool: sys.prefix != sys.base_prefix), "prefix", "base_prefix",
     "executable" (sys.executable), "site_packages" (sysconfig purelib path)

Interviewers look for
---------------------
* pyproject.toml as THE config file (PEP 621 metadata, build-system, tool config);
  [project.dependencies] = what the package needs (ranges), lock files = what an
  app deploys (exact). Extras vs dependency groups (dev tools aren't published).
* tomllib.load needs a BINARY file object ("rb") - a frequent TypeError.
* How a venv works: its own prefix + site-packages; `sys.prefix != sys.base_prefix`;
  "activate" just puts its bin/ first on PATH.
* importlib.metadata to inspect what's installed (what pip freeze reads).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectInfo:
    name: str
    version: str | None = None
    requires_python: str | None = None
    dependencies: list[str] = field(default_factory=list)
    optional: dict[str, list[str]] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)


def load_project(path: str | Path) -> ProjectInfo:
    raise NotImplementedError


def collect_requirements(
    info: ProjectInfo, extras: Iterable[str] = (), groups: Iterable[str] = ()
) -> list[str]:
    raise NotImplementedError


def requirement_name(req: str) -> str:
    raise NotImplementedError


def check_installed(requirements: Iterable[str], installed: Mapping[str, str]) -> dict[str, str]:
    raise NotImplementedError


def installed_versions() -> dict[str, str]:
    raise NotImplementedError


def environment_info() -> dict[str, object]:
    raise NotImplementedError
