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

import importlib.metadata
import re
import sys
import sysconfig
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProjectInfo:
    name: str
    version: str | None = None
    requires_python: str | None = None
    dependencies: list[str] = field(default_factory=list)
    optional: dict[str, list[str]] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)


def _resolve_group(
    name: str, raw: Mapping[str, list[Any]], done: dict[str, list[str]], stack: tuple[str, ...]
) -> list[str]:
    if name in done:
        return done[name]
    if name in stack:
        raise ValueError(f"dependency-group cycle: {' -> '.join((*stack, name))}")
    if name not in raw:
        raise ValueError(f"unknown dependency group {name!r}")
    out: list[str] = []
    for entry in raw[name]:
        if isinstance(entry, dict):
            out.extend(_resolve_group(entry["include-group"], raw, done, (*stack, name)))
        else:
            out.append(entry)
    done[name] = out
    return out


def load_project(path: str | Path) -> ProjectInfo:
    with open(path, "rb") as f:  # tomllib requires bytes
        data = tomllib.load(f)
    project = data.get("project")
    if project is None:
        raise ValueError(f"{path}: no [project] table")
    raw_groups = data.get("dependency-groups", {})
    resolved: dict[str, list[str]] = {}
    for g in raw_groups:
        _resolve_group(g, raw_groups, resolved, ())
    return ProjectInfo(
        name=project["name"],
        version=project.get("version"),
        requires_python=project.get("requires-python"),
        dependencies=list(project.get("dependencies", [])),
        optional={k: list(v) for k, v in project.get("optional-dependencies", {}).items()},
        groups={g: resolved[g] for g in raw_groups},
    )


def collect_requirements(
    info: ProjectInfo, extras: Iterable[str] = (), groups: Iterable[str] = ()
) -> list[str]:
    reqs = list(info.dependencies)
    for extra in extras:
        reqs += info.optional[extra]
    for group in groups:
        reqs += info.groups[group]
    return list(dict.fromkeys(reqs))  # ordered de-duplication


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(req: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    if not m:
        raise ValueError(f"invalid requirement: {req!r}")
    return _normalize(m.group(1).rstrip("._-"))


def check_installed(requirements: Iterable[str], installed: Mapping[str, str]) -> dict[str, str]:
    have = {_normalize(k): v for k, v in installed.items()}
    problems: dict[str, str] = {}
    for req in requirements:
        name = requirement_name(req)
        if name not in have:
            problems[name] = "missing"
            continue
        spec = req.split(";", 1)[0]
        pin = re.search(r"(?<![=!<>~])==\s*([^\s,;]+)\s*$", spec)
        if pin and have[name] != pin.group(1):
            problems[name] = f"installed {have[name]}, required =={pin.group(1)}"
    return problems


def installed_versions() -> dict[str, str]:
    return {
        _normalize(dist.metadata["Name"]): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }


def environment_info() -> dict[str, object]:
    return {
        "in_venv": sys.prefix != sys.base_prefix,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "executable": sys.executable,
        "site_packages": sysconfig.get_paths()["purelib"],
    }
