"""
Exercise 13.1 - Parsing requirements.txt                         level: ★☆☆
topics: packaging.requirements, packaging.pip, packaging.dependencies

Problem
-------
Before a deploy you want to lint requirements.txt: are all versions pinned? Write a small
parser for the common subset of the PEP 508 / pip format (no `packaging` library).

    # LLM stack
    openai==1.51.0
    LangChain_Core >= 0.3, <0.4      # loose!
    uvicorn[standard,  socks]==0.30.6
    uvloop==0.20.0 ; sys_platform != "win32"
    mypkg @ https://example.com/mypkg-1.0.tar.gz
    -r dev-requirements.txt
    --index-url https://pypi.org/simple
    numpy==2.1.1 \\
        --hash=sha256:abc

1. normalize_name(name) -> str
     PEP 503: lower-case, runs of "-", "_", "." become one "-"  ("LangChain_Core" ->
     "langchain-core")

2. parse_requirement(line) -> Requirement | None       (Requirement dataclass is given)
     * comments: "#" at the start of the line or preceded by whitespace starts a comment
     * blank / comment-only lines and pip option lines (starting with "-") -> None
     * NAME [ "[" extras "]" ] [ SPECIFIER | "@" URL ] [ ";" MARKER ]
         - name normalized; extras normalized, de-duplicated, sorted tuple
         - specifier: comma-separated clauses, each OP VERSION with OP in
           ~= == != <= >= < > === ; returned with ALL whitespace removed (">=0.3,<0.4"),
           "" if absent
         - url: text after "@" (stripped) or None; marker: text after the first ";"
           (stripped) or None
         - any trailing pip per-requirement options ("--hash=...") are dropped
     * anything malformed (no name, unclosed "[", a bad specifier clause such as
       "=1.0" or ">=") -> ValueError mentioning the line

3. parse_requirements(text) -> list[Requirement]
     * a line ending in "\\" continues on the next line; None results are skipped

4. unpinned(reqs) -> list[str]
     names (in order) whose specifier is not exactly one "==" clause without wildcard
     ("==1.*" is not a pin) and that are not URL requirements

Interviewers look for
---------------------
* requirements.txt is pip *input*: it can be a loose list or a fully pinned lock
  (pip freeze / pip-compile output). Know the difference and why deploys need pins
  (+ hashes) while libraries declare ranges.
* Name normalization - "LangChain_Core" and "langchain-core" are the same project.
* Environment markers for platform-specific deps; extras for optional features.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    name: str
    extras: tuple[str, ...] = ()
    specifier: str = ""
    marker: str | None = None
    url: str | None = None


def normalize_name(name: str) -> str:
    raise NotImplementedError


def parse_requirement(line: str) -> Requirement | None:
    raise NotImplementedError


def parse_requirements(text: str) -> list[Requirement]:
    raise NotImplementedError


def unpinned(reqs: list[Requirement]) -> list[str]:
    raise NotImplementedError
