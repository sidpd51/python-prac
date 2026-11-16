r"""
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
    numpy==2.1.1 \
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
     * a line ending in "\" continues on the next line; None results are skipped

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

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    name: str
    extras: tuple[str, ...] = ()
    specifier: str = ""
    marker: str | None = None
    url: str | None = None


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _strip_comment(line: str) -> str:
    m = re.search(r"(^|\s)#", line)
    return line[: m.start()] if m else line


def parse_requirement(line: str) -> Requirement | None:
    text = _strip_comment(line).strip()
    if not text or text.startswith("-"):
        return None
    text = re.split(r"\s--\S", " " + text, maxsplit=1)[0].strip()  # drop --hash=... etc.

    m = re.match(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)\s*(.*)$", text, re.S)
    if not m:
        raise ValueError(f"invalid requirement (no name): {line!r}")
    name, rest = m.group(1), m.group(2).strip()

    extras: tuple[str, ...] = ()
    if rest.startswith("["):
        end = rest.find("]")
        if end == -1:
            raise ValueError(f"unclosed extras in: {line!r}")
        items = {normalize_name(e.strip()) for e in rest[1:end].split(",") if e.strip()}
        extras = tuple(sorted(items))
        rest = rest[end + 1 :].strip()

    marker = None
    if ";" in rest:
        rest, marker = (part.strip() for part in rest.split(";", 1))
        marker = marker or None

    url = None
    specifier = ""
    if rest.startswith("@"):
        url = rest[1:].strip() or None
        if url is None:
            raise ValueError(f"missing URL after '@' in: {line!r}")
    elif rest:
        clauses = [c.strip() for c in rest.split(",")]
        for clause in clauses:
            if not re.match(r"^(~=|===|==|!=|<=|>=|<|>)\s*([A-Za-z0-9][A-Za-z0-9.*+!_-]*)$", clause):
                raise ValueError(f"invalid specifier {clause!r} in: {line!r}")
        specifier = ",".join(re.sub(r"\s+", "", c) for c in clauses)

    return Requirement(normalize_name(name), extras, specifier, marker, url)


def parse_requirements(text: str) -> list[Requirement]:
    joined = re.sub(r"\\\r?\n", " ", text)  # backslash line continuations
    reqs = (parse_requirement(line) for line in joined.splitlines())
    return [r for r in reqs if r is not None]


def unpinned(reqs: list[Requirement]) -> list[str]:
    out: list[str] = []
    for r in reqs:
        if r.url is not None:
            continue
        clauses = r.specifier.split(",") if r.specifier else []
        pinned = len(clauses) == 1 and clauses[0].startswith("==") and "*" not in clauses[0]
        if not pinned:
            out.append(r.name)
    return out
