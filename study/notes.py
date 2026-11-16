"""Revision notes: one markdown file per module at notes/<track>/<NN_module>.md.

Format (see docs/NOTES_GUIDE.md):
  * `# Title` then an intro paragraph
  * `## Section` headings; the line right under each carries `<!-- topics: slug1, slug2 -->`
    naming the curriculum topics (slugs of that module) the section teaches
  * fenced ```python blocks; the first line may be `# norun` (compile only) or `# nocheck` (skip)

`study notes-check` verifies coverage, unknown slugs, and that python blocks compile and (with --run) execute.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .curriculum import Module
from .paths import ROOT

NOTES_DIR = ROOT / "notes"
_TOPICS = re.compile(r"<!--\s*topics:\s*(.*?)\s*-->")
_FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)


def note_path(m: Module) -> Path:
    return NOTES_DIR / m.track_id / f"{m.dirname}.md"


@dataclass
class Section:
    title: str
    topics: list[str]
    body: str


def split_sections(text: str) -> tuple[str, list[Section]]:
    """Return (preamble before first '## ', sections). Fenced code is respected."""
    pre: list[str] = []
    secs: list[Section] = []
    cur: list[str] | None = None
    title = ""
    in_fence = False
    for ln in text.split("\n"):
        if ln.startswith("```"):
            in_fence = not in_fence
        if not in_fence and ln.startswith("## "):
            if cur is not None:
                secs.append(_make(title, cur))
            title, cur = ln[3:].strip(), []
            continue
        (cur if cur is not None else pre).append(ln)
    if cur is not None:
        secs.append(_make(title, cur))
    return "\n".join(pre), secs


def _make(title: str, lines: list[str]) -> Section:
    body = "\n".join(lines)
    m = _TOPICS.search(body)
    topics = [t.strip() for t in m.group(1).split(",") if t.strip()] if m else []
    return Section(title, topics, body)


def python_blocks(text: str) -> list[tuple[int, str]]:
    out = []
    for m in _FENCE.finditer(text):
        line = text[: m.start()].count("\n") + 2
        out.append((line, m.group(1)))
    return out


def check_module(m: Module, run: bool = False) -> list[str]:
    path = note_path(m)
    rel = path.relative_to(ROOT)
    if not path.exists():
        return [f"{rel}: missing"]
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    pre, secs = split_sections(text)
    if not pre.lstrip().startswith("# "):
        problems.append(f"{rel}: must start with a '# Title' line")
    if len(secs) < 4:
        problems.append(f"{rel}: only {len(secs)} sections (expect at least 4)")
    slugs = {t.slug for t in m.topics}
    covered: set[str] = set()
    for s in secs:
        if not s.topics and not re.match(r"(quick )?recap|summary|interview", s.title, re.I):
            problems.append(f"{rel}: section '{s.title}' has no <!-- topics: ... --> line")
        for t in s.topics:
            if t not in slugs:
                problems.append(f"{rel}: section '{s.title}' names unknown topic '{t}'")
            covered.add(t)
    missing = sorted(slugs - covered)
    if missing:
        problems.append(f"{rel}: topics not covered: {', '.join(missing)}")
    for line, code in python_blocks(text):
        first = code.lstrip().split("\n", 1)[0].strip()
        if first == "# nocheck":
            continue
        try:
            compile(code, f"{rel}:{line}", "exec")
        except SyntaxError as e:
            problems.append(f"{rel}:{line + (e.lineno or 1) - 1}: syntax error: {e.msg}")
            continue
        if run and first != "# norun":
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp, capture_output=True,
                                       text=True, timeout=20)
                except subprocess.TimeoutExpired:
                    problems.append(f"{rel}:{line}: block timed out (add '# norun' if intentional)")
                    continue
            if r.returncode != 0:
                last = (r.stderr.strip().splitlines() or ["?"])[-1]
                problems.append(f"{rel}:{line}: block raised: {last} (add '# norun' if intentional)")
    return problems
