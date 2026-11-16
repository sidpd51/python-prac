"""Data the browser app needs: curriculum, questions, revision notes, exercises.

Everything is embedded into the generated pages (data.js / page HTML), so the site works from
file:// with no server. Progress lives in the browser's localStorage (see site_assets/app.js).
"""

from __future__ import annotations

import ast
import html
import json
import re
from datetime import date

from .curriculum import Curriculum, Module
from .notes import note_path, split_sections
from .paths import ROOT

_MARKER = re.compile(r"(```python[ \t]*\n)# (?:norun|nocheck)[ \t]*\n")


def app_modules(cur: Curriculum) -> list[Module]:
    """Modules shown in the app: the ones already covered / being learned (not 'planned')."""
    return [m for m in cur.modules.values() if m.default_status in ("covered", "learning")]


def note_page_name(m: Module) -> str:
    return f"revise-{m.track_id}-{m.slug}.html"


def render_note(m: Module) -> tuple[str, list[dict]] | None:
    """Render a module's note to HTML. Returns (html, sections[{id,title,topics}]) or None."""
    from .site import _render

    path = note_path(m)
    if not path.exists():
        return None
    text = _MARKER.sub(r"\1", path.read_text(encoding="utf-8"))
    _, secs = split_sections(text)
    body, _ = _render(text)
    ids = re.findall(r'<h2 id="([^"]+)">', body)
    sections = []
    for i, hid in enumerate(ids):
        sec = secs[i] if i < len(secs) else None
        topics = [f"{m.id}.{t}" for t in (sec.topics if sec else [])]
        sections.append({"id": hid, "title": sec.title if sec else hid, "topics": topics})
        body = body.replace(f'<h2 id="{hid}">', f'<h2 id="{hid}" data-topics="{",".join(topics)}">', 1)
    return body, sections


def build_payload(cur: Curriculum, notes: dict[str, list[dict]]) -> dict:
    from .questions import load_module_questions
    from .site import _render

    modules, questions = [], []
    for m in app_modules(cur):
        modules.append({
            "id": m.id, "track": m.track_id, "num": m.num, "slug": m.slug, "title": m.title,
            "page": note_page_name(m) if m.id in notes else None,
            "topics": [{"id": t.id, "slug": t.slug, "title": t.title} for t in m.topics],
        })
        for q in load_module_questions(cur, m):
            qh, _ = _render(q.q)
            ah, _ = _render(q.a)
            questions.append({
                "id": q.qid, "m": m.id, "t": q.topic_id, "l": q.level, "k": q.kind, "q": qh, "a": ah,
                "f": [html.escape(f) for f in q.followups],
            })
    return {"built": date.today().isoformat(), "modules": modules, "questions": questions, "sections": notes}


def payload_js(payload: dict) -> str:
    return "window.AIPREP = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"


# ------------------------------------------------------------------ exercises
_HEADER = re.compile(r"Exercise\s+([\d.]+)\s*-\s*(.*?)\s+level:\s*(\S+)")


def _topic_ids(raw: str, m: Module, cur: Curriculum) -> list[str]:
    out = []
    for t in (x.strip() for x in raw.split(",") if x.strip()):
        for cand in (t, f"{m.track_id}.{t}", f"{m.id}.{t}"):
            if cand in cur.topics:
                out.append(cand)
                break
    return out


def _split_doc(src: str) -> tuple[str, str]:
    """(module docstring, code after it) using the parser, so any quoting style works."""
    tree = ast.parse(src)
    doc = ast.get_docstring(tree, clean=False) or ""
    first = tree.body[0] if tree.body else None
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
        code = "\n".join(src.split("\n")[first.end_lineno:])
    else:
        code = src
    return doc, code.lstrip("\n")


def exercises_html(cur: Curriculum) -> str:
    """One collapsible block per exercise, grouped by module, built from solutions/ (specs + code)."""
    from .site import _code, _render
    from .stubgen import make_stub

    blocks: list[str] = []
    for m in app_modules(cur):
        files = sorted(m.solution_dir.glob("ex*.py")) if m.solution_dir.exists() else []
        if not files:
            continue
        items = []
        for f in files:
            src = f.read_text(encoding="utf-8")
            doc, solution = _split_doc(src)
            lines = doc.strip().split("\n")
            head = _HEADER.search(lines[0]) if lines else None
            num, title, level = (head.group(1), head.group(2), head.group(3)) if head else (f.stem, f.stem, "")
            topics: list[str] = []
            body_start = 1
            for i, ln in enumerate(lines[1:6], 1):
                if ln.startswith("topics:"):
                    topics = _topic_ids(ln[len("topics:"):], m, cur)
                    body_start = i + 1
            spec_html, _ = _render("\n".join(lines[body_start:]).strip())
            spec_html = re.sub(r"<h([1-6])", "<h4", spec_html)
            spec_html = re.sub(r"</h[1-6]>", "</h4>", spec_html)
            _, stub = _split_doc(make_stub(src))
            ex_id = f"{m.id}.{f.stem}"
            items.append(
                f'<details class="ex" data-ex="{ex_id}" data-topics="{",".join(topics)}">'
                f'<summary><span class="ex-title">{html.escape(title)}</span><span class="ex-meta">{html.escape(level)} · {html.escape(f.stem)}</span>'
                f'<span class="ex-status" data-status></span></summary><div class="ex-body">'
                f'<div class="ex-spec">{spec_html}</div>'
                f'<p class="ex-where">Work in <code>{html.escape(str((m.exercise_dir / f.name).relative_to(ROOT)))}</code> '
                f"(or copy the starter code into any scratch file).</p>"
                f'<div class="tabset"><div class="tabpanel" data-label="Starter code">{_code(stub)}</div>'
                f'<div class="tabpanel" data-label="Reference solution">{_code(solution)}</div></div>'
                f'<div class="ex-rate"><span>How did it go?</span>'
                f'<button type="button" data-r="4">Solved on my own</button><button type="button" data-r="3">Solved with hints</button>'
                f'<button type="button" data-r="2">Needed the solution</button><button type="button" data-r="clear" class="ghost">Clear</button></div>'
                f"</div></details>"
            )
        blocks.append(
            f'<section class="exmod" data-module="{m.id}"><h2 id="ex-{m.slug}">{html.escape(m.title)}'
            f'<span class="exmod-count" data-count></span></h2>{"".join(items)}</section>'
        )
    return "".join(blocks)
