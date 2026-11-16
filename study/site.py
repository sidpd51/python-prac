"""Build the offline notes site: markdown files -> docs/site/*.html (no network needed to read).

    study site          regenerate docs/site/   (then open docs/site/index.html in any browser)

Pages share docs/site/style.css and site.js and use only relative links, so the folder works
from file:// on any machine that has the repo. Interactive widgets (study/site_assets/widgets/*.html)
are inserted next to the sections they explain; they are plain HTML + vanilla JS, no network.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

from .paths import ROOT

SITE_DIR = ROOT / "docs" / "site"
ASSETS = Path(__file__).parent / "site_assets"

# (page file, tab label, markdown source, one-line description)
PAGES = [
    ("readme.html", "Start here", "README.md", "How the system works, the daily loop and every command."),
    ("reference.html", "Reference", "docs/REFERENCE.md", "Every command, how scoring works, adding topics, folder layout."),
    ("content-guide.html", "Content guide", "docs/CONTENT_GUIDE.md", "How exercises and question banks are written for any module."),
    ("roadmap.html", "Capstone roadmap", "capstone/ROADMAP.md", "DocMind, the project that grows with every track."),
    ("explainers.html", "Visual explainers", None, "Step-through demos of lazy pipelines, async, retries, plus a predict-the-output deck."),
    ("decisions.html", "Decision log", "journal/decisions.md", "Template for the choices that become interview stories."),
]

WIDGETS = ASSETS / "widgets"

# page file -> list of (where, heading id, widget name); widgets with a leading "!" are built in code
INSERTS: dict[str, list[tuple[str, str, str]]] = {
    "readme.html": [
        ("after", "every-day", "daily-loop"),
    ],
    "reference.html": [
        ("before", "read-these-notes-in-a-browser-interactive", "system-map"),
        ("before", "commands", "srs"),
        ("after", "commands", "commands"),
    ],
    "content-guide.html": [
        ("after", "workflow-for-a-module", "flow"),
        ("before", "question-bank-questionstracknn_moduletoml", "!anatomy"),
        ("before", "question-bank-questionstracknn_moduletoml", "badtest"),
    ],
    "roadmap.html": [
        ("before", "how-to-work-a-milestone", "arch"),
        ("before", "how-to-work-a-milestone", "tracker"),
        ("after", "phase-0-python-foundations-current-track-py-tests-provided", "chunker"),
    ],
    "decisions.html": [("before", "yyyy-mm-dd-", "decision-form")],
}


def _widget(name: str) -> str:
    if name == "!anatomy":
        return _anatomy()
    return (WIDGETS / f"{name}.html").read_text(encoding="utf-8")


def _insert(body: str, where: str, heading_id: str, fragment: str) -> str:
    m = re.search(rf'<h[23] id="{re.escape(heading_id)}">.*?</h[23]>', body, re.S)
    if not m:
        raise ValueError(f"heading id not found for widget insert: {heading_id}")
    at = m.start() if where == "before" else m.end()
    return body[:at] + "\n" + fragment + "\n" + body[at:]


def _code(code: str) -> str:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    return highlight(code, PythonLexer(), HtmlFormatter(cssclass="hl", wrapcode=True))


def _anatomy() -> str:
    """One real exercise (generators ex01) shown as spec / solution / generated stub / test."""
    import ast

    from .stubgen import make_stub

    sol_path = ROOT / "solutions/py/08_generators/ex01_log_pipeline.py"
    test_path = ROOT / "exercises/py/08_generators/test_ex01_log_pipeline.py"
    if not (sol_path.exists() and test_path.exists()):
        return ""
    src = sol_path.read_text(encoding="utf-8")
    spec = ast.get_docstring(ast.parse(src), clean=False) or ""
    code_only = src[src.index('"""', 3) + 3 :].lstrip("\n")
    stub = make_stub(src)
    stub = stub[stub.index('"""', 3) + 3 :].lstrip("\n")
    tsrc = test_path.read_text(encoding="utf-8")
    tree = ast.parse(tsrc)
    lines = tsrc.split("\n")
    wanted = {"CountingSource", "test_pipeline_pulls_minimum", "test_take_does_not_overconsume_shared_iterator"}
    parts = []
    for node in tree.body:
        if getattr(node, "name", None) in wanted:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
            parts.append("\n".join(lines[start : node.end_lineno]))
    tests = "\n\n\n".join(parts)
    return f"""<div class="widget" data-widget="anatomy">
<h3 class="w-title">One exercise, four views</h3>
<p class="w-sub">Real files from <code>py.generators</code>, exercise 08.1. The stub is generated from the solution, never written by hand.</p>
<div class="tabset">
<div class="tabpanel" data-label="Spec"><pre><code>{html.escape(spec.strip())}</code></pre>
<p class="tabcap">The docstring is the whole problem statement. You can solve the exercise without reading the tests.</p></div>
<div class="tabpanel" data-label="Solution">{_code(code_only)}<p class="tabcap">Idiomatic, typed code you compare with after your own attempt. Private helpers (leading underscore) would be removed from the stub.</p></div>
<div class="tabpanel" data-label="Stub (what you get)">{_code(stub)}<p class="tabcap">Signatures and imports survive, bodies become <code>NotImplementedError</code>. Tests report this as TODO, not as a failure.</p></div>
<div class="tabpanel" data-label="Test excerpt">{_code(tests)}<p class="tabcap">Tests check the concept, here laziness and not over-consuming, and are tagged with the topic ids that update your mastery.</p></div>
</div></div>"""


def _predict_deck() -> str:
    """Predict-the-output questions pulled live from questions/**/*.toml."""
    from .curriculum import load_curriculum
    from .questions import load_all_questions

    cur = load_curriculum()
    by_module: dict[str, list] = {}
    for q in load_all_questions(cur):
        if q.kind == "predict":
            by_module.setdefault(cur.topics[q.topic_id].module_id, []).append(q)
    picked = []
    for mod_id, qs in by_module.items():
        qs.sort(key=lambda q: (-q.level, q.qid))
        picked += [(cur.modules[mod_id], q) for q in qs[:2]]
    picked.sort(key=lambda mq: (mq[0].track_id, mq[0].num, mq[1].qid))
    opts = "".join(f'<option value="{m.id}">{html.escape(m.title)}</option>' for m in dict.fromkeys(m for m, _ in picked))
    items = []
    for m, q in picked:
        qh, _ = _render(q.q)
        ah, _ = _render(q.a)
        stars = "★" * q.level + "☆" * (3 - q.level)
        items.append(
            f'<details class="pq" data-module="{m.id}"><summary><span>{html.escape(cur.topics[q.topic_id].title)}</span>'
            f'<span class="pq-meta">{stars} · {html.escape(m.title)}</span></summary>'
            f'<div class="pq-body">{qh}<div class="pq-ans"><strong>Answer</strong>{ah}</div></div></details>'
        )
    return (
        f'<div data-widget="predict"><div class="predict-bar"><select aria-label="Filter by module"><option value="">All modules ({len(items)} questions)</option>{opts}</select>'
        f'<button type="button" class="btn ghost" data-open>Open / close all</button></div>{"".join(items)}</div>'
    )


def _explainers_body() -> str:
    def w(name: str) -> str:
        return (WIDGETS / f"{name}.html").read_text(encoding="utf-8")

    return (
        "<h1 id=\"visual-explainers\">Visual explainers</h1>"
        "<p>Four ideas that come up in almost every AI-engineering interview, as things you can poke. "
        "Each one maps to exercises in the Python track.</p>"
        '<h2 id="lazy-pipelines">Lazy pipelines (generators)</h2>'
        "<p>A generator pipeline runs when something <em>pulls</em> from the end, and each stage asks the stage before it for one item. "
        "That is why memory stays flat on a 50 GB file, and why <code>take(2, ...)</code> must not read one item too many. "
        "Practice: <code>study test generators</code>.</p>" + w("lazy") +
        '<h2 id="async-concurrency">Async concurrency</h2>'
        "<p>Awaiting calls one by one adds their latencies. <code>gather</code> overlaps them, but unbounded fan-out gets you rate-limited, "
        "so production code caps in-flight work with a semaphore. Practice: <code>study test async</code>.</p>" + w("timeline") +
        '<h2 id="retries-and-backoff">Retries and backoff</h2>'
        "<p>Retry only what can succeed later (429, 5xx, timeouts), wait longer each time, add jitter, and honour <code>Retry-After</code>. "
        "The DocMind <code>HttpEmbedder</code> implements exactly this. Practice: <code>study test http</code>.</p>" + w("backoff") +
        '<h2 id="predict-the-output">Predict the output</h2>'
        "<p>Read the code, say the output aloud, then open the answer. Pulled from your question banks, so it grows as the banks do.</p>" + _predict_deck()
    )


def _fix_lists(text: str) -> str:
    """Markdown needs a blank line before a list that follows a paragraph; notes often omit it."""
    out: list[str] = []
    prev = ""
    for ln in text.split("\n"):
        if ln.startswith("- ") and prev.strip() and not prev.startswith(("- ", "  ")):
            out.append("")
        out.append(ln)
        prev = ln
    return "\n".join(out)


def _render(text: str) -> tuple[str, list[tuple[str, str]]]:
    import markdown  # imported lazily: only needed to build the site

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False, "css_class": "hl"}, "toc": {"toc_depth": "2"}},
    )
    body = md.convert(_fix_lists(text))
    body = re.sub(r"<li>\[ \]\s*", '<li class="task"><span class="box" aria-label="open"></span>', body)
    body = re.sub(r"<li>\[x\]\s*", '<li class="task done"><span class="box" aria-label="done"></span>', body)
    body = body.replace("<td>[ ]</td>", '<td class="tick"><span class="box" aria-label="open"></span></td>')
    body = body.replace("<th>✓</th>", '<th class="tick">Done</th>')
    body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
    headings: list[tuple[str, str]] = []
    for top in md.toc_tokens:
        for tok in [top, *top.get("children", [])] if top["level"] == 1 else [top]:
            if tok["level"] == 2:
                headings.append((tok["id"], re.sub(r"<[^>]+>", "", tok["name"])))
    return body, headings


def _shell(title: str, current: str, inner: str) -> str:
    tabs = "".join(
        f'<a class="tab" href="{f}"{" aria-current=\"page\"" if f == current else ""}>{html.escape(label)}</a>'
        for f, label, _, _ in PAGES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="wrap">
<header class="top">
  <div class="brand"><a href="index.html">AI Engineering Prep Notes</a><span>learn · practise · prove · revise</span></div>
  <nav class="tabs" aria-label="Documents">{tabs}</nav>
</header>
{inner}
</div>
<script src="site.js"></script>
</body>
</html>
"""


def build_site() -> list[Path]:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in ("style.css", "site.js"):
        shutil.copyfile(ASSETS / name, SITE_DIR / name)
        written.append(SITE_DIR / name)

    cards = []
    for filename, label, src, blurb in PAGES:
        if src is None:
            body = _explainers_body()
        else:
            body, _ = _render((ROOT / src).read_text(encoding="utf-8"))
            for where, hid, name in INSERTS.get(filename, []):
                body = _insert(body, where, hid, _widget(name))
        headings = [(i, re.sub(r"<[^>]+>", "", t)) for i, t in re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)]
        toc = "".join(f'<li><a href="#{i}">{html.escape(t)}</a></li>' for i, t in headings)
        shown = src or "generated page: widgets plus your question banks"
        inner = (
            f'<p class="src">{html.escape(shown)}</p>'
            f'<div class="layout"><nav class="toc" aria-label="On this page"><p class="toc-h">On this page</p><ol>{toc}</ol></nav>'
            f'<article class="prose">{body}</article></div>'
        )
        path = SITE_DIR / filename
        path.write_text(_shell(f"{label} - AI Eng Prep Notes", filename, inner), encoding="utf-8")
        written.append(path)
        cards.append(
            f'<a class="card" href="{filename}"><h2>{html.escape(label)}</h2><p>{html.escape(blurb)}</p>'
            f"<small>{html.escape(src or 'interactive')}</small></a>"
        )

    index = (
        '<section class="hero"><h1>Notes for the AI engineering prep system</h1>'
        "<p>Everything you need to run the daily loop, add new modules and build the DocMind capstone, readable in any browser.</p></section>"
        f'<div class="cards">{"".join(cards)}</div>'
        '<p class="note">These pages are generated from the markdown files shown on each card. After editing one, run '
        "<code>study site</code> to refresh them.</p>"
    )
    idx = SITE_DIR / "index.html"
    idx.write_text(_shell("AI Eng Prep Notes", "index.html", index), encoding="utf-8")
    written.append(idx)
    return written


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def export_to_windows(folder_name: str = "AI-Prep-Notes") -> tuple[Path, str] | None:
    """Copy docs/site into the Windows Documents folder (WSL only).

    Returns (linux path, windows path) of the copy, or None when not on WSL / not possible.
    The copy is self-contained, so it opens by double-click even when WSL is not running.
    """
    import subprocess

    if not is_wsl():
        return None
    try:
        docs_win = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "[Environment]::GetFolderPath('MyDocuments')"],
            capture_output=True, text=True, timeout=30, cwd="/mnt/c",
        ).stdout.strip()
        docs = Path(subprocess.run(["wslpath", "-u", docs_win], capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None
    if not docs_win or not docs.is_dir():
        return None
    target = docs / folder_name
    shutil.copytree(SITE_DIR, target, dirs_exist_ok=True)
    return target, f"{docs_win}\\{folder_name}"
