"""`study` - your revision + progress helper.

Everyday commands:
  study              what to do now (first run: a short welcome)
  study today        today's plan
  study quiz         practise interview questions
  study test         run your exercises
  study report       what you're strong and weak at
  study notes        open the notes in your browser
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from .curriculum import Curriculum, CurriculumError, Module, load_curriculum
from .paths import MISTAKES_FILE, QUIZ_LOG, ROOT
from .progress import Progress, TopicStats, today
from .questions import Question, load_all_questions, load_module_questions

console = Console()

GRADE_HELP = (
    "0 = blank / no idea   1 = wrong, recognised answer   2 = partial, big gaps\n"
    "3 = correct with effort   4 = correct, could explain to an interviewer   5 = instant, could teach it + edge cases"
)
BUCKET_STYLE = {"strong": "green", "ok": "yellow", "weak": "red", "unknown": "dim"}


# ----------------------------------------------------------------------------- helpers
def active_modules(cur: Curriculum, prog: Progress, track: str | None = None) -> list[Module]:
    mods = [
        m
        for m in cur.modules.values()
        if prog.module_status(m.id, m.default_status) in ("covered", "learning")
    ]
    if track:
        mods = [m for m in mods if m.track_id == track]
    return mods


def bar(value: int | None, width: int = 10) -> str:
    if value is None:
        return "[dim]" + "·" * width + "   ?[/dim]"
    filled = round(value / 100 * width)
    color = "green" if value >= 80 else "yellow" if value >= 60 else "red"
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (width - filled)}[/dim] {value:>3}"


def module_summary(prog: Progress, m: Module) -> dict:
    stats = [prog.stats(t.id) for t in m.topics]
    known = [s.mastery for s in stats if s.mastery is not None]
    return {
        "stats": stats,
        "mastery": round(sum(known) / len(known)) if known else None,
        "coverage": len(known) / len(stats) if stats else 0,
        "weak": sum(s.bucket == "weak" for s in stats),
        "unknown": sum(s.bucket == "unknown" for s in stats),
        "due": sum(1 for s in stats if s.due and s.due <= today()),
        "passed": sum(s.tests_passed for s in stats),
        "failed": sum(s.tests_failed for s in stats),
    }


def exercise_files(m: Module) -> list[Path]:
    if not m.exercise_dir.exists():
        return []
    return sorted(p for p in m.exercise_dir.glob("ex*.py"))


def exercise_status(prog: Progress, m: Module) -> dict[str, tuple[str, int, int, int]]:
    """filename -> (status, passed, failed, todo) based on recorded test runs."""
    rel_dir = m.exercise_dir.relative_to(ROOT).as_posix()
    agg: dict[str, list[str]] = defaultdict(list)
    for nodeid, rec in prog.data["tests"].items():
        path = nodeid.split("::", 1)[0]
        if path.startswith(rel_dir + "/"):
            name = Path(path).name.removeprefix("test_")
            agg[name].append(rec["status"])
    out = {}
    for f in exercise_files(m):
        sts = agg.get(f.name)
        if not sts:
            out[f.name] = ("never run", 0, 0, 0)
            continue
        p, fl, td = sts.count("passed"), sts.count("failed"), sts.count("todo")
        status = "solved" if p and not fl and not td else "failing" if fl else "todo" if not p else "partial"
        out[f.name] = (status, p, fl, td)
    return out


def priority(s: TopicStats) -> float:
    """Lower = more urgent. Unknown topics rank just after genuinely weak ones."""
    base = s.mastery if s.mastery is not None else 55
    return base - min(s.overdue_days, 20) - 5 * s.tests_failed


# ----------------------------------------------------------------------------- commands
def cmd_today(args: argparse.Namespace) -> None:
    cur, prog = load_curriculum(), Progress.load()
    mods = active_modules(cur, prog)
    all_stats = [prog.stats(t.id) for m in mods for t in m.topics]
    d = today()

    console.rule(f"[bold]Study plan for {d:%A %d %b %Y}")

    due = sorted(
        [s for s in all_stats if s.due and s.due <= d], key=lambda s: (-s.overdue_days, s.mastery or 0)
    )
    t = Table(title="1. Spaced-repetition reviews due (15-20 min)", show_lines=False, title_justify="left")
    t.add_column("topic")
    t.add_column("mastery")
    t.add_column("overdue", justify="right")
    for s in due[: args.n]:
        t.add_row(
            f"{cur.topics[s.topic_id].title}  [dim]{s.topic_id}[/dim]", bar(s.mastery), f"{s.overdue_days}d"
        )
    if due:
        console.print(t)
        console.print(
            f"   → [cyan]study quiz --due -n {min(len(due), args.n)}[/cyan]   ({len(due)} due in total)\n"
        )
    else:
        console.print("[green]1. No reviews due.[/green]\n")

    due_ids = {s.topic_id for s in due[: args.n]}
    weak = sorted((s for s in all_stats if s.topic_id not in due_ids), key=priority)[:3]
    console.print("[bold]2. Deliberate practice on weak spots (40-60 min)[/bold]")
    by_mod: dict[str, list[TopicStats]] = defaultdict(list)
    for s in weak:
        by_mod[cur.topics[s.topic_id].module_id].append(s)
    for mod_id, ss in by_mod.items():
        m = cur.modules[mod_id]
        console.print(f"  [bold]{m.title}[/bold] [dim]({m.id})[/dim]")
        for s in ss:
            label = s.bucket if s.mastery is not None else "no evidence yet"
            console.print(
                f"    • {cur.topics[s.topic_id].title}  {bar(s.mastery)}  [{BUCKET_STYLE[s.bucket]}]{label}[/]"
            )
        status = exercise_status(prog, m)
        todo = [f for f, (st, *_) in status.items() if st != "solved"]
        if todo:
            console.print(
                f"    → solve: [cyan]{m.exercise_dir.relative_to(ROOT)}/{todo[0]}[/cyan]  then  [cyan]study test {m.slug}[/cyan]"
            )
        console.print(f"    → drill: [cyan]study quiz -m {m.slug} --weak -n 3[/cyan]")
    console.print()

    console.print(
        "[bold]3. Build (30-60 min)[/bold]  → next unchecked milestone in [cyan]capstone/ROADMAP.md[/cyan]"
    )
    console.print(
        '[bold]4. Close the loop (5 min)[/bold] → log anything you got wrong: [cyan]study mistake "..." -t <topic>[/cyan]\n'
    )


def cmd_report(args: argparse.Namespace) -> None:
    cur, prog = load_curriculum(), Progress.load()
    if args.module:
        m = cur.module(args.module)
        t = Table(title=f"{m.title} ({m.id}) - status: {prog.module_status(m.id, m.default_status)}")
        for col in ("topic", "mastery", "self", "tests p/f/todo", "due"):
            t.add_column(col)
        for topic in m.topics:
            s = prog.stats(topic.id)
            due = "" if not s.due else ("[red]today[/red]" if s.due <= today() else s.due.isoformat())
            t.add_row(
                f"{topic.title} [dim]{topic.slug}[/dim]",
                bar(s.mastery),
                "-" if s.self_avg is None else f"{s.self_avg:.1f}",
                f"{s.tests_passed}/{s.tests_failed}/{s.tests_todo}",
                due,
            )
        console.print(t)
        st = exercise_status(prog, m)
        if st:
            console.print("Exercises: " + "  ".join(f"{f}={v[0]}" for f, v in st.items()))
        return

    tracks = [cur.track(args.track)] if args.track else cur.tracks
    for track in tracks:
        mods = [
            m for m in track.modules if args.all or prog.module_status(m.id, m.default_status) != "planned"
        ]
        if not mods:
            planned = len(track.modules)
            console.print(
                f"[dim]{track.title}: {planned} modules planned (study report --all --track {track.id})[/dim]"
            )
            continue
        t = Table(title=track.title, title_justify="left")
        for col in ("module", "mastery", "evidence", "weak", "unknown", "due", "tests ✓/✗"):
            t.add_column(col)
        for m in mods:
            s = module_summary(prog, m)
            status = prog.module_status(m.id, m.default_status)
            t.add_row(
                f"{m.num:02d} {m.title}" + ("" if status == "covered" else f" [dim]({status})[/dim]"),
                bar(s["mastery"]),
                f"{s['coverage']:.0%}",
                str(s["weak"]) if s["weak"] else "",
                str(s["unknown"]),
                str(s["due"]) if s["due"] else "",
                f"{s['passed']}/{s['failed']}",
            )
        console.print(t)

    active = active_modules(cur, prog)
    stats = [prog.stats(t.id) for m in active for t in m.topics]
    known = [s for s in stats if s.mastery is not None]
    if known:
        weakest = sorted(known, key=lambda s: s.mastery or 0)[:8]
        strongest = sorted(known, key=lambda s: -(s.mastery or 0))[:5]
        console.print(
            "[bold red]Weakest:[/bold red] "
            + ", ".join(f"{cur.topics[s.topic_id].title} ({s.mastery})" for s in weakest)
        )
        console.print(
            "[bold green]Strongest:[/bold green] "
            + ", ".join(f"{cur.topics[s.topic_id].title} ({s.mastery})" for s in strongest)
        )
    console.print(
        f"[dim]evidence = share of topics with any rating/quiz/test.  {len(stats) - len(known)} topics have no evidence yet → `study baseline`.[/dim]"
    )


def cmd_baseline(args: argparse.Namespace) -> None:
    cur, prog = load_curriculum(), Progress.load()
    mods = [cur.module(args.module)] if args.module else active_modules(cur, prog, "py")
    console.print(
        Panel(GRADE_HELP + "\n\nEnter = skip, q = save & quit", title="Rate honestly - this is for you")
    )
    for m in mods:
        console.rule(f"{m.num:02d} {m.title}")
        for topic in m.topics:
            if not args.redo and prog.reviews(topic.id):
                continue
            ans = Prompt.ask(f"  {topic.title}", default="", show_default=False).strip().lower()
            if ans == "q":
                prog.save()
                return
            if ans.isdigit() and 0 <= int(ans) <= 5:
                prog.record_review(topic.id, int(ans), "baseline")
        prog.save()
    console.print("[green]Saved.[/green] Now run [cyan]study report[/cyan].")


def _pick_questions(args: argparse.Namespace, cur: Curriculum, prog: Progress) -> list[Question]:
    if args.module:
        qs = load_module_questions(cur, cur.module(args.module))
    else:
        active = {m.id for m in active_modules(cur, prog)}
        qs = [q for q in load_all_questions(cur) if cur.topics[q.topic_id].module_id in active]
    if args.topic:
        tid = cur.topic(args.topic).id
        qs = [q for q in qs if q.topic_id == tid]
    if args.level:
        qs = [q for q in qs if q.level == args.level]
    if args.kind:
        qs = [q for q in qs if q.kind == args.kind]
    if args.due:
        qs = [q for q in qs if (s := prog.stats(q.topic_id)).due and s.due <= today()]
    if not qs:
        return []
    if args.random:
        random.shuffle(qs)
        return qs[: args.n]

    def score(q: Question) -> float:
        s = prog.stats(q.topic_id)
        rec = prog.data["questions"].get(q.qid)
        never = rec is None
        last_grade = rec["grades"][-1] if rec else 0
        days_since = (today() - datetime.fromisoformat(rec["last"]).date()).days if rec else 99
        sc = (s.mastery if s.mastery is not None else 50) + 8 * last_grade - min(days_since, 30)
        if args.weak:
            sc -= 0 if s.mastery is None else (100 - s.mastery) / 2
        if never:
            sc -= 15
        return sc + random.random() * 10  # jitter so sessions vary

    return sorted(qs, key=score)[: args.n]


def cmd_quiz(args: argparse.Namespace) -> None:
    cur, prog = load_curriculum(), Progress.load()
    qs = _pick_questions(args, cur, prog)
    if not qs:
        console.print("[yellow]No questions match.[/yellow] (Question banks live in questions/<track>/.)")
        return
    console.print(
        Panel(
            "Answer OUT LOUD or type it, as if in an interview. Then compare with the key points.\n"
            + GRADE_HELP,
            title=f"Quiz - {len(qs)} questions",
        )
    )
    QUIZ_LOG.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, q in enumerate(qs, 1):
        topic = cur.topics[q.topic_id]
        stars = "★" * q.level + "☆" * (3 - q.level)
        console.rule(f"Q{i}/{len(qs)}  {stars}  {q.kind}  · {topic.title} [dim]{q.qid}[/dim]")
        console.print(Markdown(q.q))
        typed = Prompt.ask(
            "\n[dim]your answer (Enter to reveal, q to quit)[/dim]", default="", show_default=False
        )
        if typed.strip().lower() == "q":
            break
        console.print(Panel(Markdown(q.a), title="key points", border_style="green"))
        if q.followups:
            console.print("[bold]Likely follow-ups:[/bold]")
            for f in q.followups:
                console.print(f"  • {f}")
        grade = IntPrompt.ask("grade 0-5", choices=[str(x) for x in range(6)], show_choices=False)
        prog.record_review(q.topic_id, grade, "quiz")
        prog.record_question(q.qid, grade)
        results.append((q, grade))
        with QUIZ_LOG.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "at": datetime.now().isoformat(timespec="seconds"),
                        "qid": q.qid,
                        "grade": grade,
                        "answer": typed,
                    }
                )
                + "\n"
            )
        prog.save()
    if results:
        avg = sum(g for _, g in results) / len(results)
        console.rule(f"Done: average {avg:.1f}/5")
        low = [q for q, g in results if g <= 2]
        if low:
            console.print("[red]Revise:[/red] " + ", ".join(cur.topics[q.topic_id].title for q in low))
            console.print(
                "[dim]Tip: write one sentence about WHY you got it wrong with `study mistake`.[/dim]"
            )


def cmd_rate(args: argparse.Namespace) -> None:
    cur, prog = load_curriculum(), Progress.load()
    topic = cur.topic(args.topic)
    prog.record_review(topic.id, args.grade, "rate", note=args.note or "")
    prog.save()
    s = prog.stats(topic.id)
    console.print(f"{topic.title}: mastery {bar(s.mastery)}  next review {s.due}")


def cmd_mistake(args: argparse.Namespace) -> None:
    cur, prog = load_curriculum(), Progress.load()
    topic = cur.topic(args.topic)
    prog.record_review(topic.id, 1, "mistake", note=args.text)
    prog.save()
    MISTAKES_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not MISTAKES_FILE.exists()
    with MISTAKES_FILE.open("a", encoding="utf-8") as f:
        if new:
            f.write(
                "# Mistake log\n\nRe-read this before every interview. Each entry = one misconception you fixed.\n\n"
            )
        f.write(f"- **{today()}** `{topic.id}` - {args.text}\n")
    console.print(f"Logged. {topic.title} scheduled for review tomorrow.")


def _run_pytest(paths: list[str], extra: list[str], solutions: bool = False) -> int:
    env = dict(os.environ)
    if solutions:
        env["STUDY_SOLUTIONS"] = "1"
    cmd = [sys.executable, "-m", "pytest", *paths, *extra]
    return subprocess.call(cmd, cwd=ROOT, env=env)


def _target_paths(cur: Curriculum, target: str | None, solutions: bool = False) -> list[str]:
    if not target:
        return ["exercises"]
    if target == "capstone":
        return ["capstone/tests"]
    p = Path(target)
    if p.exists():
        return [str(p)]
    m = cur.module(target)
    return [str(m.exercise_dir.relative_to(ROOT))]


def cmd_test(args: argparse.Namespace) -> None:
    cur = load_curriculum()
    sys.exit(_run_pytest(_target_paths(cur, args.target), args.pytest_args))


def cmd_verify(args: argparse.Namespace) -> None:
    cur = load_curriculum()
    sys.exit(_run_pytest(_target_paths(cur, args.target), ["--no-record", *args.pytest_args], solutions=True))


def cmd_exercises(args: argparse.Namespace) -> None:
    cur, prog = load_curriculum(), Progress.load()
    mods = [cur.module(args.module)] if args.module else active_modules(cur, prog)
    icon = {"solved": "✅", "failing": "❌", "partial": "🟡", "todo": "⬜", "never run": "·"}
    total = solved = 0
    for m in mods:
        st = exercise_status(prog, m)
        if not st:
            continue
        console.print(f"[bold]{m.num:02d} {m.title}[/bold]  [dim]{m.exercise_dir.relative_to(ROOT)}[/dim]")
        for f, (status, p, fl, td) in st.items():
            total += 1
            solved += status == "solved"
            first = (m.exercise_dir / f).read_text(encoding="utf-8").split("\n")
            title = next((ln.strip() for ln in first[1:6] if ln.strip()), "")
            console.print(f"  {icon[status]} {f:<34} [dim]{title[:70]}[/dim]  {p}✓ {fl}✗ {td}…")
    console.print(f"\n{solved}/{total} solved.  Run `study test` once to register all exercises.")


def cmd_topics(args: argparse.Namespace) -> None:
    cur, prog = load_curriculum(), Progress.load()
    mods = [cur.module(args.module)] if args.module else list(cur.modules.values())
    for m in mods:
        console.print(
            f"[bold]{m.id}[/bold]  {m.title}  [dim]{prog.module_status(m.id, m.default_status)}[/dim]"
        )
        if args.module or args.verbose:
            for t in m.topics:
                console.print(f"   {t.id:<50} {t.title}")


def cmd_set_status(args: argparse.Namespace, status: str) -> None:
    cur, prog = load_curriculum(), Progress.load()
    m = cur.module(args.module)
    prog.set_module_status(m.id, status)
    prog.save()
    console.print(f"{m.id} → {status}")
    if status == "learning" and not m.exercise_dir.exists():
        console.print(f"[dim]Tip: `study scaffold {m.id}` creates exercise + question files for it.[/dim]")


def cmd_stubs(args: argparse.Namespace) -> None:
    from .stubgen import generate_capstone_stubs, generate_stubs

    if args.module == "capstone":
        for p in generate_capstone_stubs(force=args.force):
            console.print(f"wrote {p.relative_to(ROOT)}")
        return
    cur = load_curriculum()
    mods = [cur.module(args.module)] if args.module else list(cur.modules.values())
    for m in mods:
        for p in generate_stubs(m, force=args.force):
            console.print(f"wrote {p.relative_to(ROOT)}")


def cmd_check(args: argparse.Namespace) -> None:
    """Validate content: question banks parse, topic ids exist, every exercise has test + solution."""
    cur = load_curriculum()
    mods = [cur.module(args.module)] if args.module else list(cur.modules.values())
    problems: list[str] = []
    t = Table(title="Content check")
    for col in ("module", "exercises", "tests", "questions", "L1/L2/L3", "topics w/o any content"):
        t.add_column(col)
    for m in mods:
        try:
            qs = load_module_questions(cur, m)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{m.questions_file.relative_to(ROOT)}: {e}")
            qs = []
        exs = exercise_files(m)
        tests = sorted(m.exercise_dir.glob("test_*.py")) if m.exercise_dir.exists() else []
        for f in exs:
            if not (m.exercise_dir / f"test_{f.name}").exists():
                problems.append(f"{f.relative_to(ROOT)}: missing test_{f.name}")
            if not (m.solution_dir / f.name).exists():
                problems.append(f"{f.relative_to(ROOT)}: missing solution")
        if not exs and not qs:
            continue
        covered = {q.topic_id for q in qs}
        for tf in tests:
            covered |= set(re.findall(r'"([a-z_]+\.[a-z_]+\.[a-z0-9_]+)"', tf.read_text(encoding="utf-8")))
        missing = [tp.slug for tp in m.topics if tp.id not in covered]
        lv = [sum(q.level == i for q in qs) for i in (1, 2, 3)]
        t.add_row(
            m.id,
            str(len(exs)),
            str(len(tests)),
            str(len(qs)),
            "/".join(map(str, lv)),
            ", ".join(missing) if missing else "[green]none[/green]",
        )
    console.print(t)
    for p in problems:
        console.print(f"[red]✗ {p}[/red]")
    if problems:
        sys.exit(1)
    console.print("[green]content OK[/green]")


def cmd_site(args: argparse.Namespace) -> None:
    from .site import SITE_DIR, build_site

    try:
        build_site()
    except ModuleNotFoundError as e:
        console.print(f"[red]{e}[/red]\nRun `uv sync` to install the docs dependencies (markdown, pygments).")
        sys.exit(1)
    console.print(f"Built {SITE_DIR.relative_to(ROOT)}/  → open [cyan]{SITE_DIR / 'index.html'}[/cyan] in a browser")


def cmd_notes_check(args: argparse.Namespace) -> None:
    from .notes import check_module, note_path

    cur = load_curriculum()
    mods = [cur.module(args.module)] if args.module else [m for m in cur.modules.values() if note_path(m).exists()]
    bad = 0
    for m in mods:
        probs = check_module(m, run=args.run)
        bad += len(probs)
        console.print(("[red]✗[/red] " if probs else "[green]✓[/green] ") + m.id)
        for p in probs:
            console.print(f"   {p}")
    sys.exit(1 if bad else 0)


def cmd_notes(args: argparse.Namespace) -> None:
    import subprocess
    import webbrowser

    from .site import SITE_DIR, build_site, export_to_windows

    index = SITE_DIR / "index.html"
    try:
        build_site()
    except ModuleNotFoundError:
        if not index.exists():
            console.print("[red]Notes site missing.[/red] Run `./study.sh` once (it installs what's needed), then try again.")
            sys.exit(1)

    exported = export_to_windows()
    if exported:  # WSL: put a copy in Windows Documents and open it in the Windows browser
        linux_path, win_path = exported
        console.print(f"Notes copied to Windows: [cyan]{win_path}[/cyan]")
        console.print("(Double-click [bold]index.html[/bold] there any time, even without WSL running.)")
        try:
            subprocess.run(["cmd.exe", "/c", "start", "", f"{win_path}\\index.html"], cwd="/mnt/c", check=False,
                           capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            console.print("Couldn't open the browser automatically; open that file yourself.")
        return
    console.print(f"Opening [cyan]{index}[/cyan]")
    if not webbrowser.open(index.as_uri()):
        console.print("Couldn't open a browser automatically. Open that file by double-clicking it.")


def cmd_welcome(args: argparse.Namespace) -> None:
    from .paths import PROGRESS_FILE

    if PROGRESS_FILE.exists():
        args.n = 6
        cmd_today(args)
        return
    console.print(Panel(
        "[bold]Welcome![/bold] This tool tracks what you're good at and what to revise.\n\n"
        "1. Rate yourself on topics you've studied (about 10 minutes, no right answers).\n"
        "2. Run [cyan]study[/cyan] each day. It tells you what to do.\n"
        "3. Run [cyan]study notes[/cyan] any time to read the guide in your browser.",
        title="First time here",
    ))
    if Prompt.ask("Rate your topics now?", choices=["y", "n"], default="y") == "y":
        args.module, args.redo = None, False
        cmd_baseline(args)
        console.print("\nNow try [cyan]study report[/cyan] to see where you stand, then [cyan]study[/cyan] tomorrow.")
    else:
        console.print("No problem. Run [cyan]study baseline[/cyan] whenever you're ready.")


def cmd_scaffold(args: argparse.Namespace) -> None:
    from .scaffold import scaffold

    cur = load_curriculum()
    for p in scaffold(cur.module(args.module)):
        console.print(f"created {p.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- entrypoint
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="study", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", metavar="command")
    ap.set_defaults(func=cmd_welcome)

    p = sub.add_parser("today", help="daily plan")
    p.add_argument("-n", type=int, default=6, help="max reviews to show")
    p.set_defaults(func=cmd_today)

    p = sub.add_parser("report", help="strengths & weaknesses")
    p.add_argument("module", nargs="?")
    p.add_argument("--track")
    p.add_argument("--all", action="store_true", help="include planned modules")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("baseline", help="self-rate every covered topic once")
    p.add_argument("module", nargs="?")
    p.add_argument("--redo", action="store_true", help="re-rate topics that already have ratings")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("quiz", help="interview-question drill")
    p.add_argument("-m", "--module")
    p.add_argument("-t", "--topic")
    p.add_argument("-l", "--level", type=int, choices=[1, 2, 3])
    p.add_argument("-k", "--kind", choices=["concept", "predict", "debug", "design", "compare"])
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--weak", action="store_true", help="bias towards weak topics")
    p.add_argument("--due", action="store_true", help="only topics due for review")
    p.add_argument("--random", action="store_true")
    p.set_defaults(func=cmd_quiz)

    p = sub.add_parser("rate", help=argparse.SUPPRESS)
    p.add_argument("topic")
    p.add_argument("grade", type=int, choices=range(6))
    p.add_argument("--note")
    p.set_defaults(func=cmd_rate)

    p = sub.add_parser("mistake", help="log a misconception")
    p.add_argument("text")
    p.add_argument("-t", "--topic", required=True)
    p.set_defaults(func=cmd_mistake)

    for name, func, hlp in (
        ("test", cmd_test, "run your exercises"),
        ("verify", cmd_verify, "run tests against reference solutions"),
    ):
        p = sub.add_parser(name, help=hlp if name == "test" else argparse.SUPPRESS)
        p.add_argument("target", nargs="?", help="module (e.g. generators), 'capstone', or a path")
        p.add_argument("pytest_args", nargs=argparse.REMAINDER)
        p.set_defaults(func=func)

    p = sub.add_parser("exercises", help=argparse.SUPPRESS)
    p.add_argument("module", nargs="?")
    p.set_defaults(func=cmd_exercises)

    p = sub.add_parser("topics", help=argparse.SUPPRESS)
    p.add_argument("module", nargs="?")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_topics)

    p = sub.add_parser("start", help=argparse.SUPPRESS)
    p.add_argument("module")
    p.set_defaults(func=lambda a: cmd_set_status(a, "learning"))
    p = sub.add_parser("finish", help=argparse.SUPPRESS)
    p.add_argument("module")
    p.set_defaults(func=lambda a: cmd_set_status(a, "covered"))

    p = sub.add_parser("scaffold", help=argparse.SUPPRESS)
    p.add_argument("module")
    p.set_defaults(func=cmd_scaffold)

    p = sub.add_parser("notes", help="open the notes in your browser")
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("notes-check", help=argparse.SUPPRESS)
    p.add_argument("module", nargs="?")
    p.add_argument("--run", action="store_true", help="also execute python blocks")
    p.set_defaults(func=cmd_notes_check)

    p = sub.add_parser("site", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_site)

    p = sub.add_parser("check", help=argparse.SUPPRESS)
    p.add_argument("module", nargs="?")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("stubs", help=argparse.SUPPRESS)
    p.add_argument("module", nargs="?")
    p.add_argument(
        "--force", action="store_true", help="overwrite existing exercise files (destroys your work!)"
    )
    p.set_defaults(func=cmd_stubs)
    # hide advanced/author commands from --help (they still work)
    sub._choices_actions = [a for a in sub._choices_actions if a.help is not argparse.SUPPRESS]
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except CurriculumError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(2)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]bye[/dim]")


if __name__ == "__main__":
    main()
