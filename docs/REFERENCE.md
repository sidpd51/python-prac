# Reference: how everything works

You don't need the terminal to revise or practise: the browser app (Today, Revise, Quiz, Exercises) does that and saves
progress in the browser. The terminal helper below is optional, for running exercise tests and rebuilding the pages.
This page has the details: every command, how mastery is scored,
how to add new topics, and the folder layout. You can ignore it until you need it.

In this page, `study X` means `./study.sh X` (or plain `study X` if you activated the venv with `source .venv/bin/activate`).

A personal system for going from "I've covered the syllabus" to "I can do this in an interview
for a 3-4 YOE AI-engineer role". It has four parts:

| Part | Where | What it gives you |
|------|-------|-------------------|
| **Curriculum** | `curriculum/*.toml` | Every topic of every track (Python → DSA → ML/DL → NLP → LLMs → RAG → Agents → Prod → Interviews) with a stable id like `py.generators.send` |
| **Exercises** | `exercises/<track>/<NN_module>/` | Stubs you implement + pytest tests. **Running the tests updates your progress automatically.** Reference solutions are in `solutions/` (don't peek too early) |
| **Question bank** | `questions/<track>/*.toml` | Interview questions (concept / predict-the-output / debug / design / compare) at 3 levels, drilled with spaced repetition |
| **Capstone** | `capstone/` | **DocMind**, one project that grows phase by phase into a production RAG + agents platform. See `capstone/ROADMAP.md` |

Everything feeds one tracker (`progress/progress.json`), which computes **mastery per topic**
from your self-ratings, quiz grades, logged mistakes and test results, and schedules
reviews with spaced repetition (SM-2). So you always know **what's weak** and **what to revise today**.

---

## Read these notes in a browser (interactive)

Run **`./study.sh notes`**, or open **`docs/site/index.html`** in any browser: double-click it, or run
`xdg-open docs/site/index.html` (Linux), `open docs/site/index.html` (macOS), `explorer.exe docs\site\index.html` (Windows/WSL).
It works offline, on any machine that has this repo, with no server and no install.

| Page | What's interactive |
|------|--------------------|
| **Start here** | clickable system map, 90-minute day, spaced-repetition + mastery explorer, searchable command list |
| **Content guide** | module workflow diagram, one real exercise as spec / solution / stub / test, "spot the bad test" quiz |
| **Capstone roadmap** | clickable DocMind architecture, chunker playground, milestone tracker (ticks are saved in that browser only) |
| **Visual explainers** | lazy-pipeline stepper, async timeline, retry-backoff visualiser, predict-the-output deck from the question banks |
| **Decision log** | a form that writes the markdown entry for you |

The pages are generated from the markdown files (`README.md`, `docs/CONTENT_GUIDE.md`,
`capstone/ROADMAP.md`, `journal/decisions.md`) and your question banks. **After editing any of them, run
`study site`** to refresh the HTML. Widgets live in `study/site_assets/widgets/`, styles and behaviour in
`study/site_assets/style.css` and `site.js`; they are plain HTML/CSS/JS with no dependencies.

---

## Setup (once, on each machine)

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone <your repo url> ~/python-pract   # or copy the folder
cd ~/python-pract
uv sync --all-packages              # creates .venv with everything
source .venv/bin/activate           # now `study`, `pytest`, `docmind` are on PATH
study test                          # registers all exercises (they show as TODO)
study baseline                      # 10 min: honestly rate every topic 0-5
study report                        # your first strengths/weaknesses map
```

(Without activating, prefix commands with `uv run`, e.g. `uv run study today`.)

Your progress lives in `progress/progress.json` and `journal/`. Commit them, or copy them to the
other machine, to carry your history with you. Reading the HTML notes needs none of this setup.

## The daily loop (60-120 min)

```bash
study today
```
prints your plan, in this order:

1. **Review (15-20 min)**: `study quiz --due`. Answer **out loud** or type it, as if an
   interviewer were in front of you. Then compare with the key points and grade yourself 0-5
   honestly. Low grades bring the topic back tomorrow; high grades push it further out.
2. **Deliberate practice (40-60 min)**: solve 1-2 exercises in your weakest topics.
   Read the docstring spec, write the code, then `study test <module>`.
   - Stuck for 15 min → read the Hints section. Stuck for 30 → read the solution, close it, and
     **re-implement it from memory tomorrow**.
3. **Build (30-60 min)**: the next capstone milestone (`study test capstone`).
4. **Close the loop (5 min)**: anything you got wrong becomes one sentence:
   `study mistake "asyncio.gather cancels nothing on error unless..." -t gather_exceptions`
   It goes to `journal/mistakes.md`, which you re-read before every interview.

### Weekly
- **Sunday review**: `study report`, then re-do 2 exercises you solved ≥ 7 days ago **from scratch**
  (delete your file: `git stash` it or copy it aside), then `study quiz --random -n 15 -l 3`.
- **Mock interview** (1×/week, 45 min): pick 3 L3 questions + 1 design question, set a timer,
  answer aloud while recording yourself. Then listen back: is it structured? Are there trade-offs?
  Concrete numbers?
- **DSA**: 3-5 LeetCode problems/week by pattern (track: `study start dsa.arrays_hashing`).

## Commands

```
study today                       daily plan
study report [MODULE]             module table, or a per-topic view (e.g. `study report async`)
study baseline [MODULE] [--redo]  quick self-rating pass
study quiz [-m MODULE] [-t TOPIC] [-l 1|2|3] [-k predict|debug|design|concept|compare]
           [-n N] [--weak] [--due] [--random]
study test [MODULE|capstone|PATH] run tests → progress recorded
study exercises [MODULE]          exercise list with ✅ ❌ 🟡 ⬜ status
study rate TOPIC 0-5 [--note ..]  log a self-rating after revising something
study mistake "TEXT" -t TOPIC     log a misconception (review tomorrow)
study topics [MODULE] [-v]        topic ids
study start|finish MODULE         move a module to learning / covered (new tracks!)
study scaffold MODULE             create exercise + question templates for a module
study verify [MODULE|capstone]    run tests against reference solutions
study check [MODULE]              validate content, show topics with no practice
study site                        build offline HTML notes -> open docs/site/index.html
study stubs [MODULE|capstone]     regenerate missing stubs from solutions (never overwrites)
```
Topic/module refs can be short: `generators.send`, `send` (if unique), `async`, `08_generators`.

**Grading scale:** 0 blank · 1 wrong · 2 partial with big gaps · 3 correct with effort ·
4 correct, could explain it to an interviewer · 5 instant, could teach it with edge cases.

**Mastery** (0-100) = average of (recent self/quiz grades) and (exercise pass rate), decayed
when a review is overdue. **≥80 strong · 60-79 ok · <60 weak · ? = no evidence** (no evidence
also counts as a weakness: you haven't proven it yet).

## Working on the DocMind capstone

DocMind is the project you build alongside the exercises. Milestone specs are the docstrings at the
top of each stub file in `capstone/src/docmind/`; the order and later phases are in `capstone/ROADMAP.md`.

```bash
study test capstone                 # all milestone tests (unfinished ones show as TODO)
pytest capstone/tests/test_m02_loaders.py   # one milestone
docmind ingest ./some_docs          # after M0.7: the real CLI (also `python -m docmind.cli`)
docmind search "your question" -k 5
```

1. Pick the next unchecked milestone (tick it on the roadmap page when done).
2. Implement it; `models.py` and `interfaces.py` are given, so read `interfaces.py` first.
3. Stuck 30+ minutes? Compare with `solutions/capstone/src/docmind/...`, close it, redo it tomorrow.
4. Log what you chose and rejected in `journal/decisions.md` (the Decision log page has a form).

Don't run `study stubs capstone --force`: it overwrites your capstone code. Plain `study stubs` is always safe.

## Adding future topics (ML, LLMs, RAG, agents, ...)

All future tracks are already in `curriculum/` as `planned`, so they don't pollute your stats.
When you start one:

```bash
study start llm.apis               # module now counts in today/report
study scaffold llm.apis            # creates exercises/llm/04_apis/, solutions/..., questions/llm/04_apis.toml
# write exercises/questions following docs/CONTENT_GUIDE.md (or ask Claude to generate them)
study finish llm.apis              # when done learning; it stays in spaced repetition forever
```
Need a topic that isn't listed? Add a line to the module's `topics = [...]` in the TOML.
Need a whole new area (e.g. computer vision)? Add `curriculum/10_cv.toml`. Nothing else changes.

Capstone tests can (and should) be tagged with new-track topics too, e.g.
`@pytest.mark.topic("rag.evaluation.retrieval_metrics")`, so building DocMind proves mastery.

## Layout

```
curriculum/        tracks → modules → topics (TOML)
exercises/         your work: stubs + tests            ← you edit stubs here
solutions/         reference solutions (+ capstone reference)
questions/         interview question banks
capstone/          DocMind: src/docmind (yours), tests/, ROADMAP.md
journal/           mistakes.md, decisions.md, quiz_log.jsonl (your typed answers)
progress/          progress.json: commit it and watch yourself improve
study/             the tracker itself (CLI, pytest plugin, SM-2, mastery)
docs/              CONTENT_GUIDE.md (how to write exercises/questions)
                   site/ = these notes as interactive offline HTML (open index.html; rebuild with `study site`)
```

## Troubleshooting

- **`study: command not found`**: activate the venv (`source .venv/bin/activate`) or use `uv run study ...`.
- **A test shows as skipped / TODO**: expected for unsolved stubs. It turns into pass or fail once you replace the `raise NotImplementedError`.
- **`study site` says a module is missing**: run `uv sync --all-packages` (installs `markdown` and `pygments`).
- **HTML pages look stale**: you edited a markdown file or question bank; run `study site`.
- **Python version**: the venv uses CPython 3.13 here; a few questions note 3.14 differences.
- **Reset your progress**: delete `progress/progress.json` and `journal/quiz_log.jsonl`.

## What "3-4 years of experience" actually means in interviews

Syllabus coverage gets you through L1 questions. What separates you at L3 is:

1. **Failure-mode thinking.** Not "I call the API" but "timeouts, retries with jitter,
   idempotency, partial failures, rate limits, what happens on cancellation". Every async/http
   exercise here tests those paths.
2. **Trade-offs with numbers.** "Generators: O(1) memory but single-pass and errors surface late."
   "Chunk size 512 vs 1024: here is the recall@5 I measured."
3. **Production hygiene.** Tests, typing, logging, config, secrets, packaging, CI. Interviewers
   infer experience from how you structure code, not from what you know.
4. **Evaluation.** "How do you know your RAG system got better?" is *the* senior GenAI question.
   Phase 3 of the capstone makes this concrete.
5. **Stories.** Bugs you debugged, decisions you made and why. `journal/decisions.md` +
   `journal/mistakes.md` + the capstone give you these. Write them down as you go, because you will
   not remember them later.
