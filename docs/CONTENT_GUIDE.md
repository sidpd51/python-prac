# Content guide - how a module's practice material is built

Use this when adding exercises/questions for any module (now or for future tracks:
ML, LLM, RAG, agents, prod...). The reference module is **`py.generators`**:

- `solutions/py/08_generators/ex0N_*.py`  - reference solutions (spec lives in the docstring)
- `exercises/py/08_generators/test_ex0N_*.py` - tests
- `exercises/py/08_generators/ex0N_*.py` - stubs, **generated** from solutions
- `questions/py/08_generators.toml` - interview question bank

## Workflow for a module

```bash
study scaffold <module>                # only for brand-new modules (creates templates)
# 1. write solutions/<track>/<NN_module>/exNN_name.py   (docstring = full problem spec)
# 2. write exercises/<track>/<NN_module>/test_exNN_name.py
study verify <module>                  # tests must ALL pass against the solutions
study stubs <module>                   # generates exercise stubs (never overwrites)
pytest exercises/<track>/<NN_module> --no-record   # every test must show as skipped (TODO)
# 3. write questions/<track>/<NN_module>.toml
study check <module>                   # validates; aim for "topics w/o any content: none"
```

## Exercises

**File naming**: `exNN_short_name.py` + `test_exNN_short_name.py`. 4-7 exercises per module,
ordered easy -> hard. Each exercise may contain 1-4 functions/classes.

**Docstring header** (copy this shape exactly - `study exercises` shows line 2):

```
"""
Exercise 08.2 - RAG chunker with metadata                       level: ★★☆
topics: generators.rag_chunking, generators.chunk_metadata

Problem
-------
precise spec: signatures, inputs/outputs, edge cases, what must raise what, rules
("don't use X"), so a learner can solve it WITHOUT reading the tests.

Interviewers look for
---------------------
* the concept underneath, the trade-off, the classic bug this exercise guards against

Hints (only after 15 honest minutes)       <- optional
------------------------------------
1. ...
"""
```

Levels: ★☆☆ fundamentals, ★★☆ working-engineer, ★★★ senior/3-4 YOE interview level.
Aim for roughly 1-2 / 2-3 / 1-2 across a module. **Prefer realistic, AI-engineering-flavoured
scenarios** (documents, chunks, embeddings, LLM responses, API clients, configs, logs, eval
results) over toy "foo/bar" - but the exercise must still isolate the module's concept.

Good exercise shapes:
- **implement** - a function/class with tricky edge cases (LRU cache, retry decorator, rate limiter)
- **fix the bug** - solution docstring contains the buggy version in the spec; learner writes the fixed one
- **refactor** - make something lazy/thread-safe/typed while keeping behaviour
- **write tests** (testing module) - learner writes a test suite; our tests run it against the
  correct implementation (must pass) and against mutants (must fail)

**Solutions**: idiomatic, typed, production-quality Python 3.12+ (the venv runs 3.13). They are
the learner's reference for "how a senior would write it". Comments explain *why*, sparingly.
Private helpers named `_x` are dropped from stubs; use `__stub_keep__ = ["Name"]` for scaffolding
(given classes/helpers) that must stay verbatim in the stub.

**Tests**:
- import via `from study.loader import load; ex = load(__file__, "exNN_name")` - never import directly.
- every test has `@pytest.mark.topic("<track>.<module>.<slug>", ...)` with **full** topic ids
  that exist in `curriculum/*.toml` (unknown ids abort the run). Tests in one module may also tag
  topics of other modules when genuinely exercised (e.g. `py.decorators.wraps`).
- test behaviour + the concept (laziness, eager validation, no over-consumption, ordering,
  cancellation, cleanup ran...), not implementation details. Test edge cases.
- **never** `pytest.raises(Exception)` / `RuntimeError` / `BaseException` / bare excepts - they'd
  swallow the stub's `NotImplementedError` (a subclass of `RuntimeError`) and pass on unsolved
  stubs. Use the specific exception, and make a valid call first when testing bad-call errors.
- tag topics with **literal** ids (`"py.x.y"`), never f-strings: `study check` greps for literals.
- stubs keep class bodies, constants and signatures verbatim: put answers only inside function
  bodies, or in `_private` top-level helpers (dropped from stubs).
- deterministic & fast: whole module < ~10 s; async timing tests use small sleeps (≤0.2 s) and
  generous tolerances; seed randomness; no network (use `httpx.MockTransport`); files via `tmp_path`.
- async tests: plain `async def test_...` (pytest-asyncio auto mode is on).
- ProcessPoolExecutor: functions defined in exercise modules can't be pickled by child processes
  (modules are loaded by path); design such exercises so workers are stdlib/builtin callables or
  keep them thread-based.

## Question bank (`questions/<track>/<NN_module>.toml`)

```toml
[[q]]
id = "gen-01"            # <short module prefix>-NN, unique in file
topic = "functions"      # topic slug of THIS module (or a full id from another module)
level = 1                # 1 fundamentals | 2 mid | 3 senior (3-4 YOE)
kind = "concept"         # concept | predict | debug | design | compare
q = '''question; may include ```python fenced code```'''
a = '''model answer as bullet points - exactly what a strong candidate says, incl. gotchas'''
followups = ["what the interviewer asks next"]     # optional
```

Per module: 12-20 questions, about 30% L1 / 45% L2 / 25% L3, with at least 3 `predict`
(show code, ask exact output), 2 `debug`, and `design`/`compare` where meaningful.
Answers must be **correct** - run every `predict` snippet with the venv Python (and 3.14 if it may differ) and paste the real
output. Use `'''` for q/a strings that contain `"""`.

Together, exercises + questions should touch **every topic** in the module (`study check`).


## Python version

The venv runs CPython 3.13; `/usr/bin/python3.14` exists separately. Run `predict` snippets on
both when behaviour could differ, and say so in the answer.
