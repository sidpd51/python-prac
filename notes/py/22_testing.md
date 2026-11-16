# Testing and Code Quality

Tests are how you change code without fear, and in AI engineering they matter more than usual:
your code wraps slow, paid, non-deterministic model calls, so you must be able to test everything
*around* the model without calling it. Interviewers probe four things: do you know what to test at
which level, do you understand mocking (especially where to patch), can you test async code, and do you
have a credible answer to "how do you test an LLM app?". This note also covers logging, debugging and
the tooling (ruff, mypy, PEP 8) that keeps a codebase healthy.

## The test pyramid, unit versus integration
<!-- topics: unit, integration -->

```text
 few   end-to-end: whole system, slow, flaky
 some  integration: real collaborators (files, DB, HTTP stack)
 many  unit: one function/class, no I/O, milliseconds
```

The shape is cost versus feedback: cheap tests run on every save, expensive ones on CI or nightly.

* **Unit test**: one unit of logic in isolation. Collaborators that do I/O (network, clock, disk, model)
  are replaced by test doubles. It tells you *which function* broke.
* **Integration test**: several real parts together (your code plus a real temp file, a real SQLite DB,
  an `httpx` client talking to a `MockTransport`). It catches what mocks hide: wrong encoding, wrong
  JSON shape, a bad SQL query, wiring mistakes.
* An LLM app adds an **eval layer** beside the pyramid (see the last section): real-model runs on a
  golden set, on demand or nightly, never on every commit.

> **Interview** "Tests pass, production breaks." Usually a mock encoded a wrong assumption. Add an
> integration test against the real thing.

## pytest mechanics
<!-- topics: pytest -->

pytest finds tests by convention: files named `test_*.py` or `*_test.py`, functions named `test_*`,
and methods `test_*` in classes named `Test*` (no `__init__`). A test is a plain function; a test
**passes if it does not raise**. You write bare `assert`.

```python
# norun
import pytest

def test_add():
    assert 1 + 2 == 3

def test_raises():
    with pytest.raises(ZeroDivisionError, match="division"):
        1 / 0
    assert 0.1 + 0.2 == pytest.approx(0.3)
```

**Assert rewriting.** Normally `assert a == b` only says `AssertionError`. When pytest imports your
test modules, it rewrites the assert statements in the AST so a failure shows both sides, the
operator and a diff of lists/dicts. This only works for test modules and `conftest.py`, not for helper
modules (register them with `pytest.register_assert_rewrite`).

Handy flags: `-x` (stop at first failure), `-k "chunk and not slow"` (select by name), `--lf` (rerun
failures), `-s` (show prints), `--pdb` (debugger on failure), `file.py::test_name` (one test).

> **Gotcha** Never compare floats with `==` (use `pytest.approx` or `math.isclose`), and never
> `assert (cond, "message")`: a non-empty tuple is always true. Write `assert cond, "message"`.

## Fixtures: setup, teardown, scope
<!-- topics: fixtures -->

A **fixture** is a function that provides a dependency to a test. The test asks for it by putting
its name in the parameters; pytest builds it, passes it in, and (for `yield` fixtures) runs the code
after the `yield` as teardown, even if the test failed.

```python
# norun
import pytest

@pytest.fixture
def store():
    s = {}              # setup
    yield s             # value given to the test
    s.clear()           # teardown, always runs

@pytest.fixture(scope="session")
def embedder():         # built once for the whole run (expensive things)
    return FakeEmbedder()

def test_put(store):
    store["a"] = 1
    assert store == {"a": 1}
```

Scopes: `function` (default, per test: use for anything mutable), `class`/`module`, and `session`
(once per run: model loading, a DB container).

Fixtures can depend on other fixtures; shared ones live in `conftest.py` (auto-discovered). Built-ins:

* `tmp_path`: a fresh `pathlib.Path` directory per test. Use it for every file test.
* `monkeypatch`: set/delete env vars (`setenv`, `delenv`), attributes (`setattr`), dict items; **undone
  automatically** after the test. This is the safe way to fake `os.environ`.
* `caplog` (captured log records), `capsys` (captured stdout/stderr).

```python
# norun
def test_reads_key(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("API_KEY", "k-123")       # undone after the test
    (tmp_path / "doc.txt").write_text("héllo", encoding="utf-8")
    run(tmp_path / "doc.txt")
    assert "loaded 1 doc" in caplog.text
```

> **Gotcha** A wide-scoped fixture returning a mutable object leaks state between tests ("passes alone,
> fails in the suite"). Keep mutable fixtures function-scoped or return a factory.

## Parametrized tests
<!-- topics: parametrize -->

`@pytest.mark.parametrize` runs one test body over a table of inputs. Each row becomes its own test
(own pass/fail, own id), so one bad case does not hide the others. That is "table-driven testing".

```python
# norun
import pytest

@pytest.mark.parametrize(
    "text, expected",
    [
        ("score: 7", 7),
        ("SCORE=10/10", 10),
        pytest.param("no number", None, id="missing"),
    ],
)
def test_extract_score(text, expected):
    assert extract_score(text) == expected
```

Good habits: include edge cases (empty, unicode, boundaries), give odd rows an `id`, stack decorators for a
cross product, and parametrize inputs, not logic. Without pytest it is just `for text, want in cases: assert f(text) == want`.

## Mocking: doubles, fakes, stubs
<!-- topics: mocking, clean_code -->

A **test double** replaces a real collaborator. Vocabulary:
* **stub**: returns canned answers. **fake**: a simplified but working implementation (in-memory store,
  scripted LLM). **mock**: records calls so you can assert on interactions. **spy**: real object plus recording.

Prefer **state-based** tests (check the result) with stubs and fakes over **interaction-based** tests
(check which methods were called): the latter pin down implementation details, so every refactor
breaks them. Use a mock when the interaction *is* the behaviour ("the email was sent once", "we did not
call the paid API on a cache hit").

The enabler is **dependency injection**: pass collaborators in instead of creating them inside.

```python
import json

class FakeLLM:
    """Scripted fake: returns replies in order and records prompts."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)

def summarize(text, llm):                 # llm injected, not constructed inside
    return json.loads(llm.complete(f"Summarize as JSON: {text}"))["summary"]

fake = FakeLLM(['{"summary": "short"}'])
assert summarize("long text", fake) == "short"
assert "long text" in fake.prompts[0]
print("ok")   # prints: ok
```

Code that calls `datetime.now()`, reads `os.environ` or builds `OpenAI()` inline is hard to test; take the
clock, the settings or the client as parameters (with a sensible default) and a test hands in a fake.
This is also the clean-code payoff: small functions, one responsibility, explicit inputs and outputs, no
hidden globals. Testable code and clean code are nearly the same thing.

## unittest.mock: Mock, patch, autospec
<!-- topics: unittest_mock -->

`Mock` accepts any attribute or call and records it; `MagicMock` also supports magic methods
(`len`, `in`, `with`, iteration). Configure with `return_value` and `side_effect` (a list returns items
in order, an exception instance is raised, a function computes the result).

```python
from unittest.mock import Mock, MagicMock

m = Mock()
m.complete.return_value = "hi"
print(m.complete("x"))             # prints: hi
print(m.complete.call_args)        # prints: call('x')
m.complete.assert_called_once_with("x")

print(m.complet.called)            # prints: False   (typo silently accepted!)

flaky = Mock(side_effect=[1, ValueError("boom")])
print(flaky())                     # prints: 1
try:
    flaky()
except ValueError as e:
    print(e)                       # prints: boom

print(len(MagicMock()))            # prints: 0 (MagicMock supports magic methods; Mock does not)
```

**autospec** makes the mock copy the real object's signature, so wrong attribute names or wrong
arguments fail like they would in production:

```python
from unittest.mock import create_autospec

class Client:
    def complete(self, prompt: str, *, temperature: float = 0.0) -> str: ...

s = create_autospec(Client, instance=True)
s.complete("a", temperature=1)          # fine
try:
    s.complete()
except TypeError as e:
    print(e)                            # prints: missing a required argument: 'prompt'
try:
    s.compleet
except AttributeError as e:
    print(e)                            # prints: Mock object has no attribute 'compleet'
```

### patch and the where-to-patch rule

`patch("target")` temporarily replaces a name and restores it on exit (context manager or decorator).
The rule: **patch the name where it is looked up, not where it is defined.**

```text
app/llm.py        defines   call_model
app/service.py    does      from app.llm import call_model   -> service.call_model is its OWN reference
```

Patch `app.service.call_model` (where it is looked up). Patching `app.llm.call_model` changes only the
original. If `service` does `import app.llm` and calls `app.llm.call_model(...)`, patching `app.llm` works.

```python
from unittest.mock import patch
import os

with patch("os.getenv", return_value="k") as g:
    print(os.getenv("X"))     # prints: k
    print(g.call_args)        # prints: call('X')
print(os.getenv("X_NOPE"))    # prints: None   (restored)
```

> **Gotcha** Use `autospec=True` with `patch`, and `monkeypatch`/`patch.dict` for `os.environ`.

<details markdown="1"><summary>Check yourself: `service.py` has `from llm import call`. Why does `patch("llm.call")` not affect it?</summary>

`from llm import call` copies the reference into `service`'s namespace at import time. Patching
`llm.call` rebinds the name in `llm` only. Patch `service.call` (the place it is used).

</details>

## Async testing and AsyncMock
<!-- topics: async_testing, testing_async_functions -->

An `async def` test needs an event loop to run it. With `pytest-asyncio` you mark the test (or set
`asyncio_mode = "auto"` in config) and pytest awaits it for you:

```python
# norun
@pytest.mark.asyncio
async def test_embed_all():
    out = await embed_all(["a", "b"], client=fake_client)
    assert len(out) == 2
```

Under the hood that is `asyncio.run(test())`, which also lets you demo async logic without pytest.

`AsyncMock` is a mock whose call returns an awaitable. `await mock(...)` yields `return_value`, or
raises/steps through `side_effect`; it also records `await_count` and `assert_awaited_once_with`.

```python
import asyncio
from unittest.mock import AsyncMock

embed = AsyncMock(return_value=[1])
print(asyncio.run(embed("q")))     # prints: [1]
print(embed.await_count)           # prints: 1

# simulate "rate limited once, then success" - essential for retry tests
api = AsyncMock(side_effect=[RuntimeError("429"), "ok"])

async def with_retry():
    for attempt in range(3):
        try:
            return await api()
        except RuntimeError as e:
            print("retrying after", e)

print(asyncio.run(with_retry()))
# prints: retrying after 429
# prints: ok
print(api.await_count)             # prints: 2
```

`patch` on an `async def` target auto-creates an `AsyncMock`. Inject the sleep function
(`sleep=AsyncMock()`) so retry tests are instant, and assert the back-off values it received.

> **Gotcha** Calling an `async def` without `await` gives a truthy coroutine, so `assert fn()` tests nothing.

## Coverage and its limits
<!-- topics: coverage -->

Coverage measures **which lines (and optionally branches) ran** during the tests. With `pytest-cov`:

```bash
pytest --cov=mypkg --cov-branch --cov-report=term-missing
pytest --cov=mypkg --cov-fail-under=85     # CI gate
```

`term-missing` lists untested line numbers; branch coverage checks both sides of each `if`.

The limit: coverage proves code *ran*, never that the result was *checked*. A test with no assertions
gives the same coverage as a perfect one. 100% coverage with weak assertions is worth little, and 70%
with sharp tests on the risky paths is worth more. A good check is **mutation testing** thinking: if
you changed `>` to `>=`, would some test fail? If not, the test is too weak.

> **Remember** Coverage finds untested code, not missing tests or bad assertions. A floor, not a goal.

## Debugging with pdb
<!-- topics: debugging -->

`breakpoint()` (Python 3.7+) drops into the debugger at that line; it calls `pdb.set_trace()` unless
`PYTHONBREAKPOINT` says otherwise (`PYTHONBREAKPOINT=0` disables all of them, handy in CI).

Commands: `n` next, `s` step into, `c` continue, `r` run to return, `p expr` / `pp` print, `l` list source,
`w` stack, `u`/`d` move up/down, `q` quit.

`pytest --pdb` opens the debugger at the failing assert; `python -m pdb script.py` for any script.

Often cheaper: read the traceback bottom-up, shrink the failing input, write the reproduction as a test first.

> **Gotcha** A leftover `breakpoint()` hangs a service or CI job waiting for stdin. Lint rules
> (ruff `T100`) flag them.

## The logging module
<!-- topics: logging, logging_module -->

`print` is for scripts; services use `logging`: levels, destinations, structure and per-module control.
Moving parts: loggers (named, levelled), handlers (where), formatters (how), propagation.

```text
logger.info("x") -> Logger "app.rag" (level check) -> its Handlers (each with a Formatter)
                    -> if propagate: parent Logger "app" handlers -> ... -> root
```

Loggers form a hierarchy by dotted name: `app.rag` is a child of `app`, which is a child of root. A
logger with no level of its own uses its nearest ancestor's (**effective level**). Levels in order:
`DEBUG(10) < INFO(20) < WARNING(30) < ERROR(40) < CRITICAL(50)`; the root default is WARNING.

```python
import io
import logging

log = logging.getLogger("app.rag")
print(log.getEffectiveLevel(), log.propagate)   # prints: 30 True

buf = io.StringIO()
handler = logging.StreamHandler(buf)
handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
parent = logging.getLogger("app")
parent.addHandler(handler)
parent.setLevel(logging.INFO)

log.debug("hidden")                # below INFO
log.info("loaded %d docs", 3)      # handled by the parent's handler (propagation)
log.warning("slow")
print(buf.getvalue(), end="")
# prints: INFO app.rag: loaded 3 docs
# prints: WARNING app.rag: slow

log.propagate = False
log.info("not delivered")
print(len(buf.getvalue().splitlines()))   # prints: 2
```

Rules worth memorising:

* **`logger = logging.getLogger(__name__)` in every module.** Never configure handlers in library code;
  only the application entry point calls `logging.basicConfig(...)` or `dictConfig`.
* **Lazy formatting**: `log.info("got %s", x)`, not an f-string, so the string is only built if the
  level is enabled.
* `log.exception("failed")` inside `except` logs at ERROR with the traceback.
* **Never log secrets** (API keys, tokens, full prompts with PII). Redact or log a prefix/hash.

Testing logs: use `caplog` (`assert "429" in caplog.text`, or check `record.levelname` and
`record.name`). Treat levels and message templates as a contract, since alerts depend on them.

> **AI use** Log per LLM call: model, latency, token counts, retries, request id. Not the key.

## Formatting, linting, type checking and PEP 8
<!-- topics: formatting, linting, type_checking, pep8 -->

Three different jobs, often confused:

| job | tool | question it answers |
|---|---|---|
| **format** | `ruff format` (or black) | how should this code look? (automatic, no debate) |
| **lint** | `ruff check` (flake8, pylint) | does this code smell? unused imports, bugs, style |
| **type check** | `mypy` (or pyright) | do the types line up? |

```bash
ruff format .            # rewrite files to a canonical style
ruff check . --fix       # report (and auto-fix) lint problems
mypy src/                # static type check, no code is run
```

**PEP 8** is the style guide: 4-space indents, `snake_case` for functions/variables, `PascalCase` for
classes, `UPPER_CASE` for constants, line length around 79-100 (the formatter decides), two blank lines
around top-level definitions, imports grouped stdlib / third-party / local, `is None` not `== None`,
no bare `except:`. Formatters enforce layout; linters enforce the rest. Put the configuration in
`pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`) and run all three in CI and a pre-commit hook.

mypy never runs your code, and annotations are not enforced at runtime: `total_tokens(["a"])` on a
`list[int]` function fails only under mypy (`List item 0 has incompatible type "str"; expected "int"`).

Add types where they pay off: public functions, data shapes (`TypedDict`, dataclass, pydantic), `Optional`.

> **Interview** "Linter vs type checker?" The linter looks at syntax patterns and style without
> understanding types; the type checker does flow analysis on annotations. They catch different bugs,
> so you run both.

## Dependency management
<!-- topics: dependency_management -->

The goal is a **reproducible environment** on your laptop, CI and production.

* Use a **virtual environment** per project (`python -m venv .venv`, or `uv venv`) so projects do not
  fight over versions.
* Declare **direct dependencies with version ranges** in `pyproject.toml` (`httpx>=0.27,<1`).
* Pin the **full resolved tree** in a lock file (`uv.lock`, `poetry.lock`, or a `pip-compile` output).
  Install from the lock in CI and production. Ranges say what you accept; the lock says what you tested.
* Separate **dev dependencies** (pytest, ruff, mypy) from runtime ones (optional groups or extras).
* Upgrade deliberately (bot PR plus test suite); audit with `pip-audit`.

> **Gotcha** An unpinned `openai` or `langchain` will eventually break your build; the lock makes upgrades deliberate.

## Testing non-deterministic LLM apps
<!-- topics: unit, integration, mocking -->

A model's output varies run to run, so exact string equality is the wrong assertion, and calling the
real model in unit tests is slow, costly and flaky. Split the problem in layers:

1. **Test everything around the model deterministically** with a fake client: prompt construction,
   parsing, retries, validation, caching, fallbacks. Most of your bugs live here.
2. **Assert on properties, not text**: output is valid JSON matching a schema, the answer cites a
   provided source, length is in bounds, no banned phrases, the score is in 1..5.
3. **Golden set evals**: 30-200 curated inputs with expected facts or labels, scored automatically
   (exact match on extracted fields, retrieval recall@k, or an LLM judge for open text). Run on demand
   and in CI nightly, compare against a stored baseline, and fail on regression beyond a threshold.
4. **Pin what you can** (temperature 0, fixed model version, seed) and **record/replay** real responses as fixtures.

```python
import json

class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
    def complete(self, prompt):
        return next(self.replies)

def extract_person(text, llm, retries=1):
    for _ in range(retries + 1):
        raw = llm.complete(f"Return JSON with name and age: {text}")
        try:
            data = json.loads(raw)
            if isinstance(data.get("name"), str) and isinstance(data.get("age"), int):
                return data
        except json.JSONDecodeError:
            pass
    raise ValueError("model never produced valid output")

# the model misbehaves first (chatty prose), then complies: test the repair path offline
llm = FakeLLM(["Sure! Here you go.", '{"name": "Ada", "age": 36}'])
print(extract_person("Ada is 36", llm))   # prints: {'name': 'Ada', 'age': 36}
```

```

> **Interview** "How do you test an LLM feature?" Fake client for deterministic code, schema/property
> assertions, a versioned golden set with thresholds, LLM-as-judge only with a spot-checked rubric, then
> production monitoring.

## Quick recap
- Pyramid: many fast isolated unit tests, fewer integration tests with real parts, few end-to-end, plus an eval layer for LLMs.
- pytest discovers `test_*`, rewrites asserts for readable diffs, passes if nothing raises.
- Fixtures inject dependencies, `yield` gives teardown, scope controls reuse; keep mutable ones function-scoped.
- `tmp_path`, `monkeypatch`, `caplog` give isolation for files, env vars and logs.
- Parametrize one test over a table of cases; each row is its own test.
- Prefer fakes and state-based assertions; use mocks when the interaction is the behaviour; inject dependencies.
- Patch where the name is looked up, use `autospec` so mocks keep the real signature.
- `AsyncMock` plus `side_effect` simulates failures; inject `sleep` so retry tests are instant; never forget `await`.
- Coverage shows what ran, not what was verified.
- `breakpoint()` / `pytest --pdb` for debugging; reproduce with a test first.
- Logging: `getLogger(__name__)`, hierarchy and propagation, handlers/formatters configured only by the app, lazy `%s` args, never secrets.
- Format (ruff format), lint (ruff check), type check (mypy) are three separate jobs; PEP 8 is the style baseline.
- Reproducible installs: venv, ranges in pyproject, lock file.
- Test LLM apps by faking the client, asserting properties/schemas, and running golden-set evals with thresholds.
