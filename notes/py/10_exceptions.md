# Exceptions

An exception is Python's way of saying "this line cannot continue", and it travels up the call stack until
something handles it. Good exception handling is what separates a script that dies on the first flaky API
response from a service that retries rate limits, fails fast on bad input and leaves a readable traceback when
something truly unexpected happens. In AI engineering you spend your life around failing calls (timeouts,
429s, malformed JSON from an LLM), so interviewers probe hierarchy, chaining, `finally` semantics and what
you log.

## Syntax errors versus runtime errors
<!-- topics: syntax_vs_runtime -->

A **syntax error** is found while the source is being compiled, before any line runs. A **runtime error**
(an exception) happens while executing valid code.

```python
try:
    compile("x = (1 +", "<demo>", "exec")      # the parser refuses this
except SyntaxError as e:
    print(type(e).__name__, "-", e.msg)
# prints: SyntaxError - '(' was never closed

try:
    1 / 0                                       # valid syntax, fails when run
except ZeroDivisionError as e:
    print(e)                                    # prints: division by zero
```

Consequence: a file with a syntax error on its last line runs *nothing*, not even the first line. And you
normally cannot catch a `SyntaxError` in the same file that contains it (it happens at import or `compile`
time, so you can catch it around `import`, `eval`, `exec`). `IndentationError` and `TabError` are subclasses.

> **Gotcha** `NameError` and `TypeError` are runtime errors, not syntax errors. A misspelled variable inside
> a function that is never called never fails. That is the gap type checkers (see the type hints module) fill.

## try, except, else, finally
<!-- topics: try_except -->

```text
try:      risky code
except:   runs only if the try raised a matching exception
else:     runs only if the try raised NOTHING (and did not return/break out)
finally:  ALWAYS runs, last, on every exit path
```

```python
def parse(s):
    try:
        n = int(s)
    except ValueError:
        print("bad int")
        return None
    else:
        print("parsed")          # only on success
        return n
    finally:
        print("cleanup")         # even though return already happened

print(parse("7"))    # prints: parsed, cleanup, then 7
print(parse("x"))    # prints: bad int, cleanup, then None
```

Why `else`? It keeps the `try` body as small as possible. Code in `else` that raises is **not** caught by the
sibling `except`, so you do not accidentally swallow a bug in the success path.

**`finally` and `return`.** `finally` runs before the function actually returns, and a `return` inside
`finally` *overrides* everything, including a pending exception:

```python
def f():
    try:
        return "try"
    finally:
        print("finally runs first")

print(f())          # prints: finally runs first, then try

def g():
    try:
        raise RuntimeError("lost")
    finally:
        return "swallowed"      # the exception vanishes

print(g())          # prints: swallowed
```

> **Gotcha** `return`, `break` or `continue` inside `finally` silently discards exceptions. Python 3.14
> (PEP 765) emits a `SyntaxWarning` for it at compile time; 3.13 stays silent. Never do it.

> **Remember** Use `finally` (or better, `with`) for cleanup that must happen; use `else` for code that
> belongs to the success path only.

<details markdown="1"><summary>Check yourself: what does `f()` return if `try` has `return 1` and `finally` has `print("x")`?</summary>

It prints `x` and then returns 1. The return value is computed first, `finally` runs, then the function returns it.

</details>

## Handling multiple exceptions
<!-- topics: multiple -->

Handlers are tried **top to bottom** and the **first matching one wins**. Match several types in one clause
with a tuple. Put specific types before general ones.

```python
def classify(value):
    try:
        return int(value) + [1, 2][value if isinstance(value, int) else 0]
    except (ValueError, TypeError) as e:       # one handler, two types
        return f"input problem: {type(e).__name__}"
    except IndexError:
        return "index problem"

print(classify("x"))   # prints: input problem: ValueError
print(classify(5))     # prints: index problem
print(classify(None))  # prints: input problem: TypeError
```

Reverse the order (`except Exception` first) and the specific handlers become dead code. Python does not warn.

`except*` and **ExceptionGroup** (3.11+) handle *several unrelated errors raised at once*, which is what
`asyncio.TaskGroup` and concurrent fan-out produce:

```python
def run():
    raise ExceptionGroup("batch failed", [ValueError("bad row"), TimeoutError("slow"), KeyError("k")])

try:
    run()
except* ValueError as eg:
    print("values:", [str(e) for e in eg.exceptions])   # prints: values: ['bad row']
except* (TimeoutError, KeyError) as eg:
    print("others:", len(eg.exceptions))                # prints: others: 2
```

Every `except*` clause sees only the matching sub-group, and each runs (unlike plain `except`, where the first
match ends the search). Unhandled leftovers are re-raised as a group. You cannot mix `except` and `except*`
in one `try`.

> **Python 3.14** PEP 758 allows `except A, B:` without parentheses when there is no `as`. Parenthesised
> tuples still work everywhere and are what you should write for 3.12/3.13 code.

> **AI use** Firing 20 embedding requests with `asyncio.TaskGroup` and getting 3 different failures back as one
> `ExceptionGroup` is routine. `except*` lets you retry the `RateLimitError`s and propagate the rest.

## The exception hierarchy
<!-- topics: hierarchy -->

Every exception derives from `BaseException`. Almost everything you catch derives from `Exception`.

```text
BaseException
 +-- SystemExit, KeyboardInterrupt, GeneratorExit, (asyncio.CancelledError)
 +-- Exception
      +-- ArithmeticError  (ZeroDivisionError, OverflowError)
      +-- LookupError      (KeyError, IndexError)
      +-- OSError          (FileNotFoundError, PermissionError, TimeoutError, ConnectionError)
      +-- ValueError       (UnicodeError, json.JSONDecodeError)
      +-- TypeError, AttributeError, NameError, ImportError, RuntimeError, StopIteration ...
```

Why it matters: `except Exception` deliberately does **not** catch Ctrl-C (`KeyboardInterrupt`), `sys.exit()`
(`SystemExit`) or task cancellation. A bare `except:` or `except BaseException` does, making your program
impossible to stop.

```python
print(issubclass(KeyError, LookupError))            # prints: True
print(issubclass(KeyboardInterrupt, Exception))     # prints: False
print(issubclass(FileNotFoundError, OSError))       # prints: True
print([c.__name__ for c in ConnectionResetError.__mro__][:4])
# prints: ['ConnectionResetError', 'ConnectionError', 'OSError', 'Exception']
import json
print(issubclass(json.JSONDecodeError, ValueError)) # prints: True
```

Catching a parent catches all children: `except LookupError` handles both `KeyError` and `IndexError`;
`except OSError` handles missing files, refused connections and timeouts alike.

> **Gotcha** `asyncio.CancelledError` is a `BaseException` since 3.8. Swallowing it with `except Exception` is
> fine, but `except BaseException` that does not re-raise breaks task cancellation and timeouts.

## Raising exceptions
<!-- topics: raising -->

`raise Cls("message")` creates and raises. A bare `raise` inside an `except` block **re-raises the current
exception with its original traceback**, which is what you want after logging or cleanup.

```python
def load(cfg):
    if "model" not in cfg:
        raise KeyError("model")           # raise an instance (or a class: Python instantiates it)
    return cfg["model"]

def wrapper(cfg):
    try:
        return load(cfg)
    except KeyError:
        print("logging then re-raising")
        raise                             # same exception, same traceback

try:
    wrapper({})
except KeyError as e:
    print("caught again:", e)
# prints: logging then re-raising
# prints: caught again: 'model'
```

Prefer `raise` over `raise e`: the latter adds a redundant traceback frame. Pick the *right built-in* type:
`ValueError` (right type, bad value), `TypeError` (wrong type), `KeyError`/`IndexError` (missing key/position),
`RuntimeError` (state is wrong), `NotImplementedError` (abstract hook). Raising bare `Exception("...")` forces
callers to catch everything.

> **Gotcha** The name bound by `except E as e` is **deleted** when the block ends. Use it inside the block or
> copy it out: `saved = e`.

## Custom exceptions
<!-- topics: custom -->

Define your own when callers need to *distinguish* failure kinds and react differently. Make one base class per
library or app, then subclass it, so callers can catch "anything from my package" with one clause.

```python
class LLMError(Exception):
    """Base for everything our client raises."""

class RateLimitError(LLMError):
    def __init__(self, retry_after: float):
        super().__init__(f"rate limited, retry in {retry_after}s")
        self.retry_after = retry_after

class InvalidRequestError(LLMError):
    """Permanent: retrying will not help."""

def call(status):
    if status == 429:
        raise RateLimitError(2.5)
    if status == 400:
        raise InvalidRequestError("prompt too long")
    return "ok"

for s in (200, 429, 400):
    try:
        print(call(s))
    except RateLimitError as e:
        print("retry after", e.retry_after)
    except LLMError as e:
        print("give up:", e)
# prints: ok
# prints: retry after 2.5
# prints: give up: prompt too long
```

Rules of thumb: inherit from `Exception` (never `BaseException`), name them `...Error`, keep structured data
as attributes (status code, retry_after, request id) rather than parsing the message. The OpenAI and Anthropic
SDKs expose exactly this shape: a base `APIError`, with `RateLimitError`, `APITimeoutError`,
`AuthenticationError` beneath it.

> **Interview** "Which errors do you retry?" Transient ones (rate limit, timeout, 5xx, connection reset) with
> exponential backoff and jitter. Never 4xx validation or auth errors. A class hierarchy makes that decision a
> one-line `except`.

## Exception chaining
<!-- topics: chaining -->

When you raise a new exception while handling another, Python links them so the root cause is not lost.

* **Implicit** (`__context__`): raising inside an `except` block. Traceback says "During handling of the
  above exception, another exception occurred".
* **Explicit** (`raise New from old`, sets `__cause__`): "The above exception was the direct cause of...".
* **Suppress** (`raise New from None`): hides the context, sets `__suppress_context__`.

```python
import json

class ConfigError(Exception): ...

def read(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError("invalid config") from e

try:
    read("{bad")
except ConfigError as e:
    print(type(e.__cause__).__name__)      # prints: JSONDecodeError
    print(e.__context__ is e.__cause__)    # prints: True

try:
    try:
        1 / 0
    except ZeroDivisionError:
        raise ValueError("hidden") from None
except ValueError as e:
    print(e.__cause__, e.__suppress_context__)   # prints: None True
```

Why wrap at all? Layer boundaries: your service should raise `ConfigError`, not leak `JSONDecodeError`
implementation details, yet still keep the original for debugging. Use `from None` only to drop *noise*
(for example a `KeyError` you converted to a friendlier message), never to hide a real cause.

> **Remember** Inside `except`, new `raise X` -> implicit context; `raise X from e` -> explicit cause. Always
> chain when translating errors.

## assert
<!-- topics: assert -->

`assert cond, "message"` raises `AssertionError` if `cond` is falsy. It is for **bugs and invariants**, not for
validating input. Running `python -O` strips every `assert` statement entirely, so a check that guards real
data disappears in optimised mode.

```python
def mean(xs):
    assert len(xs) > 0, "mean() of empty list is a programming error"
    return sum(xs) / len(xs)

try:
    mean([])
except AssertionError as e:
    print("assertion:", e)     # prints: assertion: mean() of empty list is a programming error
```

| Use | `assert` | `raise ValueError` |
|---|---|---|
| Internal invariant ("cannot happen") | yes | no |
| User / API / file input | **no** | yes |
| Survives `python -O` | no | yes |

> **Gotcha** `assert (cond, "msg")` asserts a non-empty tuple, which is always true. Python warns with a
> `SyntaxWarning`. Write `assert cond, "msg"`. Pytest asserts are a separate, rewritten mechanism.

## Contextual error handling
<!-- topics: contextual -->

Handle an error **where you have enough information to do something useful**, and add context as it travels up.
Low-level code usually cannot decide; the caller that knows the *intent* can.

```python
def read_line(raw):
    return int(raw)                      # no handling: it cannot know what to do

def load_scores(lines):
    scores, bad = [], []
    for i, raw in enumerate(lines, start=1):
        try:
            scores.append(read_line(raw))
        except ValueError as e:
            bad.append((i, str(e)))      # this layer knows "skip and report"
    return scores, bad

print(load_scores(["1", "x", "3"]))
# prints: ([1, 3], [(2, "invalid literal for int() with base 10: 'x'")])
```

Add context with chaining (`raise ... from e`) or, in 3.11+, `e.add_note()`, which appears in the traceback
without changing the type:

```python
try:
    try:
        int("abc")
    except ValueError as e:
        e.add_note("while parsing doc_id=42")
        raise
except ValueError as e:
    print(e.__notes__)     # prints: ['while parsing doc_id=42']
```

Also: **EAFP** ("ask forgiveness") is idiomatic Python: try the operation and handle the exception, instead of
pre-checking (LBYL), which can race (the file can vanish between `exists()` and `open()`).

> **AI use** In a batch job, catch per-record failures, record `(doc_id, error)` in a dead-letter list and
> continue. One poisoned PDF should not kill a 10 000-document run.

## Logging exceptions
<!-- topics: logging -->

`print` loses the traceback and goes nowhere in production. Use `logging`. Inside an `except`,
`logger.exception("msg")` logs at ERROR level **with the full traceback**. `logger.error("msg", exc_info=True)`
is equivalent.

```python
import logging, sys

logging.basicConfig(stream=sys.stdout, format="%(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("ingest")

try:
    int("x")
except ValueError:
    log.exception("failed to parse")
# prints: ERROR failed to parse
# prints: Traceback (most recent call last): ... (the full traceback follows)
```

Do not do `log.error(str(e))` (no traceback) and do not log-and-raise at every layer (you get the same error
logged five times). **Log once, at the boundary that handles it**, or re-raise without logging. Use
lazy formatting: `log.warning("retry %d for %s", n, url)`, not f-strings.

> **Gotcha** Logging then swallowing hides failures. If you log and continue, be sure continuing is
> actually correct.

## Best practices
<!-- topics: best_practices -->

* **Catch narrow, try small.** `except Exception: pass` hides typos and real bugs. Wrap only the line that can
  fail.
* **Never bare `except:`** (catches `KeyboardInterrupt`). At a true top-level boundary (worker loop, request
  handler) `except Exception` plus logging is acceptable.
* **Fail fast on programmer errors**, recover only from *expected, recoverable* conditions.
* **Do not use exceptions for ordinary control flow** when a plain check is clearer, but note that
  `StopIteration`, `KeyError` fallbacks and EAFP are normal Python.
* **Clean up with `with`/`finally`**, not by hoping the happy path runs.
* **Chain** when translating; **preserve** types callers rely on.
* **Retry with a bounded loop**, backoff and jitter, only on transient errors:

```python
import time

class Transient(Exception): ...

def with_retry(fn, attempts=3, base=0.0):
    for n in range(1, attempts + 1):
        try:
            return fn()
        except Transient:
            if n == attempts:
                raise                     # out of budget: let the caller see it
            time.sleep(base * 2 ** (n - 1))

calls = []
def flaky():
    calls.append(1)
    if len(calls) < 3:
        raise Transient
    return "ok"

print(with_retry(flaky), len(calls))   # prints: ok 3
```

> **Interview** "Is `except Exception: pass` ever OK?" Almost never; a strong answer names the one place
> (best-effort cleanup such as closing a socket in `finally`) and says it should still log at debug level.

## Quick recap
- Syntax errors stop the file before it runs; runtime errors are exceptions raised during execution.
- `else` = success-only code; `finally` = always runs; `return` in `finally` swallows exceptions (3.14 warns).
- First matching `except` wins; order specific before general. `except*` handles groups from concurrent work.
- `Exception` is the app-level base; `BaseException` adds `KeyboardInterrupt`, `SystemExit`, cancellation.
- Bare `raise` re-raises with the original traceback; the `as e` name vanishes after the block.
- One app base class + specific subclasses with structured attributes makes retry policies trivial.
- `raise X from e` sets `__cause__`; implicit chaining sets `__context__`; `from None` suppresses.
- `assert` is for invariants and disappears under `-O`; validate real input with exceptions.
- `logger.exception` in an `except` block logs the traceback; log once, at the handling boundary.
- Catch narrow, keep `try` small, retry only transient errors with bounded backoff.
