# Context managers

A context manager is an object that sets something up when a `with` block starts and tears it down when the block
ends, no matter how it ends. That guarantee is how Python handles files, locks, database transactions, HTTP
sessions, temporary state and timers without leaks. In AI engineering you meet them everywhere: `with open`,
`with httpx.Client()`, `with torch.no_grad()`, `with tracer.span(...)`, `with timer()`. Interviews ask what `with`
really expands to, what `__exit__` returning True means, and how `@contextmanager` and `ExitStack` work.

## The protocol
<!-- topics: protocol -->

Any object with `__enter__` and `__exit__` methods is a context manager.

```text
with EXPR as NAME:        manager = EXPR
    BODY                  NAME = manager.__enter__()        <- return value goes to NAME
                          BODY runs
                          manager.__exit__(exc_type, exc, tb)   <- ALWAYS called
```

```python
class Demo:
    def __enter__(self):
        print("enter")
        return "resource"           # becomes the `as` target

    def __exit__(self, exc_type, exc, tb):
        print("exit", exc_type.__name__ if exc_type else None)
        return False                # do not swallow exceptions

with Demo() as r:
    print("body", r)
# prints: enter
# prints: body resource
# prints: exit None
```

## __enter__ and __exit__ semantics
<!-- topics: enter, exit -->

**`__enter__`** acquires the resource and returns what you want bound to the `as` name. It often returns `self`,
but it does not have to (`open` returns the file; a lock returns `True`; a transaction might return a cursor). If
`__enter__` raises, `__exit__` is **not** called, because setup never completed.

**`__exit__(exc_type, exc_value, traceback)`** gets `(None, None, None)` on a clean exit, otherwise the exception
details. Its **return value** matters:

| Returns | Effect |
|---|---|
| falsy (`None`, `False`) | the exception, if any, **propagates** after cleanup |
| truthy (`True`) | the exception is **suppressed**; execution continues after the `with` |

```python
class Suppress:
    def __enter__(self): return self
    def __exit__(self, et, ev, tb):
        print("suppressing", et.__name__)
        return True

with Suppress():
    raise ValueError("ignored")
print("still running")
# prints: suppressing ValueError
# prints: still running
```

> **Gotcha** Forgetting `return` in `__exit__` returns `None`, so exceptions propagate. That is the safe default.
> Returning True by accident (for example `return self.cleanup()` that happens to be truthy) silently eats bugs.
> Also never `raise` the received exception yourself; just return falsy and Python re-raises it.

## What the with statement really does
<!-- topics: with_internals -->

`with` is not just `try/finally`. Roughly, Python expands it to:

```text
mgr = EXPR
enter, exit = type(mgr).__enter__, type(mgr).__exit__    # looked up on the TYPE
value = enter(mgr)
try:
    NAME = value
    BODY
except BaseException as e:
    if not exit(mgr, type(e), e, e.__traceback__):
        raise                                         # falsy -> re-raise
else:
    exit(mgr, None, None, None)                       # clean exit
```

Consequences you can state in an interview:

* The methods are looked up on the **class**, so assigning `obj.__enter__ = ...` on an instance does nothing.
* `__exit__` runs on `return`, `break` and `continue` inside the body too.
* If `__exit__` itself raises, that new exception replaces the original (chained via `__context__`).
* `BaseException` is covered, so Ctrl-C still triggers cleanup.
* Several managers in one statement nest and exit in **reverse** order:

```python
from contextlib import contextmanager

@contextmanager
def tag(name):
    print("open", name)
    try:
        yield
    finally:
        print("close", name)

with tag("a"), tag("b"):
    print("inside")
# prints: open a
# prints: open b
# prints: inside
# prints: close b
# prints: close a
```

Parenthesised multi-line `with (open(a) as x, open(b) as y):` is supported from 3.10.

<details markdown="1"><summary>Check yourself: `__enter__` raises. Is `__exit__` called?</summary>

No. The `with` only guarantees `__exit__` once `__enter__` has succeeded. If `__enter__` acquires two things,
it must clean up the first itself if the second fails.

</details>

## Writing a class-based context manager
<!-- topics: custom -->

Use a class when the manager has state, is reusable, or needs both a rich API and cleanup.

```python
import time

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, et, ev, tb):
        self.elapsed = time.perf_counter() - self.start
        return False

with Timer() as t:
    sum(range(1000))
print(t.elapsed >= 0)     # prints: True
```

A transaction manager shows the point of `exc_type`: commit on success, roll back on error, never swallow.

```python
class Transaction:
    def __init__(self, log): self.log = log
    def __enter__(self):
        self.log.append("begin"); return self
    def __exit__(self, et, ev, tb):
        self.log.append("rollback" if et else "commit")
        return False

log = []
try:
    with Transaction(log):
        raise RuntimeError("db error")
except RuntimeError:
    pass
with Transaction(log):
    pass
print(log)   # prints: ['begin', 'rollback', 'begin', 'commit']
```

Reusability: a manager that stores state on `self` in `__enter__` (like `Timer`) breaks if you nest or reuse the
same instance concurrently. Return a fresh object, or store state per entry, if it must be reentrant. Async
code uses `__aenter__`/`__aexit__` with `async with`.

## contextlib.contextmanager
<!-- topics: contextmanager -->

`@contextmanager` turns a **generator with exactly one `yield`** into a context manager. Code before `yield` is
`__enter__`; the yielded value is the `as` target; code after is `__exit__`.

```python
from contextlib import contextmanager

@contextmanager
def opened(name):
    print("setup", name)
    try:
        yield name.upper()
    finally:
        print("teardown", name)

with opened("db") as x:
    print("using", x)
# prints: setup db
# prints: using DB
# prints: teardown db
```

If the body raises, the exception is **thrown into the generator at the `yield`**. Without `try/finally` your
teardown is skipped. To suppress, catch it and do not re-raise:

```python
from contextlib import contextmanager

@contextmanager
def ignore(*types):
    try:
        yield
    except types as e:
        print("ignored", type(e).__name__)

with ignore(KeyError):
    {}["x"]
print("continued")
# prints: ignored KeyError
# prints: continued
```

The most common mistake: omitting `try/finally`.

```python
from contextlib import contextmanager

@contextmanager
def bad():
    print("setup")
    yield
    print("cleanup")       # skipped when the body raises

try:
    with bad():
        raise ValueError
except ValueError:
    print("no cleanup ran")
# prints: setup
# prints: no cleanup ran
```

The standard library ships useful ones: `contextlib.suppress(...)`, `redirect_stdout`, `closing(obj)` (calls
`.close()`), `nullcontext()` (no-op placeholder, handy for optional managers), and `chdir`.

| Choose | When |
|---|---|
| `@contextmanager` | short setup/teardown, no extra API |
| class with `__enter__/__exit__` | state, reuse, extra methods, subclassing |

> **Interview** "How does `@contextmanager` work?" A wrapper object whose `__enter__` calls `next()` on your
> generator and whose `__exit__` either resumes it (clean exit) or `throw()`s the exception into it.

## ExitStack
<!-- topics: exitstack -->

`ExitStack` is a context manager that manages a **dynamic number** of other context managers (or callbacks), then
unwinds them in reverse order. Use it when you cannot write the `with` statements statically.

```python
from contextlib import ExitStack, contextmanager

@contextmanager
def res(n):
    print("open", n)
    yield n
    print("close", n)

with ExitStack() as stack:
    items = [stack.enter_context(res(i)) for i in range(3)]
    stack.callback(print, "callback last registered, runs first")
    print("items", items)
# prints: open 0
# prints: open 1
# prints: open 2
# prints: items [0, 1, 2]
# prints: callback last registered, runs first
# prints: close 2
# prints: close 1
# prints: close 0
```

Typical uses: open N files whose count is only known at runtime (`for p in paths: stack.enter_context(open(p))`),
conditionally enter a manager, and register arbitrary cleanup with `callback()`. If entering the 3rd resource
fails, the first two are still closed. `stack.pop_all()` transfers ownership so you can keep resources open past
the `with` (used inside `__enter__` of your own classes).

> **Remember** `ExitStack` = a `with` for "I do not know how many". Exits run in LIFO order, like nested `with`.

## Resource management
<!-- topics: resources -->

A **resource** is anything you must give back: file descriptors, sockets, locks, DB connections, GPU memory,
temporary state. Rely on `with`, not on garbage collection or `__del__`, because release timing in `__del__` is
unspecified and it can run during interpreter shutdown.

The pattern "change global state, then restore it" is a perfect fit:

```python
import os
from contextlib import contextmanager

@contextmanager
def env(key, value):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[key]
        else:
            os.environ[key] = old

with env("MODEL", "small"):
    print(os.environ["MODEL"])      # prints: small
print("MODEL" in os.environ)        # prints: False
```

Other familiar examples: `threading.Lock` (`with lock:` releases on error, so no deadlock from a forgotten
release), `decimal.localcontext()`, `unittest.mock.patch`, `torch.no_grad()`, `httpx.Client()`.

Generators, by contrast, hold resources until closed: `with closing(gen)` or keeping `open` inside a `with`
around the loop avoids leaking a file behind an abandoned generator.

> **Gotcha** `x = open(p).read()` leaves closing to the garbage collector, and `ResourceWarning` shows when
> warnings are enabled. Always wrap in `with` (or use `Path.read_text`).

> **AI use** A pooled `httpx.Client` or `AsyncClient` in a `with` block reuses TCP connections for many LLM
> calls and closes them deterministically. Creating a new client per request wastes connections and latency.

## Database and file cleanup
<!-- topics: cleanup -->

Cleanup contexts guarantee **commit/rollback** and **close** even when code between raises. Beware that the
driver's own context manager differs from what you may assume. `sqlite3.Connection` as a context manager manages
the *transaction* (commit or rollback) but does **not** close the connection:

```python
import sqlite3
from contextlib import closing

with closing(sqlite3.connect(":memory:")) as conn:    # closing -> conn.close() at the end
    conn.execute("create table t (x integer)")
    try:
        with conn:                                    # transaction scope
            conn.execute("insert into t values (1)")
            raise RuntimeError("fail after insert")
    except RuntimeError:
        pass
    print(conn.execute("select count(*) from t").fetchone()[0])   # prints: 0  (rolled back)

    with conn:
        conn.execute("insert into t values (2)")
    print(conn.execute("select count(*) from t").fetchone()[0])   # prints: 1  (committed)
```

Layering is the pattern: outer manager for the connection lifetime, inner one for each transaction. For files, the
same ideas appear as: close on error, flush before success is reported, and write to a temp file then rename
(see the files module). When cleanup itself can fail, put the cleanup in `finally`/`__exit__` but keep it from
masking the original error: log the secondary failure instead of raising it.

> **Interview** "Does `with sqlite3.connect()` close the connection?" No, it commits or rolls back. Wrap it in
> `contextlib.closing` or close it explicitly. SQLAlchemy sessions and psycopg connections have their own rules,
> so read what `__exit__` does.

## Quick recap
- `with` calls `__enter__` (its return value is the `as` target) then guarantees `__exit__(type, exc, tb)`.
- `__exit__` returning truthy suppresses the exception; falsy or `None` lets it propagate. Default to falsy.
- If `__enter__` raises, `__exit__` does not run. Methods are looked up on the type.
- Multiple managers exit in reverse (LIFO) order.
- `@contextmanager`: one `yield`; wrap it in `try/finally` or teardown is skipped on error.
- Use a class for state and reuse, a generator for short setup/teardown.
- `ExitStack` handles a runtime-determined number of managers or callbacks, unwinding LIFO.
- Prefer `with` to `__del__` or GC for any resource; restore global state in `finally`.
- `sqlite3` connection-as-manager handles transactions only; add `closing()` to close.
