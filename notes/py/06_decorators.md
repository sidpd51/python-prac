# Decorators

A decorator is a function that takes a function and returns a replacement for it, usually one that
does the same thing plus something extra: timing, retrying, caching, logging, access control. The
`@` line is only syntax sugar. Decorators are everywhere in AI engineering code: `@retry` around flaky
provider calls, `@timed` and `@observe` for tracing, `@lru_cache` on embeddings, `@tool` to register
functions for an agent, `@app.post` in FastAPI. Interviewers ask you to write one from scratch and
then probe `functools.wraps`, decorators with arguments, and stacking order. This note builds on
closures (see the closures note): every wrapper below is a closure over the decorated function.

## Functions are first-class objects
<!-- topics: first_class, higher_order -->

In Python a function is an ordinary object: you can bind it to a name, put it in a list or dict, pass it
as an argument, return it from another function and attach attributes to it.

```python
def shout(s):
    return s.upper() + "!"

handlers = {"loud": shout, "len": len}      # stored in a dict
print(handlers["loud"]("hi"))               # prints: HI!

def apply(fn, value):                       # passed as an argument
    return fn(value)
print(apply(str.title, "hello world"))      # prints: Hello World

shout.calls = 0                             # attributes on a function object
print(shout.calls)                          # prints: 0
```

A **higher-order function** takes functions and/or returns functions. `map`, `sorted(key=...)`,
`functools.reduce` take functions; factories like `make_multiplier` return them. A decorator is exactly
a higher-order function that takes one function and returns another. Without first-class functions there
would be no decorators.

> **Remember** `f` (the name) is the function object, `f()` is a call. Decorating is about passing around
> and replacing the *object*, so forgetting or adding parentheses is the most common decorator bug.

## Writing a wrapper
<!-- topics: wrappers, wrapping, args_kwargs -->

The pattern has three parts: an outer function that receives the target, an inner **wrapper** that runs
around the call, and a return of the wrapper. The wrapper is a closure over `func`.

```python
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__} with {args} {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result!r}")
        return result
    return wrapper

def add(a, b=0):
    return a + b

add = log_calls(add)      # manual decoration
add(2, b=3)
# prints: calling add with (2,) {'b': 3}
# prints: add returned 5
```

Why `*args, **kwargs`? The wrapper does not know the signature of what it wraps, so it accepts anything and
forwards it unchanged. If you write `def wrapper(x):` the decorator breaks every function that takes a
different number of arguments, or keyword arguments:

```python
def bad(func):
    def wrapper(x):
        return func(x)
    return wrapper

@bad
def power(base, exp=2):
    return base ** exp

try:
    power(3, exp=3)
except TypeError as e:
    print(e)   # prints: bad.<locals>.wrapper() got an unexpected keyword argument 'exp'
```

Three habits of a correct wrapper: take `*args, **kwargs`, forward them untouched, and **return the
result** (forgetting `return` makes every decorated function return `None`).

> **Gotcha** After decoration the name `add` refers to `wrapper`, not the original. Anything you
> do not forward (return value, exceptions, arguments) is changed for all callers.

## The @ syntax and when things run
<!-- topics: syntax -->

This:

```python
# nocheck
@decorator
def f(): ...
```

is exactly this:

```python
# nocheck
def f(): ...
f = decorator(f)
```

The decorator expression runs **once, when the `def` statement executes** (import time for a module
level function). The wrapper runs **on every call**.

```python
def announce(func):
    print(f"decorating {func.__name__}")      # runs at definition time
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")     # runs at call time
        return func(*args, **kwargs)
    return wrapper

@announce                                     # prints: decorating hello
def hello():
    return "hi"

print("defined")                              # prints: defined
hello()                                       # prints: calling hello
hello()                                       # prints: calling hello
```

Because of this, any expensive setup in the decorator body (reading config, opening a client) happens at
import time, and a decorator with a side effect such as registering the function runs even if the function
is never called. That is how `@tool` and `@app.get(...)` registries work: the decorator stores `func` in a dict
and often returns it unchanged.

```python
TOOLS = {}

def tool(func):
    TOOLS[func.__name__] = func     # register, return the original unchanged
    return func

@tool
def search(q):
    return f"results for {q}"

print(list(TOOLS), TOOLS["search"]("cats"))   # prints: ['search'] results for cats
```

> **Interview** "What does `@decorator` do and when does it run?" It rebinds the name to
> `decorator(func)` at definition time; the returned wrapper replaces the function, and runs per call.

## functools.wraps and metadata
<!-- topics: wraps, metadata -->

A plain wrapper hides the original: `__name__`, `__doc__`, annotations and the signature all become the
wrapper's. That breaks debugging, logs, `help()`, pytest, and frameworks that introspect (FastAPI,
tool-calling schemas generated from a function's signature and docstring).

```python
def plain(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@plain
def embed(text: str) -> list[float]:
    """Return an embedding."""
    return [0.0]

print(embed.__name__, embed.__doc__)   # prints: wrapper None
```

`functools.wraps(func)` is itself a decorator that copies `__module__`, `__name__`, `__qualname__`,
`__doc__`, `__annotations__` (and updates `__dict__`) from the original onto the wrapper, and sets
`wrapper.__wrapped__ = func`.

```python
import functools
import inspect

def good(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@good
def embed(text: str, dim: int = 3) -> list[float]:
    """Return an embedding."""
    return [0.0] * dim

print(embed.__name__, embed.__doc__)   # prints: embed Return an embedding.
print(embed.__wrapped__.__name__)      # prints: embed
print(inspect.signature(embed))        # prints: (text: str, dim: int = 3) -> list[float]
```

`inspect.signature` follows `__wrapped__`, so tools see the *original* parameters even though the real
callable takes `*args, **kwargs`. That is why `wraps` matters for frameworks: they build request
validation or an LLM tool schema from the signature. `embed.__wrapped__` also lets you reach the
undecorated function in tests, bypassing e.g. a cache or retry.

> **Gotcha** `wraps` copies metadata, it does not make the wrapper's real signature match; calling with
> wrong arguments still fails inside the wrapper or the original. Also, `__wrapped__` is how
> `inspect.unwrap` peels stacked decorators.

<details markdown="1"><summary>Check yourself: what does `embed.__name__` return without `@wraps`?</summary>

`"wrapper"`. The name `embed` points at the wrapper function object, which has its own `__name__`.
Logs, tracebacks and registries then all say "wrapper".

</details>

## A timing decorator
<!-- topics: timing -->

The standard exercise. Details that interviewers look for: use `time.perf_counter` (monotonic,
high resolution) rather than `time.time` (wall clock, can jump when NTP adjusts it), measure in a
`try/finally` so failures are timed too (slow timeouts matter most), and use `wraps`.

```python
import functools
import time

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            wrapper.last_duration = time.perf_counter() - start
            print(f"{func.__qualname__} finished")   # a real one would log the duration
    wrapper.last_duration = 0.0
    return wrapper

@timed
def fake_llm_call(prompt):
    time.sleep(0.05)
    return f"echo: {prompt}"

print(fake_llm_call("hi"))
# prints: fake_llm_call finished
# prints: echo: hi
print(fake_llm_call.last_duration >= 0.05)   # prints: True
```

An `async def` function needs an
`async` wrapper that `await`s the call; a sync wrapper around a coroutine function would time only the
creation of the coroutine object, not the work.

> **AI use** Production LLM code logs latency, token counts, model name and error type per call. An
> `@observe`-style decorator does this: it should record in `finally`, never swallow exceptions, avoid
> logging full prompts (PII, size), and be safe for both sync and async functions.

## Decorators with arguments
<!-- topics: with_arguments -->

`@retry(times=3)` does not fit the one-level pattern: `retry(times=3)` runs first and must *return the
decorator*. So there are three levels:

```text
retry(times=3)      -> decorator       (level 1: takes the configuration)
decorator(func)     -> wrapper         (level 2: takes the function)
wrapper(*a, **kw)   -> result          (level 3: runs on each call)
```

```python
import functools

def retry(times=3, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"attempt {attempt} failed: {e}")
                    if attempt == times:
                        raise
        return wrapper
    return decorator

calls = {"n": 0}

@retry(times=3, exceptions=(ConnectionError,))
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise ConnectionError("boom")
    return "ok"

print(flaky())
# prints: attempt 1 failed: boom
# prints: attempt 2 failed: boom
# prints: ok
```

`@retry(times=3)` desugars to `flaky = retry(times=3)(flaky)`. The classic bug is forgetting the
parentheses: `@retry` alone passes `flaky` as `times`, and you get a decorator back instead of a function.

To allow both `@retry` and `@retry(times=3)`, accept an optional function first:

```python
import functools

def log(func=None, *, level="INFO"):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*a, **k):
            print(f"[{level}] {f.__name__}")
            return f(*a, **k)
        return wrapper
    if func is not None:          # used bare: @log
        return decorator(func)
    return decorator              # used with arguments: @log(level="DEBUG")

@log
def a(): pass

@log(level="DEBUG")
def b(): pass

a(); b()
# prints: [INFO] a
# prints: [DEBUG] b
```

> **Interview** Real retry decorators: only retry *transient* errors (timeouts, 429, 5xx), not
> programming errors or 4xx; use exponential backoff with jitter; cap total time; make the
> operation idempotent first, since retrying a non-idempotent call can double-charge or double-send.

<details markdown="1"><summary>Check yourself: why are there three nested functions in `@retry(times=3)`?</summary>

`retry(times=3)` is evaluated first and must return the real decorator. The decorator then receives the
function and returns the wrapper. Config, function and call arguments each need their own level.

</details>

## Stacking decorators
<!-- topics: stacking -->

Multiple decorators apply **bottom to top** (closest to the `def` first), so the top one is the
outermost layer. At call time the outermost wrapper runs first.

```python
def tag(name):
    def decorator(func):
        print(f"decorate with {name}")
        def wrapper(*a, **k):
            print(f"{name} before")
            result = func(*a, **k)
            print(f"{name} after")
            return result
        return wrapper
    return decorator

@tag("A")
@tag("B")
def work():
    print("work")

work()
# prints: decorate with B
# prints: decorate with A
# prints: A before
# prints: B before
# prints: work
# prints: B after
# prints: A after
```

`@A @B def f` means `f = A(B(f))`. Think of onion layers: calls travel inward A -> B -> f and results come
back outward.

Order is a design decision, not a style choice. For an LLM client:

```text
@rate_limited      outermost: reject or queue before doing any work
@retry             re-runs everything below it on a transient failure
@timed             measures each individual attempt (above @retry it would time the total)
@cache             innermost: a cache hit skips the call; never cache failures
def call_llm(...)
```

Typical reasoning: put `cache` *inside* `retry` (retry only the real network call, a hit never retries),
put `rate_limited` *outside* `retry` if retries should consume quota (inside if you want each retry
throttled). If `@timed` is above `@retry` you measure total latency including backoff; below it you
measure one attempt. State which one your dashboard needs.

> **Gotcha** Order matters with built-in descriptor decorators too: `@staticmethod` / `@classmethod` /
> `@property` should normally be the *outermost* (top) decorator, because they expect a plain function
> beneath them, and a custom wrapper placed above them receives a descriptor object that is not callable the
> same way.

## Class-based decorators
<!-- topics: class_based -->

A class works as a decorator if instances are callable (`__call__`). The class receives the function in
`__init__` and acts as the wrapper in `__call__`. The big win is **natural, inspectable state**.

```python
import functools

class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)   # like @wraps, for an instance
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)

@CountCalls
def ping():
    return "pong"

ping(); ping()
print(ping.calls, ping.__name__, ping())   # prints: 2 ping pong
```

Choose a class when the decorator holds meaningful state (call counts, a cache, a token bucket), needs
extra methods (`ping.reset()`), or has a lot of configuration. Choose a function otherwise; it is
shorter.

> **Gotcha** A class-based decorator on a *method* breaks: instances of your class are not functions, so
> they have no `__get__` and do not bind `self`. Methods get called without the instance. Either
> implement `__get__` or use a function decorator for methods. With arguments, `__init__` takes the
> config and `__call__(self, func)` returns the wrapper.

## Decorators you already use
<!-- topics: higher_order, syntax -->

`@property`, `@staticmethod`, `@classmethod` and `@functools.cached_property` are decorators too, but
they return **descriptor objects** rather than function wrappers (see the OOP note). `@functools.lru_cache`
and `@functools.cache` return a wrapper object with `.cache_info()` and `.cache_clear()`:

```python
import functools

@functools.lru_cache(maxsize=2)
def square(n):
    print("computing", n)
    return n * n

print(square(3), square(3))   # prints: 9 9   (after a single 'computing 3' line)
print(square.cache_info())    # prints: CacheInfo(hits=1, misses=1, maxsize=2, currsize=1)
```

`lru_cache` requires hashable arguments, and on methods it caches on `self` too and so keeps instances
alive (a memory leak trap). Decorators can also decorate classes (`@dataclass`): the decorator receives
the class and returns a class.

> **AI use** Cache embeddings by text hash with a bounded cache; never `@cache` an LLM completion
> function whose temperature is above 0 unless you really want the same answer back.

## Quick recap
- Functions are objects: pass, return, store and attach attributes to them. A decorator is a higher-order function.
- `@d def f` is `f = d(f)`, executed once at definition; the wrapper runs every call.
- Wrapper = closure; use `*args, **kwargs`, forward everything, return the result.
- Always `@functools.wraps(func)`: keeps `__name__`/`__doc__`, sets `__wrapped__`, lets `inspect.signature` see the real parameters.
- Timing: `perf_counter`, `try/finally`, handle async separately.
- Decorators with arguments are three levels (`config -> decorator -> wrapper`); `@retry` without parentheses is the classic bug.
- Stacking applies bottom to top, calls run top to bottom (`@A @B f` is `A(B(f))`); choose order deliberately (rate limit, retry, timing, cache).
- Class-based decorators suit stateful or multi-method wrappers but need `__get__` to work on methods.
- `property`, `classmethod`, `staticmethod`, `lru_cache` are decorators too.
