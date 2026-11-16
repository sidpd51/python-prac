# Functions

Functions are the unit of reuse, and in Python they are also ordinary objects: you can store them in a dict, pass them
around, and return them. Almost every interesting Python feature (decorators, callbacks, dependency injection, tool
dispatch for LLM agents) rests on two things you must understand precisely: how arguments are bound to parameters,
and how names are resolved (scope). Interviews probe both with small traps: mutable defaults, `*args` ordering,
`f` versus `f()`, and `UnboundLocalError`.

## Defining and calling functions, return values
<!-- topics: definition, return_values -->

`def` is an **executable statement**: when it runs, it creates a function object and binds the name to it. The body runs
only when the function is *called*. A call evaluates the arguments, binds them to parameters in a fresh local scope, runs
the body, and produces the return value.

```python
def add(a: int, b: int = 0) -> int:
    """Return a + b."""
    return a + b

print(add(2, 3), add.__name__, add.__doc__)   # prints: 5 add Return a + b.
print(add.__annotations__)                    # prints: {'a': <class 'int'>, 'b': <class 'int'>, 'return': <class 'int'>}
```

Type hints are metadata, not enforcement. The docstring is the first string literal in the body.

**Return values.** A function with no `return` (or a bare `return`) returns `None`. To return several values you
return **one tuple** and the caller unpacks it. `return` ends the function immediately, so early returns replace deep nesting.

```python
def stats(xs):
    if not xs:
        return None                      # early exit, explicit
    return min(xs), max(xs), sum(xs) / len(xs)

print(stats([2, 4, 6]))                  # prints: (2, 6, 4.0)
lo, hi, mean = stats([2, 4, 6])
print(stats([]))                         # prints: None

def log(msg):
    print(msg)                           # no return statement

print(log("hi"))                         # prints: hi  then None
```

> **Gotcha** Functions that mutate their argument and also return it (like `sorted` versus `list.sort`) confuse callers.
> Convention: mutate and return `None`, or build and return a new object. Not both.

> **Remember** `return` and `print` are different. A function that prints but does not return gives its caller `None`.

## Positional and keyword arguments
<!-- topics: positional_args, keyword_args -->

**Positional** arguments bind left to right by position. **Keyword** arguments bind by parameter name, in any order,
and make call sites self-documenting. Rules: positional first, then keyword; a parameter may not be bound twice.

```python
def call_llm(model, prompt, temperature=0.7, max_tokens=256):
    return f"{model}|{prompt}|{temperature}|{max_tokens}"

print(call_llm("m1", "hi"))                              # prints: m1|hi|0.7|256
print(call_llm("m1", "hi", max_tokens=10))               # prints: m1|hi|0.7|10
print(call_llm(prompt="hi", model="m1", temperature=0))  # prints: m1|hi|0|256

for bad in (lambda: call_llm("m1"), lambda: call_llm("m1", "hi", model="x"), lambda: call_llm("m1", "hi", top_p=1)):
    try:
        bad()
    except TypeError as e:
        print(e)
# prints: call_llm() missing 1 required positional argument: 'prompt'
# prints: call_llm() got multiple values for argument 'model'
# prints: call_llm() got an unexpected keyword argument 'top_p'
```

`TypeError` for a bad call is raised **before** the body runs. Prefer keyword arguments for anything with more than two
parameters or boolean flags (`retry(True, 3)` is unreadable; `retry(jitter=True, attempts=3)` is not).

## Default arguments
<!-- topics: default_args -->

Defaults are evaluated **once, when `def` executes**, and stored on the function object (`f.__defaults__`). The same
object is reused on every call that omits the argument. For immutable defaults (numbers, strings, `None`, tuples) that is
harmless. For mutable ones it is a shared global in disguise.

```python
def collect(item, bucket=[]):
    bucket.append(item)
    return bucket

print(collect(1), collect(2))            # prints: [1, 2] [1, 2]
print(collect.__defaults__)              # prints: ([1, 2],)
```

Both calls returned the same list object, so the first result already shows the second append. The idiom is a `None`
sentinel, created fresh inside the body:

```python
def collect(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket

print(collect(1), collect(2))            # prints: [1] [2]

_MISSING = object()                      # use a private sentinel when None is a valid value
def get(key, default=_MISSING):
    if default is _MISSING:
        return f"required:{key}"
    return default
print(get("a"), get("a", None))          # prints: required:a None
```

Another consequence: a default like `ts=time.time()` is frozen at definition time, not call time. Parameters with defaults
must come after those without (`def f(a=1, b)` is a `SyntaxError`).

> **Interview** "Why does the default list persist between calls?" Because `def` runs once and the default is an attribute
> of the function object; the call does not re-evaluate it. Show `__defaults__`.

> **AI use** A shared mutable default such as `messages=[]` in a chat helper leaks one user's conversation into the next
> request, a real production bug. Default to `None` and build the list inside.

## *args and **kwargs
<!-- topics: args, kwargs -->

`*args` collects extra **positional** arguments into a **tuple**; `**kwargs` collects extra **keyword** arguments into a
**dict**. The names are convention; the stars are the syntax.

```python
def show(a, *args, **kwargs):
    print(a, args, kwargs)

show(1)                         # prints: 1 () {}
show(1, 2, 3, x=4, y=5)         # prints: 1 (2, 3) {'x': 4, 'y': 5}

def total(*nums):
    return sum(nums)
print(total(1, 2, 3))           # prints: 6
```

Their main job is **forwarding**: wrappers and decorators accept whatever the wrapped function accepts and pass it
through without knowing the signature.

```python
import functools

def logged(fn):
    @functools.wraps(fn)                    # keeps fn.__name__ and __doc__
    def wrapper(*args, **kwargs):
        print("calling", fn.__name__, args, kwargs)
        return fn(*args, **kwargs)
    return wrapper

@logged
def greet(name, punct="!"):
    return "hi " + name + punct

print(greet("ada", punct="?"))
# prints: calling greet ('ada',) {'punct': '?'}
# prints: hi ada?
```

Parameter order in a signature is fixed: normal, `*args`, keyword-only, `**kwargs`. **Overriding an SDK call** is the
typical use: `def chat(prompt, **overrides): return client.create(model="m", **{"temperature": 0.2, **overrides})`.

> **Gotcha** `**kwargs` swallows typos silently (`temprature=0`) if you forward it to something permissive. Validate
> known keys or avoid `**kwargs` in public APIs where explicit parameters are clearer.

## Argument unpacking
<!-- topics: unpacking -->

The stars also work at the **call site**: `*iterable` expands into positional arguments, `**mapping` expands into keyword
arguments. The same operators unpack in assignments and build collections.

```python
def area(w, h, scale=1):
    return w * h * scale

dims = (3, 4)
opts = {"scale": 2}
print(area(*dims), area(*dims, **opts))     # prints: 12 24

first, *middle, last = [1, 2, 3, 4, 5]
print(first, middle, last)                  # prints: 1 [2, 3, 4] 5
print([*range(3), *"ab"], {**{"a": 1}, **{"a": 2, "b": 3}})
# prints: [0, 1, 2, 'a', 'b'] {'a': 2, 'b': 3}
```

Later keys win in `{**a, **b}`. Duplicate keyword arguments across several `**` in a *call* raise `TypeError`, unlike in a dict
literal. A common LLM pattern is building a request dict, then `client.create(**request)`.

> **Remember** `*` unpacks any iterable into positionals; `**` unpacks a mapping with **string keys** into keywords.

## Positional-only and keyword-only parameters
<!-- topics: pos_kw_only -->

A bare `*` or `*args` in a signature makes everything after it **keyword-only**. A `/` makes everything before it
**positional-only** (3.8+).

```text
def f(a, b, /, c, *, d):
        ───────  ─  ───
   positional-only | keyword-only
            positional-or-keyword
```

```python
def f(a, b, /, c, *, d):
    return (a, b, c, d)

print(f(1, 2, 3, d=4), f(1, 2, c=3, d=4))   # prints: (1, 2, 3, 4) (1, 2, 3, 4)
for call in (lambda: f(a=1, b=2, c=3, d=4), lambda: f(1, 2, 3, 4)):
    try:
        call()
    except TypeError as e:
        print(e)
# prints: f() got some positional-only arguments passed as keyword arguments: 'a, b'
# prints: f() takes 3 positional arguments but 4 were given
```

Why use them in a library:

* **Keyword-only** (`*`) forces call sites to name risky parameters (`timeout=`, `strict=`), and lets you add
  parameters later without breaking positional callers. Great for boolean flags.
* **Positional-only** (`/`) lets you rename a parameter without breaking callers and frees the name for `**kwargs`:
  `def update(self, key, /, **kwargs)` works even if someone passes `key="x"` as a keyword.

Built-ins use it everywhere: `len(obj, /)`, and `sorted(iterable, /, *, key=None, reverse=False)`.

## First-class and higher-order functions
<!-- topics: first_class, higher_order -->

**First-class** means functions are objects like any other: you can assign them to names, put them in collections, pass them
as arguments and return them. A **higher-order function** takes or returns a function.

```python
def shout(s): return s.upper()
def whisper(s): return s.lower()

styles = {"loud": shout, "quiet": whisper}        # functions in a dict
print(styles["loud"]("Hello"))                    # prints: HELLO
print(shout.__name__, callable(shout), type(shout).__name__)   # prints: shout True function

def apply_twice(fn, x):
    return fn(fn(x))
print(apply_twice(lambda n: n * 3, 2))            # prints: 18

def multiplier(k):                                # returns a function
    def mul(x):
        return x * k
    return mul
triple = multiplier(3)
print(triple(5))                                  # prints: 15
```

Built-in higher-order functions: `sorted(key=)`, `max(key=)`, `map`, `filter`, `functools.partial`. Practical uses in
an LLM application:

```python
import functools

# 1. Tool dispatch: model returns a tool NAME and arguments; look the function up.
def get_weather(city): return f"sunny in {city}"
def add(a, b): return a + b
TOOLS = {"get_weather": get_weather, "add": add}

call = {"name": "add", "arguments": {"a": 2, "b": 3}}      # as parsed from model output
print(TOOLS[call["name"]](**call["arguments"]))            # prints: 5

# 2. Retry wrapper: a higher-order function that adds behaviour to any call.
def retry(fn, attempts=3):
    def run(*a, **kw):
        for i in range(attempts):
            try:
                return fn(*a, **kw)
            except ConnectionError:
                if i == attempts - 1:
                    raise
    return run

state = {"n": 0}
def flaky():
    state["n"] += 1
    if state["n"] < 3:
        raise ConnectionError
    return "ok"
print(retry(flaky)(), state["n"])                          # prints: ok 3

# 3. Pre-configure with partial.
embed = functools.partial(lambda model, text: f"{model}:{text}", "ada-002")
print(embed("hello"))                                      # prints: ada-002:hello
```

> **Interview** "What does first-class mean and why care?" Functions are objects, so behaviour can be data: dispatch tables
> replace `if/elif` chains, and decorators, callbacks and strategy injection all rely on it. Mention `callable()`.

> **Gotcha** In the retry above, catching `Exception` instead of `ConnectionError` would also retry bugs. Catch only
> transient errors, add backoff plus jitter, and only retry idempotent calls.

## Function reference vs function call
<!-- topics: reference_vs_call -->

`f` is the function **object**; `f()` **calls** it and evaluates to its return value. Passing a callback means passing
`f`, not `f()`, otherwise you pass the *result* (often `None`) and the work runs immediately.

```python
def make_greeting():
    print("computing")
    return "hello"

ref = make_greeting                  # no call, nothing printed
print(callable(ref))                 # prints: True
value = make_greeting()              # prints: computing
print(value)                         # prints: hello

def run_later(callback):
    return callback()

print(run_later(make_greeting))      # prints: computing  then hello
try:
    run_later(make_greeting())       # computing runs now, then 'str' is not callable
except TypeError as e:
    print(e)                         # prints: computing  then 'str' object is not callable
```

The same distinction explains `sorted(xs, key=len)` (not `len(xs)`), `threading.Thread(target=work)` (not `work()`),
and `@decorator` versus `@decorator()` (a decorator *factory* is called first and returns the real decorator).
Bound methods are references too: `obj.method` is a callable object that remembers `obj`.

> **Gotcha** `default_factory=list`, not `list()`; `key=str.lower`, not `str.lower()`. A frequent bug is writing
> `button.on_click(handler())` so the handler fires at setup time.

## Recursion
<!-- topics: recursion -->

A recursive function calls itself on a smaller problem until it reaches a **base case**. Every call gets its own frame
(locals) on the call stack. Without a base case that always gets reached, the stack overflows.

```python
def factorial(n):
    if n <= 1:                       # base case
        return 1
    return n * factorial(n - 1)      # recursive case: smaller input

print(factorial(5))                  # prints: 120

def depth(obj):
    """Nesting depth of lists."""
    if not isinstance(obj, list):
        return 0
    return 1 + max((depth(x) for x in obj), default=0)
print(depth([1, [2, [3]], []]))      # prints: 3
```

Recursion fits naturally on **trees and nested data** (JSON, ASTs, file systems, deep merge). Python limits depth
(default 1000, `sys.getrecursionlimit()`) and **does not do tail-call optimisation**, so deep linear recursion
raises `RecursionError`.

```python
import sys
print(sys.getrecursionlimit())       # prints: 1000

def count_down(n):
    return 0 if n == 0 else count_down(n - 1)

try:
    count_down(10_000)
except RecursionError:
    print("too deep")                # prints: too deep

def count_down_loop(n):
    while n:
        n -= 1
    return 0
print(count_down_loop(10_000))       # prints: 0
```

Workarounds, in order of preference: rewrite as a **loop** (or with an explicit stack list), use `functools.cache` to
memoise overlapping subproblems (Fibonacci), or use an iterative algorithm. Raising the limit with
`sys.setrecursionlimit` risks a real C-stack crash. (Since 3.12 the limit counts only Python frames; C-level recursion has its own separate limit.)

```python
from functools import cache

@cache
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
print(fib(80))                       # prints: 23416728348467685
```

Without `@cache`, naive `fib` is O(2^n); with it, O(n).

> **Interview** "Recursion vs iteration?" Recursion mirrors tree-shaped problems and reads well; iteration avoids the
> depth limit and call overhead. Always name the base case and the progress step first.

## Variable scope (LEGB)
<!-- topics: legb -->

Python resolves a name by searching four scopes in order: **L**ocal (the current function), **E**nclosing (outer
functions), **G**lobal (module), **B**uilt-in (`len`, `print`, ...). The first hit wins.

```python
x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"
        return x
    return inner(), x

print(outer(), x)                    # prints: ('local', 'enclosing') global
print(len)                           # prints: <built-in function len>
```

Key rule: a name is **local to the whole function if it is assigned anywhere in it** (decided at compile time).
Reading a global is fine; assigning makes it local and breaks reads that come first.

```python
counter = 0

def bad():
    counter += 1                     # assigns -> counter is local -> read fails

try:
    bad()
except UnboundLocalError as e:
    print(e)    # prints: cannot access local variable 'counter' where it is not associated with a value

def ok():
    global counter                   # declare: rebind the module-level name
    counter += 1

ok(); ok()
print(counter)                       # prints: 2

def make_counter():
    n = 0
    def tick():
        nonlocal n                   # rebind the enclosing function's variable
        n += 1
        return n
    return tick

t = make_counter()
print(t(), t(), t())                 # prints: 1 2 3
```

`global` and `nonlocal` are needed only to **rebind** a name. *Mutating* an object (`lst.append`, `d[k] = v`) needs neither,
because you are not assigning the name. Class bodies are not an enclosing scope for methods, and comprehension variables
live in their own scope. `nonlocal` is the basis of closures (see the closures module).

> **Gotcha** Mutable global state (a module-level dict you `.append` to) is a hidden dependency that makes tests order-dependent
> and breaks under multiple workers. Prefer passing state in and returning it.

<details markdown="1"><summary>Check yourself: `def f(): print(x); x = 1` with a global `x` raises. Why?</summary>

`x` is assigned in `f`, so the compiler makes it local to the whole function; the `print(x)` reads the local before it
has a value, raising `UnboundLocalError`, instead of falling back to the global.

</details>

## Quick recap
- `def` runs at definition time and creates a function object; the body runs only on call. No `return` means `None`.
- Multiple returns are one tuple. Bad calls raise `TypeError` before the body runs.
- Defaults are evaluated once at `def`; use a `None` (or private sentinel) default for anything mutable.
- `*args` is a tuple, `**kwargs` is a dict; `*`/`**` at a call site unpack; signature order is normal, `*args`, keyword-only, `**kwargs`.
- `/` makes earlier parameters positional-only; `*` makes later ones keyword-only.
- Functions are objects: store them in dicts, pass them, return them. `f` is the object, `f()` is the result.
- Recursion needs a base case, hits `RecursionError` near 1000 frames, has no tail-call optimisation: use loops or `functools.cache`.
- LEGB: assignment anywhere in a function makes the name local; `global`/`nonlocal` rebind, mutation does not need them.
