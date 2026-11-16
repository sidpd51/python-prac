# Closures and Scope

A closure is a function that remembers variables from the function it was defined in, even after
that outer function has returned. It is the mechanism behind decorators, callbacks, factories and
`functools.partial`-style code, and it is the first place most people meet Python's scoping rules for
real. In AI engineering you use closures constantly without naming them: a retry wrapper that
remembers `max_attempts`, a callback that remembers which request it belongs to, a token-budget
tracker created per session. Interviewers probe closures to see whether you know *how names are
resolved* and what a function object really carries around.

## Nested functions and lexical scoping
<!-- topics: nested_functions, lexical_scoping -->

You can define a function inside another function. The inner function is created each time the outer
one runs, and it can read the outer function's variables.

```python
def make_greeter(greeting):
    def greet(name):
        return f"{greeting}, {name}!"   # `greeting` comes from the enclosing function
    return greet

hi = make_greeter("Hi")
hello = make_greeter("Hello")
print(hi("Ada"))      # prints: Hi, Ada!
print(hello("Ada"))   # prints: Hello, Ada!
```

**Lexical scoping** means names are resolved by where the code is *written*, not by who calls it. The
inner function looks outward through the nesting you can see in the source. It does not look at the
caller's variables (that would be *dynamic* scoping, which Python does not have).

```python
x = "global"

def outer():
    x = "outer"
    def inner():
        return x
    return inner

def caller():
    x = "caller"          # irrelevant: inner was *defined* inside outer
    return outer()()

print(caller())   # prints: outer
```

> **Remember** Python scoping is lexical and decided at compile time: the text of the program, not the
> call stack, determines which variable a name refers to.

## LEGB, local and global variables
<!-- topics: local_global -->

When Python meets a name it searches four scopes in order, stopping at the first hit:

```text
L  Local      names assigned inside the current function (and its parameters)
E  Enclosing  locals of any outer *function* (not classes), innermost first
G  Global     module-level names
B  Built-in   len, print, ValueError ...
```

```python
total = "global total"          # G

def f():
    label = "enclosing label"   # E for g()
    def g():
        local = "local"         # L
        print(local, "|", label, "|", total, "|", len("abc"))
    g()

f()   # prints: local | enclosing label | global total | 3
```

Reading a global from a function needs nothing special. **Assigning** is the catch: any assignment
makes the name local to that function, so to rebind a module-level name you must say `global`.

```python
calls = 0

def bump():
    global calls     # "this name lives at module level"
    calls += 1

bump(); bump()
print(calls)   # prints: 2
```

Mutating a global object does not need `global`, because you are not rebinding the name:

```python
registry = []

def register(x):
    registry.append(x)   # no assignment to `registry` -> no `global` needed

register("a")
print(registry)   # prints: ['a']
```

> **Gotcha** `global` is usually a design smell (hidden shared state, hard to test). Reach for a
> closure, a class, or an explicit argument first. Class bodies are not an "E" scope: a method cannot see
> a class variable by bare name, it must write `self.x` or `ClassName.x`.

## Closures and free variables
<!-- topics: closures, free_variables -->

A **free variable** is a name used in a function but defined in an enclosing *function* scope (not local,
not global). A **closure** is the function object bundled with the cells for its free variables.

```python
def make_multiplier(factor):      # `factor` is a local of make_multiplier
    def multiply(x):
        return x * factor         # `factor` is FREE in multiply
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
del make_multiplier               # the factory is gone; the closures still work
print(double(10), triple(10))     # prints: 20 30
```

How does `factor` survive after `make_multiplier` returned? The compiler notices that an inner function
uses `factor`, so it stores `factor` not in a plain stack slot but in a **cell**, a small box holding the
value. The inner function object keeps references to those cells.

```text
make_multiplier(2) call         make_multiplier(3) call
  cell_A -> 2                     cell_B -> 3
     ^                               ^
  double.__closure__             triple.__closure__
```

Each call to the factory makes **new cells**, which is why two closures from the same factory are
independent. The closure shares the *variable* (the cell), not a copy of the value at definition time,
a distinction that matters in the next sections.

> **Interview** "What is a closure and what does it remember?" Strong answer: a function plus the cells
> of the enclosing-scope variables it references (only the ones it actually uses). It remembers
> *variables*, not values, so it sees later rebinding of them.

<details markdown="1"><summary>Check yourself: is a function that only uses globals a closure?</summary>

No. Globals are looked up in the module dict, not through cells. A closure needs at least one free
variable from an enclosing function; without one `f.__closure__` is `None`.

</details>

## nonlocal and rebinding
<!-- topics: nonlocal -->

A closure can *read* a free variable freely. To *rebind* it you must declare `nonlocal`; otherwise the
assignment creates a new local instead.

```python
def make_counter(start=0, step=1):
    count = start - step
    def next_value():
        nonlocal count          # rebind the enclosing variable, not a new local
        count += step
        return count
    return next_value

a = make_counter(100, 10)
b = make_counter()
print(a(), a(), a())   # prints: 100 110 120
print(b(), b())        # prints: 0 1
```

`global` targets module scope; `nonlocal` targets the nearest enclosing *function* scope and is a
SyntaxError if no such variable exists. If you only mutate an object (`items.append(x)`), neither is needed.

| You want to | Need |
|---|---|
| read an outer variable | nothing |
| mutate an outer object (`.append`, `d[k] = v`) | nothing |
| rebind an outer function's variable (`n += 1`, `n = ...`) | `nonlocal n` |
| rebind a module-level variable | `global n` |

Closures replace a one-method class when the state is tiny. A running average, in O(1) memory:

```python
def make_averager():
    total = 0.0
    count = 0
    def average(value):
        nonlocal total, count
        total += value
        count += 1
        return total / count
    return average

avg = make_averager()
print(avg(10), avg(20), avg(30))   # prints: 10.0 15.0 20.0
```

> **Gotcha** Is a closure counter thread-safe? No. `count += 1` is read, add, write (several bytecodes),
> so two threads can interleave and lose updates; guard with a `threading.Lock`. In `asyncio` (single
> thread) it is safe as long as there is no `await` between the read and the write.

## UnboundLocalError and compile-time scoping
<!-- topics: unboundlocalerror -->

Python decides whether a name is local **when it compiles the function**, by scanning the whole body
for assignments (`=`, `+=`, `for x`, `import x`, `def x`, `with ... as x`...). If the name is assigned anywhere in the
function it is local *everywhere* in that function, including lines before the assignment.

```python
count = 10

def broken():
    count += 1     # `count` is local (because of +=), but has no value yet
    return count

try:
    broken()
except UnboundLocalError as e:
    print(e)   # prints: cannot access local variable 'count' where it is not associated with a value
```

A subtler version: it fails even when the assignment comes later and the read is first.

```python
level = "global"

def surprise():
    print(level)       # looks like it reads the global...
    level = "local"    # ...but this line made `level` local for the whole function

try:
    surprise()
except UnboundLocalError as e:
    print("UnboundLocalError")   # prints: UnboundLocalError
```

Why compile time? So the compiler can allocate fixed local slots (fast) and decide cell vs. local layout
before running anything; resolving each name dynamically on every access would be slower and
unpredictable. Fix by declaring `global`/`nonlocal`, by passing the value in as a parameter, or by
renaming.

> **Remember** `UnboundLocalError` is a subclass of `NameError`. Message wording was tweaked in 3.11+
> ("cannot access local variable ... where it is not associated with a value"); older versions said
> "referenced before assignment".

<details markdown="1"><summary>Check yourself: why does `x = x + 1` inside a function fail when `x` is a global?</summary>

The assignment makes `x` local for the entire function, so the right-hand `x` is read from an empty
local slot. Use `global x`, or pass `x` in and return the new value.

</details>

## Late binding in loops
<!-- topics: closures, free_variables -->

Because closures capture the *variable* (the cell) and not its value, closures created in a loop all
share the same loop variable, and see its **final** value.

```python
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])   # prints: [2, 2, 2]
```

After the loop `i` is 2 and all three lambdas look at the same cell. Three fixes:

```python
# 1. default argument: evaluated NOW, stored per function
funcs = [lambda i=i: i for i in range(3)]
print([f() for f in funcs])   # prints: [0, 1, 2]

# 2. factory function: each call gets its own cell
def make(i):
    return lambda: i
funcs = [make(i) for i in range(3)]
print([f() for f in funcs])   # prints: [0, 1, 2]

# 3. functools.partial: binds the value at creation
from functools import partial
funcs = [partial(int.__add__, i) for i in range(3)]
print([f(10) for f in funcs])   # prints: [10, 11, 12]
```

The same trap hits real code: registering a callback per tool name, per retry attempt or per
request inside a loop. If every handler reports the last item, think "late binding".

> **Gotcha** The default-argument trick changes the signature (callers can override `i`). Prefer a
> factory or `partial` in public APIs.

Late binding also means a closure sees rebinding of the outer variable after creation, which is
sometimes exactly what you want and sometimes a surprise:

```python
def outer():
    x = 1
    def get():
        return x
    x = 2          # rebinds the same cell
    return get

print(outer()())   # prints: 2
```

## Inspecting closures
<!-- topics: inspection, co_freevars -->

A function carries its closure with it, so you can look inside. This is how you debug "what did this
factory capture?".

```python
def make_client(base_url, timeout):
    retries = 3                       # local, but not used by the inner function
    def call(path):
        return f"{base_url}{path} (timeout={timeout})"
    return call

call = make_client("https://api.example.com", 30)
print(call.__code__.co_freevars)                      # prints: ('base_url', 'timeout')
print([c.cell_contents for c in call.__closure__])    # prints: ['https://api.example.com', 30]
print(call.__code__.co_varnames)                      # prints: ('path',)
print(make_client.__code__.co_cellvars)               # prints: ('base_url', 'timeout')
```

* `__code__.co_freevars`: names of the free variables this function uses, in the same order as...
* `__closure__`: a tuple of **cell** objects (`None` for non-closures); read each with `.cell_contents`.
* `co_cellvars` (on the *outer* function): its locals that inner functions capture.

`retries` is absent: only variables the inner function actually references are captured.
`inspect.getclosurevars(func)` gives the same information as a friendly named tuple (nonlocals, globals, builtins).

```python
import inspect

x = 10
def outer(y):
    def inner():
        return x + y + len("ab")
    return inner

cv = inspect.getclosurevars(outer(5))
print(cv.nonlocals, cv.globals, sorted(cv.builtins))
# prints: {'y': 5} {'x': 10} ['len']
```

Cells are writable (`cell.cell_contents = ...`), which is occasionally used in tests but is a hack in
production code.

## Closures versus classes and partial
<!-- topics: closures, nested_functions -->

Three ways to bundle behaviour with state or pre-filled arguments:

```python
from functools import partial

# closure
def make_adder(n):
    return lambda x: x + n

# class with __call__
class Adder:
    def __init__(self, n):
        self.n = n
    def __call__(self, x):
        return x + self.n

# partial (pre-fill arguments of an existing function)
add5 = partial(int.__add__, 5)

print(make_adder(5)(1), Adder(5)(1), add5(1))   # prints: 6 6 6
```

| | Best when | Downside |
|---|---|---|
| closure | one behaviour, small private state, decorators and callbacks | state is hidden: hard to inspect, test or pickle |
| class with `__call__` | several methods, visible attributes, needs `repr`/pickling, inheritance | more boilerplate |
| `functools.partial` | fixing some arguments of an existing function | no extra logic or state |

> **Interview** "Closure vs class?" A class when you need more than one operation on the state, want it
> inspectable (`obj.count`), serialisable (closures and lambdas cannot be pickled, which matters for
> `multiprocessing`) or subclassable. A closure when it is one function with a bit of private state.

> **AI use** A per-session token budget tracker is a classic: one factory returns `track()`, `remaining()`
> and `report()` closures that all share the same `used` cell. It is also why decorators like
> `@retry(times=3)` work: `times` lives in a cell.

## Closures in long-running services
<!-- topics: closures, free_variables -->

A closure keeps every captured object alive as long as the closure itself is alive. So a callback that
captured a huge object holds it in memory:

```python
import gc
import weakref

class BigPayload:
    pass

def make_callback():
    payload = BigPayload()
    ref = weakref.ref(payload)
    def callback():
        return payload        # captures the payload in a cell
    return callback, ref

cb, ref = make_callback()
gc.collect()
print(ref() is not None)   # prints: True   (the closure keeps it alive)
del cb
gc.collect()
print(ref() is None)       # prints: True   (freed once the closure is gone)
```

Typical leaks: a global registry or event-bus list that keeps appending callbacks per request; an
`lru_cache` or global dict holding closures that captured whole request objects or message histories;
bound-method/closure reference cycles with large objects.

Avoid by capturing only what you need (copy out the id, not the whole request), unsubscribing
callbacks, using `weakref`/`WeakSet` for observers, and bounding caches.

> **AI use** A streaming handler that captures the entire conversation history "just for logging"
> and is stored on a global list retains every conversation. Capture the request id, not the object.

## Quick recap
- Python is lexically scoped; resolution is LEGB and decided at compile time per function.
- Any assignment in a function body makes that name local to the whole function; reading it first gives `UnboundLocalError`.
- `global` rebinds a module name, `nonlocal` rebinds an enclosing function's variable; mutating an object needs neither.
- A closure = function + cells for the free variables it uses; each factory call creates fresh cells.
- Closures capture variables, not values: loop-created lambdas see the final value. Fix with a default arg, a factory or `partial`.
- Inspect with `__code__.co_freevars`, `__closure__[i].cell_contents`, `inspect.getclosurevars`.
- Closure for one behaviour with tiny state, class for several operations or picklability, `partial` for pre-filled arguments.
- Captured objects live as long as the closure: watch registries and caches.
- Closure counters are not thread-safe without a lock.
