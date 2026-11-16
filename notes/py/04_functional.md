# Functional Programming

Python is not a functional language, but it borrows enough ideas to be useful: functions as values, `map`/`filter`/`reduce`,
lazy iterators, and a style that favours pure functions and immutable data. The value for you is not syntax trivia; it is
a design habit. Code that computes results from inputs without touching hidden state is easy to test, cache, run in
parallel and debug, which is exactly what you want for ranking retrieved chunks, transforming LLM responses and building
agent loops with a small impure shell. The traps (single-use iterators, `reduce` quadratics, late-binding lambdas) are
common interview material.

## Lambda functions
<!-- topics: lambda -->

A `lambda` is an anonymous function limited to **one expression**; the value of that expression is returned. It is
exactly the same kind of object as a `def` function, only with no name, no statements and no annotations.

```python
square = lambda x: x * x
print(square(4), type(square).__name__, square.__name__)   # prints: 16 function <lambda>

pairs = [("a", 3), ("b", 1), ("c", 2)]
print(sorted(pairs, key=lambda p: p[1]))     # prints: [('b', 1), ('c', 2), ('a', 3)]
print(max(pairs, key=lambda p: p[1])[0])     # prints: a
```

Use a lambda for a short throwaway key or callback, passed inline. Use `def` when it needs a name, a docstring,
statements, type hints or reuse. Assigning a lambda to a name (as `square` above) is discouraged style: the traceback shows
`<lambda>`, so use `def`. `operator.itemgetter(1)` and `attrgetter("score")` are faster, clearer replacements for the most
common lambdas.

```python
from operator import itemgetter
pairs = [("a", 3), ("b", 1), ("c", 2)]
print(sorted(pairs, key=itemgetter(1)) == sorted(pairs, key=lambda p: p[1]))   # prints: True
```

> **Gotcha** Lambdas capture **variables, not values** (late binding). In a loop, every lambda sees the final value:

```python
fns = [lambda: i for i in range(3)]
print([f() for f in fns])                    # prints: [2, 2, 2]
fns = [lambda i=i: i for i in range(3)]      # default argument freezes the value now
print([f() for f in fns])                    # prints: [0, 1, 2]
```

## map, filter and reduce
<!-- topics: map, filter, reduce -->

**`map(fn, *iterables)`** applies `fn` to each item. **`filter(pred, it)`** keeps items for which `pred` is truthy (`None`
as predicate keeps truthy items). Both return **lazy iterators** in Python 3, not lists. **`reduce(fn, it, initial)`**
(from `functools`) folds a sequence into one value by repeatedly applying a two-argument function.

```python
from functools import reduce

nums = [1, 2, 3, 4, 5]
print(list(map(lambda x: x * 2, nums)))           # prints: [2, 4, 6, 8, 10]
print(list(map(pow, [2, 3], [3, 2])))             # prints: [8, 9]   (several iterables, stops at the shortest)
print(list(filter(lambda x: x % 2, nums)))        # prints: [1, 3, 5]
print(list(filter(None, [0, "a", "", None, 7])))  # prints: ['a', 7]
print(reduce(lambda acc, x: acc + x, nums))       # prints: 15
print(reduce(lambda acc, x: acc * x, nums, 10))   # prints: 1200
print(map(str, nums))                             # prints: <map object at 0x...>  (address varies)
```

`reduce` evaluates `fn(fn(fn(a, b), c), d)`. With an `initial` value it also works on an empty sequence (returns the
initial); **without one, an empty iterable raises `TypeError`**.

```python
from functools import reduce
try:
    reduce(lambda a, b: a + b, [])
except TypeError as e:
    print(e)             # prints: reduce() of empty iterable with no initial value
print(reduce(lambda a, b: a + b, [], 0))   # prints: 0
```

**Comprehension vs `map`/`filter`.** `[f(x) for x in xs if p(x)]` is usually clearer than `map(f, filter(p, xs))`, and
`map` is only neat when you already have a named function (`map(str.strip, lines)`). Use a generator expression for laziness.
For summing and joining prefer the dedicated tools: `sum`, `max`, `min`, `any`, `all`, `"".join`.

> **Gotcha** Using `reduce(operator.add, list_of_lists)` or `sum(list_of_lists, [])` to flatten is **O(n^2)**: each
> `+` creates a new list and copies everything so far. Use `itertools.chain.from_iterable(list_of_lists)` or a nested comprehension.

```python
from itertools import chain
lol = [[1, 2], [3], [4, 5]]
print(list(chain.from_iterable(lol)))   # prints: [1, 2, 3, 4, 5]
print([x for sub in lol for x in sub])  # prints: [1, 2, 3, 4, 5]
```

> **Interview** "Why did `reduce` move out of the built-ins?" Guido found most uses unreadable compared to an explicit
> loop; it lives in `functools`. A strong answer shows one good use (composition, products, merging dicts) and the empty-input rule.

## Lazy evaluation
<!-- topics: lazy_evaluation -->

`map`, `filter`, `zip`, `enumerate`, `reversed`, `range` (a lazy sequence) and generator expressions compute values **on
demand**. Nothing happens when you build the pipeline; work happens when a consumer (`for`, `list`, `sum`, `next`) pulls.

```python
def noisy(x):
    print("computing", x)
    return x * 10

m = map(noisy, [1, 2, 3])      # nothing printed: no work done yet
print(next(m))                 # prints: computing 1  then 10
print(list(m))                 # prints: computing 2, computing 3  then [20, 30]
```

Laziness also defers **exceptions**: a bug in `fn` fires at consumption, far from where you built the `map`. Use it for big
or streaming data (LLM token streams, JSONL files); use a list when you need random access, a length or several passes.
`any()`/`all()` stop at the first decisive item, which is the payoff of laziness (and why side effects inside can be skipped).

## Iterator consumption
<!-- topics: iterator_consumption -->

The matching trap: lazy iterators are **single-use**. They are consumed as you read them, and once exhausted they yield
nothing again, silently.

```python
evens = filter(lambda n: n % 2 == 0, range(10))
print(list(evens))             # prints: [0, 2, 4, 6, 8]
print(list(evens))             # prints: []
print(sum(evens))              # prints: 0

it = map(str, [1, 2, 3])
print("2" in it, list(it))     # prints: True ['3']   (the membership test consumed 1 and 2)
```

Related traps:

* `len(map(...))` and indexing fail: iterators have no length or position.
* Sharing one iterator between two consumers (or calling `list(x)` for logging, then passing `x` on) steals items.
* A generator expression evaluates its **outermost iterable immediately** but everything else lazily, so later changes to
  variables it uses are visible.

```python
lines = (l.strip() for l in ["a", "", "b"])
print(sum(1 for _ in lines), list(lines))   # prints: 3 []
```

Fixes: **materialise** once (`data = list(it)`), recreate the pipeline from a function, or `itertools.tee` (buffers
memory).

> **AI use** Streaming a chat completion yields chunks once. If you iterate it to print and then try to join the text
> afterwards, the second pass is empty. Collect pieces while you consume.

<details markdown="1"><summary>Check yourself: why does `print(list(m), list(m))` show an empty second list?</summary>

`m` is an iterator and the first `list(m)` exhausted it. Iterators do not rewind. An iterable like a list would produce a
fresh iterator on each `list()` call; a `map` object is its own iterator.

</details>

## Functional programming concepts
<!-- topics: concepts -->

Functional style rests on a few ideas, and Python supports them to varying degrees:

* **Functions as values** (first-class, higher-order): fully supported.
* **Pure functions** and **immutable data**: supported by convention, not enforced.
* **Declarative pipelines** (`map`/`filter`, comprehensions, generators, `itertools`, `functools`): well supported.
* **Composition and currying**: `functools.partial` gives partial application; compose with a small helper or `reduce`.
* **Recursion as the loop**: poorly supported (no tail-call optimisation, depth limit 1000).
* **Pattern matching, algebraic types**: partial (`match`, dataclasses, `Enum`).

```python
from functools import partial, reduce

def compose(*fns):
    """compose(f, g, h)(x) == f(g(h(x)))"""
    return reduce(lambda f, g: lambda x: f(g(x)), fns)

clean = compose(str.title, str.strip)
print(clean("  hello world "))        # prints: Hello World

def power(base, exp):
    return base ** exp

square_it = partial(power, exp=2)       # partial application: fix one argument
print(square_it(7))                          # prints: 49
```

The practical middle ground ("pragmatic Python"): write small pure functions for the transformations, keep I/O at the
edges, prefer comprehensions to `map`/`filter`, and do not contort code to avoid a plain loop or a local mutable accumulator.
A local variable that no one else can see is fine, because mutation that cannot be observed is not a side effect.

> **Interview** "Where does Python fall short as a functional language?" No tail-call optimisation, no enforced
> immutability, a one-expression lambda, no persistent data structures, and `map`/`filter` are less composable than method chains.

## Pure functions
<!-- topics: pure_functions -->

A function is **pure** if (1) its result depends only on its arguments, and (2) it has no observable **side effects**.
Same input always gives the same output, so it can be cached, tested with a one-line assert, reordered or run in parallel.

A **side effect** is anything beyond returning a value: mutating an argument or global, writing a file or DB, printing,
logging, making an HTTP call, reading the clock, generating random numbers, or reading mutable global state.

```python
import random, time

# Impure: depends on the clock, randomness, and mutates its argument
def stamp(record):
    record["at"] = time.time()
    record["id"] = random.random()
    return record

# Pure: same input, same output; the input is untouched
def with_fields(record, at, id_):
    return {**record, "at": at, "id": id_}

original = {"q": "hi"}
print(with_fields(original, 1.0, "x1"))      # prints: {'q': 'hi', 'at': 1.0, 'id': 'x1'}
print(original)                              # prints: {'q': 'hi'}
```

## Side effects and functional core
<!-- topics: side_effects -->

Pure code can still *use* time and randomness: **pass them in** (`now: float`, a `random.Random(seed)`) so the function
stays deterministic. Mutating an argument is the sneakiest side effect:

```python
def normalise_bad(scores):
    scores.sort()                  # mutates caller's list
    top = scores[-1]
    return [s / top for s in scores]

def normalise(scores):
    top = max(scores)
    return [s / top for s in scores]   # builds new; caller's list unchanged

data = [3, 1, 2]
print(normalise(data), data)       # prints: [1.0, 0.3333333333333333, 0.6666666666666666] [3, 1, 2]
```

**Functional core, imperative shell**: keep decisions (parse a model reply, choose the next tool, rank results) in pure
functions that take data and return data; confine I/O (HTTP, DB, logging, `input`) to a thin outer layer.

```python
# pure core: decide what to do from the model's reply
def next_action(reply: dict) -> tuple[str, dict]:
    calls = reply.get("tool_calls") or []
    if calls:
        return "tool", calls[0]
    return "answer", {"text": reply.get("content") or ""}

# impure shell would loop: reply = call_llm(...); kind, payload = next_action(reply); run tool / print answer
print(next_action({"tool_calls": [{"name": "search"}]}))   # prints: ('tool', {'name': 'search'})
print(next_action({"content": "42"}))                      # prints: ('answer', {'text': '42'})
```

The core is testable with plain dicts and no mocks, while only the shell needs fakes or recorded responses.
`functools.lru_cache` is only safe on pure functions: cache a function that reads a file, and you serve stale data.

> **Gotcha** A function that looks pure but reads a global config, a module-level cache or `os.environ` is not. A
> function that returns a *mutable* object it also keeps (a cached list) lets callers corrupt the cache.

## Immutability
<!-- topics: immutability -->

Python has no `const`, but offers tools with different depth. The key is that most guarantees are **shallow**: the
container is frozen, the objects inside may not be.

| Tool | What it freezes | Depth |
|---|---|---|
| `tuple` | which objects it holds | shallow |
| `frozenset` | membership | shallow (elements must be hashable) |
| `@dataclass(frozen=True)` | attribute assignment | shallow |
| `NamedTuple` | fields | shallow |
| `types.MappingProxyType(d)` | read-only **view** of a dict | view only; the underlying `d` can still change |
| `str`, `int`, `bytes` | the value | fully immutable |

```python
from dataclasses import dataclass, replace
from types import MappingProxyType

@dataclass(frozen=True)
class Config:
    model: str
    tags: tuple = ()
    opts: list | None = None       # a mutable field slips through the freeze

cfg = Config("m1", ("a",))
try:
    cfg.model = "m2"
except Exception as e:
    print(type(e).__name__)        # prints: FrozenInstanceError
cfg2 = replace(cfg, model="m2")    # "modify" by building a new object
print(cfg.model, cfg2.model)       # prints: m1 m2

leaky = Config("m1", opts=[1])
leaky.opts.append(2)               # allowed: the freeze is shallow
print(leaky.opts)                  # prints: [1, 2]
```

Read-only views and frozen sets:

```python
from types import MappingProxyType

base = {"model": "m1"}
view = MappingProxyType(base)
try:
    view["model"] = "m2"
except TypeError as e:
    print(e)                       # prints: 'mappingproxy' object does not support item assignment
base["model"] = "m3"               # the view is live: it reflects changes to the original
print(view["model"])               # prints: m3

fs = frozenset({1, 2})
print(fs | {3}, hash(fs) == hash(frozenset({2, 1})))   # prints: frozenset({1, 2, 3}) True
```

To get *deep* immutability you must convert nested data as well (lists to tuples, dicts to `MappingProxyType`
or frozen dataclasses) or `copy.deepcopy` on the way in and out of a function boundary. The pragmatic habit:

* Treat function arguments as read-only; return new objects (`{**d, "k": v}`, `[*xs, x]`, `replace(obj, f=1)`).
* Freeze shared configuration (`frozen=True` dataclass, `tuple` instead of `list`) so one request cannot change settings
  for the next.
* Frozen/hashable values can be cache keys and set members; they are also safe to share across threads.

Cost: building new objects instead of mutating costs time and memory (copying a 1M-element list per step is O(n) each).
Mutate locally inside a function if the data never escapes; be immutable at the boundaries.

> **Remember** `frozen=True` stops `obj.attr = x`, not `obj.attr.append(x)`. Immutability is only as deep as the types you chose.

> **AI use** Freeze the model/system-prompt config object at startup. A request handler that mutates a shared config
> dict ("just for this call") causes bugs that appear only under concurrent traffic.

<details markdown="1"><summary>Check yourself: is a tuple containing a list immutable?</summary>

The tuple cannot change *which* objects it holds, but the list inside can still be mutated. It is also unhashable, because
hashing a tuple hashes its items. Immutability is shallow.

</details>

## Quick recap
- `lambda` is a one-expression anonymous function; prefer `def` for anything named, long or reused; loop lambdas bind late (fix with `i=i`).
- `map` and `filter` return lazy iterators; `reduce` folds to one value and needs an initial value for empty input.
- Comprehensions are usually clearer than `map`/`filter`; avoid `reduce(add, lists)` and `sum(lists, [])`, which are O(n^2); use `chain.from_iterable`.
- Lazy iterators do work only when consumed, defer errors to consumption, and are single-use: a second pass is silently empty.
- Materialise with `list()`, rebuild the pipeline, or `tee` when you need more than one pass.
- Pure: output depends only on arguments, no side effects. Pass time, randomness and config in as arguments.
- Side effects: mutation of arguments or globals, I/O, logging, clock, randomness.
- Functional core, imperative shell: decisions in pure functions, I/O at the edges; easy to test with plain data.
- Immutability tools (`tuple`, `frozenset`, frozen dataclass, `NamedTuple`, `MappingProxyType`) are shallow; return new objects instead of mutating inputs.
