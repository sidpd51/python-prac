# Memory management

Python hides memory management, but interviews and production incidents do not. "Why did my list
change when I edited the copy?", "why does this service grow 200 MB an hour?", "why is a million small
objects so heavy?" are all questions about the object model, reference counting and the garbage
collector. In AI work the same ideas decide whether an embedding job fits in RAM, whether a model cache
leaks, and whether a config shared between requests gets silently mutated.

## The object model: everything is an object
<!-- topics: object_model -->

Every value in Python, an int, a function, a class, a module, is an **object** living on the heap.
Each object has three things: an **identity** (`id()`, its address in CPython), a **type**, and a
**value**. Identity and type never change; the value may change if the type is mutable.

```text
CPython object header (every object has it)
+-----------+-----------+------------------------+
| refcount  | type ptr  |  payload (int digits,  |
| (8 bytes) | (8 bytes) |  list of pointers ...) |
+-----------+-----------+------------------------+
```

That header is why even a tiny int is not 4 bytes:

```python
import sys

print(sys.getsizeof(1))      # prints: 28
print(sys.getsizeof(1.0))    # prints: 24
print(sys.getsizeof("a"))    # prints: 42
print(sys.getsizeof([]))     # prints: 56
```

(Sizes are for 64-bit CPython 3.13; they are the same on 3.14.) A list does not contain its items; it contains
**pointers** to objects. `getsizeof` reports only the object itself, not what it points to.

> **Remember** Python variables are not boxes holding values. Objects exist on their own; variables
> are labels attached to them. Mutable vs immutable is a property of the *type*, not the name.

## Names, references and identity
<!-- topics: references -->

Assignment never copies. `b = a` makes a second **name** for the **same object**.

```text
a = [1, 2]        a ---+
b = a                  +--> [1, 2]      (one object, two names)
                  b ---+
```

```python
a = [1, 2]
b = a
b.append(3)
print(a)            # prints: [1, 2, 3]
print(a is b)       # prints: True
print(a == [1, 2, 3], a is [1, 2, 3])   # prints: True False
```

`==` asks "equal values?" (calls `__eq__`); `is` asks "same object?" (compares `id`). Use `is` only for
singletons: `None`, `True`, `False`, sentinels.

Rebinding vs mutating is the distinction that explains most aliasing bugs:

```python
def add_a(items):
    items.append("a")     # mutates the caller's list
def rebind(items):
    items = items + ["b"] # creates a NEW list, only the local name changes

xs = []
add_a(xs); rebind(xs)
print(xs)           # prints: ['a']
```

Python passes **object references by value**: the function gets a new name for the same object.

> **Gotcha** Small ints (-5..256) and some strings are cached or interned, so `a is b` can be `True`
> for equal numbers and `False` for larger ones. That is an implementation detail, never rely on it:

```python
a = int("1000"); b = int("1000")
print(a == b, a is b)   # prints: True False
```

> **Gotcha** `[[0] * 3] * 2` repeats the *same* inner list, and a mutable default argument
> (`def f(x=[])`) is one object shared by every call.

```python
grid = [[0] * 2] * 2
grid[0][0] = 9
print(grid)         # prints: [[9, 0], [9, 0]]
```

<details markdown="1"><summary>Check yourself: why does `def f(cfg={})` leak state between calls?</summary>

The default is evaluated once, at function definition, and stored on the function. Every call that
does not pass `cfg` gets that same dict object. Use `None` and create the dict inside.

</details>

## Reference counting
<!-- topics: refcounting -->

CPython's primary mechanism: every object stores how many references point at it. When the count
reaches **zero** the object is freed immediately, and its `__del__` runs.

```python
import sys

x = []
print(sys.getrefcount(x))   # prints: 2
y = x
print(sys.getrefcount(x))   # prints: 3
del y
print(sys.getrefcount(x))   # prints: 2
```

Why 2 for a single name? `getrefcount(x)` itself receives `x` as an argument, which is a temporary extra
reference. Things that add a reference: assignment, being stored in a container, being passed as an argument,
a closure capturing it. Things that remove one: `del name`, rebinding, the container dying, the function returning.

Deterministic cleanup is the big advantage:

```python
class Resource:
    def __init__(self, name): self.name = name
    def __del__(self): print("freed", self.name)

r = Resource("a")
r = Resource("b")    # prints: freed a      (old object hits zero on rebind)
del r                # prints: freed b
```

Costs and limits: every assignment touches a counter (this is part of why Python is slow, and why the GIL
exists, see the internals note), and refcounting alone **cannot free cycles**. Some objects (`None`, small
ints, interned strings) are *immortal* since 3.12: `sys.getrefcount(None)` is a huge constant and never
reaches zero.

> **Gotcha** Don't use `__del__` for cleanup you depend on (files, locks, sockets): timing is
> implementation-specific and it runs at interpreter shutdown unpredictably. Use `with` / `try/finally`.

## Stack vs heap
<!-- topics: stack_heap -->

Conceptually: the **call stack** holds one *frame* per active function call (local names, the current
instruction, the evaluation stack); the **heap** holds every object. In Python the split is simpler than in C:
**all objects are on the heap**, and frames only hold *references* to them.

```text
  call stack (frames)                 heap (objects)
  +------------------+
  | frame: main      |  data ------>  [ 0.1, 0.2, ... ]  (the list object)
  |   data           |
  +------------------+
  | frame: process   |  rows ------>  same list (a new name, not a copy)
  |   rows           |
  +------------------+
```

When `process` returns, its frame is popped and the references it held are dropped; the list survives because
`main` still refers to it. Consequences you can use in interviews:

* Returning a big list from a function does not copy it, it hands out another reference.
* Deep recursion exhausts the *frame* limit (`sys.getrecursionlimit()`, default 1000) long before memory.
* A paused generator or a closure keeps its frame/cell alive on the heap, which keeps referenced objects alive.

## Cycles and the garbage collector
<!-- topics: cycles, gc -->

A **reference cycle** is objects that point at each other, so none of them ever reaches count zero:

```python
import gc

class Node:
    def __init__(self): self.other = None

gc.collect()                    # start clean
a, b = Node(), Node()
a.other, b.other = b, a         # a <-> b
del a, b                        # names gone, but each still referenced by the other
print(gc.collect() >= 2)        # prints: True   (the cycle collector found and freed them)
```

The **cyclic garbage collector** (`gc` module) periodically scans *container* objects (lists, dicts, instances;
ints and strs cannot form cycles), finds groups reachable only from each other, and frees them.

```text
generation 0 (new objects) --survive--> generation 1 --survive--> generation 2 (long-lived)
collected most often                                              collected rarely
```

It is **generational** because most objects die young: young objects are scanned often and cheaply, survivors
are promoted and scanned rarely. A collection is triggered when allocations minus deallocations exceed a
threshold.

```python
import gc
print(gc.get_threshold())   # prints: (2000, 10, 10)
```

That is the 3.13 value (3.14 prints `(2000, 10, 0)` because 3.14 introduced an incremental collector with two
generations of work). The API is the same: `gc.collect()`, `gc.disable()`, `gc.get_stats()`,
`gc.get_referrers(obj)`.

> **Interview** "Does Python have garbage collection?" Strong answer: refcounting frees almost everything
> immediately; the generational cycle GC is a backstop for cycles only. Since 3.4, objects with `__del__`
> in cycles are collected too.

> **AI use** Classic leak: a chat-session object holds `history`, messages hold a back-reference to the
> session. Plain refcounting won't free it; the cycle GC eventually will, but a global registry or a
> `lru_cache` pinning sessions never lets them go. Look for *who holds the strong reference*.

<details markdown="1"><summary>Check yourself: a cycle exists but `__del__` never printed. Bug?</summary>

Not necessarily. Cycles are freed only when the GC runs, which is non-deterministic. Call `gc.collect()` to
force it in a test. If it still never frees, something else (a global, a cache, a traceback) holds a reference.

</details>

## Weak references
<!-- topics: weakref -->

A **weak reference** points at an object without increasing its refcount. When the last *strong* reference
disappears the object is freed and the weakref goes dead (returns `None`). This is how you build caches and
registries that must not keep things alive.

```python
import weakref

class Model:
    def __init__(self, name): self.name = name

m = Model("bert")
ref = weakref.ref(m)
print(ref().name)    # prints: bert
del m
print(ref())         # prints: None   (freed as soon as the count hit zero)
```

Helpers: `WeakValueDictionary` (value vanishes when nothing else holds it, ideal for "share one loaded model
while anyone is using it"), `WeakKeyDictionary`, `WeakSet`, `weakref.finalize(obj, callback)`.

```python
import weakref

class Model:
    def __init__(self, name): self.name = name

cache = weakref.WeakValueDictionary()
m = Model("reranker")
cache["rr"] = m
print("rr" in cache)   # prints: True
del m
print("rr" in cache)   # prints: False
```

> **Gotcha** Not everything can be weakly referenced: `int`, `str`, `list`, `dict`, `tuple` instances raise
> `TypeError`. User-defined classes work (unless they use `__slots__` without `"__weakref__"`).

> **AI use** A registry that `get(name)` loads a model once, returns the shared instance while any request holds
> it, and lets it be freed afterwards: `WeakValueDictionary` plus a lock. A plain dict would pin gigabytes forever.

## Memory allocation and what things cost
<!-- topics: allocation -->

Python asks the OS for large **arenas**; its small-object allocator (`pymalloc`) carves them into pools of
fixed-size blocks for objects up to 512 bytes. Freed blocks go back to the pool for reuse, which is fast but
means **RSS does not always shrink** when you `del` things: memory is reusable by Python, not necessarily
returned to the OS.

Containers over-allocate so that `append` is amortised O(1):

```python
import sys

sizes = []
lst = []
for i in range(10):
    lst.append(i)
    if not sizes or sys.getsizeof(lst) != sizes[-1]:
        sizes.append(sys.getsizeof(lst))
print(sizes)   # prints: [88, 120, 184]
```

(It jumps from 4 slots to 8 to 16 pointers; the empty list is 56 bytes.)

Every instance normally carries a `__dict__`. For millions of small records, `__slots__` removes it:

```python
import sys

class Plain:
    def __init__(self): self.doc_id = 1; self.score = 0.5
class Slotted:
    __slots__ = ("doc_id", "score")
    def __init__(self): self.doc_id = 1; self.score = 0.5

p, s = Plain(), Slotted()
print(sys.getsizeof(p) + sys.getsizeof(p.__dict__))   # prints: 344
print(sys.getsizeof(s))                               # prints: 48
```

`getsizeof` is shallow. To measure a nested structure, walk it (tracking `id`s so shared objects are counted
once) or use `tracemalloc` (next sections). `__slots__` instances cannot gain new attributes, which is also a
correctness feature. For numeric data the real fix is to leave Python objects entirely: a NumPy `float32`
array stores 4 bytes per number, a list of floats stores an 8-byte pointer plus a 24-byte object.

## Shallow copy vs deep copy
<!-- topics: copy -->

Because assignment aliases, you need explicit copies. They differ in how far they go:

```text
shallow:  new outer container, SAME inner objects
deep:     new outer container, recursively NEW inner objects
```

```python
import copy

cfg = {"model": "m1", "params": {"temp": 0.2}, "stop": ["\n"]}

shallow = cfg.copy()               # same as copy.copy(cfg), dict(cfg)
deep = copy.deepcopy(cfg)

shallow["model"] = "m2"            # rebinding a key on the copy: original untouched
shallow["params"]["temp"] = 0.9    # mutating a shared inner dict: original changes!
print(cfg["model"], cfg["params"]["temp"])   # prints: m1 0.9
print(deep["params"] is cfg["params"])       # prints: False
```

Slicing `xs[:]`, `list(xs)`, `dict(d)`, `d.copy()` are all shallow. `deepcopy` handles cycles (it keeps a memo
of what it has already copied) and respects `__deepcopy__`; it is slow and fails on things like open files and
locks.

```python
import copy

a = [1]
a.append(a)                   # self-referencing list
b = copy.deepcopy(a)
print(b[1] is b, b is a)      # prints: True False
```

> **Gotcha** The classic bug: a default config dict is shallow-copied per request, then one request edits a
> nested dict (`cfg["params"]["temp"] = ...`) and every later request sees it. Deep copy the config, or
> build immutable configs (frozen dataclasses, pydantic `model_copy(deep=True)`).

> **Interview** "Shallow vs deep copy?" Say it in terms of nested mutable objects. Add that copies of
> immutables (tuples of ints, strings) are free because sharing is safe.

## Measuring memory: profiling with tracemalloc
<!-- topics: profiling -->

Do not guess where memory goes. `tracemalloc` (stdlib) records every Python allocation with its traceback.

```python
import tracemalloc

tracemalloc.start()
before, _ = tracemalloc.get_traced_memory()
data = [str(i) * 10 for i in range(100_000)]
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print((current - before) > 5_000_000)   # prints: True
print(peak >= current)                  # prints: True
```

`get_traced_memory()` returns `(current, peak)` in bytes: **peak** is what decides whether a job fits in RAM.
To find *where*, take a snapshot and group by line:

```python
import tracemalloc

tracemalloc.start()
data = [bytes(1000) for _ in range(1000)]
snap = tracemalloc.take_snapshot()
top = snap.statistics("lineno")[0]
tracemalloc.stop()
print(top.size >= 1_000_000, top.count >= 1000)   # prints: True True
```

Comparing two snapshots (`snap2.compare_to(snap1, "lineno")`) finds leaks: lines whose size keeps growing.
Other tools: `sys.getsizeof` (shallow), `pympler.asizeof` / `memory_profiler` (third-party, line-by-line),
`resource.getrusage` / `psutil` for process RSS. `tracemalloc` sees Python allocations only, not native memory
inside NumPy or torch (use RSS for that).

> **Remember** Measure **peak**, not the final size, and measure relative to a baseline taken just before
> the call.

## Memory-efficient processing and generators
<!-- topics: efficient_processing, generator_memory -->

The rule: keep **one record (or one batch)** in memory, not the dataset. Three levers:

1. **Stream** with generators / iterators (`for line in file`, never `file.read()` or `readlines()`).
2. **Batch** so you pay per-call overhead once per N items (embedding APIs take batches).
3. **Compact representations**: arrays/`float32`, `__slots__`, categorical codes, only the columns you need.

```python
import tracemalloc

def eager(n): return [i * 2 for i in range(n)]
def lazy(n): return (i * 2 for i in range(n))

def peak_of(fn, n):
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    total = sum(fn(n))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return total, peak - base

total_e, peak_e = peak_of(eager, 200_000)
total_l, peak_l = peak_of(lazy, 200_000)
print(total_e == total_l)        # prints: True
print(peak_e > 100 * peak_l)     # prints: True
```

Same answer, three orders of magnitude different peak. Batching a stream keeps that property:

```python
from itertools import batched   # Python 3.12+

for batch in batched(range(7), 3):
    print(batch)
# prints: (0, 1, 2)
# prints: (3, 4, 5)
# prints: (6,)
```

> **Gotcha** A generator saves memory only if the consumer also streams. `sorted(gen)` or `list(gen)`
> materialises everything again. Streaming also needs the *source* to stream: `json.load` of a 5 GB file
> does not; JSON Lines read line by line does.

> **AI use** Corpus ingestion: read lines, parse, chunk, `batched(..., 64)`, embed, upsert, repeat. Peak
> memory is one batch, whether the corpus is 1 MB or 100 GB.

## Quick recap
- Variables are names bound to objects; assignment aliases, it never copies. `is` = identity, `==` = value.
- Mutating shares across names, rebinding does not. Mutable defaults are shared across calls.
- Refcount hits zero means immediate free; `sys.getrefcount` reads one too high; `__del__` is not for cleanup.
- All objects live on the heap; frames hold references. Returning a big object is O(1).
- Cycles defeat refcounting; the generational GC (`gc.collect()`) reclaims them. Leaks are usually a strong
  reference held by a global, cache or registry.
- `weakref` / `WeakValueDictionary` let caches avoid keeping objects alive; ints, strs, lists cannot be weak-referenced.
- `getsizeof` is shallow; lists over-allocate; `__slots__` cuts per-instance cost; arrays beat lists of floats.
- Shallow copy shares nested mutables; `deepcopy` does not and handles cycles.
- `tracemalloc` gives current/peak and per-line stats; measure peak against a baseline.
- Stream, batch, and use compact types; a generator helps only if every stage streams.
