# Generators

A generator is a function that can pause and resume. Instead of building a whole result and
returning it, it hands you values one at a time and remembers where it stopped. That one idea gives
you constant memory on huge inputs, infinite streams, and the lazy pipelines behind most document
ingestion code: read a file, clean it, chunk it, embed it, without ever holding the corpus in memory.
Interviewers use generators to check whether you understand *when code actually runs*.

## Generator functions and yield
<!-- topics: functions, yield -->

A function containing `yield` is a **generator function**. Calling it does **not** run its body. It
returns a **generator object** that is paused before the first line.

```python
def countdown(n):
    print("start")
    while n > 0:
        yield n
        n -= 1
    print("done")

g = countdown(2)       # nothing printed yet
print(type(g).__name__)  # prints: generator
print(next(g))         # prints: start, then 2
print(next(g))         # prints: 1
```

`yield value` does two things: it hands `value` to whoever called `next()`, and it **freezes the
whole frame** (local variables, loop position) until the next `next()`.

> **Remember** `yield` makes the *function* a generator. `return value` inside it just ends the
> iteration (the value rides on `StopIteration`, see `yield from` below).

> **Gotcha** Because the body does not run on call, validation at the top of a generator is
> deferred until the first `next()`. See "Eager validation" later.

<details markdown="1"><summary>Check yourself: what does `countdown(2)` print by itself?</summary>

Nothing. Calling it only creates the generator object. "start" appears on the first `next()`.

</details>

## Suspension and resumption
<!-- topics: suspension, resumption -->

Think of the generator as a paused frame:

```text
next(g)  ->  run until the next `yield`  ->  hand out the value  ->  PAUSE (frame kept alive)
next(g)  ->  resume right after that `yield`  ->  ...
body ends ->  raise StopIteration
```

You can watch the states with `inspect`:

```python
import inspect

def gen():
    yield 1

g = gen()
print(inspect.getgeneratorstate(g))   # prints: GEN_CREATED
next(g)
print(inspect.getgeneratorstate(g))   # prints: GEN_SUSPENDED
try:
    next(g)
except StopIteration:
    pass
print(inspect.getgeneratorstate(g))   # prints: GEN_CLOSED
```

The paused frame lives on the heap, which is why local variables survive between calls and why an
unfinished generator can keep resources (a file, a DB cursor) open.

## Lazy evaluation, list versus generator
<!-- topics: lazy_evaluation, list_vs_generator, memory_efficiency -->

Lazy means: compute a value only when somebody asks for it. A list comprehension is eager, a
generator is lazy.

```python
import sys

eager = [n * n for n in range(1_000_000)]
lazy = (n * n for n in range(1_000_000))
print(sys.getsizeof(eager) > 1_000_000)  # prints: True   (megabytes)
print(sys.getsizeof(lazy) < 300)         # prints: True   (a few hundred bytes)
print(next(lazy))                        # prints: 0
```

| | list | generator |
|---|---|---|
| Memory | O(n) | O(1) |
| Reusable | yes | **one pass** |
| `len()`, index, slice | yes | no |
| Errors surface | at creation | **at consumption** |
| Can be infinite | no | yes |

> **Interview** "When is a generator the wrong choice?" Random access, multiple passes, `len()`,
> sorting, tiny data where the list is simpler, and when you need the work done *now* (e.g. a
> resource will be closed before you consume it).

> **AI use** Streaming a 50 GB JSONL corpus into an embedding job only works because each stage
> keeps one record in memory.

## Generator expressions
<!-- topics: expressions -->

`(expr for x in it if cond)` is the lazy cousin of a list comprehension. Parentheses can be dropped
when it is the only argument:

```python
print(sum(n * n for n in range(4)))      # prints: 14
print(any(len(w) > 5 for w in ["a", "bbbbbbb"]))  # prints: True  (stops at the first hit)
```

The **outermost iterable is evaluated immediately**; everything else is evaluated lazily:

```python
nums = [1, 2, 3]
g = (n * 10 for n in nums)
nums.append(4)           # the list object is shared, so the generator sees the 4
print(list(g))           # prints: [10, 20, 30, 40]

x = 5
h = (x + i for i in range(2))
x = 100                  # x is looked up lazily
print(list(h))           # prints: [100, 101]
```

## Exhaustion: one pass only
<!-- topics: exhaustion -->

Once a generator has produced everything it is **exhausted**. Asking again gives nothing, and
there is no error.

```python
squares = (n * n for n in range(4))
print(sum(squares))     # prints: 14
print(sum(squares))     # prints: 0
print(list(squares))    # prints: []
```

> **Gotcha** Passing one generator to two consumers silently gives the second an empty stream.
> Fix: materialise with `list()`, recreate it, or use `itertools.tee` (which buffers).

An **iterable** (list, `range`) hands out a fresh iterator each time; an **iterator** (generator)
is its own iterator and can only be used once. `for` calls `iter()` first, which is why a list works
in two loops and a generator does not.

## Pipelines and chaining
<!-- topics: pipelines, chaining -->

Generators compose into **pull-based pipelines**: the last stage asks the previous one for an item,
which asks the one before it, and so on.

```python
from itertools import islice

def parse(lines):
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            level, _, msg = line.partition(" ")
            yield level.upper(), msg

def only(records, level):
    return (r for r in records if r[0] == level)

lines = ["# header", "info ok", "error disk", "warn slow", "error oom", "error never reached"]
print(list(islice(only(parse(lines), "ERROR"), 2)))
# prints: [('ERROR', 'disk'), ('ERROR', 'oom')]
```

```text
islice(2)  <-  only("ERROR")  <-  parse  <-  lines
   pulls          pulls            pulls       (stops reading after the 2nd error)
```

Each stage holds one item at a time, so memory is O(1) whatever the input size, and the pipeline
stops reading as soon as the last stage has enough. Standard helpers: `itertools.chain` (concatenate),
`islice` (first N), `tee` (split, buffers), `heapq.merge` (merge sorted streams).

<details markdown="1"><summary>Check yourself: why does `take(2, ...)` have to be careful about when it stops?</summary>

If it calls `next()` one extra time after getting 2 items, it consumes (and loses) an item from a
shared source iterator. It must stop *before* pulling a third.

</details>

## yield from
<!-- topics: yield_from -->

`yield from sub` delegates to another iterable. It is more than `for x in sub: yield x`:

1. it forwards `send()`, `throw()` and `close()` into the sub-generator, and
2. it **evaluates to the sub-generator's return value**.

```python
def inner():
    yield 1
    yield 2
    return "inner result"

def outer():
    result = yield from inner()
    print("got:", result)
    yield 3

print(list(outer()))
# prints: got: inner result
# prints: [1, 2, 3]
```

Typical use: recursive generators such as walking a nested structure.

```python
def flatten(items):
    for item in items:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

print(list(flatten([1, [2, [3, 4]], 5])))   # prints: [1, 2, 3, 4, 5]
```

## send, throw and close
<!-- topics: send, throw, close -->

Generators are two-way channels. These three methods are how a caller talks *into* a paused generator.

* `g.send(v)` resumes it and makes the paused `yield` **expression evaluate to `v`**; it returns the next yielded value.
* `g.throw(exc)` raises `exc` **at the paused `yield`**.
* `g.close()` raises `GeneratorExit` there so `finally` blocks run. Yielding after that is a `RuntimeError`.

```python
def averager():
    total = 0.0
    count = 0
    average = None
    try:
        while True:
            value = yield average
            total += value
            count += 1
            average = total / count
    finally:
        print("closing after", count, "values")

avg = averager()
next(avg)             # "prime" it: run to the first yield
print(avg.send(10))   # prints: 10.0
print(avg.send(20))   # prints: 15.0
avg.close()           # prints: closing after 2 values
```

> **Gotcha** A fresh generator must be primed with `next(g)` (or `send(None)`) first. Sending a real
> value to a generator that is not paused at a `yield` raises `TypeError`.

> **Interview** "How are generators related to `async`?" `await` is built on the same suspend/resume
> machinery: a coroutine is a generator-like object an event loop resumes when its awaited result is ready.

## Eager validation: a generator trap
<!-- topics: suspension -->

Since the body is deferred, argument checks inside a generator function fire late:

```python
def chunks(size):
    if size <= 0:
        raise ValueError("size must be positive")
    yield "chunk"

g = chunks(0)          # no error here, the bug is silently stored
try:
    next(g)
except ValueError as e:
    print("late:", e)  # prints: late: size must be positive
```

Fix: a normal function validates, then returns the generator.

```python
def chunks(size):
    if size <= 0:
        raise ValueError("size must be positive")
    return _chunks(size)

def _chunks(size):
    yield "chunk"

try:
    chunks(0)
except ValueError as e:
    print("early:", e)   # prints: early: size must be positive
```

## Documents, chunking and metadata
<!-- topics: document_processing, rag_chunking, chunk_metadata -->

RAG ingestion is a generator pipeline: documents in, chunks out, each carrying where it came from.

```python
def chunk_text(text, size, overlap=0):
    step = size - overlap
    for index, start in enumerate(range(0, len(text), step)):
        end = min(start + size, len(text))
        yield {"index": index, "start": start, "end": end, "text": text[start:end]}
        if end == len(text):
            return

def chunk_documents(docs, size, overlap=0):
    for doc in docs:
        for chunk in chunk_text(doc["text"], size, overlap):
            chunk["doc_id"] = doc["id"]          # keep provenance with every chunk
            chunk["source"] = doc.get("source")
            yield chunk

docs = [{"id": "a", "source": "a.md", "text": "abcdefghij"}]
for c in chunk_documents(docs, size=4, overlap=1):
    print(c["doc_id"], c["index"], c["text"])
# prints: a 0 abcd
# prints: a 1 defg
# prints: a 2 ghij
```

Why carry metadata (doc id, source, page, offsets)? It enables **citations**, **metadata filters** in the vector
database (by tenant or date), **re-indexing** a changed document, and debugging "which chunk did the model see?".
Copy the metadata dict per chunk; sharing one mutable dict lets one chunk's edit leak into the others.

> **Interview** "Chunk size and overlap?" Start around 300-800 tokens with 10-15% overlap, measured in
> tokens not characters, then decide with retrieval metrics (recall@k) on a golden question set. Chunking is a
> hyperparameter you test, not a guess.

## Quick recap
- Calling a generator function runs nothing; `next()` runs to the next `yield` and pauses.
- Lazy = O(1) memory, one pass, errors at consumption; list = eager, reusable, O(n).
- A second consumer of an exhausted generator silently sees nothing.
- Pipelines pull: they read only as much input as the last stage needs.
- `yield from` forwards send/throw/close and returns the sub-generator's value.
- `send` feeds values in, `throw` injects exceptions, `close` triggers cleanup; prime a new generator first.
- Validate eagerly by splitting into a checking function and an inner generator.
- In ingestion pipelines, carry provenance metadata through every stage.
