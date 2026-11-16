# Iterators

The iterator protocol is the small contract behind every `for` loop, comprehension, `list()`,
`sum()`, unpacking and `in` test in Python. Understanding it explains why a list can be looped twice
but a file or generator cannot, why `StopIteration` exists, and how generators (see the generators note)
are really just a convenient way to write iterators. In AI engineering it matters for streaming
LLM tokens, paginated vector-DB APIs and lazy dataset loading, where you consume values one at a time and
sometimes only once. Interviewers ask "what does a `for` loop actually do?" to find out how deep your
model of Python goes.

## Iterable versus iterator
<!-- topics: iterable_vs_iterator -->

Two roles, two different protocols:

* An **iterable** is anything you can loop over: it has `__iter__()` that returns an iterator
  (or `__getitem__` with integers from 0, the legacy route). Lists, tuples, strings, dicts, sets, `range`.
* An **iterator** is the object that does the walking: it has `__next__()` that returns the next item or
  raises `StopIteration`, and `__iter__()` that returns **itself**. Generators, file objects, `zip`,
  `map`, `enumerate`.

```text
iterable  --iter()-->  iterator  --next()-->  item, item, ..., StopIteration
(re-usable)            (one position, one pass)
```

Every iterator is an iterable (because `__iter__` returns itself), but not the other way round. The
key difference: an iterable gives you a **fresh iterator each time**, so you can loop it repeatedly;
an iterator holds the position and is used up.

```python
nums = [1, 2, 3]
it = iter(nums)

print(iter(nums) is iter(nums))   # prints: False   (list makes a new iterator each time)
print(iter(it) is it)             # prints: True    (an iterator is its own iterator)
print(type(nums).__name__, type(it).__name__)   # prints: list list_iterator
print(list(it), list(it))         # prints: [1, 2, 3] []
```

> **Remember** iterable = "can give me an iterator"; iterator = "is the cursor". `list` is iterable
> but not an iterator: `next([1, 2])` raises `TypeError`.

## iter(), next() and StopIteration
<!-- topics: iter, next, stopiteration -->

`iter(obj)` calls `obj.__iter__()` and returns the iterator (TypeError if `obj` is not iterable).
`next(it)` calls `it.__next__()`. When there is nothing left, `__next__` raises `StopIteration`, which is
the normal end-of-data *signal*, not an error.

```python
it = iter(["a", "b"])
print(next(it))     # prints: a
print(next(it))     # prints: b
try:
    next(it)
except StopIteration:
    print("done")   # prints: done
```

`next(it, default)` returns `default` instead of raising: the clean way to "get the first item or a
fallback".

```python
print(next(iter([]), None))          # prints: None
print(next(iter([10, 20]), None))    # prints: 10
words = ["x", "apple", "avocado"]
print(next((w for w in words if w.startswith("a")), "none"))   # prints: apple
```

There is also a two-argument form, `iter(callable, sentinel)`, which calls `callable()` repeatedly until it
returns `sentinel`. Perfect for reading fixed-size blocks:

```python
import io

stream = io.BytesIO(b"abcdefgh")
chunks = list(iter(lambda: stream.read(3), b""))
print(chunks)   # prints: [b'abc', b'def', b'gh']
```

> **Gotcha** `StopIteration` raised *inside* a generator body is converted to `RuntimeError` (PEP 479),
> so it cannot silently end the generator early. In your own `for_each`-style helpers keep the `try`
> around `next()` only, otherwise a `StopIteration` raised by your callback is mistaken for end of data.

<details markdown="1"><summary>Check yourself: why is `next(it, None)` often better than try/except?</summary>

It expresses "first item or default" in one expression and cannot accidentally catch a `StopIteration`
raised by unrelated code. Use a unique `object()` sentinel if `None` could be a real value.

</details>

## How a for loop works
<!-- topics: for_internals -->

```python
# nocheck
for x in iterable:
    body
```

is equivalent to:

```python
# nocheck
it = iter(iterable)                 # ONE call to __iter__
while True:
    try:
        x = next(it)                # __next__ each round
    except StopIteration:
        break                       # normal exit; `else:` clause runs only if no break
    body
```

You can prove it by implementing the loop by hand:

```python
def manual_for(iterable, fn):
    it = iter(iterable)
    count = 0
    while True:
        try:
            item = next(it)
        except StopIteration:
            return count
        fn(item)               # outside the try: its own StopIteration must propagate
        count += 1

seen = []
print(manual_for("abc", seen.append), seen)   # prints: 3 ['a', 'b', 'c']
```

The same protocol drives comprehensions, `list(x)`, `tuple(x)`, `sum`, `max`, `sorted`, `in`
(falls back to iterating), unpacking `a, b = x`, `*args` expansion and `dict(pairs)`. Because the loop calls
`iter()` once up front, **mutating a list while looping** misbehaves: the list iterator holds an index.

```python
items = [2, 4, 6, 8]
for x in items:
    if x % 2 == 0:
        items.remove(x)
print(items)   # prints: [4, 8]
```

Every element is even, yet half survive: after removing 2 the list shifted left but the iterator's index
moved on, skipping 4. Iterate over a copy (`for x in items[:]`) or build a new list instead.

## The iterator protocol: __iter__ and __next__
<!-- topics: protocol, dunder_iter, dunder_next -->

To be an **iterator** a class implements both methods:

* `__iter__(self)` returns `self`.
* `__next__(self)` returns the next value or raises `StopIteration`.

To be merely **iterable** a class implements only `__iter__` and returns a (new) iterator, which can be a
generator.

```python
from collections.abc import Iterable, Iterator

class Countdown:                      # an iterator
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n + 1

c = Countdown(3)
print(list(c))                                   # prints: [3, 2, 1]
print(isinstance(c, Iterator), isinstance(c, Iterable))   # prints: True True
print(isinstance([1], Iterator), isinstance([1], Iterable))   # prints: False True
```

`collections.abc.Iterator` checks structurally (the presence of `__iter__` and `__next__`); you do not
need to inherit from it.

> **Interview** "`isinstance(obj, Iterable)` returned False, but `for x in obj` works. How?" The class has
> no `__iter__` but defines `__getitem__`, the legacy sequence protocol: `iter()` then builds an iterator that
> calls `obj[0]`, `obj[1]`, ... until `IndexError`. The reliable test is simply `try: iter(obj)`.

```python
class Legacy:
    def __getitem__(self, i):
        if i >= 3:
            raise IndexError
        return i * 10

from collections.abc import Iterable
print(isinstance(Legacy(), Iterable), list(Legacy()))   # prints: False [0, 10, 20]
```

## Custom iterator classes
<!-- topics: custom -->

Write a class when the iterator needs **state you want to expose** (`pages_fetched`), methods besides
`__next__`, or lazy resources with explicit lifecycle. A cursor-paginated API is the textbook case:

```python
class DocumentPager:
    """Yields every document across pages; fetches a page only when needed."""
    def __init__(self, fetch_page):
        self._fetch = fetch_page          # fetch_page(cursor) -> (items, next_cursor)
        self._buffer = []
        self._cursor = None
        self._started = False
        self._done = False
        self.pages_fetched = 0

    def __iter__(self):
        return self

    def __next__(self):
        while not self._buffer:           # loop: empty pages may still have a next cursor
            if self._done:
                raise StopIteration
            items, nxt = self._fetch(self._cursor)
            self.pages_fetched += 1
            self._buffer = list(reversed(items))
            self._cursor = nxt
            self._done = nxt is None
        return self._buffer.pop()

pages = {None: ([1, 2], "c2"), "c2": ([], "c3"), "c3": ([3], None)}
pager = DocumentPager(lambda cursor: pages[cursor])
print(list(pager), pager.pages_fetched)   # prints: [1, 2, 3] 3
print(list(pager))                        # prints: []
```

Rules for a well-behaved iterator:

1. `__iter__` returns `self`.
2. After `StopIteration` it must **keep raising** it on every later `next()` (never "restart").
3. Be lazy: constructing it does no work; fetch on demand.
4. Do not mutate or double-consume the source.

> **AI use** For production pagination also think about: page size, retries with backoff on a failed
> page (without losing the cursor), a max-items cap, and not holding every page in memory.
> Exposing `pages_fetched` makes it testable ("did it stay lazy?").

A **generator** gives you rules 1-3 for free and is usually shorter:

```python
pages = {None: ([1, 2], "c2"), "c2": ([], "c3"), "c3": ([3], None)}

def pager_gen(fetch_page):
    cursor = None
    while True:
        items, cursor = fetch_page(cursor)
        yield from items
        if cursor is None:
            return

print(list(pager_gen(lambda c: pages[c])))   # prints: [1, 2, 3]
```

Prefer the generator unless you need extra public state/methods, a custom `__repr__`, or to implement
`__iter__` and `__next__` for performance in a hot library path.

## Iterable containers that produce fresh iterators
<!-- topics: iterable_vs_iterator, dunder_iter -->

A container should be *iterable but not an iterator*, so it can be looped many times. Implement
`__iter__` as a generator or return a new iterator object each call:

```python
class Corpus:
    def __init__(self, docs):
        self.docs = list(docs)
    def __iter__(self):                 # a generator function: a new generator per loop
        for d in self.docs:
            yield d.lower()

corpus = Corpus(["A", "B"])
print(list(corpus), list(corpus))   # prints: ['a', 'b'] ['a', 'b']
```

The trap is the opposite design, where `__iter__` returns `self` on a container: the second loop is silently
empty.

```python
class BadCorpus:
    def __init__(self, docs):
        self.docs, self.i = list(docs), 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i >= len(self.docs):
            raise StopIteration
        self.i += 1
        return self.docs[self.i - 1]

bad = BadCorpus(["A", "B"])
print(list(bad), list(bad))   # prints: ['A', 'B'] []
```

Many APIs check `iter(x) is x` to detect "this is a one-shot iterator" (e.g. to decide whether to
materialise). A dataset class passed to `DataLoader`-style code should be re-iterable per epoch.

## Exhaustion and one-pass iterators
<!-- topics: exhaustion, one_pass -->

An exhausted iterator stays exhausted and gives no error on later loops. File objects, generators,
`map`, `zip`, `filter`, `enumerate`, `csv.reader` and network streams are all **one-pass**.

```python
m = map(str.upper, ["a", "b"])
print(list(m))   # prints: ['A', 'B']
print(list(m))   # prints: []

z = zip([1, 2], "ab")
print(max(z), list(z))   # prints: (2, 'b') []
```

Consuming partially is also consuming: `next()`, `in`, `any()`, `min()` and a `break` out of a `for` all
advance the same shared iterator.

```python
it = iter(range(10))
print(3 in it)           # prints: True   (consumed 0, 1, 2, 3)
print(next(it))          # prints: 4
```

Options when you need to traverse **twice**:

| Option | Cost |
|---|---|
| `list(it)` (materialise) | O(n) memory, simplest, fine if it fits |
| `itertools.tee(it, 2)` | buffers the gap between the two consumers; O(n) worst case, and you must not use the original afterwards |
| re-create the iterator from the source | recompute / re-read; needs a re-iterable source or a factory function |
| single pass computing everything | O(1) memory: gather count, sum and your output in one loop |

```python
from itertools import tee

it = iter([1, 2, 3, 4])
a, b = tee(it)
print(sum(a), list(b))   # prints: 10 [1, 2, 3, 4]
```

> **Gotcha** `tee` is not free: if one branch runs far ahead, everything it consumed stays buffered, so it is
> no better than `list()` for "stats first, then stream".

> **AI use** A streamed LLM response is a one-pass iterator of chunks. If you need both to show tokens live
> and keep the full text, accumulate into a list *as you iterate* instead of trying to loop twice.

<details markdown="1"><summary>Check yourself: why does `sum(z)` then `list(z)` give nothing for `z = zip(...)`?</summary>

`zip` returns an iterator. `sum` consumed it to the end, so the second pass starts at the end position
and `list(z)` is empty. Nothing raises.

</details>

## Infinite iterators and useful tools
<!-- topics: iter, next -->

Iterators are lazy, so they may be infinite as long as consumers stop themselves. `itertools` provides the
standard building blocks:

```python
from itertools import count, cycle, islice, chain, takewhile

print(list(islice(count(10, 5), 3)))                  # prints: [10, 15, 20]
print(list(islice(cycle("ab"), 5)))                   # prints: ['a', 'b', 'a', 'b', 'a']
print(list(chain([1], (2, 3))))                       # prints: [1, 2, 3]
print(list(takewhile(lambda n: n < 3, count())))      # prints: [0, 1, 2]
```

`enumerate(it)`, `zip(a, b)` and `reversed(seq)` are iterators too. `islice` is the iterator
equivalent of slicing (lists support `[a:b]`, iterators do not). `zip` stops at the shortest input
(`zip(..., strict=True)` raises on mismatched lengths; `itertools.zip_longest` pads).

> **Interview** "Iterator vs generator vs iterable?" Iterable: has `__iter__`. Iterator: has `__next__`
> and `__iter__` returning self. Generator: an iterator created by a function with `yield` or a generator
> expression; all generators are iterators, not all iterators are generators.

## Quick recap
- Iterable: `__iter__` returns a fresh iterator (list, dict, str, range). Iterator: `__next__` + `__iter__` returning self (generator, file, map, zip).
- `iter(x)` -> iterator; `next(it)` -> item or `StopIteration`; `next(it, default)` avoids the exception; `iter(callable, sentinel)` reads until a sentinel.
- `for` = one `iter()` call, then `next()` in a loop, ending on `StopIteration`; comprehensions, unpacking and `list()` use the same machinery.
- A custom iterator implements `__iter__` (return `self`) and `__next__`; once exhausted it must keep raising `StopIteration`.
- Containers should be iterable-not-iterator (`__iter__` creating a new iterator); `__iter__` returning `self` makes them one-shot.
- Iterators are one-pass: a second loop is silently empty. Re-traverse with `list`, `tee`, or by re-creating.
- Reliable iterability test: `iter(obj)` (the legacy `__getitem__` route is iterable without `Iterable`).
- Prefer a generator; write a class when you need exposed state or extra methods.
