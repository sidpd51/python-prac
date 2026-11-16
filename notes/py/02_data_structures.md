# Data Structures

Choosing the right container is most of the performance work you will ever do in Python. A list, a dict, a set and a deque
differ in what they make cheap, and an interviewer asking "list or set?" is really asking whether you know what is
happening underneath. In AI engineering this shows up as deduplicating millions of chunk ids, counting tokens, building
inverted indexes, caching embeddings with an LRU, and keeping a sliding window of conversation messages. This note covers
the built-ins, the `collections` extras, searching, and Big-O so you can justify each choice.

## Lists
<!-- topics: lists -->

A `list` is a **dynamic array** of references. CPython keeps a contiguous block of pointers and over-allocates, so
`append` is *amortised* O(1): occasionally it must grow and copy, but averaged over many appends the cost is constant.

```text
list object ─► [ ptr0 | ptr1 | ptr2 | ... spare capacity ]
                  │      │      └► 3.5 (a float object elsewhere)
                  └► "hi"
```

| Operation | Cost | Why |
|---|---|---|
| `lst[i]`, `lst[i] = x` | O(1) | pointer arithmetic |
| `append(x)`, `pop()` | O(1) amortised | touches the end only |
| `insert(0, x)`, `pop(0)`, `del lst[0]` | O(n) | shifts every element |
| `x in lst`, `lst.index(x)`, `lst.count(x)` | O(n) | linear scan with `==` |
| `lst.sort()` | O(n log n) | Timsort, stable |
| `lst[a:b]` | O(b - a) | builds a new list |

```python
lst = [3, 1, 2]
lst.append(4); lst.extend([5, 6]); lst.insert(0, 0)
print(lst)                      # prints: [0, 3, 1, 2, 4, 5, 6]
print(lst.pop(), lst.pop(0))    # prints: 6 0
lst.sort(reverse=True)
print(lst)                      # prints: [5, 4, 3, 2, 1]
print(sorted(["b", "A", "c"], key=str.lower))   # prints: ['A', 'b', 'c']
print([1, 2] + [3], [0] * 3)    # prints: [1, 2, 3] [0, 0, 0]
```

`list.sort()` sorts in place and returns `None`; `sorted(x)` returns a new list from any iterable. Both accept
`key=` (called once per element) and are **stable**: equal keys keep their original order, so you can sort by a
secondary key first and then the primary key.

```python
docs = [("b", 0.9), ("a", 0.9), ("c", 0.5)]
print(sorted(docs, key=lambda d: (-d[1], d[0])))   # prints: [('a', 0.9), ('b', 0.9), ('c', 0.5)]
```

> **Gotcha** `lst.remove(x)` removes the first match only and raises `ValueError` if absent. Removing while iterating
> skips elements. Use a comprehension to filter: `lst = [x for x in lst if keep(x)]`.

> **Interview** "Cost of `pop(0)` in a loop?" O(n) each, so O(n^2) overall. Use `collections.deque` (O(1) `popleft`).

## Dictionaries and dict lookup
<!-- topics: dicts, dict_lookup -->

A `dict` is a **hash table**. To store `d[k] = v`, Python computes `hash(k)`, uses it to pick a slot in an index array,
and stores the key, hash and value in a compact entries array. Lookup repeats the hash, jumps to the slot, and confirms
with `==` (after an identity shortcut). Average cost is **O(1)**; collisions are resolved by probing, and heavy
collisions can degrade to O(n), which is why hash quality matters.

```text
d[k]  ─►  hash(k) ─► slot in index table ─► entry (hash, key, value) ─► compare key ─► value
```

Consequences:

* **Keys must be hashable**: immutable and hashing consistently with equality. `str`, `int`, `tuple` of hashables,
  `frozenset` and `None` work; `list`, `dict`, `set` raise `TypeError: unhashable type`.
* Since Python 3.7 a dict **preserves insertion order** (a language guarantee). Updating an existing key keeps its position.
* `1`, `1.0` and `True` are equal with equal hashes, so they are the *same key*.

```python
d = {"a": 1, "b": 2}
d["c"] = 3
print(d["a"], d.get("z"), d.get("z", 0))      # prints: 1 None 0
print(d.pop("a"), d.setdefault("d", 4), d)    # prints: 1 4 {'b': 2, 'c': 3, 'd': 4}
print(list(d), list(d.values()), list(d.items())[0])   # prints: ['b', 'c', 'd'] [2, 3, 4] ('b', 2)
print({1: "x", 1.0: "y", True: "z"})          # prints: {1: 'z'}
try:
    {[1]: "x"}
except TypeError as e:
    print(e)                                  # prints: unhashable type: 'list'
```

`d[k]` raises `KeyError` on a miss; `d.get(k, default)` never does. `k in d` tests keys in O(1). The `keys()`, `values()`
and `items()` views are live (they reflect later changes), and `keys()`/`items()` support set operations (`a.keys() & b.keys()`).
Merging: `{**a, **b}` or `a | b` (3.9+), right side wins.

```python
a, b = {"x": 1, "y": 2}, {"y": 20, "z": 30}
print(a | b)                       # prints: {'x': 1, 'y': 20, 'z': 30}
print(a.keys() & b.keys())         # prints: {'y'}
view = a.keys()
a["new"] = 0
print(list(view))                  # prints: ['x', 'y', 'new']
```

> **Gotcha** Do not change a dict's size while iterating over it (`RuntimeError: dictionary changed size during
> iteration`); iterate over `list(d)` instead. And `d.get(k, expensive())` evaluates `expensive()` every time.

> **Remember** `__eq__` and `__hash__` contract: equal objects must have equal hashes. A mutable key whose hash changes
> after insertion becomes unfindable.

## Tuples
<!-- topics: tuples -->

A `tuple` is an **immutable** sequence. Use it for a *fixed-shape record* (a heterogeneous "row"), where position has
meaning; use a list for a *homogeneous collection* of variable length. Because they are immutable (and their items are
hashable) tuples can be dict keys and set members; lists cannot.

```python
point = (3, 4)
x, y = point
print(x, y)                              # prints: 3 4
print(type((1)).__name__, type((1,)).__name__)   # prints: int tuple
cache = {("gpt", 0.2): "result"}         # tuple as a composite key
print(cache[("gpt", 0.2)])               # prints: result
def min_max(xs):
    return min(xs), max(xs)              # returns ONE tuple, unpacked by the caller
lo, hi = min_max([4, 9, 2])
print(lo, hi)                            # prints: 2 9
print(hash((1, 2)) == hash((1, 2)))      # prints: True
```

The comma makes the tuple, not the parentheses: `(1)` is an int, `(1,)` is a tuple, `()` is empty. Tuples are slightly
smaller and faster to build than lists. Immutability is **shallow**: `([1], 2)` is hashable-looking but raises
`TypeError` on `hash()` because it contains a list.

> **Interview** "Tuple vs list?" Immutability (hashable, safe to share), semantic meaning (record vs sequence), small
> performance edge. Not "tuples are faster at everything".

## Sets, set operations and deduplication
<!-- topics: sets, set_operations, dedup -->

A `set` is an unordered collection of **unique, hashable** items, implemented as a hash table with only keys. Membership,
`add` and `remove` are O(1) average. `frozenset` is the immutable, hashable variant.

```python
a, b = {1, 2, 3}, {3, 4}
print(a | b, a & b, a - b, a ^ b)       # prints: {1, 2, 3, 4} {3} {1, 2} {1, 2, 4}
print({1, 2} <= a, a.isdisjoint({9}))   # prints: True True
s = set()
s.add(5); s.discard(99)                 # discard never raises, remove does
print(s, type({}).__name__, type(set()).__name__)   # prints: {5} dict set
```

`{}` is an empty **dict**; an empty set is `set()`. Operators: union `|`, intersection `&`, difference `-`, symmetric
difference `^`, subset `<=`. The method forms (`a.union(iterable)`) accept any iterable.

**Deduplication** is the classic use. `set(items)` drops duplicates but **loses order**. To keep first-seen order, use
`dict.fromkeys` (dicts are ordered) or a `seen` set in a loop:

```python
chunk_ids = ["c3", "c1", "c3", "c2", "c1"]
print(list(dict.fromkeys(chunk_ids)))        # prints: ['c3', 'c1', 'c2']

seen = set()
unique = [c for c in chunk_ids if not (c in seen or seen.add(c))]
print(unique)                                # prints: ['c3', 'c1', 'c2']

records = [{"id": 1, "t": "a"}, {"id": 1, "t": "a"}, {"id": 2, "t": "b"}]
print(len({tuple(sorted(r.items())) for r in records}))   # prints: 2   (dicts are unhashable; convert)
```

Scale: for exact duplicates of 10M documents, store a **content hash** (`hashlib.sha256(text.encode()).hexdigest()`)
in the set rather than the full text. For *near*-duplicates you need MinHash/SimHash, not a set.

> **Gotcha** Set iteration order is arbitrary (string hashes are randomised per process), so never depend on it; use
> `sorted(s)` for stable output. Elements must be hashable: you cannot put a list or dict in a set.

> **AI use** "Which of these 100k chunk ids were already embedded?" is `new_ids - embedded_ids` with a set: O(n + m),
> versus O(n * m) with a list.

## List vs dict vs set: choosing
<!-- topics: list_vs_dict -->

Choose by the question you ask most often.

| Need | Use | Why |
|---|---|---|
| ordered, indexable, duplicates allowed | `list` | position matters |
| "is x present?" on many items | `set` | O(1) membership |
| find a value by key, or count/group by key | `dict` | O(1) key to value |
| presence plus stored data (id to embedding) | `dict` | a set cannot hold values |
| fixed record / composite key | `tuple` | immutable, hashable |

```python
import time

ids = list(range(20_000))
as_set = set(ids)
probe = list(range(19_000, 21_000))

t0 = time.perf_counter(); hits = sum(1 for p in probe if p in ids);    t_list = time.perf_counter() - t0
t0 = time.perf_counter(); hits2 = sum(1 for p in probe if p in as_set); t_set = time.perf_counter() - t0
print(hits == hits2, t_list > t_set * 10)    # prints: True True
```

Building the set costs O(n) once, so convert when you will do many lookups. For a single lookup in a small list, the
scan is cheaper than building anything. Sets and dicts cost more memory per element than lists.

> **Interview** "Check 100k incoming ids against 1M known ids?" A set (or dict if you need the stored embedding). A list
> would be about 10^11 comparisons.

## Searching and complexity
<!-- topics: linear_search, binary_search, searching, complexity -->

**Big-O** describes how cost grows with input size n, ignoring constants. Memorise the ladder:
O(1) < O(log n) < O(n) < O(n log n) < O(n^2). "Amortised" means averaged over a sequence of operations;
"average" for hashing means assuming a good hash distribution.

| Structure and op | Cost |
|---|---|
| list index, append, pop() | O(1) |
| list insert/pop at front, `in`, `index` | O(n) |
| dict/set get, set, `in`, delete | O(1) average |
| `deque` append/popleft either end | O(1) |
| `sorted()`, `.sort()` | O(n log n) |
| `bisect.bisect_left` on sorted list | O(log n) |
| `heapq.heappush/heappop` | O(log n) |
| `heapq.nlargest(k, xs)` | O(n log k) |

**Linear search** checks items one by one: O(n), works on anything, needs no ordering. It is the right answer for small
or one-off lookups, unsorted data, or when you need a predicate (`next(x for x in xs if x.ok)`).

**Binary search** needs a **sorted** sequence and halves the search range each step: O(log n). 1M items need about 20 steps.

```python
def binary_search(xs: list[int], target: int) -> int:
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

xs = [1, 3, 5, 7, 9, 11]
print(binary_search(xs, 7), binary_search(xs, 4))   # prints: 3 -1

import bisect
print(bisect.bisect_left(xs, 7), bisect.bisect_left(xs, 4), bisect.bisect_right(xs, 7))
# prints: 3 2 4
bisect.insort(xs, 4)
print(xs)                                           # prints: [1, 3, 4, 5, 7, 9, 11]
```

`bisect_left` returns the leftmost insertion point (so it also finds "the first element >= x"); `bisect_right` goes after
equal items. Use the standard library `bisect`, not a hand-rolled version, in real code. Typical uses: bucketing a score
into a grade, finding the chunk containing token offset N from a sorted list of offsets.

> **Gotcha** Binary search on unsorted data returns garbage with no error. Sorting first costs O(n log n), so it only
> pays off with many searches. Off-by-one errors live in `lo <= hi` and `mid + 1` / `mid - 1`.

<details markdown="1"><summary>Check yourself: you search a list once; is sorting then binary-searching faster?</summary>

No. Sorting is O(n log n), more than the single O(n) scan. Sort (or build a set/dict) only when you will search repeatedly.

</details>

## Stack and queue, and deque
<!-- topics: stack_queue, deque -->

A **stack** is LIFO (last in, first out): undo history, DFS, matching brackets, call stacks. A **queue** is FIFO
(first in, first out): task scheduling, BFS, request buffers. A list is a fine stack (`append`/`pop` at the end are O(1))
but a poor queue (`pop(0)` is O(n)). `collections.deque` is a doubly linked block list with O(1) at **both** ends.

```python
from collections import deque

stack = []
stack.append("a"); stack.append("b")
print(stack.pop())                 # prints: b

q = deque()
q.append("job1"); q.append("job2")
print(q.popleft(), len(q))         # prints: job1 1

window = deque(maxlen=3)           # fixed-size sliding window
for n in range(6):
    window.append(n)
print(window)                      # prints: deque([3, 4, 5], maxlen=3)
q2 = deque([1, 2, 3]); q2.rotate(1); q2.appendleft(0)
print(q2)                          # prints: deque([0, 3, 1, 2])
```

With `maxlen`, appending to a full deque silently discards from the opposite end: ideal for "last N messages" in a
chat history or a rolling latency average. A deque is O(n) for indexing in the middle, so it is not a list replacement.

Which queue? `deque` is a plain data structure (not thread-safe for coordination). `queue.Queue` is thread-safe with
blocking `put/get` for threads. `asyncio.Queue` is for coroutines (`await q.get()`). `multiprocessing.Queue` crosses
processes. `heapq`/`PriorityQueue` order by priority, not arrival.

```python
from collections import deque

def balanced(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs and (not stack or stack.pop() != pairs[ch]):
            return False
    return not stack

print(balanced("{[()]}"), balanced("([)]"))   # prints: True False

graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
order, seen, frontier = [], {"a"}, deque(["a"])
while frontier:                      # BFS uses a queue
    node = frontier.popleft()
    order.append(node)
    for nxt in graph[node]:
        if nxt not in seen:
            seen.add(nxt); frontier.append(nxt)
print(order)                         # prints: ['a', 'b', 'c', 'd']
```

> **Interview** "Implement a queue with a list?" Say it works but `pop(0)` is O(n); the right tool is `deque`.

## Counter and defaultdict
<!-- topics: counter, defaultdict -->

Both remove the "check whether the key exists first" boilerplate.

`Counter` is a dict subclass mapping items to counts. Missing keys count as `0` (without being inserted).

```python
from collections import Counter

c = Counter("mississippi")
print(c)                               # prints: Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})
print(c.most_common(2), c["zzz"])      # prints: [('i', 4), ('s', 4)] 0
c.update(["m", "m"]); c.subtract("ss")
print(c["m"], c["s"])                  # prints: 3 2
print(Counter(a=3, b=1) - Counter(a=1, b=5))   # prints: Counter({'a': 2})
tokens = "the cat the dog the end".split()
print(Counter(tokens).most_common(1))  # prints: [('the', 3)]
```

`most_common(n)` is sorted by count (ties in first-seen order). Subtraction operators drop zero and negative counts;
`subtract()` keeps them. Great for token frequency, label distributions, and comparing bag-of-words.

`defaultdict(factory)` calls `factory()` to create the value the first time a missing key is **accessed**.

```python
from collections import defaultdict

groups = defaultdict(list)
for word in ["apple", "avocado", "banana", "blueberry", "cherry"]:
    groups[word[0]].append(word)
print(dict(groups))
# prints: {'a': ['apple', 'avocado'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}

index = defaultdict(set)               # inverted index: term -> doc ids
for doc_id, text in enumerate(["red fox", "red hen", "blue fox"]):
    for term in text.split():
        index[term].add(doc_id)
print(sorted(index["red"]), sorted(index["fox"]))   # prints: [0, 1] [0, 2]

nested = defaultdict(lambda: defaultdict(int))
nested["a"]["x"] += 1
print({k: dict(v) for k, v in nested.items()})      # prints: {'a': {'x': 1}}
```

> **Gotcha** Merely *reading* `dd[missing]` inserts the key, so `if dd[k]:` and `len(dd)` can surprise you. Use `k in dd`
> or `dd.get(k)` (which does not insert) to test. The factory must be a zero-argument callable: `list`, `int`, `set`, a lambda.

## OrderedDict and namedtuple
<!-- topics: ordereddict, namedtuple -->

Plain dicts keep insertion order, so `OrderedDict` is now niche. It still differs in three ways: equality between
two OrderedDicts **is order-sensitive**, it has `move_to_end(key, last=True)` and `popitem(last=False)` for cheap reordering
(the basis of a hand-rolled **LRU cache**).

```python
from collections import OrderedDict

print(OrderedDict(a=1, b=2) == OrderedDict(b=2, a=1))   # prints: False
print({"a": 1, "b": 2} == {"b": 2, "a": 1})             # prints: True

class LRU:
    def __init__(self, capacity: int):
        self.capacity, self.data = capacity, OrderedDict()
    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)             # mark most recently used
        return self.data[key]
    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)      # evict least recently used

cache = LRU(2)
cache.put("a", 1); cache.put("b", 2); cache.get("a"); cache.put("c", 3)
print(list(cache.data))                        # prints: ['a', 'c']
```

(For function results, `functools.lru_cache` already does this.)

`namedtuple` creates a tuple subclass with named fields: lightweight, immutable, and still unpackable and indexable.

```python
from collections import namedtuple
from typing import NamedTuple

Hit = namedtuple("Hit", ["doc_id", "score"])
h = Hit("d1", 0.83)
print(h.doc_id, h[1], h)               # prints: d1 0.83 Hit(doc_id='d1', score=0.83)
doc, score = h
print(h._replace(score=0.9), h._asdict())
# prints: Hit(doc_id='d1', score=0.9) {'doc_id': 'd1', 'score': 0.83}

class Result(NamedTuple):              # typed, class syntax, supports defaults and methods
    doc_id: str
    score: float = 0.0
print(Result("d2"))                    # prints: Result(doc_id='d2', score=0.0)
```

Choosing a record type: `tuple` (throwaway), `namedtuple`/`NamedTuple` (immutable, tiny, positional), `@dataclass`
(mutable or `frozen=True`, methods, defaults), `TypedDict` (typing for a plain dict you will JSON-serialise),
pydantic `BaseModel` (validate untrusted input such as LLM output).

> **Remember** namedtuples are tuples: `h == ("d1", 0.83)` is `True` and they unpack positionally, which is convenient
> and occasionally surprising.

## Quick recap
- A list is a dynamic array: O(1) index and append, O(n) for insert/pop at the front and for `in`.
- A dict is a hash table: O(1) average, keys must be hashable, insertion order is guaranteed (3.7+).
- Tuples are immutable records usable as dict keys; the comma makes the tuple; immutability is shallow.
- Sets give O(1) membership and the algebra `| & - ^`; `{}` is a dict; dedupe preserving order with `dict.fromkeys`.
- Pick the container by the query: membership means set, key to value means dict, ordered/positional means list.
- Binary search needs sorted data and is O(log n); use `bisect`. A single search is cheaper unsorted.
- `deque` gives O(1) at both ends and `maxlen` windows; list as queue is O(n) per `pop(0)`.
- `Counter` counts and `most_common`; `defaultdict` groups and builds inverted indexes; reading a missing key inserts it.
- `OrderedDict` is order-sensitive and has `move_to_end` (LRU); `namedtuple` is a cheap named immutable record.
