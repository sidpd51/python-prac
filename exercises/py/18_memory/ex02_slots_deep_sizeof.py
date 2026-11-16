"""
Exercise 18.2 - __slots__ and the true size of an object             level: ★★☆
topics: memory.object_model, memory.allocation, memory.references

Problem
-------
You hold millions of retrieval hits in memory and want to know what they really cost.

1. EmbeddingRecord - a class with exactly the attributes `doc_id`, `vector`, `score`
   (score defaults to 0.0), declared with __slots__ so instances have NO per-instance
   __dict__:
     * assigning an undeclared attribute (rec.foo = 1) raises AttributeError
     * __repr__ -> "EmbeddingRecord(doc_id='d1', score=0.5)" (vector omitted: it's big)
   `PlainRecord` (given, kept in the stub) is the same thing without __slots__, for
   comparison.

2. deep_sizeof(obj) -> int
   sys.getsizeof() is SHALLOW: a list of 1000 long strings reports only the list's own
   pointer array. Write the deep version:
     * sum sys.getsizeof() over obj and everything reachable from it through
       dict keys and values, list/tuple/set/frozenset elements, an instance's __dict__
       (count the dict itself too) and its __slots__ attributes (only the ones that are set;
       walk __slots__ of every class in type(obj).__mro__)
     * count every object ONCE, by identity - shared references are not double counted,
       and reference cycles must terminate
     * don't count or descend into classes, functions or modules
     * don't recurse: a 100_000-deep nested list must not hit RecursionError
       (use an explicit stack)

Interviewers look for
---------------------
* Everything in CPython is a heap-allocated PyObject with a header (refcount + type
  pointer); a list stores pointers, not values.
* __slots__ replaces the per-instance dict with fixed descriptors: less memory, faster
  attribute access, no ad-hoc attributes (and no weakrefs unless '__weakref__' is
  listed). Since 3.11+ plain instances store attributes inline too, so the gap is smaller
  than old blog posts claim - but it's still real, measure it.
* Identity-based visited set - the same trick copy.deepcopy uses with its memo.
"""

from typing import Any


class PlainRecord:
    def __init__(self, doc_id: str, vector: Any, score: float = 0.0) -> None:
        self.doc_id = doc_id
        self.vector = vector
        self.score = score


class EmbeddingRecord:
    __slots__ = ("doc_id", "vector", "score")

    def __init__(self, doc_id: str, vector: Any, score: float = 0.0) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


def deep_sizeof(obj: Any) -> int:
    raise NotImplementedError
