"""
Exercise 07.3 - Iterable vs iterator: a re-iterable eval dataset   level: ★★☆
topics: iterators.iterable_vs_iterator, iterators.one_pass, iterators.dunder_iter, iterators.iter

Problem
-------
An eval harness loops over the same dataset once per model under test. The first version
was written like this - and every model after the first scored 0/0:

    class BadDataset:
        def __init__(self, rows):
            self.rows, self.i = rows, 0
        def __iter__(self):
            return self                  # the container IS its own iterator...
        def __next__(self):
            if self.i == len(self.rows):
                raise StopIteration
            self.i += 1
            return self.rows[self.i - 1]

Fix the design by separating the ITERABLE from the ITERATOR:

1. DatasetIterator(rows: Sequence) - the iterator: __iter__ returns self, __next__ returns
   rows in order then raises StopIteration (and keeps raising it).
2. EvalDataset(rows: Iterable) - the iterable container:
   * stores the rows as a tuple (so a one-shot generator passed in is consumed once, here,
     and the dataset still works for many passes)
   * __iter__ returns a NEW DatasetIterator every call (no yield in __iter__ - use the class)
   * __len__ -> number of rows
   Nested loops over the same dataset must work (pairwise comparisons).
3. is_iterable(obj) -> bool: True if iter(obj) succeeds. Use iter() itself - NOT
   isinstance(obj, collections.abc.Iterable), which misses classes that only implement the
   old sequence protocol (__getitem__ with 0, 1, 2, ... until IndexError).
4. is_iterator(obj) -> bool: True if obj is iterable, has __next__, and iter(obj) is obj.

Interviewers look for
---------------------
* Iterable = has __iter__ returning a fresh iterator (list, dict, str, range, your dataset).
  Iterator = has __next__ and __iter__ returning self; it is single-use.
* Why `for` calls iter() first, and why a container must not be its own iterator.
* A list can be traversed many times; list(iterator) only once.
"""

from collections.abc import Iterable, Sequence
from typing import Any


class DatasetIterator:
    def __init__(self, rows: Sequence[Any]) -> None:
        self._rows = rows
        self._index = 0

    def __iter__(self) -> "DatasetIterator":
        return self

    def __next__(self) -> Any:
        if self._index >= len(self._rows):
            raise StopIteration
        row = self._rows[self._index]
        self._index += 1
        return row


class EvalDataset:
    def __init__(self, rows: Iterable[Any]) -> None:
        self._rows = tuple(rows)

    def __iter__(self) -> DatasetIterator:
        return DatasetIterator(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def is_iterable(obj: Any) -> bool:
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def is_iterator(obj: Any) -> bool:
    return is_iterable(obj) and hasattr(obj, "__next__") and iter(obj) is obj
