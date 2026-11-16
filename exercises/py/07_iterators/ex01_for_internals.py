"""
Exercise 07.1 - What a for loop really does: iter(), next(), StopIteration   level: ★☆☆
topics: iterators.for_internals, iterators.iter, iterators.next, iterators.stopiteration

Problem
-------
1. for_each(iterable, fn) -> int
   Call fn(item) for every item and return how many items were processed - WITHOUT a
   `for` statement or comprehension: use iter(), next() and catch StopIteration yourself,
   exactly like the interpreter does.
   * call iter() exactly once
   * a StopIteration raised by `fn` itself must NOT be mistaken for the end of the data -
     it must propagate to the caller (only the next() call belongs in the try block).

2. first(iterable, default=<missing>) -> item
   Return the first item. Empty iterable: return `default` if given, else raise ValueError.
   Use next() (hint: next has a 2-argument form). Must not consume more than one item
   from an iterator. Note that None is a valid default.

3. read_records(stream, size) -> Iterator[bytes]
   Return an iterator over fixed-size chunks of a binary stream (e.g. a file or BytesIO),
   stopping when stream.read(size) returns b"". Use the TWO-ARGUMENT form
   iter(callable, sentinel) - no while loop, no generator function.
   size < 1 -> ValueError (eagerly).

Interviewers look for
---------------------
* `for x in it:` == `_it = iter(it)` then `while True: try: x = next(_it) except
  StopIteration: break` + body.
* Keeping the try block tight so you don't swallow StopIteration from the loop body.
* iter(callable, sentinel) for polling/reading loops; next(it, default) to avoid try/except.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import Any, BinaryIO

_MISSING = object()


def for_each(iterable: Iterable[Any], fn: Callable[[Any], Any]) -> int:
    raise NotImplementedError


def first(iterable: Iterable[Any], default: Any = _MISSING) -> Any:
    raise NotImplementedError


def read_records(stream: BinaryIO, size: int) -> Iterator[bytes]:
    raise NotImplementedError
