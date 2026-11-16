"""
Exercise 09.3 - An immutable Embedding vector with dunder methods   level: ★★☆
topics: oop.dunder, oop.slots, oop.encapsulation, oop.property

Problem
-------
Write `Embedding`, an immutable, hashable vector type that behaves like a built-in:

* __init__(values, *, model="") - store the values as a tuple of floats; empty -> ValueError.
  Use __slots__ = ("_values", "_model") (millions of these may live in memory), and expose
  read-only properties `values` (the tuple) and `model`. Assigning to e.values / e.model or
  to any new attribute must raise AttributeError.
* __repr__:
      dim <= 4: "Embedding([0.1, 0.2], model='m')"
      dim  > 4: "Embedding([0.1, 0.2, ..., 0.9], dim=10, model='m')"  (first two, last one)
* __len__ (dimension), __iter__ (over the floats), __getitem__:
  an int index -> float; a slice -> a new Embedding with the same model.
* __eq__: equal if same values AND same model. Comparing with a non-Embedding returns
  NotImplemented (so `e == [0.1, 0.2]` is simply False). __hash__ consistent with __eq__.
* __add__ / __sub__: element-wise. Different dimension or different model -> ValueError.
  Non-Embedding operand -> NotImplemented (Python then raises TypeError).
* __mul__ / __rmul__ by a number (int/float, not bool): 2 * e and e * 2. Otherwise NotImplemented.
* __matmul__: e1 @ e2 -> dot product (float); dims must match (ValueError).
* __abs__ -> L2 norm; __bool__ -> False only for the all-zeros vector.
* cosine(other) -> float in [-1, 1]; a zero vector on either side -> ValueError.

Interviewers look for
---------------------
* Returning NotImplemented (not raising) lets Python try the reflected operation
  (__rmul__ etc.) and produce the standard TypeError.
* __eq__ without __hash__ makes the class unhashable; equal objects MUST hash equal;
  hashable objects should be immutable.
* __slots__: no per-instance __dict__ -> less memory, faster attribute access, no stray
  attributes - at the price of flexibility (no dynamic attrs, care with inheritance).
* Real systems use NumPy for the math - this is about the data-model protocol.
"""

from collections.abc import Iterable, Iterator
from typing import Any


class Embedding:
    __slots__ = ("_values", "_model")

    def __init__(self, values: Iterable[float], *, model: str = "") -> None:
        raise NotImplementedError

    @property
    def values(self) -> tuple[float, ...]:
        raise NotImplementedError

    @property
    def model(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[float]:
        raise NotImplementedError

    def __getitem__(self, index: int | slice) -> Any:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    def __add__(self, other: object) -> "Embedding":
        raise NotImplementedError

    def __sub__(self, other: object) -> "Embedding":
        raise NotImplementedError

    def __mul__(self, scalar: object) -> "Embedding":
        raise NotImplementedError

    __rmul__ = __mul__

    def __matmul__(self, other: object) -> float:
        raise NotImplementedError

    def __abs__(self) -> float:
        raise NotImplementedError

    def __bool__(self) -> bool:
        raise NotImplementedError

    def cosine(self, other: "Embedding") -> float:
        raise NotImplementedError
