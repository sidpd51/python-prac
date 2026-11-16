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

import math
from collections.abc import Iterable, Iterator
from typing import Any


class Embedding:
    __slots__ = ("_values", "_model")

    def __init__(self, values: Iterable[float], *, model: str = "") -> None:
        vals = tuple(float(v) for v in values)
        if not vals:
            raise ValueError("an embedding needs at least one value")
        self._values = vals
        self._model = model

    @property
    def values(self) -> tuple[float, ...]:
        return self._values

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        v = self._values
        if len(v) <= 4:
            return f"Embedding({list(v)!r}, model={self._model!r})"
        return f"Embedding([{v[0]!r}, {v[1]!r}, ..., {v[-1]!r}], dim={len(v)}, model={self._model!r})"

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return Embedding(self._values[index], model=self._model)
        return self._values[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Embedding):
            return NotImplemented
        return self._values == other._values and self._model == other._model

    def __hash__(self) -> int:
        return hash((self._values, self._model))

    def _check_compatible(self, other: "Embedding") -> None:
        if len(self) != len(other):
            raise ValueError(f"dimension mismatch: {len(self)} vs {len(other)}")
        if self._model != other._model:
            raise ValueError(f"model mismatch: {self._model!r} vs {other._model!r}")

    def __add__(self, other: object) -> "Embedding":
        if not isinstance(other, Embedding):
            return NotImplemented
        self._check_compatible(other)
        return Embedding((a + b for a, b in zip(self, other, strict=True)), model=self._model)

    def __sub__(self, other: object) -> "Embedding":
        if not isinstance(other, Embedding):
            return NotImplemented
        self._check_compatible(other)
        return Embedding((a - b for a, b in zip(self, other, strict=True)), model=self._model)

    def __mul__(self, scalar: object) -> "Embedding":
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            return NotImplemented
        return Embedding((v * scalar for v in self._values), model=self._model)

    __rmul__ = __mul__

    def __matmul__(self, other: object) -> float:
        if not isinstance(other, Embedding):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("dimension mismatch")
        return math.fsum(a * b for a, b in zip(self, other, strict=True))

    def __abs__(self) -> float:
        return math.sqrt(self @ self)

    def __bool__(self) -> bool:
        return any(self._values)

    def cosine(self, other: "Embedding") -> float:
        denom = abs(self) * abs(other)
        if denom == 0:
            raise ValueError("cosine similarity is undefined for a zero vector")
        return (self @ other) / denom
