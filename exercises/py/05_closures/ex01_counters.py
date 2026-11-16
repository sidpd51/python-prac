"""
Exercise 05.1 - Counters and running averages with nonlocal       level: ★☆☆
topics: closures.nested_functions, closures.closures, closures.nonlocal

Problem
-------
Write two closure factories. No classes, no globals, no `function.attribute` state and
no "mutable container" trick (`count = [0]`) - keep the state in plain variables of the
enclosing function and rebind them with `nonlocal`.

1. make_counter(start=0, step=1) -> Callable[[], int]
   Each call returns the next value, starting AT `start`:
       next_id = make_counter(100, 10)
       next_id(), next_id(), next_id()   -> 100, 110, 120
   Two counters made by two calls are completely independent.
   step == 0 -> ValueError raised by make_counter itself (eagerly).

2. make_averager() -> Callable[[float], float]
   Running mean of every value seen so far, in O(1) memory (don't store the values):
       avg = make_averager()
       avg(10) -> 10.0 ; avg(20) -> 15.0 ; avg(30) -> 20.0
   Keep the state in two variables called `total` and `count`.

Interviewers look for
---------------------
* A closure = function + the cells of the enclosing scope it references; each call to the
  factory creates NEW cells, which is why counters are independent.
* Without `nonlocal`, `count += 1` makes `count` local to the inner function and raises
  UnboundLocalError on the first call.
* When a closure beats a class (one behaviour, tiny state) and when it doesn't.
"""

from collections.abc import Callable


def make_counter(start: int = 0, step: int = 1) -> Callable[[], int]:
    raise NotImplementedError


def make_averager() -> Callable[[float], float]:
    raise NotImplementedError
