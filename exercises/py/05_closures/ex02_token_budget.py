"""
Exercise 05.2 - Token-usage tracker: several closures, one state  level: ★★☆
topics: closures.closures, closures.nonlocal, closures.free_variables

Problem
-------
An agent loop calls an LLM many times; you want a hard token budget per user session.

make_usage_tracker(budget: int) -> UsageTracker (a NamedTuple of three functions, given below)

    track(prompt_tokens: int, completion_tokens: int) -> int
        Record one LLM call and return the tokens REMAINING after it.
        * negative token counts -> ValueError
        * if the call would push usage above `budget`, raise BudgetExceeded (given below)
          and DO NOT record anything (state must be unchanged after a rejected call).
          Spending exactly the budget is allowed.
    remaining() -> int
        Tokens left.
    report() -> dict
        {"calls": int, "prompt_tokens": int, "completion_tokens": int, "remaining": int}
        (counts only successful calls; return a fresh dict each time)

All three functions are closures over the SAME state (no class instance, no globals, no
mutable-container trick). budget < 0 -> ValueError raised by the factory.

Interviewers look for
---------------------
* Sibling closures created in the same factory call share cells: a `nonlocal` rebind in one
  is seen by the others.
* Validate BEFORE mutating, so a failed call can't leave half-updated state.
* Returning internal state? Return a copy (fresh dict), never the live object.
"""

from collections.abc import Callable
from typing import NamedTuple


class BudgetExceeded(Exception):
    """Raised when a call would exceed the token budget."""


class UsageTracker(NamedTuple):
    track: Callable[[int, int], int]
    remaining: Callable[[], int]
    report: Callable[[], dict]


def make_usage_tracker(budget: int) -> UsageTracker:
    raise NotImplementedError
