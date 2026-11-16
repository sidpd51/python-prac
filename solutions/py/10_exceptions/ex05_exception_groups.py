"""
Exercise 10.5 - Batch failures with ExceptionGroup / except*      level: ★★★
topics: exceptions.multiple, exceptions.hierarchy, exceptions.contextual,
        exceptions.best_practices

Problem
-------
Embedding 10 document batches concurrently can fail in several *different* ways at once.
Python 3.11+ models that with ExceptionGroup and `except*`.
TransientError is given.

1. run_all(tasks: Mapping[str, Callable[[], T]]) -> dict[str, T]
     * run EVERY task (in order) even after some fail; return {name: result} if all succeed
     * if any failed: raise ExceptionGroup(f"{n_failed} of {n_total} tasks failed", errors)
       with errors in task order; before grouping, add the note f"task={name}" to each
     * only Exception failures are grouped - a BaseException (KeyboardInterrupt) stops
       immediately and propagates as-is (ExceptionGroup can't even hold one)

2. task_name(exc: BaseException) -> str | None
     the name from a "task=<name>" note, or None

3. flatten(exc: BaseException) -> list[BaseException]
     leaf exceptions of (possibly nested) groups, depth-first, in order; a non-group
     exception -> [exc]

4. triage(fn: Callable[[], object]) -> Triage        (Triage dataclass given)
     * call fn(); use `except*` to sort what it raised:
         TransientError -> task names into Triage.retry
         ValueError     -> task names into Triage.invalid
       (task names via task_name(); in order of appearance)
     * anything else must propagate - and Python does that for you: the unmatched
       remainder is re-raised as an ExceptionGroup
     * a *naked* TransientError / ValueError (not in a group) is handled the same way
     * no error -> empty Triage
     * note: `return` is not allowed inside an except* block - why?

Interviewers look for
---------------------
* except* runs EVERY matching clause (unlike except, which picks one), each with a
  sub-group of the matching leaves; unmatched leaves are re-raised automatically.
* asyncio.TaskGroup raises ExceptionGroup; code catching `except ValueError` around it
  silently stops working - you need except*.
* Attaching context (notes) so a group of 7 failures is debuggable.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")

__stub_keep__ = ["TransientError"]


class TransientError(Exception):
    """Retryable failure (rate limit / 503)."""


@dataclass
class Triage:
    retry: list[str | None] = field(default_factory=list)
    invalid: list[str | None] = field(default_factory=list)


def run_all(tasks: Mapping[str, Callable[[], T]]) -> dict[str, T]:
    results: dict[str, T] = {}
    errors: list[Exception] = []
    for name, task in tasks.items():
        try:
            results[name] = task()
        except Exception as e:
            e.add_note(f"task={name}")
            errors.append(e)
    if errors:
        raise ExceptionGroup(f"{len(errors)} of {len(tasks)} tasks failed", errors)
    return results


def task_name(exc: BaseException) -> str | None:
    notes: list[str] = getattr(exc, "__notes__", [])
    for note in notes:
        if note.startswith("task="):
            return note.removeprefix("task=")
    return None


def flatten(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in flatten(sub)]
    return [exc]


def triage(fn: Callable[[], object]) -> Triage:
    result = Triage()
    try:
        fn()
    except* TransientError as eg:
        result.retry.extend(task_name(e) for e in flatten(eg))
    except* ValueError as eg:
        result.invalid.extend(task_name(e) for e in flatten(eg))
    return result
