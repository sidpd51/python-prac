'''
Exercise 04.4 - Pure functions and immutable conversation state     level: ★★★
topics: functional.pure_functions, functional.side_effects, functional.immutability,
        functional.reduce, functional.concepts

Problem
-------
Part A - refactor this impure function into a pure one:

```python
SEEN = set()

def normalize_scores(results):
    top = max(r["score"] for r in results)
    for r in results:
        r["score"] = r["score"] / top        # mutates the caller's dicts
        SEEN.add(r["id"])                     # hidden global side effect
    return results
```

1. normalize_scores(results) -> tuple[dict, ...]
     * min-max normalise "score" to [0, 1]: (s - lo) / (hi - lo); if all scores are equal
       every score becomes 1.0; empty input -> ()
     * returns NEW dicts (other keys copied) in input order; never touches the inputs;
       no global state; same input -> same output

Part B - conversation state as immutable values + a pure reducer (Redux/event-sourcing
style). `Message` and `ChatState` are given below (both immutable).

2. apply(state, event) -> ChatState                               (pure, never mutates)
     * {"type": "user" | "assistant", "content": str, "tokens": int}
           -> append Message(role, content, tokens); tokens_used += tokens
     * {"type": "truncate", "keep": n}
           -> keep only the last n messages (n == 0 -> none); tokens_used = sum of the kept
              messages' tokens
     * {"type": "reset"} -> ChatState()
     * anything else -> ValueError

3. replay(events, initial=EMPTY) -> ChatState
     * fold all events over the initial state with functools.reduce
     * (EMPTY is one shared ChatState() used as a default - why is that safe here when a
       shared `[]` default would not be?)

4. history(events, initial=EMPTY) -> list[ChatState]
     * every intermediate state, starting with `initial` (len == len(events) + 1);
       because states are immutable, older states are never affected by newer ones
     * hint: itertools.accumulate

Interviewers look for
---------------------
* Pure = output depends only on inputs + no observable side effects (no mutation of args,
  globals, I/O). Pure code is trivially testable, cacheable, parallelisable, replayable.
* "Functional core, imperative shell": keep I/O (LLM calls, DB writes, logging) at the edges
  and the decision logic pure.
* frozen dataclass + tuples: mutation raises FrozenInstanceError; "updates" are
  dataclasses.replace(...) returning new objects that share unchanged parts.
* Immutable defaults are safe; immutable state makes undo/time-travel debugging, caching and
  concurrent reads easy. Cost: allocation/copying - fine at chat scale.
'''

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from typing import Any, NamedTuple


class Message(NamedTuple):
    role: str
    content: str
    tokens: int


@dataclass(frozen=True)
class ChatState:
    messages: tuple[Message, ...] = ()
    tokens_used: int = 0


EMPTY = ChatState()  # one shared instance is fine: it can never change


def normalize_scores(results: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if not results:
        return ()
    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    return tuple({**r, "score": (r["score"] - lo) / span if span else 1.0} for r in results)


def apply(state: ChatState, event: Mapping[str, Any]) -> ChatState:
    match event.get("type"):
        case "user" | "assistant" as role:
            msg = Message(role, event["content"], event["tokens"])
            return replace(state, messages=(*state.messages, msg), tokens_used=state.tokens_used + msg.tokens)
        case "truncate":
            n = event["keep"]
            kept = state.messages[len(state.messages) - n :] if n > 0 else ()
            return ChatState(kept, sum(m.tokens for m in kept))
        case "reset":
            return ChatState()
        case other:
            raise ValueError(f"unknown event type {other!r}")


def replay(events: Iterable[Mapping[str, Any]], initial: ChatState = EMPTY) -> ChatState:
    return reduce(apply, events, initial)


def history(events: Iterable[Mapping[str, Any]], initial: ChatState = EMPTY) -> list[ChatState]:
    return list(itertools.accumulate(events, apply, initial=initial))
