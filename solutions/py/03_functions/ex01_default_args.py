'''
Exercise 03.1 - Fix the default-argument and return-value bugs      level: ★☆☆
topics: functions.default_args, functions.definition, functions.return_values,
        functions.keyword_args

Problem
-------
A chat backend has three helpers. They pass a quick manual test and break in production.

```python
def new_conversation(system_prompt, messages=[]):
    messages.insert(0, {"role": "system", "content": system_prompt})
    return messages

def make_event(name, payload={}, created_at=time.time()):
    payload["name"] = name
    return {"payload": payload, "created_at": created_at}

def parse_model_id(model_id):
    provider, rest = model_id.split("/")
    name, version = rest.split(":")
    print(provider, name, version)
```

Write the fixed versions:

1. new_conversation(system_prompt, messages=None) -> list[dict]
     * returns a NEW list: the system message followed by the given messages (if any)
     * never mutates the caller's list; independent calls never share state

2. make_event(name, payload=None, *, clock=time.time) -> dict
     * returns {"name": name, "payload": <copy of payload or {}>, "created_at": clock()}
     * created_at is taken at CALL time, every call
     * the caller's payload dict is not mutated, and two events never share a payload dict
     * `clock` is keyword-only (inject a fake clock in tests)

3. parse_model_id(model_id) -> tuple[str, str, str | None]
     * "anthropic/claude-x:2024-10" -> ("anthropic", "claude-x", "2024-10")
     * "openai/gpt-4o"              -> ("openai", "gpt-4o", None)
     * split on the FIRST "/" and the FIRST ":" after it - the remainder may contain more
       ("local/llama:8b:q4" -> ("local", "llama", "8b:q4"), "hf/org/model" -> ("hf", "org/model", None))
     * missing "/" or empty provider/name -> ValueError
     * RETURN the values (a function that prints returns None)

Interviewers look for
---------------------
* Default values are evaluated ONCE, when `def` runs, and stored on the function
  (fn.__defaults__). A mutable default is shared by every call; `time.time()` is frozen at
  import time.
* The idiom: `param=None`, then `if param is None: param = []`. (`param or []` also works
  here only because we copy anyway - if you meant to mutate the caller's empty list, `or`
  would silently swap it for a new one.)
* Mutating an argument the caller owns is a side effect - copy or build new.
* Return a tuple (or a NamedTuple/dataclass) for multiple values; the caller unpacks it.
'''

import time
from collections.abc import Callable
from typing import Any


def new_conversation(
    system_prompt: str, messages: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, *(messages or [])]


def make_event(
    name: str,
    payload: dict[str, Any] | None = None,
    *,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    return {"name": name, "payload": dict(payload or {}), "created_at": clock()}


def parse_model_id(model_id: str) -> tuple[str, str, str | None]:
    provider, sep, rest = model_id.partition("/")
    name, colon, version = rest.partition(":")
    if not sep or not provider or not name:
        raise ValueError(f"expected 'provider/name[:version]', got {model_id!r}")
    return provider, name, version if colon else None
