"""
Exercise 09.6 - Composition over inheritance + shallow vs deep copy   level: ★★★
topics: oop.composition, oop.copy, oop.polymorphism, oop.encapsulation

Problem
-------
The first version of the chat history was `class Conversation(FakeLLMClient)` - every
conversation WAS a client (its own connection pool), and "branching" a conversation with
copy.deepcopy crashed with `TypeError: cannot pickle '_thread.lock' object`. Rewrite it with
composition: a Conversation HAS a client (any object with complete(messages) -> str).

Conversation(client, system_prompt)
* self.client = client (shared, never copied); self.messages = a list of message dicts,
  starting with {"role": "system", "content": system_prompt, "metadata": {}}
* add(role, content, **metadata) - role must be one of ROLES (ValueError otherwise);
  appends {"role": role, "content": content, "metadata": dict(metadata)}
* ask(text) -> str - add the user message, call self.client.complete(self.messages),
  add the reply as an "assistant" message, return the reply (delegation).
* __len__ -> number of messages
* __copy__ -> copy.copy(conv): a new Conversation with the SAME client, a NEW messages list
  that holds the SAME message dicts (appending to one conversation doesn't affect the other,
  but editing a shared message dict does - that's what "shallow" means).
* __deepcopy__(memo) -> copy.deepcopy(conv): a new Conversation with the SAME client and a
  fully independent deep copy of the messages (nested metadata included). Pass `memo` on
  when deep-copying the messages.
* fork() -> a deep copy (a "what if" branch of the conversation).
* Build copies WITHOUT re-running __init__ (that would add a second system message):
  e.g. new = type(self).__new__(type(self)), then set its attributes.

FakeLLMClient (given) holds a threading.Lock, like real SDK clients hold connection pools.

Interviewers look for
---------------------
* Composition: Conversation delegates to a client instead of being one -> share one client
  across thousands of conversations, swap providers/mocks freely (duck typing).
* `b = a` (alias) vs copy.copy (new outer container, shared inner objects) vs copy.deepcopy
  (recursively copied) - and which bug each one causes.
* Customising copying with __copy__ / __deepcopy__ (and why some objects - locks, sockets,
  clients - must be shared, not copied).
"""

import threading
from typing import Any

ROLES = frozenset({"system", "user", "assistant", "tool"})


class FakeLLMClient:
    """Expensive, shared resource. Holds a lock, so it can't be deep-copied or pickled."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0

    def complete(self, messages: list[dict[str, Any]]) -> str:
        with self._lock:
            self.requests += 1
        return f"echo: {messages[-1]['content']}"


class Conversation:
    def __init__(self, client: Any, system_prompt: str) -> None:
        raise NotImplementedError

    def add(self, role: str, content: str, **metadata: Any) -> None:
        raise NotImplementedError

    def ask(self, text: str) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __copy__(self) -> "Conversation":
        raise NotImplementedError

    def __deepcopy__(self, memo: dict[int, Any]) -> "Conversation":
        raise NotImplementedError

    def fork(self) -> "Conversation":
        raise NotImplementedError
