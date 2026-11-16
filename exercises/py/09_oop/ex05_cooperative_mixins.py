"""
Exercise 09.5 - Cooperative mixins, MRO and super() in a diamond    level: ★★★
topics: oop.mro, oop.super, oop.inheritance, oop.overriding

Problem
-------
An LLM client is assembled from mixins that all inherit from `Component` (a diamond):

    RedactMixin, LoggingMixin, CacheMixin  -- each one subclasses Component
    SafeClient(RedactMixin, LoggingMixin, CacheMixin)                      <- given

Each layer's process(text) does its bit and hands over with super().process(...). Every
layer appends a marker to self.trace so the order is observable.

* Component
  - __init__(**kwargs): if any keyword arguments are left over -> TypeError whose message
    contains the sorted leftover names (e.g. "unexpected config: colour"); sets
    self.trace = [].
  - process(text): appends "core", returns f"<{text}>" (stands in for the model call).
* RedactMixin(Component)
  - __init__(*, redact=(), **kwargs): stores tuple(redact) as self.redact
  - process(text): appends "redact", replaces every term in self.redact with "[REDACTED]",
    passes the result on.
* LoggingMixin(Component)
  - __init__(*, log_prefix="llm", **kwargs): stores self.log_prefix, self.logs = []
  - process(text): appends "log", passes text on unchanged, then appends
    f"{log_prefix}: {text} -> {result}" to self.logs (text as THIS layer received it).
* CacheMixin(Component)
  - __init__(**kwargs): self.cache = {}
  - process(text): on a hit append "cache-hit" and return the cached result WITHOUT calling
    further down; on a miss append "cache-miss", call on, store, return.

Rules: each __init__ takes the keyword arguments it understands and passes the REST on via
super().__init__(**kwargs) - set your attributes AFTER that call (Component.__init__ resets
trace). Never call a specific parent class directly (no `Component.__init__(self)`):
mixins must keep working when someone inserts another class into the MRO.

Interviewers look for
---------------------
* super() means "next class in the MRO of type(self)", not "my parent". That is what
  makes the diamond call Component.__init__ exactly once.
* C3 linearization: SafeClient, RedactMixin, LoggingMixin, CacheMixin, Component, object.
* Cooperative signatures: keyword-only args + **kwargs, and a root that rejects leftovers.
* Base order is behaviour: with Logging before Redact, raw PII reaches the logs.
* When composition (a list of middleware objects) would be simpler than mixins.
"""

from typing import Any


class Component:
    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError

    def process(self, text: str) -> str:
        raise NotImplementedError


class RedactMixin(Component):
    def __init__(self, *, redact: tuple[str, ...] = (), **kwargs: Any) -> None:
        raise NotImplementedError

    def process(self, text: str) -> str:
        raise NotImplementedError


class LoggingMixin(Component):
    def __init__(self, *, log_prefix: str = "llm", **kwargs: Any) -> None:
        raise NotImplementedError

    def process(self, text: str) -> str:
        raise NotImplementedError


class CacheMixin(Component):
    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError

    def process(self, text: str) -> str:
        raise NotImplementedError


class SafeClient(RedactMixin, LoggingMixin, CacheMixin):
    """Redact first so that neither the logs nor the cache ever see raw PII."""
