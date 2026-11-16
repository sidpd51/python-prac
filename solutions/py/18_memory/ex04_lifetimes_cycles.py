"""
Exercise 18.4 - Refcounting, the cycle collector and a leaky chat history   level: ★★☆
topics: memory.refcounting, memory.gc, memory.cycles, memory.weakref, memory.stack_heap

Problem
-------
CPython frees most objects the instant their reference count hits zero; only objects
stuck in reference CYCLES need the cyclic garbage collector (module `gc`).

1. added_references(obj, action) -> int
   Return how many references to `obj` were created (positive) or dropped (negative)
   by calling action() - the change in sys.getrefcount(obj) across the call.
       cache = []
       added_references(obj, lambda: cache.extend([obj, obj]))  -> 2
   (Absolute getrefcount values vary between Python versions - only the DELTA is
   meaningful.)

2. classify_lifetime(factory) -> "refcount" | "gc" | "leak"
   Call factory() to build an object, keep only a weakref.ref to it, drop the strong
   reference, and report how it dies:
     * "refcount" - already dead right after the last strong reference was dropped
     * "gc"       - still alive, but dead after gc.collect()   (it was in a cycle)
     * "leak"     - still alive even after gc.collect()  (something else holds it)
   The automatic collector must NOT run in the middle of your measurement (it could
   turn a "gc" into a "refcount"): disable it for the duration and restore the previous
   gc.isenabled() state afterwards, even on error. If the object doesn't support weak
   references, let the TypeError propagate.

3. Conversation / Message - fix the cycle.
   A conversation holds its messages; each message needs to reach its conversation.
   The naive version (self.conversation = conversation) creates a cycle for every
   conversation, so dropped conversations pile up until the GC runs.
     * Conversation(title); .add(role, text) -> creates, stores and returns a Message;
       .messages -> list of Message in insertion order
     * Message(conversation, role, text); attributes role, text; property
       `conversation` returns the Conversation while it is alive, else None.
       Hold it via weakref.ref so a Conversation (with messages) is freed by
       refcounting alone:
           classify_lifetime(lambda: build_conversation_with_messages()) == "refcount"

Interviewers look for
---------------------
* Refcounting is deterministic and immediate; the generational cycle GC (thresholds:
  gc.get_threshold()) is periodic. `del x` deletes a NAME, not the object.
* Cycles are not leaks in Python - they're freed later - but "later" can mean big
  memory spikes, delayed __del__/close, and GC pauses. Break parent<->child cycles with
  weakrefs (parent owns children strongly; child points back weakly).
* Names live in frames (conceptually "the stack"); all objects live on the heap.
"""

import gc
import sys
import weakref
from collections.abc import Callable
from typing import Any, Literal


def added_references(obj: Any, action: Callable[[], Any]) -> int:
    before = sys.getrefcount(obj)
    action()
    return sys.getrefcount(obj) - before


def classify_lifetime(factory: Callable[[], Any]) -> Literal["refcount", "gc", "leak"]:
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        obj = factory()
        ref = weakref.ref(obj)
        del obj  # drop our only strong reference (the name, not the object)
        if ref() is None:
            return "refcount"
        gc.collect()
        return "gc" if ref() is None else "leak"
    finally:
        if was_enabled:
            gc.enable()


class Conversation:
    def __init__(self, title: str) -> None:
        self.title = title
        self.messages: list[Message] = []

    def add(self, role: str, text: str) -> "Message":
        msg = Message(self, role, text)
        self.messages.append(msg)
        return msg


class Message:
    def __init__(self, conversation: Conversation, role: str, text: str) -> None:
        self._conversation = weakref.ref(conversation)
        self.role = role
        self.text = text

    @property
    def conversation(self) -> Conversation | None:
        return self._conversation()
