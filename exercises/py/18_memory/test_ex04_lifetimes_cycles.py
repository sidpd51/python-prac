import gc

import pytest

from study.loader import load

ex = load(__file__, "ex04_lifetimes_cycles")

REGISTRY: list = []  # module-level holder -> objects stored here "leak"


class Doc:
    def __init__(self, text: str = "") -> None:
        self.text = text


@pytest.mark.topic("py.memory.refcounting")
def test_added_references():
    obj = Doc()
    cache: list = []
    index: dict = {}
    assert ex.added_references(obj, lambda: cache.extend([obj, obj])) == 2
    assert ex.added_references(obj, lambda: index.update(a=obj)) == 1
    assert ex.added_references(obj, cache.clear) == -2
    assert ex.added_references(obj, lambda: [obj] * 3) == 0  # temporary list is freed


@pytest.mark.topic("py.memory.refcounting", "py.memory.stack_heap")
def test_refcount_lifetime():
    assert ex.classify_lifetime(lambda: Doc("hello")) == "refcount"


@pytest.mark.topic("py.memory.cycles", "py.memory.gc")
def test_cycle_needs_gc():
    def make_cycle():
        a, b = Doc("a"), Doc("b")
        a.other, b.other = b, a
        return a

    def self_cycle():
        d = Doc()
        d.me = d
        return d

    assert ex.classify_lifetime(make_cycle) == "gc"
    assert ex.classify_lifetime(self_cycle) == "gc"


@pytest.mark.topic("py.memory.gc")
def test_leak_detected():
    def registered():
        d = Doc("kept")
        REGISTRY.append(d)
        return d

    try:
        assert ex.classify_lifetime(registered) == "leak"
    finally:
        REGISTRY.clear()


@pytest.mark.topic("py.memory.gc")
def test_gc_state_restored():
    assert gc.isenabled()
    ex.classify_lifetime(Doc)
    assert gc.isenabled(), "automatic GC must be re-enabled"

    gc.disable()
    try:
        ex.classify_lifetime(Doc)
        assert not gc.isenabled(), "must restore the PREVIOUS state, not blindly enable"
    finally:
        gc.enable()

    with pytest.raises(TypeError):
        ex.classify_lifetime(lambda: 42)  # ints can't be weakly referenced
    assert gc.isenabled()


@pytest.mark.topic("py.memory.weakref", "py.memory.cycles")
def test_conversation_back_reference():
    convo = ex.Conversation("support chat")
    m1 = convo.add("user", "hi")
    m2 = convo.add("assistant", "hello!")
    assert convo.messages == [m1, m2]
    assert (m2.role, m2.text) == ("assistant", "hello!")
    assert m1.conversation is convo
    del convo
    assert m1.conversation is None, "message must not keep its conversation alive"


@pytest.mark.topic("py.memory.weakref", "py.memory.cycles", "py.memory.refcounting")
def test_conversation_freed_by_refcount_alone():
    def build():
        c = ex.Conversation("t")
        for i in range(5):
            c.add("user", f"msg {i}")
        return c

    assert ex.classify_lifetime(build) == "refcount"
