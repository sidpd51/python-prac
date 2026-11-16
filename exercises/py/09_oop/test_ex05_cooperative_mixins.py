import pytest

from study.loader import load

ex = load(__file__, "ex05_cooperative_mixins")

EMAIL = "jane@corp.com"


@pytest.mark.topic("py.oop.mro")
def test_mro_and_process_order():
    client = ex.SafeClient(redact=[EMAIL])
    assert [c.__name__ for c in ex.SafeClient.__mro__] == [
        "SafeClient",
        "RedactMixin",
        "LoggingMixin",
        "CacheMixin",
        "Component",
        "object",
    ]
    assert client.process(f"mail {EMAIL} now") == "<mail [REDACTED] now>"
    assert client.trace == ["redact", "log", "cache-miss", "core"]


@pytest.mark.topic("py.oop.super", "py.oop.inheritance")
def test_init_kwargs_routed_through_the_chain():
    client = ex.SafeClient(redact=("x",), log_prefix="svc")
    assert client.redact == ("x",)
    assert client.log_prefix == "svc"
    assert client.logs == []
    assert client.cache == {}
    assert client.trace == []
    with pytest.raises(TypeError, match="colour"):
        ex.SafeClient(colour="red")


@pytest.mark.topic("py.oop.overriding")
def test_cache_hit_short_circuits():
    client = ex.SafeClient()
    client.process("hello")
    client.trace.clear()
    assert client.process("hello") == "<hello>"
    assert client.trace == ["redact", "log", "cache-hit"]
    assert client.logs == ["llm: hello -> <hello>", "llm: hello -> <hello>"]


@pytest.mark.topic("py.oop.mro", "py.oop.overriding")
def test_base_order_is_behaviour():
    class LeakyClient(ex.LoggingMixin, ex.RedactMixin, ex.CacheMixin):
        pass

    safe = ex.SafeClient(redact=[EMAIL])
    leaky = LeakyClient(redact=[EMAIL])
    safe.process(f"to {EMAIL}")
    leaky.process(f"to {EMAIL}")
    assert all(EMAIL not in line for line in safe.logs)
    assert any(EMAIL in line for line in leaky.logs), "logging before redaction leaks PII"
    assert leaky.trace == ["log", "redact", "cache-miss", "core"]


@pytest.mark.topic("py.oop.super", "py.oop.mro")
def test_cooperative_with_foreign_class_in_mro():
    init_calls = []

    class Counting(ex.Component):
        def __init__(self, **kwargs):
            init_calls.append(type(self).__name__)
            super().__init__(**kwargs)

    class Upper(ex.Component):
        def process(self, text):
            self.trace.append("upper")
            return super().process(text.upper())

    class Custom(ex.RedactMixin, Upper, ex.LoggingMixin, ex.CacheMixin, Counting):
        pass

    c = Custom(redact=["secret"], log_prefix="p")
    assert init_calls == ["Custom"], "every __init__ must use super(), so Counting runs once"
    assert c.process("a secret") == "<A [REDACTED]>"
    assert c.trace == ["redact", "upper", "log", "cache-miss", "core"]
    assert c.logs == ["p: A [REDACTED] -> <A [REDACTED]>"]


@pytest.mark.topic("py.oop.inheritance")
def test_single_mixins_work_alone():
    class JustCache(ex.CacheMixin):
        pass

    j = JustCache()
    assert j.process("q") == "<q>"
    assert j.process("q") == "<q>"
    assert j.trace == ["cache-miss", "core", "cache-hit"]
    assert isinstance(j, ex.Component)
    assert not isinstance(j, ex.LoggingMixin)
