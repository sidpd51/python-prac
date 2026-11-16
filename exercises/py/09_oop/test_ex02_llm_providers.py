import pytest

from study.loader import load

ex = load(__file__, "ex02_llm_providers")


@pytest.mark.topic("py.oop.abc", "py.oop.abstraction")
def test_base_is_abstract():
    echo = ex.EchoLLM("echo-1")  # concrete subclasses work...
    assert isinstance(echo, ex.BaseLLM)
    with pytest.raises(TypeError):
        ex.BaseLLM("m")  # ...the abstraction itself can't be instantiated

    class HalfDone(ex.BaseLLM):
        def complete(self, prompt):
            return prompt

    with pytest.raises(TypeError):
        HalfDone("m")  # count_tokens missing


@pytest.mark.topic("py.oop.inheritance", "py.oop.abstraction")
def test_generate_template_method():
    c = ex.EchoLLM("echo-1").generate("  what is RAG  ")
    assert c == ex.Completion("what is RAG", "echo-1", 3, 3)
    s = ex.ShoutLLM("loud").generate("hello there")
    assert s == ex.Completion("HELLO THERE!", "loud", 3, 3)
    with pytest.raises(ValueError):
        ex.EchoLLM("e").generate("   ")


@pytest.mark.topic("py.oop.overriding", "py.oop.super")
def test_shout_overrides_init_and_complete():
    s = ex.ShoutLLM("loud", 0.5, suffix="!!!")
    assert s.model == "loud"
    assert s.temperature == 0.5
    assert s.complete("hi") == "HI!!!"
    assert s.count_tokens("abcdefghi") == 3
    with pytest.raises(ValueError):
        ex.ShoutLLM("")  # base validation still runs


@pytest.mark.topic("py.oop.property", "py.oop.encapsulation")
def test_temperature_property_validates():
    llm = ex.EchoLLM("e", temperature=1)
    assert llm.temperature == 1.0
    assert isinstance(vars(ex.BaseLLM)["temperature"], property)
    llm.temperature = 2.0
    assert llm.temperature == 2.0
    for bad, err in [(2.5, ValueError), (-0.1, ValueError), ("0.5", TypeError), (True, TypeError)]:
        with pytest.raises(err):
            llm.temperature = bad
    assert llm.temperature == 2.0, "a rejected value must not be stored"
    with pytest.raises(ValueError):
        ex.EchoLLM("e", temperature=3)  # __init__ goes through the setter


@pytest.mark.topic("py.oop.dunder")
def test_repr_uses_real_class_name():
    assert repr(ex.EchoLLM("echo-1")) == "EchoLLM(model='echo-1', temperature=0.0)"
    assert repr(ex.ShoutLLM("loud", 1.5)) == "ShoutLLM(model='loud', temperature=1.5)"


@pytest.mark.topic("py.oop.class_methods")
def test_from_config_dispatch():
    llm = ex.BaseLLM.from_config({"provider": "shout", "model": "loud", "suffix": "?"})
    assert type(llm) is ex.ShoutLLM
    assert llm.complete("ok") == "OK?"
    e = ex.EchoLLM.from_config({"model": "echo-2", "temperature": 0.3})
    assert type(e) is ex.EchoLLM
    assert e.temperature == 0.3
    with pytest.raises(ValueError):
        ex.BaseLLM.from_config({"provider": "nope", "model": "m"})
    with pytest.raises(ValueError):
        ex.BaseLLM.from_config({"model": "m"})


@pytest.mark.topic("py.oop.polymorphism")
def test_run_all_polymorphism_and_duck_typing():
    class FakeVendor:  # not a BaseLLM subclass
        def generate(self, prompt):
            return ex.Completion(prompt[::-1], "fake", 0, 0)

        def __repr__(self):
            return "FakeVendor()"

    out = ex.run_all([ex.EchoLLM("e"), ex.ShoutLLM("s"), FakeVendor()], "abc")
    assert out == {
        "EchoLLM(model='e', temperature=0.0)": "abc",
        "ShoutLLM(model='s', temperature=0.0)": "ABC!",
        "FakeVendor()": "cba",
    }
