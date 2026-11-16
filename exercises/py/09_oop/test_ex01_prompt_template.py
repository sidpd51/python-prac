import pytest

from study.loader import load

ex = load(__file__, "ex01_prompt_template")

QA = "Context: {context}\nQuestion: {question}"


@pytest.mark.topic("py.oop.classes_objects", "py.oop.init", "py.oop.instance_vars")
def test_init_sets_instance_variables():
    t = ex.PromptTemplate("qa", QA, tags=("rag", "v2"))
    assert isinstance(t, ex.PromptTemplate)
    assert (t.name, t.template, t.tags) == ("qa", QA, ["rag", "v2"])
    assert ex.PromptTemplate("x", "hi").tags == []
    with pytest.raises(ValueError):
        ex.PromptTemplate("", QA)


@pytest.mark.topic("py.oop.instance_vars")
def test_tags_are_not_shared():
    tags = ["rag"]
    a = ex.PromptTemplate("a", QA, tags=tags)
    b = ex.PromptTemplate("b", QA)
    c = ex.PromptTemplate("c", QA)
    a.tags.append("x")
    b.tags.append("y")
    assert tags == ["rag"], "copy the argument"
    assert c.tags == []
    assert a.tags is not b.tags


@pytest.mark.topic("py.oop.class_vars")
def test_created_counter_is_class_level():
    class ChatTemplate(ex.PromptTemplate):
        pass

    before = ex.PromptTemplate.created
    t = ex.PromptTemplate("a", QA)
    ChatTemplate("b", QA)
    assert ex.PromptTemplate.created == before + 2
    assert "created" not in vars(t), "self.created += 1 creates an instance attribute"
    assert "created" not in vars(ChatTemplate), "subclasses must update the base counter"


@pytest.mark.topic("py.oop.static_methods")
def test_variables_static_method():
    assert ex.PromptTemplate.variables(QA) == {"context", "question"}
    assert ex.PromptTemplate.variables("no vars, {{literal}}") == set()
    t = ex.PromptTemplate("qa", QA)
    assert t.variables("{a}{b}{a}") == {"a", "b"}


@pytest.mark.topic("py.oop.instance_methods")
def test_render():
    t = ex.PromptTemplate("qa", QA)
    assert t.render(context="Paris is in France.", question="Where is Paris?", extra=1) == (
        "Context: Paris is in France.\nQuestion: Where is Paris?"
    )
    with pytest.raises(ValueError, match="missing variables: context, question"):
        t.render()
    with pytest.raises(ValueError, match="missing variables: question$"):
        t.render(context="c")


@pytest.mark.topic("py.oop.class_methods")
def test_from_dict_respects_subclass():
    class ChatTemplate(ex.PromptTemplate):
        pass

    t = ex.PromptTemplate.from_dict({"name": "qa", "template": QA, "tags": ["a"]})
    assert type(t) is ex.PromptTemplate
    assert t.tags == ["a"]
    c = ChatTemplate.from_dict({"name": "chat", "template": "{msg}"})
    assert type(c) is ChatTemplate
    assert c.tags == []


@pytest.mark.topic("py.oop.dunder")
def test_contains_and_repr():
    t = ex.PromptTemplate("qa", QA)
    assert "question" in t
    assert "answer" not in t
    assert repr(t) == "PromptTemplate(name='qa', variables=['context', 'question'])"

    class ChatTemplate(ex.PromptTemplate):
        pass

    assert repr(ChatTemplate("c", "{b}{a}")) == "ChatTemplate(name='c', variables=['a', 'b'])"
