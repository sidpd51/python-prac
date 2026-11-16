import copy

import pytest

from study.loader import load

ex = load(__file__, "ex06_conversation_copy")


def make_conv(client=None):
    conv = ex.Conversation(client or ex.FakeLLMClient(), "You are a helpful assistant.")
    conv.ask("What is RAG?")
    return conv


@pytest.mark.topic("py.oop.composition")
def test_has_a_client_and_delegates():
    client = ex.FakeLLMClient()
    conv = ex.Conversation(client, "sys")
    assert not issubclass(ex.Conversation, ex.FakeLLMClient), "use composition"
    assert conv.client is client
    assert conv.messages == [{"role": "system", "content": "sys", "metadata": {}}]
    assert conv.ask("hi") == "echo: hi"
    assert [m["role"] for m in conv.messages] == ["system", "user", "assistant"]
    assert len(conv) == 3
    assert client.requests == 1


@pytest.mark.topic("py.oop.composition", "py.oop.polymorphism")
def test_any_client_works_and_clients_are_shared():
    class ScriptedClient:  # duck typing: no common base class
        def __init__(self):
            self.seen = []

        def complete(self, messages):
            self.seen.append(len(messages))
            return "42"

    scripted = ScriptedClient()
    assert ex.Conversation(scripted, "s").ask("q") == "42"
    shared = ex.FakeLLMClient()
    a, b = ex.Conversation(shared, "a"), ex.Conversation(shared, "b")
    a.ask("1")
    b.ask("2")
    a.ask("3")
    assert shared.requests == 3
    assert scripted.seen == [2]


@pytest.mark.topic("py.oop.encapsulation")
def test_add_validates_role_and_copies_metadata():
    conv = ex.Conversation(ex.FakeLLMClient(), "s")
    meta = {"source": "doc-7"}
    conv.add("tool", "result", **meta)
    assert conv.messages[-1] == {"role": "tool", "content": "result", "metadata": {"source": "doc-7"}}
    with pytest.raises(ValueError):
        conv.add("admin", "rm -rf")


@pytest.mark.topic("py.oop.copy")
def test_shallow_copy_semantics():
    conv = make_conv()
    shallow = copy.copy(conv)
    assert isinstance(shallow, ex.Conversation)
    assert shallow.client is conv.client
    assert shallow.messages is not conv.messages, "new outer list"
    assert all(a is b for a, b in zip(shallow.messages, conv.messages, strict=True)), "same inner dicts"
    shallow.add("user", "only in the copy")
    assert len(conv) == 3 and len(shallow) == 4
    shallow.messages[0]["content"] = "EDITED"
    assert conv.messages[0]["content"] == "EDITED", "inner dicts are shared in a shallow copy"


@pytest.mark.topic("py.oop.copy")
def test_deepcopy_shares_client_copies_messages():
    conv = make_conv()
    conv.messages[1]["metadata"]["tags"] = ["a"]
    deep = copy.deepcopy(conv)  # must not try to copy the client's lock
    assert deep.client is conv.client
    assert deep.messages == conv.messages
    deep.messages[0]["content"] = "EDITED"
    deep.messages[1]["metadata"]["tags"].append("b")
    assert conv.messages[0]["content"] == "You are a helpful assistant."
    assert conv.messages[1]["metadata"]["tags"] == ["a"]
    with pytest.raises(TypeError):
        copy.deepcopy(ex.FakeLLMClient())  # why the client must be shared


@pytest.mark.topic("py.oop.copy")
def test_deepcopy_respects_memo():
    conv = make_conv()
    pair = copy.deepcopy([conv, conv])
    assert pair[0] is pair[1], "the same object deep-copied twice in one call -> one copy"
    assert pair[0] is not conv


@pytest.mark.topic("py.oop.copy", "py.oop.composition")
def test_fork_branches_independently():
    conv = make_conv()
    branch = conv.fork()
    assert branch is not conv
    branch.ask("Explain like I'm five")
    conv.ask("Give me the formal definition")
    assert len(conv) == len(branch) == 5
    assert conv.messages[3]["content"] == "Give me the formal definition"
    assert branch.messages[3]["content"] == "Explain like I'm five"
    assert [m["role"] for m in branch.messages].count("system") == 1
    assert branch.client.requests == 3  # one shared client saw every request
