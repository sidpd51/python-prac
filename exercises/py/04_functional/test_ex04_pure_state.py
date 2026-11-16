import copy
import dataclasses

import pytest

from study.loader import load

ex = load(__file__, "ex04_pure_state")

RESULTS = [
    {"id": "a", "score": 12.0, "src": "x"},
    {"id": "b", "score": 4.0},
    {"id": "c", "score": 8.0},
]

EVENTS = [
    {"type": "user", "content": "hi", "tokens": 3},
    {"type": "assistant", "content": "hello!", "tokens": 5},
    {"type": "user", "content": "explain RAG", "tokens": 7},
]


@pytest.mark.topic("py.functional.pure_functions")
def test_normalize_scores():
    out = ex.normalize_scores(RESULTS)
    assert isinstance(out, tuple)
    assert [r["score"] for r in out] == pytest.approx([1.0, 0.0, 0.5])
    assert out[0] == {"id": "a", "score": 1.0, "src": "x"}


@pytest.mark.topic("py.functional.pure_functions", "py.functional.side_effects")
def test_normalize_scores_is_pure():
    snapshot = copy.deepcopy(RESULTS)
    first = ex.normalize_scores(RESULTS)
    second = ex.normalize_scores(RESULTS)
    assert snapshot == RESULTS, "inputs were mutated"
    assert first == second, "same input must give the same output"
    assert all(a is not b for a, b in zip(first, RESULTS, strict=True))
    assert not hasattr(ex, "SEEN")


@pytest.mark.topic("py.functional.pure_functions")
def test_normalize_scores_edges():
    assert ex.normalize_scores([]) == ()
    assert [r["score"] for r in ex.normalize_scores([{"score": 3}, {"score": 3}])] == [1.0, 1.0]
    assert ex.normalize_scores([{"score": -2}]) == ({"score": 1.0},)


@pytest.mark.topic("py.functional.immutability", "py.functional.pure_functions")
def test_apply_messages():
    s0 = ex.ChatState()
    s1 = ex.apply(s0, EVENTS[0])
    s2 = ex.apply(s1, EVENTS[1])
    assert s0 == ex.ChatState() and s0.messages == () and s0.tokens_used == 0
    assert s1.messages == (ex.Message("user", "hi", 3),) and s1.tokens_used == 3
    assert s2.messages == (ex.Message("user", "hi", 3), ex.Message("assistant", "hello!", 5))
    assert s2.tokens_used == 8
    assert s1 is not s2 and len(s1.messages) == 1, "old state changed"


@pytest.mark.topic("py.functional.immutability")
def test_states_cannot_be_mutated():
    s = ex.apply(ex.ChatState(), EVENTS[0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.tokens_used = 0
    assert isinstance(s.messages, tuple)
    with pytest.raises(AttributeError):
        s.messages.append(ex.Message("user", "x", 1))


@pytest.mark.topic("py.functional.immutability", "py.functional.concepts")
@pytest.mark.parametrize(("keep", "contents", "tokens"), [(2, ["hello!", "explain RAG"], 12), (0, [], 0), (10, ["hi", "hello!", "explain RAG"], 15)])
def test_apply_truncate(keep, contents, tokens):
    full = ex.replay(EVENTS)
    out = ex.apply(full, {"type": "truncate", "keep": keep})
    assert [m.content for m in out.messages] == contents
    assert out.tokens_used == tokens
    assert len(full.messages) == 3, "truncate mutated the old state"


@pytest.mark.topic("py.functional.concepts")
def test_apply_reset_and_unknown():
    full = ex.replay(EVENTS)
    assert ex.apply(full, {"type": "reset"}) == ex.ChatState()
    assert full.tokens_used == 15
    with pytest.raises(ValueError):
        ex.apply(full, {"type": "delete_everything"})
    with pytest.raises(ValueError):
        ex.apply(full, {})


@pytest.mark.topic("py.functional.reduce", "py.functional.immutability")
def test_replay():
    state = ex.replay(EVENTS)
    assert state.tokens_used == 15
    assert [m.role for m in state.messages] == ["user", "assistant", "user"]
    assert ex.replay([]) == ex.ChatState()
    assert ex.replay([]) == ex.ChatState(), "default initial state was corrupted"
    start = ex.replay(EVENTS[:1])
    assert ex.replay(EVENTS[1:], start) == state
    assert start.tokens_used == 3


@pytest.mark.topic("py.functional.reduce", "py.functional.immutability")
def test_history_time_travel():
    states = ex.history(EVENTS + [{"type": "truncate", "keep": 1}])
    assert len(states) == 5
    assert states[0] == ex.ChatState()
    assert [s.tokens_used for s in states] == [0, 3, 8, 15, 7]
    assert [len(s.messages) for s in states] == [0, 1, 2, 3, 1]
    assert ex.history([]) == [ex.ChatState()]
