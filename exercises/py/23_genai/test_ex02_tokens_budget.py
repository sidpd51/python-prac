import copy

import pytest

from study.loader import load

ex = load(__file__, "ex02_tokens_budget")


@pytest.mark.topic("py.genai.tokenization")
def test_tokenize():
    assert ex.tokenize("Hello, world!") == ["Hello", ",", "world", "!"]
    assert ex.tokenize("  GPT-4o costs $2.50/1M\ttokens\n") == [
        "GPT", "-", "4o", "costs", "$", "2", ".", "50", "/", "1M", "tokens",
    ]  # fmt: skip
    assert ex.tokenize("") == []
    assert ex.tokenize("naïve café_au_lait") == ["naïve", "café_au_lait"]


@pytest.mark.topic("py.genai.tokenization")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, world!", 6),
        ("a", 1),
        ("abcd", 1),
        ("abcde", 2),
        ("internationalization", 5),  # 20 chars
        ("...", 3),
        ("", 0),
        ("   \n\t", 0),
    ],
)
def test_count_tokens(text, expected):
    assert ex.count_tokens(text) == expected


@pytest.mark.topic("py.genai.tokenization", "py.genai.text_processing")
def test_truncate_keeps_whole_tokens():
    text = "Retrieval augmented generation, explained."
    # Retrieval=3, augmented=3, generation=3, ","=1, explained=3, "."=1  -> 14
    assert ex.truncate_to_budget(text, 14) == text
    assert ex.truncate_to_budget(text, 13) == "Retrieval augmented generation, explained"
    assert ex.truncate_to_budget(text, 12) == "Retrieval augmented generation,"
    assert ex.truncate_to_budget(text, 9) == "Retrieval augmented generation"
    assert ex.truncate_to_budget(text, 8) == "Retrieval augmented"  # never half a word
    assert ex.truncate_to_budget(text, 2) == ""
    assert ex.truncate_to_budget(text, 0) == ""


@pytest.mark.topic("py.genai.text_processing")
def test_truncate_preserves_inner_spacing_and_drops_trailing():
    text = "line one\n\nline two   and more"
    assert ex.truncate_to_budget(text, 4) == "line one\n\nline two"
    assert ex.truncate_to_budget("  padded  ", 5) == "  padded  ", "whole text fits -> unchanged"


@pytest.mark.topic("py.genai.tokenization")
def test_truncate_validation():
    with pytest.raises(ValueError):
        ex.truncate_to_budget("x", -1)
    assert ex.truncate_to_budget("", 3) == ""


def _m(role, content):
    return {"role": role, "content": content}


@pytest.mark.topic("py.genai.tokenization")
def test_fit_messages_drops_oldest_non_system():
    msgs = [
        _m("system", "brief"),  # 2 + 4 = 6
        _m("user", "aaaa bbbb"),  # 2 + 4 = 6
        _m("assistant", "cccc"),  # 1 + 4 = 5
        _m("user", "dddd"),  # 1 + 4 = 5
        _m("assistant", "eeee"),  # 1 + 4 = 5
        _m("user", "final question"),  # 4 + 4 = 8
    ]  # total 35
    snapshot = copy.deepcopy(msgs)
    assert ex.fit_messages(msgs, 35) == msgs
    assert ex.fit_messages(msgs, 34) == [msgs[0], *msgs[2:]]
    assert ex.fit_messages(msgs, 24) == [msgs[0], msgs[3], msgs[4], msgs[5]]
    assert ex.fit_messages(msgs, 14) == [msgs[0], msgs[5]]
    assert msgs == snapshot, "input must not be mutated"
    result = ex.fit_messages(msgs, 35)
    assert result is not msgs


@pytest.mark.topic("py.genai.tokenization")
def test_fit_messages_keeps_all_system_messages_in_order():
    msgs = [
        _m("system", "rules"),  # 2 + 4 = 6
        _m("user", "old old old"),  # 3 + 4 = 7
        _m("system", "retrieved context"),  # 5 + 4 = 9
        _m("user", "q"),  # 1 + 4 = 5
    ]
    assert ex.fit_messages(msgs, 20) == [msgs[0], msgs[2], msgs[3]]


@pytest.mark.topic("py.genai.tokenization")
def test_fit_messages_impossible_budget():
    msgs = [_m("system", "brief"), _m("user", "hi"), _m("user", "final question")]
    with pytest.raises(ValueError):
        ex.fit_messages(msgs, 13)  # system 6 + last 8 = 14 > 13
    assert ex.fit_messages(msgs, 14) == [msgs[0], msgs[2]]
    assert ex.fit_messages([], 10) == []
