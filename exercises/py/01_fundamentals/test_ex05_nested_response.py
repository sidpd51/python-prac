import json

import pytest

from study.loader import load

ex = load(__file__, "ex05_nested_response")

RESPONSE = {
    "id": "resp-1",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"query": "rag", "k": 3}'}},
                    {"function": {"name": "calc", "arguments": "{'bad json'"}},
                    {"function": {"name": "noop", "arguments": "[1, 2]"}},
                    {"function": {"arguments": "{}"}},
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
}


@pytest.mark.topic("py.fundamentals.nested_structures")
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("id", "resp-1"),
        ("usage.prompt_tokens", 12),
        ("choices.0.finish_reason", "tool_calls"),
        ("choices.-1.message.role", "assistant"),
        (["choices", 0, "message", "tool_calls", 0, "function", "name"], "search"),
        (("choices", "0", "message", "role"), "assistant"),
    ],
)
def test_get_path_hits(path, expected):
    assert ex.get_path(RESPONSE, path) == expected


@pytest.mark.topic("py.fundamentals.nested_structures", "py.fundamentals.conditionals")
@pytest.mark.parametrize(
    "path",
    [
        "missing",
        "choices.1.message",  # index out of range
        "choices.first.message",  # non-numeric index into a list
        "usage.prompt_tokens.deeper",  # descending into an int
        "choices.0.message.content.text",  # descending into None
        ["choices", 5],
    ],
)
def test_get_path_misses_return_default(path):
    assert ex.get_path(RESPONSE, path) is None
    assert ex.get_path(RESPONSE, path, default="N/A") == "N/A"


@pytest.mark.topic("py.fundamentals.none")
def test_get_path_present_none_is_not_default():
    assert ex.get_path(RESPONSE, "choices.0.message.content", default="N/A") is None


@pytest.mark.topic("py.fundamentals.nested_structures")
def test_get_path_empty_path_and_tuples():
    assert ex.get_path(RESPONSE, "") is RESPONSE
    assert ex.get_path(RESPONSE, []) is RESPONSE
    assert ex.get_path({"pair": (1, 2)}, "pair.1") == 2


@pytest.mark.topic("py.fundamentals.nested_structures", "py.fundamentals.loops")
def test_flatten():
    data = {"a": {"b": 1, "c": [10, {"d": 2}]}, "e": "x"}
    flat = ex.flatten(data)
    assert flat == {"a.b": 1, "a.c.0": 10, "a.c.1.d": 2, "e": "x"}
    assert list(flat) == ["a.b", "a.c.0", "a.c.1.d", "e"]


@pytest.mark.topic("py.fundamentals.nested_structures")
def test_flatten_edges():
    assert ex.flatten({}) == {}
    assert ex.flatten({"meta": {}, "tags": [], "n": None}) == {"meta": {}, "tags": [], "n": None}
    assert ex.flatten({"a": {"b": 1}}, sep="/") == {"a/b": 1}
    with pytest.raises(TypeError):
        ex.flatten([1, 2])


@pytest.mark.topic("py.fundamentals.nested_structures", "py.fundamentals.type_conversion")
def test_extract_tool_calls():
    assert ex.extract_tool_calls(RESPONSE) == [
        ("search", {"query": "rag", "k": 3}),
        ("calc", None),  # invalid JSON
        ("noop", None),  # valid JSON but not an object
    ]


@pytest.mark.topic("py.fundamentals.none", "py.fundamentals.conditionals")
@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"content": "hi"}}]},
        {"choices": [{"message": {"content": "hi", "tool_calls": None}}]},
        json.loads('{"choices": null}'),
    ],
)
def test_extract_tool_calls_missing(response):
    assert ex.extract_tool_calls(response) == []
