import json

import pytest

from study.loader import load

ex = load(__file__, "ex01_tool_call_parsing")


@pytest.mark.topic("py.exceptions.try_except")
@pytest.mark.parametrize(
    ("raw", "name", "args"),
    [
        ('{"name": "search", "arguments": {"q": "rag"}}', "search", {"q": "rag"}),
        ('{"name": "search", "arguments": "{\\"q\\": \\"rag\\"}"}', "search", {"q": "rag"}),
        ('{"name": "now"}', "now", {}),
    ],
)
def test_valid_calls(raw, name, args):
    call = ex.parse_tool_call(raw)
    assert call == ex.ToolCall(name, args)


@pytest.mark.topic("py.exceptions.chaining", "py.exceptions.raising")
def test_invalid_json_is_chained():
    raw = '{"name": "search", '
    with pytest.raises(ex.ToolCallError) as info:
        ex.parse_tool_call(raw)
    err = info.value
    assert isinstance(err, ValueError)
    assert isinstance(err.__cause__, json.JSONDecodeError), "use `raise ... from e`"
    assert err.__suppress_context__ is True
    assert err.raw == raw


@pytest.mark.topic("py.exceptions.chaining")
def test_invalid_inner_arguments_json_is_chained():
    raw = json.dumps({"name": "search", "arguments": "{q: rag}"})
    with pytest.raises(ex.ToolCallError) as info:
        ex.parse_tool_call(raw)
    assert isinstance(info.value.__cause__, json.JSONDecodeError)


@pytest.mark.topic("py.exceptions.multiple", "py.exceptions.raising")
@pytest.mark.parametrize(
    "raw",
    [
        '{"arguments": {}}',  # KeyError inside
        '["search"]',  # TypeError inside
        '"search"',  # TypeError inside
        "42",  # TypeError inside
        '{"name": ""}',
        '{"name": 7}',
        '{"name": "search", "arguments": [1, 2]}',
        '{"name": "search", "arguments": "[1, 2]"}',
    ],
)
def test_bad_shapes_raise_domain_error_only(raw):
    with pytest.raises(ex.ToolCallError) as info:
        ex.parse_tool_call(raw)
    assert info.value.raw == raw
    assert str(info.value), "give the error a helpful message"


@pytest.mark.topic("py.exceptions.multiple")
def test_missing_name_cause_is_the_lookup_error():
    with pytest.raises(ex.ToolCallError) as info:
        ex.parse_tool_call('["search"]')
    assert isinstance(info.value.__cause__, TypeError)
    with pytest.raises(ex.ToolCallError) as info:
        ex.parse_tool_call("{}")
    assert isinstance(info.value.__cause__, KeyError)


@pytest.mark.topic("py.exceptions.try_except", "py.exceptions.best_practices")
def test_non_str_input_is_a_bug_not_wrapped():
    with pytest.raises(TypeError):
        ex.parse_tool_call(None)
    try:
        ex.parse_tool_call(None)
    except ex.ToolCallError:  # pragma: no cover - this is the failure we guard against
        pytest.fail("a TypeError for None input must not be translated into ToolCallError")
    except TypeError:
        pass


@pytest.mark.topic("py.exceptions.try_except")
def test_parse_many_collects_failures():
    raws = ['{"name": "a"}', "not json", '{"name": "b", "arguments": {"x": 1}}', "[]"]
    ok, failed = ex.parse_many(raws)
    assert ok == [ex.ToolCall("a"), ex.ToolCall("b", {"x": 1})]
    assert [i for i, _ in failed] == [1, 3]
    assert all(isinstance(e, ex.ToolCallError) for _, e in failed)


@pytest.mark.topic("py.exceptions.best_practices")
def test_parse_many_does_not_swallow_bugs():
    with pytest.raises(TypeError):
        ex.parse_many(['{"name": "a"}', None])


@pytest.mark.topic("py.exceptions.syntax_vs_runtime")
def test_syntax_error_location():
    src = "def f(x):\n    return x +\n"
    lineno, msg = ex.syntax_error_at(src)
    assert lineno == 2
    assert "syntax" in msg.lower()


@pytest.mark.topic("py.exceptions.syntax_vs_runtime")
def test_runtime_errors_are_not_syntax_errors_and_code_does_not_run(tmp_path):
    marker = tmp_path / "ran.txt"
    src = f"open({str(marker)!r}, 'w').write('x')\n1 / 0\nundefined_name\n"
    assert ex.syntax_error_at(src) is None
    assert not marker.exists(), "the code must only be compiled, never executed"


@pytest.mark.topic("py.exceptions.syntax_vs_runtime")
def test_nul_bytes_and_indentation():
    assert ex.syntax_error_at("x = 1\0") is not None
    lineno, _ = ex.syntax_error_at("if True:\nprint('hi')\n")
    assert lineno == 2
