import pytest

from study.loader import load

ex = load(__file__, "ex04_tool_dispatch")


def search(query: str, k: int = 3, *, source: str = "web") -> list[str]:
    """Search the corpus.

    Longer description that must not end up in the schema.
    """
    return [f"{source}:{query}:{i}" for i in range(k)]


def add(a: int, b: int) -> int:
    return a + b


def buggy(text: str) -> int:
    """Has a bug inside."""
    return len(text) + None  # TypeError raised INSIDE the tool


def explode() -> None:
    raise RuntimeError("disk full")


@pytest.mark.topic("py.functions.first_class", "py.functions.reference_vs_call")
def test_tools_from():
    tools = ex.tools_from(search, add)
    assert tools == {"search": search, "add": add}
    assert tools["add"] is add
    assert ex.tools_from() == {}


@pytest.mark.topic("py.functions.reference_vs_call")
def test_tools_from_rejects_call_results_and_duplicates():
    assert ex.tools_from(add) == {"add": add}
    with pytest.raises(TypeError):
        ex.tools_from(add, search("x"))  # called instead of referenced
    with pytest.raises(TypeError):
        ex.tools_from(add(1, 2))

    def add_v2(a, b):
        return a + b

    add_v2.__name__ = "add"
    with pytest.raises(ValueError):
        ex.tools_from(add, add_v2)


@pytest.mark.topic("py.functions.first_class", "py.functions.definition")
def test_tool_schema():
    assert ex.tool_schema(search) == {
        "name": "search",
        "description": "Search the corpus.",
        "required": ["query"],
        "optional": {"k": 3, "source": "web"},
    }
    assert ex.tool_schema(add) == {"name": "add", "description": "", "required": ["a", "b"], "optional": {}}


@pytest.mark.topic("py.functions.pos_kw_only", "py.functions.args", "py.functions.kwargs")
def test_tool_schema_rejects_unnamed_params():
    def pos_only(a, /, b):
        return a

    def star_args(*items):
        return items

    def star_kwargs(**opts):
        return opts

    assert ex.tool_schema(explode)["required"] == []
    for fn in (pos_only, star_args, star_kwargs):
        with pytest.raises(TypeError):
            ex.tool_schema(fn)


@pytest.mark.topic("py.functions.higher_order", "py.functions.kwargs", "py.functions.unpacking")
def test_run_tool_success_dict_and_json_args():
    tools = {"search": search, "add": add}
    assert ex.run_tool(tools, {"name": "add", "arguments": {"a": 2, "b": 3}}) == {"ok": True, "result": 5}
    out = ex.run_tool(tools, {"name": "search", "arguments": '{"query": "rag", "k": 2, "source": "kb"}'})
    assert out == {"ok": True, "result": ["kb:rag:0", "kb:rag:1"]}
    assert ex.run_tool({"explode": lambda: "fine"}, {"name": "explode"}) == {"ok": True, "result": "fine"}


@pytest.mark.topic("py.functions.higher_order")
def test_run_tool_unknown_and_bad_json():
    tools = {"add": add}
    assert ex.run_tool(tools, {"name": "nope", "arguments": {}}) == {"ok": False, "error": "unknown tool: nope"}
    out = ex.run_tool(tools, {"name": "add", "arguments": '{"a": 1, "b":'})
    assert out["ok"] is False and out["error"].startswith("bad arguments")
    out = ex.run_tool(tools, {"name": "add", "arguments": "[1, 2]"})
    assert out["ok"] is False and out["error"].startswith("bad arguments")


@pytest.mark.topic("py.functions.kwargs", "py.functions.higher_order")
@pytest.mark.parametrize(
    "arguments",
    [{"a": 1}, {"a": 1, "b": 2, "c": 3}, {"x": 1}],
)
def test_run_tool_signature_mismatch_does_not_call(arguments):
    calls = []

    def add_logged(a, b):
        calls.append((a, b))
        return a + b

    assert ex.run_tool({"add": add_logged}, {"name": "add", "arguments": {"a": 1, "b": 1}})["ok"]
    calls.clear()
    out = ex.run_tool({"add": add_logged}, {"name": "add", "arguments": arguments})
    assert out["ok"] is False
    assert out["error"].startswith("bad arguments")
    assert calls == []


@pytest.mark.topic("py.functions.higher_order", "py.functions.reference_vs_call")
def test_run_tool_errors_inside_tool():
    tools = ex.tools_from(buggy, explode)
    out = ex.run_tool(tools, {"name": "buggy", "arguments": {"text": "abc"}})
    assert out["ok"] is False
    assert out["error"].startswith("tool error: TypeError"), "a TypeError inside the tool is not a bad call"
    assert ex.run_tool(tools, {"name": "explode", "arguments": {}}) == {
        "ok": False,
        "error": "tool error: RuntimeError: disk full",
    }
