import json

import pytest

from study.loader import load

ex = load(__file__, "ex06_graphs_and_stacks")


@pytest.mark.topic("py.data_structures.stack_queue")
@pytest.mark.parametrize(
    ("partial", "suffix"),
    [
        ('{"a": [1, {"b": 2', "}]}"),
        ('{"a": 1}', ""),
        ("", ""),
        ("[[[", "]]]"),
        ('{"text": "use [brackets] and {braces', '"}'),  # ends inside a string
        ('{"q": "say \\"hi\\" {", "n": [', "]}"),  # escaped quotes inside a string
        ('["a\\\\", [', "]]"),  # escaped backslash, then the string really ends
    ],
)
def test_closing_suffix(partial, suffix):
    assert ex.closing_suffix(partial) == suffix


@pytest.mark.topic("py.data_structures.stack_queue")
def test_closing_suffix_makes_stream_prefixes_parseable():
    full = json.dumps({"answer": "see [1]", "citations": [{"doc": "a{b}", "page": 3}], "ok": True})
    # cut only right after a value/structure, where adding closers yields valid JSON
    for cut in (full.index("[{") + 2 + len('"doc": "a{b}"'), len(full) - 1, len(full)):
        prefix = full[:cut]
        json.loads(prefix + ex.closing_suffix(prefix))


@pytest.mark.topic("py.data_structures.stack_queue")
@pytest.mark.parametrize("bad", ["}", '{"a": ]', "[1, 2}", "{]"])
def test_closing_suffix_mismatch(bad):
    assert ex.closing_suffix("[") == "]"
    with pytest.raises(ValueError):
        ex.closing_suffix(bad)


GRAPH = {
    "user": ["planner"],
    "planner": ["search", "calculator"],
    "search": ["reader", "planner"],  # cycle back to planner
    "calculator": ["answer"],
    "reader": ["answer"],
    "orphan": ["user"],
}


@pytest.mark.topic("py.data_structures.deque", "py.data_structures.dicts")
def test_shortest_path():
    # 3 edges via calculator beats 4 edges via search -> reader
    assert ex.shortest_path(GRAPH, "user", "answer") == ["user", "planner", "calculator", "answer"]
    assert ex.shortest_path(GRAPH, "search", "calculator") == ["search", "planner", "calculator"]
    assert ex.shortest_path(GRAPH, "user", "user") == ["user"]


@pytest.mark.topic("py.data_structures.deque", "py.data_structures.sets")
def test_shortest_path_unreachable_and_sinks():
    assert ex.shortest_path(GRAPH, "answer", "user") is None  # "answer" has no outgoing edges
    assert ex.shortest_path(GRAPH, "user", "orphan") is None
    assert ex.shortest_path({}, "a", "b") is None


@pytest.mark.topic("py.data_structures.deque", "py.data_structures.complexity")
def test_shortest_path_tie_uses_neighbour_order_and_scales():
    g = {"s": ["b", "a"], "a": ["t"], "b": ["t"]}
    assert ex.shortest_path(g, "s", "t") == ["s", "b", "t"]
    # long chain + dense fan-out: BFS must be O(V + E)
    n = 20_000
    chain = {i: [i + 1] + list(range(max(0, i - 5), i)) for i in range(n)}
    path = ex.shortest_path(chain, 0, n)
    assert path == list(range(n + 1))


@pytest.mark.topic("py.data_structures.dicts", "py.data_structures.sets")
def test_run_order():
    deps = {"embed": ["chunk"], "chunk": ["load"], "index": ["embed"], "load": []}
    assert ex.run_order(deps) == ["load", "chunk", "embed", "index"]


@pytest.mark.topic("py.data_structures.dicts", "py.data_structures.stack_queue")
def test_run_order_ties_and_implicit_steps():
    deps = {
        "report": ["eval_b", "eval_a"],
        "eval_a": ["model"],
        "eval_b": ["model", "dataset"],
        "model": ["weights"],  # "weights" and "dataset" are only mentioned as deps
    }
    assert ex.run_order(deps) == ["dataset", "weights", "model", "eval_a", "eval_b", "report"]
    assert ex.run_order({}) == []
    assert ex.run_order({"a": ["b", "b"]}) == ["b", "a"]  # duplicate dep listed twice


@pytest.mark.topic("py.data_structures.dicts")
def test_run_order_cycle():
    assert ex.run_order({"a": []}) == ["a"]
    with pytest.raises(ValueError, match="b"):
        ex.run_order({"a": ["b"], "b": ["c"], "c": ["b"], "d": []})
    with pytest.raises(ValueError):
        ex.run_order({"self": ["self"]})
