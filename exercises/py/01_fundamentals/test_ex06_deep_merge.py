import copy

import pytest

from study.loader import load

ex = load(__file__, "ex06_deep_merge")

DEFAULTS = {
    "llm": {"model": "small", "temperature": 0.7, "stop": ["\n\n"]},
    "retrieval": {"k": 4, "rerank": {"enabled": False, "model": "mini"}},
    "debug": False,
}
PROD = {
    "llm": {"model": "large", "stop": ["</answer>"]},
    "retrieval": {"rerank": {"enabled": True}},
    "debug": None,  # unset
}


@pytest.mark.topic("py.fundamentals.nested_structures")
def test_deep_merge_basic():
    assert ex.deep_merge(DEFAULTS, PROD) == {
        "llm": {"model": "large", "temperature": 0.7, "stop": ["</answer>"]},
        "retrieval": {"k": 4, "rerank": {"enabled": True, "model": "mini"}},
    }


@pytest.mark.topic("py.fundamentals.nested_structures")
@pytest.mark.parametrize(
    ("base", "override", "expected"),
    [
        ({}, {}, {}),
        ({"a": 1}, {}, {"a": 1}),
        ({}, {"a": {"b": 1}}, {"a": {"b": 1}}),
        ({"a": [1, 2]}, {"a": [3]}, {"a": [3]}),  # lists replaced
        ({"a": {"b": 1}}, {"a": 5}, {"a": 5}),  # scalar replaces dict
        ({"a": 5}, {"a": {"b": 1}}, {"a": {"b": 1}}),  # dict replaces scalar
        ({"a": {"b": 1, "c": 2}}, {"a": {"c": None}}, {"a": {"b": 1}}),  # nested unset
        ({"a": 1}, {"zzz": None}, {"a": 1}),  # unsetting an absent key is fine
        ({"a": 0}, {"a": False}, {"a": False}),  # falsy is not None
    ],
)
def test_deep_merge_cases(base, override, expected):
    assert ex.deep_merge(base, override) == expected


@pytest.mark.topic("py.fundamentals.mutability")
def test_deep_merge_does_not_mutate_inputs():
    base, override = copy.deepcopy(DEFAULTS), copy.deepcopy(PROD)
    ex.deep_merge(base, override)
    assert base == DEFAULTS
    assert override == PROD


@pytest.mark.topic("py.fundamentals.mutability", "py.fundamentals.identity_equality")
def test_deep_merge_result_shares_nothing():
    base, override = copy.deepcopy(DEFAULTS), copy.deepcopy(PROD)
    merged = ex.deep_merge(base, override)
    merged["llm"]["stop"].append("X")
    merged["retrieval"]["rerank"]["model"] = "changed"
    merged["retrieval"]["k"] = 99
    assert base == DEFAULTS
    assert override == PROD
    src = {"tags": ["a"]}
    out = ex.deep_merge(src, {})
    out["tags"].append("b")
    assert src == {"tags": ["a"]}


@pytest.mark.topic("py.fundamentals.nested_structures")
def test_merge_all_layers():
    cli = {"llm": {"temperature": 0.0}}
    assert ex.merge_all(DEFAULTS, PROD, cli) == {
        "llm": {"model": "large", "temperature": 0.0, "stop": ["</answer>"]},
        "retrieval": {"k": 4, "rerank": {"enabled": True, "model": "mini"}},
    }
    assert ex.merge_all() == {}
    single = ex.merge_all(DEFAULTS)
    assert single == DEFAULTS and single is not DEFAULTS


@pytest.mark.topic("py.fundamentals.dict_comprehensions", "py.fundamentals.none")
def test_config_diff():
    new = ex.merge_all(DEFAULTS, PROD)
    diff = ex.config_diff(DEFAULTS, new)
    assert diff == {
        "debug": (False, ex.MISSING),
        "llm.model": ("small", "large"),
        "llm.stop": (["\n\n"], ["</answer>"]),
        "retrieval.rerank.enabled": (False, True),
    }
    assert list(diff) == sorted(diff)
    assert diff["debug"][1] is ex.MISSING


@pytest.mark.topic("py.fundamentals.dict_comprehensions", "py.fundamentals.none")
def test_config_diff_edges():
    assert ex.config_diff(DEFAULTS, copy.deepcopy(DEFAULTS)) == {}
    assert ex.config_diff({}, {"a": {"b": 1}}) == {"a": (ex.MISSING, {"b": 1})}
    assert ex.config_diff({"a": None}, {}) == {"a": (None, ex.MISSING)}
    assert ex.config_diff({"a": None}, {"a": None}) == {}
    assert ex.config_diff({"a": 1}, {"a": {"x": 1}}) == {"a": (1, {"x": 1})}
