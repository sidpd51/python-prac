import sys

import pytest

from study.loader import load

ex = load(__file__, "ex05_recursion")

TREE = {
    "title": "Guide",
    "children": [
        {
            "title": "Install",
            "children": [{"title": "Linux"}, {"title": "macOS", "children": []}],
        },
        {"title": "Usage", "children": [{"title": "CLI", "children": [{"title": "Linux"}]}]},
        {"title": "FAQ"},
    ],
}


def nested_list(levels):
    obj = []
    for _ in range(levels - 1):
        obj = [obj]
    return obj


CASES = [
    (5, 0),
    ("text", 0),
    (None, 0),
    ([], 1),
    ({}, 1),
    ([1, [2]], 2),
    ({"a": {"b": [1]}}, 3),
    ([[], [[[]]], 1], 4),
    ({"choices": [{"message": {"tool_calls": [{"function": {}}]}}]}, 7),
]


@pytest.mark.topic("py.functions.recursion", "py.functions.return_values")
@pytest.mark.parametrize(("obj", "expected"), CASES)
def test_depth(obj, expected):
    assert ex.depth(obj) == expected


@pytest.mark.topic("py.functions.recursion")
@pytest.mark.parametrize(("obj", "expected"), CASES)
def test_depth_iterative_matches(obj, expected):
    assert ex.depth_iterative(obj) == expected


@pytest.mark.topic("py.functions.recursion")
def test_depth_iterative_survives_hostile_nesting():
    levels = sys.getrecursionlimit() * 50
    assert ex.depth_iterative(nested_list(levels)) == levels


@pytest.mark.topic("py.functions.recursion")
def test_recursive_depth_hits_the_limit():
    assert ex.depth(nested_list(10)) == 10
    with pytest.raises(RecursionError):
        ex.depth(nested_list(sys.getrecursionlimit() * 5))


@pytest.mark.topic("py.functions.recursion", "py.fundamentals.fstrings")
def test_outline():
    assert ex.outline(TREE) == "\n".join(
        [
            "- Guide",
            "  - Install",
            "    - Linux",
            "    - macOS",
            "  - Usage",
            "    - CLI",
            "      - Linux",
            "  - FAQ",
        ]
    )
    assert ex.outline({"title": "Solo"}) == "- Solo"


@pytest.mark.topic("py.functions.recursion", "py.functions.return_values")
def test_find_path():
    assert ex.find_path(TREE, "Guide") == ["Guide"]
    assert ex.find_path(TREE, "macOS") == ["Guide", "Install", "macOS"]
    assert ex.find_path(TREE, "Linux") == ["Guide", "Install", "Linux"]  # first in pre-order
    assert ex.find_path(TREE, "CLI") == ["Guide", "Usage", "CLI"]
    assert ex.find_path(TREE, "FAQ") == ["Guide", "FAQ"]
    assert ex.find_path(TREE, "Windows") is None
