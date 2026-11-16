"""
Exercise 03.5 - Recursion over nested documents                     level: ★★★
topics: functions.recursion, functions.return_values, functions.definition

Problem
-------
Parsed documents are trees: JSON payloads, or heading outlines
    {"title": "Guide", "children": [{"title": "Install", "children": []}, ...]}

1. depth(obj) -> int                                                    (recursive)
     * nesting depth of dicts/lists: a scalar -> 0, [] or {} -> 1,
       [1, [2]] -> 2, {"a": {"b": [1]}} -> 3   (for dicts, look at the values)

2. depth_iterative(obj) -> int                                          (NO recursion)
     * same result as depth(), using an explicit stack, so it works on inputs nested far
       deeper than sys.getrecursionlimit() (e.g. 100_000 levels from a hostile payload)

3. outline(node) -> str
     * render the heading tree as lines "<2 spaces per level>- <title>", root at level 0,
       children in order, lines joined with newlines (no trailing newline)
     * a node without "children" is a leaf

4. find_path(node, title) -> list[str] | None
     * titles from the root down to the FIRST node (pre-order: node, then children left to
       right) whose title == title; not found -> None
     * this is the "breadcrumb" (Guide > Install > Linux) you attach to RAG chunks

Interviewers look for
---------------------
* Every recursive function needs a base case and progress toward it; the return value of
  the recursive call must be USED (a forgotten `return` -> None bubbling up).
* CPython has no tail-call optimisation; default recursion limit ~1000 -> RecursionError on
  deep inputs. Raising the limit risks a hard C-stack crash - convert to an explicit stack.
* Recursion is natural for trees; an explicit stack/queue gives control (depth limits,
  BFS vs DFS order, early exit).
* Untrusted JSON can be adversarially deep - cap the depth at the boundary.
"""

from typing import Any


def depth(obj: Any) -> int:
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, list):
        children = obj
    else:
        return 0
    return 1 + max((depth(c) for c in children), default=0)


def depth_iterative(obj: Any) -> int:
    best = 0
    stack: list[tuple[Any, int]] = [(obj, 0)]
    while stack:
        node, level = stack.pop()
        if isinstance(node, dict):
            children: Any = node.values()
        elif isinstance(node, list):
            children = node
        else:
            continue
        best = max(best, level + 1)
        stack.extend((c, level + 1) for c in children)
    return best


def outline(node: dict[str, Any]) -> str:
    return "\n".join(_outline_lines(node, 0))


def _outline_lines(node: dict[str, Any], level: int) -> list[str]:
    lines = [f"{'  ' * level}- {node['title']}"]
    for child in node.get("children", []):
        lines.extend(_outline_lines(child, level + 1))
    return lines


def find_path(node: dict[str, Any], title: str) -> list[str] | None:
    if node["title"] == title:
        return [node["title"]]
    for child in node.get("children", []):
        sub = find_path(child, title)
        if sub is not None:
            return [node["title"], *sub]
    return None
