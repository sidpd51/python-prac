"""
Exercise 02.6 - Streaming JSON repair, BFS and pipeline ordering    level: ★★★
topics: data_structures.stack_queue, data_structures.deque, data_structures.sets,
        data_structures.dicts, data_structures.complexity

Problem
-------
1. closing_suffix(partial: str) -> str                                    (a STACK)
     * an LLM streams JSON token by token; to render/parse a partial object you need the
       brackets that would close it: '{"a": [1, {"b": 2' -> '}]}'
     * track { and [ ; a } or ] must match the most recent opener, else ValueError
     * brackets INSIDE string literals don't count; inside a string a backslash escapes
       the next character (an escaped quote does not end the string)
     * if the text ends inside a string, the suffix starts with '"' to close it
     * balanced input -> ""

2. shortest_path(graph, start, goal) -> list | None                       (a QUEUE / BFS)
     * graph: Mapping[node, Iterable[node]] adjacency (directed); nodes that only appear as
       neighbours have no outgoing edges
     * return the path with the fewest edges as [start, ..., goal]; start == goal -> [start]
     * unreachable -> None
     * explore neighbours in the order given (so ties resolve deterministically)
     * use collections.deque; must handle cycles

3. run_order(deps) -> list                                                (Kahn's algorithm)
     * deps: Mapping[step, Iterable[step]] - the steps each step depends on
       {"embed": ["chunk"], "chunk": ["load"], "index": ["embed"], "load": []}
     * return every step (keys and anything mentioned as a dependency) such that each step
       comes after all its dependencies; among steps that are ready at the same time,
       pick the alphabetically smallest first (deterministic)
     * a cycle -> ValueError naming the steps involved

Interviewers look for
---------------------
* Stack (LIFO, list.append/pop) for nesting/matching; queue (FIFO, deque.popleft) for BFS.
  DFS with a stack does NOT give shortest paths in unweighted graphs; BFS does.
* Mark nodes visited when ENQUEUED, not when dequeued - otherwise a node can be queued
  many times (still correct, but wasteful on dense graphs).
* O(V + E) for BFS and Kahn's; a `visited` set gives O(1) membership.
* Topological ordering = how agent/tool DAGs and data pipelines (Airflow, Prefect) schedule
  work; leftover nodes with in-degree > 0 mean a cycle.
"""

from collections.abc import Hashable, Iterable, Mapping


def closing_suffix(partial: str) -> str:
    raise NotImplementedError


def shortest_path[N: Hashable](graph: Mapping[N, Iterable[N]], start: N, goal: N) -> list[N] | None:
    raise NotImplementedError


def run_order(deps: Mapping[str, Iterable[str]]) -> list[str]:
    raise NotImplementedError
