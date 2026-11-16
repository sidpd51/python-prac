"""
Exercise 16.9 - Structured concurrency: TaskGroup and a task DAG   level: ★★★
topics: async.taskgroup, async.dependencies, async.scheduling, async.cancellation

Problem
-------
1. `async def enrich_chunk(chunk, *, summarize, keywords, embed) -> Enriched`
   Run the three async calls on `chunk` concurrently inside an `asyncio.TaskGroup` and
   return `Enriched(summary, keywords, vector)`. If any call fails the TaskGroup cancels
   the others and raises an `ExceptionGroup` - let it propagate (don't catch it).

2. `async def run_dag(nodes) -> dict[str, Any]`
   `nodes` maps name -> `Node(deps, fn)` (given). `fn` is async and receives ONE argument:
   a dict {dep_name: dep_result} of its direct dependencies. Run the whole graph inside a
   TaskGroup with maximum parallelism: every node starts as soon as ALL of its own deps
   have finished (not "level by level" - a node must not wait for unrelated slow nodes).
   Return {name: result} for every node.
   * validate BEFORE running anything: a dep that isn't a node, or a cycle -> ValueError
   * if a node fails: every running node is cancelled, nodes not yet started never start,
     and the TaskGroup's ExceptionGroup propagates
   Typical use: an agent plan "fetch_profile, fetch_orders -> build_prompt -> llm".

Interviewers look for
---------------------
* TaskGroup (3.11+) = structured concurrency: no task outlives the `async with` block; the
  first failure cancels siblings; errors surface as ExceptionGroup (handle with except*).
* gather() vs TaskGroup: gather leaves siblings running on error (unless return_exceptions),
  TaskGroup cancels them. A Task can be awaited by many waiters (that's how deps work).
* Dependencies: await the dependency *tasks* inside each node's coroutine; topological
  order only for creating tasks / detecting cycles.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Any


@dataclass(frozen=True)
class Enriched:
    summary: str
    keywords: list[str]
    vector: list[float]


@dataclass(frozen=True)
class Node:
    fn: Callable[[dict[str, Any]], Awaitable[Any]]
    deps: tuple[str, ...] = field(default=())


async def enrich_chunk(
    chunk: str,
    *,
    summarize: Callable[[str], Awaitable[str]],
    keywords: Callable[[str], Awaitable[list[str]]],
    embed: Callable[[str], Awaitable[list[float]]],
) -> Enriched:
    raise NotImplementedError


async def run_dag(nodes: Mapping[str, Node]) -> dict[str, Any]:
    raise NotImplementedError
