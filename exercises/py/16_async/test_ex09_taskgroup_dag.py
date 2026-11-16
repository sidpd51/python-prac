import asyncio
import time

import pytest

from study.loader import load

ex = load(__file__, "ex09_taskgroup_dag")


class Tracker:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.started: dict[str, float] = {}
        self.finished: dict[str, float] = {}
        self.cancelled: list[str] = []

    def node(self, name: str, delay: float, fail: bool = False):
        async def fn(deps: dict) -> str:
            self.started[name] = time.perf_counter() - self.t0
            try:
                await asyncio.sleep(delay)
                if fail:
                    raise ValueError(f"{name} failed")
                inputs = ",".join(f"{k}={v}" for k, v in sorted(deps.items()))
                return f"{name}({inputs})"
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise
            finally:
                self.finished[name] = time.perf_counter() - self.t0

        return fn


def _part(name: str, delay: float, log: list[str], fail: bool = False):
    async def call(chunk: str):
        try:
            await asyncio.sleep(delay)
            if fail:
                raise ConnectionError(name)
            return {"summary": f"sum:{chunk}", "keywords": [chunk.upper()], "embed": [1.0, 2.0]}[name]
        except asyncio.CancelledError:
            log.append(f"cancelled {name}")
            raise

    return call


@pytest.mark.topic("py.async.taskgroup", "py.async.sequential_vs_concurrent")
async def test_enrich_runs_concurrently():
    log: list[str] = []
    start = time.perf_counter()
    out = await ex.enrich_chunk(
        "rag",
        summarize=_part("summary", 0.05, log),
        keywords=_part("keywords", 0.05, log),
        embed=_part("embed", 0.05, log),
    )
    assert time.perf_counter() - start < 0.13
    assert out == ex.Enriched("sum:rag", ["RAG"], [1.0, 2.0])


@pytest.mark.topic("py.async.taskgroup", "py.async.cancellation", "py.async.gather_exceptions")
async def test_enrich_failure_cancels_siblings_with_exception_group():
    log: list[str] = []
    with pytest.raises(ExceptionGroup) as info:
        await ex.enrich_chunk(
            "rag",
            summarize=_part("summary", 0.2, log),
            keywords=_part("keywords", 0.01, log, fail=True),
            embed=_part("embed", 0.2, log),
        )
    assert info.group_contains(ConnectionError)
    assert sorted(log) == ["cancelled embed", "cancelled summary"]


def diamond(tr: Tracker) -> dict:
    #      A
    #    /   \
    #   B     C      B is slow; D only needs C
    #         |
    #         D
    return {
        "A": ex.Node(tr.node("A", 0.03)),
        "B": ex.Node(tr.node("B", 0.15), ("A",)),
        "C": ex.Node(tr.node("C", 0.03), ("A",)),
        "D": ex.Node(tr.node("D", 0.03), ("C",)),
    }


@pytest.mark.topic("py.async.dependencies", "py.async.taskgroup")
async def test_dag_results_and_dependency_inputs():
    tr = Tracker()
    nodes = diamond(tr)
    nodes["E"] = ex.Node(tr.node("E", 0.0), ("B", "D"))
    result = await ex.run_dag(nodes)
    assert result == {
        "A": "A()",
        "B": "B(A=A())",
        "C": "C(A=A())",
        "D": "D(C=C(A=A()))",
        "E": "E(B=B(A=A()),D=D(C=C(A=A())))",
    }
    # deps finished before dependants started
    assert tr.started["B"] >= tr.finished["A"] and tr.started["C"] >= tr.finished["A"]
    assert tr.started["E"] >= max(tr.finished["B"], tr.finished["D"])


@pytest.mark.topic("py.async.dependencies", "py.async.scheduling")
async def test_dag_starts_nodes_eagerly_not_level_by_level():
    tr = Tracker()
    start = time.perf_counter()
    await ex.run_dag(diamond(tr))
    elapsed = time.perf_counter() - start
    # B and C run in parallel; D starts right after C (doesn't wait for slow B)
    assert abs(tr.started["B"] - tr.started["C"]) < 0.02
    assert tr.finished["D"] < tr.finished["B"]
    assert elapsed < 0.25  # critical path A->B = 0.18 s


@pytest.mark.topic("py.async.dependencies", "py.async.taskgroup", "py.async.cleanup_finally")
async def test_dag_failure_cancels_running_and_skips_unstarted():
    tr = Tracker()
    nodes = {
        "A": ex.Node(tr.node("A", 0.01)),
        "slow": ex.Node(tr.node("slow", 0.3), ("A",)),
        "bad": ex.Node(tr.node("bad", 0.02, fail=True), ("A",)),
        "after_bad": ex.Node(tr.node("after_bad", 0.01), ("bad",)),
    }
    start = time.perf_counter()
    with pytest.raises(ExceptionGroup) as info:
        await ex.run_dag(nodes)
    assert time.perf_counter() - start < 0.2
    assert info.group_contains(ValueError, match="bad failed")
    assert tr.cancelled == ["slow"]
    assert "after_bad" not in tr.started


@pytest.mark.topic("py.async.dependencies")
async def test_dag_validation_before_running():
    tr = Tracker()
    with pytest.raises(ValueError):
        await ex.run_dag({"A": ex.Node(tr.node("A", 0)), "B": ex.Node(tr.node("B", 0), ("missing",))})
    with pytest.raises(ValueError):
        await ex.run_dag(
            {
                "A": ex.Node(tr.node("A", 0), ("C",)),
                "B": ex.Node(tr.node("B", 0), ("A",)),
                "C": ex.Node(tr.node("C", 0), ("B",)),
                "free": ex.Node(tr.node("free", 0)),
            }
        )
    assert tr.started == {}  # nothing ran
    assert await ex.run_dag({}) == {}
