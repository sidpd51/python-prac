import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from study.loader import load

ex = load(__file__, "ex05_weak_model_cache")


def make_loader(delay: float = 0.0):
    calls: list[str] = []

    def loader(name: str):
        calls.append(name)
        if delay:
            time.sleep(delay)
        return ex.Model(name)

    return loader, calls


@pytest.mark.topic("py.memory.weakref")
def test_shared_while_in_use():
    loader, calls = make_loader()
    reg = ex.ModelRegistry(loader)
    m = reg.get("bge-small")
    same = reg.get("bge-small") is m
    assert same
    assert reg.loads == 1 and calls == ["bge-small"]
    assert "bge-small" in reg and len(reg) == 1


@pytest.mark.topic("py.memory.weakref", "py.memory.refcounting")
def test_freed_when_unused_then_reloaded():
    loader, calls = make_loader()
    reg = ex.ModelRegistry(loader)
    m = reg.get("reranker")
    assert isinstance(m, ex.Model) and m.name == "reranker"
    del m  # last strong reference -> freed immediately by refcounting
    assert "reranker" not in reg
    assert len(reg) == 0
    assert reg.evicted == ["reranker"]

    m2 = reg.get("reranker")
    assert reg.loads == 2 and m2.name == "reranker"


@pytest.mark.topic("py.memory.weakref")
def test_get_returns_model_even_if_caller_discards_it():
    reg = ex.ModelRegistry(make_loader()[0])
    assert reg.get("tmp").name == "tmp"  # the classic KeyError bug fails here
    assert reg.evicted == ["tmp"]


@pytest.mark.topic("py.memory.weakref", "py.memory.gc")
def test_eviction_order_and_close():
    reg = ex.ModelRegistry(make_loader()[0])
    a, b, c = reg.get("a"), reg.get("b"), reg.get("c")
    del b
    del a
    assert reg.evicted == ["b", "a"]
    assert len(reg) == 1

    reg.close()
    reg.close()  # idempotent
    assert len(reg) == 0
    del c
    assert reg.evicted == ["b", "a"], "finalizers must be detached by close()"


@pytest.mark.topic("py.memory.weakref", "py.internals.threading")
def test_concurrent_get_loads_once():
    loader, calls = make_loader(delay=0.05)
    reg = ex.ModelRegistry(loader)
    barrier = threading.Barrier(8)

    def worker(_):
        barrier.wait(timeout=5)
        return reg.get("embedder")

    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(worker, range(8)))
    assert calls == ["embedder"], "concurrent first access must load exactly once"
    assert all(m is models[0] for m in models)
