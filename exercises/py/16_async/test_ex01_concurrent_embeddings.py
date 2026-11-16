import asyncio
import inspect
import time

import pytest

from study.loader import load

ex = load(__file__, "ex01_concurrent_embeddings")

TEXTS = ["alpha", "be", "gamma-ray", "d", "epsilon"]
EXPECTED = [[5.0], [2.0], [9.0], [1.0], [7.0]]


@pytest.fixture(autouse=True)
def _clear_calls():
    ex.CALLS.clear()


@pytest.mark.topic("py.async.sequential_vs_concurrent", "py.async.await", "py.async.sync_vs_async")
async def test_sequential_is_sum_of_delays():
    start = time.perf_counter()
    result = await ex.embed_sequential(TEXTS)
    elapsed = time.perf_counter() - start
    assert result == EXPECTED
    assert ex.CALLS == TEXTS
    assert elapsed >= 5 * ex.EMBED_DELAY * 0.9  # 5 awaits one after another


@pytest.mark.topic(
    "py.async.sequential_vs_concurrent", "py.async.gather", "py.async.io_bound", "py.async.asyncio_sleep"
)
async def test_concurrent_is_max_of_delays():
    start = time.perf_counter()
    result = await ex.embed_concurrent(TEXTS)
    elapsed = time.perf_counter() - start
    assert result == EXPECTED
    assert sorted(ex.CALLS) == sorted(TEXTS)
    # sequential would be 0.25 s; concurrent ~0.05 s
    assert elapsed < 3 * ex.EMBED_DELAY


@pytest.mark.topic("py.async.gather_ordering", "py.async.suspension")
async def test_results_in_input_order_not_completion_order():
    delays = [0.12, 0.01, 0.06]
    result = await ex.embed_concurrent(["slow", "fast!", "medium"], delays)
    assert result == [[4.0], [5.0], [6.0]]
    assert await ex.embed_concurrent([]) == []


@pytest.mark.topic("py.async.coroutine_objects", "py.async.coroutine_functions", "py.async.async_keyword")
async def test_calling_coroutine_function_runs_nothing():
    assert inspect.iscoroutinefunction(ex.embed_concurrent)
    coro = ex.embed_concurrent(["x", "y"])
    assert inspect.iscoroutine(coro)
    await asyncio.sleep(0.01)
    assert ex.CALLS == []  # created, never awaited -> body has not started
    assert await coro == [[1.0], [1.0]]
    assert sorted(ex.CALLS) == ["x", "y"]


@pytest.mark.topic("py.async.create_task", "py.async.scheduling", "py.async.task_done")
async def test_start_prefetch_schedules_without_awaiting():
    tasks = ex.start_prefetch(["a", "bb", "ccc"])
    assert all(isinstance(t, asyncio.Task) for t in tasks)
    assert [t.get_name() for t in tasks] == ["embed-0", "embed-1", "embed-2"]
    # scheduled, but nothing runs until we yield control to the loop
    assert ex.CALLS == []
    assert not any(t.done() for t in tasks)
    await asyncio.sleep(0)
    assert ex.CALLS == ["a", "bb", "ccc"]  # started (in creation order) ...
    assert not any(t.done() for t in tasks)  # ... but still sleeping
    assert await asyncio.gather(*tasks) == [[1.0], [2.0], [3.0]]
    assert all(t.done() for t in tasks)


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited")
@pytest.mark.topic("py.async.create_task")
def test_start_prefetch_needs_running_loop():
    # not pytest.raises(RuntimeError): NotImplementedError is a RuntimeError subclass
    try:
        ex.start_prefetch(["a"])
    except NotImplementedError:
        raise
    except RuntimeError as e:
        assert "event loop" in str(e)
    else:
        pytest.fail("create_task outside a running loop must raise RuntimeError")


@pytest.mark.topic("py.async.asyncio_run", "py.async.event_loop")
def test_blocking_entrypoint_runs_its_own_loop():
    assert ex.embed_blocking_entrypoint(["hello", "hi"]) == [[5.0], [2.0]]
    # a second call works too: asyncio.run creates (and closes) a fresh loop each time
    assert ex.embed_blocking_entrypoint([]) == []
