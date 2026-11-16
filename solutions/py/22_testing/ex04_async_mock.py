"""
Exercise 22.4 - Testing async code with AsyncMock                     level: ★★★
topics: testing.async_testing, testing.testing_async_functions, testing.unittest_mock

Problem
-------
The code under test is GIVEN (`RateLimitError`, `embed_all`). You write an async,
mutation-tested suite for it (same grading as ex01/ex03: pass on the real implementation,
fail - any exception - on each mutant).

Spec of  await embed_all(client, texts, *, batch_size=2, max_attempts=3,
                         base_delay=0.1, sleep=asyncio.sleep) -> list[list[float]]
  * empty `texts` -> [] and the client is never called
  * splits texts into consecutive batches of `batch_size` (the last may be shorter) and
    awaits client.embed(batch) once per batch, IN ORDER (batch is a list[str]); each call
    returns one vector per text
  * returns all vectors flattened, in input order
  * client.embed raising RateLimitError -> `await sleep(base_delay * 2 ** (attempt - 1))`
    after failed attempt number `attempt` (1-based), then retry: delays base, 2*base, 4*base...
  * at most `max_attempts` calls per batch; if the last one fails, that RateLimitError
    propagates. No sleep before a first attempt, none after a success.
  * `sleep` is injected precisely so tests don't really wait.

1. async check_embed_all(impl) -> None
   Use AsyncMock for client.embed (return_value / side_effect lists, await_count,
   await_args_list) and an AsyncMock for `sleep` to assert the exact backoff delays.
   With pytest-asyncio you would write `async def test_...` - here you just write the
   coroutine and our async tests await it.

Interviewers look for
---------------------
* Mock() vs AsyncMock(): calling an AsyncMock returns an awaitable; a plain Mock's return
  value can't be awaited ("object Mock can't be used in 'await' expression").
* assert_awaited_* / await_args_list vs assert_called_*: called-but-never-awaited is a bug
  (a forgotten `await` just creates a coroutine and a RuntimeWarning).
* Inject time (sleep, clock) instead of really sleeping - fast AND deterministic tests.
* Pick parameters that make bugs visible: with 2 retries, linear (1,2) and exponential
  (1,2) backoff look identical; with 3+ they don't.
"""

import asyncio
from unittest.mock import AsyncMock, call

import pytest

__stub_keep__ = ["RateLimitError", "embed_all"]


class RateLimitError(Exception):
    """HTTP 429 from the embedding API."""


async def embed_all(
    client,
    texts: list[str],
    *,
    batch_size: int = 2,
    max_attempts: int = 3,
    base_delay: float = 0.1,
    sleep=asyncio.sleep,
) -> list[list[float]]:
    """Code under test - correct implementation (given)."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(1, max_attempts + 1):
            try:
                vectors.extend(await client.embed(batch))
                break
            except RateLimitError:
                if attempt == max_attempts:
                    raise
                await sleep(base_delay * 2 ** (attempt - 1))
    return vectors


def _client(side_effect=None) -> AsyncMock:
    client = AsyncMock()
    if side_effect is None:
        # one fake vector per text, derived from the text -> order is checkable
        client.embed.side_effect = lambda batch: [[float(len(t))] for t in batch]
    else:
        client.embed.side_effect = side_effect
    return client


async def check_embed_all(impl) -> None:
    # empty input: no calls
    client = _client()
    assert await impl(client, [], sleep=AsyncMock()) == []
    client.embed.assert_not_awaited()

    # batching + order, including the short last batch
    client, sleep = _client(), AsyncMock()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    out = await impl(client, texts, batch_size=2, sleep=sleep)
    assert out == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert client.embed.await_args_list == [call(["a", "bb"]), call(["ccc", "dddd"]), call(["eeeee"])]
    sleep.assert_not_awaited()

    # retry with exponential backoff, then success
    client, sleep = _client([RateLimitError(), RateLimitError(), RateLimitError(), [[9.0]]]), AsyncMock()
    assert await impl(client, ["x"], max_attempts=4, base_delay=0.5, sleep=sleep) == [[9.0]]
    assert client.embed.await_count == 4
    assert sleep.await_args_list == [call(0.5), call(1.0), call(2.0)]

    # gives up after max_attempts and re-raises; no sleep after the last failure
    client, sleep = _client(RateLimitError("429")), AsyncMock()
    with pytest.raises(RateLimitError):
        await impl(client, ["x"], max_attempts=3, base_delay=0.1, sleep=sleep)
    assert client.embed.await_count == 3
    assert sleep.await_count == 2

    # a retry in batch 2 must not disturb batch 1's results
    client, sleep = _client([[[1.0], [2.0]], RateLimitError(), [[3.0]]]), AsyncMock()
    assert await impl(client, ["a", "b", "c"], batch_size=2, sleep=sleep) == [[1.0], [2.0], [3.0]]
