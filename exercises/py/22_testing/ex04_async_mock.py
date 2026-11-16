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


async def check_embed_all(impl) -> None:
    raise NotImplementedError
