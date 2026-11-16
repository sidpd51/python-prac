import asyncio

import pytest

from study.loader import load

ex = load(__file__, "ex04_async_mock")


async def assert_suite_fails(run_suite, *args):
    """Like a red pytest run: ANY exception counts as "the suite caught the bug".
    NotImplementedError is re-raised so an unsolved stub shows up as TODO, not as a pass."""
    try:
        await run_suite(*args)
    except NotImplementedError:
        raise
    except (Exception, pytest.fail.Exception):
        return
    raise AssertionError("your suite PASSED on a buggy implementation - add a check for it")


def _make(
    *,
    retry=True,
    delay=lambda base, attempt: base * 2 ** (attempt - 1),
    extra_attempt=0,
    drop_partial=False,
    sleep_first=False,
    swallow=False,
    reverse_batches=False,
    empty_calls=False,
):
    async def impl(client, texts, *, batch_size=2, max_attempts=3, base_delay=0.1, sleep=asyncio.sleep):
        if empty_calls and not texts:
            await client.embed([])
            return []
        stop = len(texts) - len(texts) % batch_size if drop_partial else len(texts)
        starts = list(range(0, stop, batch_size))
        if reverse_batches:
            starts.reverse()
        results = []
        for start in starts:
            batch = list(texts[start : start + batch_size])
            attempts = max_attempts + extra_attempt if retry else 1
            got = []
            for attempt in range(1, attempts + 1):
                if sleep_first:
                    await sleep(base_delay)
                try:
                    got = await client.embed(batch)
                    break
                except ex.RateLimitError:
                    if attempt == attempts:
                        if swallow:
                            break
                        raise
                    if not sleep_first:
                        await sleep(delay(base_delay, attempt))
            results.append(got)
        if reverse_batches:
            results.reverse()  # results come back in input order, but calls were out of order
        return [v for r in results for v in r]

    return impl


MUTANTS = {
    "no_retry": _make(retry=False),
    "linear_backoff": _make(delay=lambda base, attempt: base * attempt),
    "constant_backoff": _make(delay=lambda base, attempt: base),
    "one_attempt_too_many": _make(extra_attempt=1),
    "drops_short_last_batch": _make(drop_partial=True),
    "sleeps_before_every_attempt": _make(sleep_first=True),
    "swallows_final_error": _make(swallow=True),
    "embeds_batches_out_of_order": _make(reverse_batches=True),
    "calls_api_for_empty_input": _make(empty_calls=True),
}


@pytest.mark.topic("py.testing.async_testing", "py.testing.testing_async_functions")
async def test_suite_passes_on_correct_impl():
    await ex.check_embed_all(ex.embed_all)


@pytest.mark.topic(
    "py.testing.async_testing", "py.testing.testing_async_functions", "py.testing.unittest_mock"
)
@pytest.mark.parametrize("name", sorted(MUTANTS))
async def test_suite_catches_mutant(name):
    await assert_suite_fails(ex.check_embed_all, MUTANTS[name])


@pytest.mark.topic("py.testing.testing_async_functions")
async def test_suite_never_really_sleeps():
    loop = asyncio.get_running_loop()
    start = loop.time()
    await ex.check_embed_all(ex.embed_all)
    assert loop.time() - start < 0.5, "your suite spent real time sleeping - inject `sleep`"
