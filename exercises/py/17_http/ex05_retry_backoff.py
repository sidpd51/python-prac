"""
Exercise 17.5 - Retries: exponential backoff, full jitter, Retry-After, idempotency   level: ★★★
topics: http.retries, http.backoff, http.retry_limits, http.status_codes,
        http.rate_limiting, http.error_handling

Problem
-------
RetryPolicy and RetriesExhausted are given.

1. backoff_delay(retry_number, policy, rng=random.random) -> float
     * "full jitter": rng() * min(policy.max_delay, policy.base_delay * 2 ** retry_number)
     * retry_number is 0 for the first retry, 1 for the second, ...
     * rng() returns a float in [0, 1)

2. retry_after_seconds(response) -> float | None
     * the Retry-After header as delta-seconds (float); missing / unparseable / negative -> None
       (also accept the non-standard "retry-after-ms" header, in milliseconds, preferred
       when present - OpenAI sends it)

3. is_idempotent(request) -> bool
     * GET, HEAD, OPTIONS, PUT, DELETE are idempotent; anything else only if the request
       carries an "Idempotency-Key" header

4. async send_with_retries(client, request, policy=RetryPolicy(), *, sleep=asyncio.sleep,
                           rng=random.random) -> httpx.Response
     Send `request` (an httpx.Request) with `await client.send(request)`; at most
     policy.max_attempts attempts in total. Decide after each attempt:

     outcome                                         retry?
     ----------------------------------------------  -------------------------------------
     response, status NOT in policy.retry_statuses   no - return it (even a 400/401/404:
                                                     the caller maps errors)
     status 429 or 503                               yes, always (server refused, did not
                                                     process the request)
     other status in retry_statuses (500/502/504)    only if is_idempotent(request)
     httpx.ConnectError / ConnectTimeout /           yes, always (request never left)
     PoolTimeout
     any other httpx.TransportError (ReadTimeout,    only if is_idempotent(request)
     RemoteProtocolError, WriteError, ...)
     any other exception                             no - propagate

     * "no" for a retryable-looking error = re-raise the exception / return the response
     * delay before retry k (k = 0, 1, ...): if the last response had a Retry-After
       -> min(that, policy.max_delay); otherwise backoff_delay(k, policy, rng)
     * await sleep(delay) - the injected function - never time.sleep
     * when attempts run out: raise RetriesExhausted(attempts, last_response) - chained
       `from` the last exception if the last attempt raised (last_response is then None)
     * policy.max_attempts < 1 -> ValueError
     * close (await response.aclose()) a response you're about to discard for a retry, so
       its connection returns to the pool

Interviewers look for
---------------------
* Retry only what can succeed next time: 408/429/5xx/connection errors - never 400/401/403/
  404/422 (they fail identically and burn quota).
* Non-idempotent POSTs (e.g. "charge card", "create fine-tune job") may have been executed
  when a read timeout / 502 happens: retrying duplicates them. Idempotency keys fix that.
* Exponential backoff spreads retries out; JITTER de-synchronises many clients that failed at
  the same moment (thundering herd). Full jitter = random in [0, cap].
* Bounded attempts AND a max delay (and in real systems an overall deadline/retry budget),
  honouring the server's Retry-After.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
ALWAYS_SAFE_STATUSES = frozenset({429, 503})
NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_statuses: frozenset[int] = field(default=frozenset({429, 500, 502, 503, 504}))


class RetriesExhausted(Exception):
    def __init__(self, attempts: int, last_response: httpx.Response | None) -> None:
        status = last_response.status_code if last_response is not None else None
        super().__init__(f"gave up after {attempts} attempt(s) (last status: {status})")
        self.attempts = attempts
        self.last_response = last_response


def backoff_delay(retry_number: int, policy: RetryPolicy, rng: Callable[[], float] = random.random) -> float:
    raise NotImplementedError


def retry_after_seconds(response: httpx.Response) -> float | None:
    raise NotImplementedError


def is_idempotent(request: httpx.Request) -> bool:
    raise NotImplementedError


async def send_with_retries(
    client: httpx.AsyncClient,
    request: httpx.Request,
    policy: RetryPolicy = RetryPolicy(),
    *,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> httpx.Response:
    raise NotImplementedError
