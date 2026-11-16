# Async HTTP and API Integration

Every LLM call, embedding job and tool invocation is an HTTP request, and in production most of the engineering is
not the happy path: it is timeouts, 429s, retries, concurrency caps, partial failures and validating what came
back. This note uses `httpx` and, for every example, `httpx.MockTransport`: a fake transport that runs your own
function instead of touching the network, so all code here runs offline and deterministically. The same trick is how
you unit-test API clients. Interviews ask: which errors do you retry, why jitter, why one shared client, how do you
cap concurrency, and what happens when 3 of 50 batches fail.

## HTTP fundamentals
<!-- topics: fundamentals -->

HTTP is request/response over a TCP connection (TLS on top for `https`). A request is a **method**, a **URL**,
**headers** and an optional **body**; a response is a **status code**, headers and a body.

```text
POST /v1/chat/completions HTTP/1.1          HTTP/1.1 200 OK
Host: api.example.com                       Content-Type: application/json
Authorization: Bearer sk-...                x-request-id: req_123
Content-Type: application/json
                                            {"choices": [...]}
{"model": "m", "messages": [...]}
```

**Safe** methods (GET, HEAD, OPTIONS) have no side effects. **Idempotent** methods (those plus PUT, DELETE) have the same
effect when repeated. **POST is neither.**

Almost every LLM endpoint is a `POST`. A `POST` retried after an ambiguous failure may run twice (two charges,
two emails sent by a tool). That is why idempotency matters so much for retries, see below.

The client section below shows a real request. `json=` serialises the body and sets `Content-Type: application/json`.

## Sync versus async, requests versus httpx
<!-- topics: sync_vs_async, requests_vs_httpx -->

A network call spends nearly all its time **waiting**. A sync call blocks the thread while waiting; 100 sequential
calls of 1 s each take 100 s. With async, the event loop runs other coroutines during each wait, so 100 concurrent
calls take about 1 s (if the server and your limits allow it). LLM work is I/O-bound, so async is the natural fit
for fan-out: embedding many chunks, parallel tool calls, many users on one server.

Concurrency is demonstrated in the gather section below.

Async does not make one request faster; it lets many waits overlap.

| | `requests` | `httpx` |
|---|---|---|
| Sync API | yes | yes |
| Async API | **no** | yes (`AsyncClient`) |
| HTTP/2 | no | optional (`http2=True`) |
| Timeouts | none by default (hangs forever) | **5 s default**, four-part `Timeout` |
| Redirects | followed by default | **not** followed by default (`follow_redirects=True`) |
| Test transport | third-party mocks | built-in `MockTransport` |

Alternatives: `aiohttp` (async only, own API) and the official provider SDKs (`openai`, `anthropic`), which sit on httpx
and add retries and typed errors.

> **Gotcha** Calling blocking `requests.post` inside an `async def` freezes the whole event loop; every other
> task stalls. Use `httpx.AsyncClient`, or `asyncio.to_thread` for unavoidable blocking code.

## Client and AsyncClient
<!-- topics: client, async_client, get_post -->

`httpx.Client` (sync) and `httpx.AsyncClient` (async) hold shared configuration (base URL, headers, timeout, auth)
and the **connection pool**. Use them as context managers so connections are closed.

```python
import asyncio, httpx

def handler(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
        return httpx.Response(200, json={"id": request.url.params["id"]})
    body = request.read()                       # raw bytes of the body
    return httpx.Response(201, json={"echo": request.headers["content-type"], "len": len(body)})

async def main():
    async with httpx.AsyncClient(base_url="https://api.test",
                                 transport=httpx.MockTransport(handler)) as client:
        g = await client.get("/items", params={"id": 7})
        p = await client.post("/items", json={"name": "x"})
        print(g.url, g.json())             # prints: https://api.test/items?id=7 {'id': '7'}
        print(p.status_code, p.json())     # prints: 201 {'echo': 'application/json', 'len': 12}

asyncio.run(main())
```

* `params=` builds the query string and encodes it. `json=` sends JSON; `data=` sends form fields;
  `content=` sends raw bytes/text; `files=` multipart.
* `await` every `AsyncClient` call (forgetting gives a coroutine, not a response). `response.json()`, `.text`, `.content`
  parse the body; a client is unusable after `async with` ends.

## Connection pooling and reuse
<!-- topics: pooling, reuse -->

Opening a connection is expensive: DNS lookup, TCP handshake, TLS handshake (several round trips, tens to
hundreds of ms). HTTP/1.1 keep-alive lets one connection carry many requests in sequence; a **pool** keeps idle
connections around and hands them to the next request.

```text
new client per request:   [DNS][TCP][TLS][request] ... close   x N     (N handshakes)
one shared client:        [DNS][TCP][TLS][req][req][req]...            (1 handshake, reused)
```

That is why you create **one `AsyncClient` per application** (at start-up, closed at shutdown) and reuse it:

```python
import httpx

limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0)
client = httpx.AsyncClient(limits=limits)          # created once, shared by every task
print(limits.max_connections, limits.max_keepalive_connections)    # prints: 100 20
```

* `max_connections` is a hard cap; extra requests **queue** for the pool (the `pool` timeout below). Idle connections
  beyond `max_keepalive_connections`, or idle longer than `keepalive_expiry`, are closed.

A shared `AsyncClient` is safe to use from many tasks at once. The mistakes: building `AsyncClient()` inside the
function that makes each call (loses reuse, leaks sockets if you forget `aclose()`), or sharing one client across
different event loops (it is bound to the loop it first ran in).

> **Interview** "Why reuse a client?" Avoids repeated DNS/TCP/TLS setup, bounds connections via `Limits`,
> centralises headers/timeouts/auth. Mention that `max_connections` is a *client-side* limit, separate from the
> provider's rate limits.

## Timeouts: connect, read, write, pool
<!-- topics: connect_timeouts, request_timeouts -->

httpx's default is a 5 s timeout, and "timeout" is really four separate limits, each a *per-operation* gap, not a
total deadline:

| Timeout | Covers | Exception |
|---|---|---|
| `connect` | establishing the TCP/TLS connection | `ConnectTimeout` |
| `read` | waiting for the next chunk of response bytes | `ReadTimeout` |
| `write` | sending the next chunk of the request | `WriteTimeout` |
| `pool` | waiting for a free connection in the pool | `PoolTimeout` |

All four subclass `httpx.TimeoutException`, which is a `TransportError`.

```python
import httpx

t = httpx.Timeout(60.0, connect=5.0)         # 60 s for everything, but only 5 s to connect
print(t.connect, t.read, t.write, t.pool)    # prints: 5.0 60.0 60.0 60.0
```

Choose them differently for LLMs: a **short connect** (a few seconds: if you cannot connect, fail fast and retry)
but a **long read** (a non-streaming completion may legitimately take 30-120 s). A single 5 s default would kill
valid long generations; `timeout=None` would hang forever on a dead server.

> **Gotcha** `read=60` does not mean "the request may last 60 s". It is the maximum silence between bytes. A
> streaming response that sends a token every second never trips it and can run for minutes. For a real total
> deadline wrap the call: `async with asyncio.timeout(90):` (3.11+) or `asyncio.wait_for`.

Per-request override: `client.post(url, timeout=120)`. Both `ConnectTimeout` and `ReadTimeout` are usually
retryable, but a `ReadTimeout` on a POST is ambiguous: the server may have processed it.

## Status codes and error handling
<!-- topics: status_codes, error_handling -->

The first digit is the family, and it decides your reaction:

| Family | Meaning | Examples | Typical reaction |
|---|---|---|---|
| 2xx | success | 200, 201, 204 | use the body |
| 3xx | redirect | 301, 302, 307 | httpx does not follow unless told to |
| 4xx | **client** error: the request is wrong | 400, 401, 403, 404, 422 | fix it; do not retry |
| 429 | rate limited | 429 | **retry** after waiting (`Retry-After`) |
| 5xx | **server** error | 500, 502, 503, 504 | retry (transient) |

Exceptions to "4xx never retry": `408` (request timeout) and `429`, sometimes `409`. LLM providers also use `529`/`503`
for overload. `401`/`403` mean credentials or permissions: retrying only hammers the API.

httpx does **not** raise on error status by default; you get a normal `Response`. Call `raise_for_status()`, which
raises `httpx.HTTPStatusError` (carrying `.response`) for 4xx/5xx.

```python
import httpx

def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/bad":
        return httpx.Response(400, json={"error": {"message": "max_tokens too large"}},
                              headers={"x-request-id": "req_9"})
    return httpx.Response(503, text="upstream unavailable")

with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.test") as c:
    r = c.get("/bad")
    print(r.status_code, r.is_error, r.is_client_error)    # prints: 400 True True
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        msg = e.response.json()["error"]["message"]
        print(e.response.status_code, msg, e.response.headers["x-request-id"])
        # prints: 400 max_tokens too large req_9
    print(c.get("/other").is_server_error)                # prints: True
```

Exception hierarchy to remember:

```text
httpx.HTTPError
 |- httpx.RequestError            (no usable response)
 |   |- httpx.TransportError  -> TimeoutException, NetworkError (ConnectError, ReadError), ProtocolError
 |- httpx.HTTPStatusError         (only raised by raise_for_status())
```

Good error handling extracts the **message and request id** from the body/headers (providers' support teams ask for
it), maps statuses to typed exceptions (`AuthError`, `RateLimitError`, `ServerError`) so callers write
`except RateLimitError`, and logs the request id.

> **Gotcha** `Retry-After` is either delta-seconds (`"2"`) or an HTTP-date. Parse both; never assume one.
> Some providers also send non-standard variants such as `retry-after-ms`.

## Retries: what to retry and idempotency
<!-- topics: retries -->

A retry is only correct if repeating the request is **safe** and the failure is **transient**.

| Outcome | Retry? | Why |
|---|---|---|
| connect error, connect timeout | yes | request never reached the server |
| read timeout / connection reset after sending | only if idempotent | server may have acted |
| 429 | yes, honour `Retry-After` | capacity, will pass |
| 500, 502, 503, 504, 529 | yes | transient server trouble |
| 400, 401, 403, 404, 422 | **no** | the request is wrong; same input, same failure |

Idempotency decides the ambiguous cases: GET, PUT, DELETE can be repeated. `POST` can not, **unless** you send an
**idempotency key**: a unique id the server uses to recognise a repeat and return the first result instead of
re-running it (Stripe popularised it; many APIs support `Idempotency-Key`).

```python
import httpx

def is_idempotent(req: httpx.Request) -> bool:
    return req.method in {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"} or "idempotency-key" in req.headers

def is_retryable_status(code: int) -> bool:
    return code in {408, 409, 429} or code >= 500

post = httpx.Request("POST", "https://api.test/charge")
keyed = httpx.Request("POST", "https://api.test/charge", headers={"Idempotency-Key": "abc-1"})
print(is_idempotent(post), is_idempotent(keyed), is_retryable_status(429), is_retryable_status(400))
# prints: False True True False
```

(Headers are case-insensitive in httpx, so `"idempotency-key" in req.headers` matches.)

> **Interview** "Is it safe to retry a POST that timed out?" Not blindly: the timeout means you do not know if it
> ran. Safe if the endpoint is idempotent or you send an idempotency key. For pure LLM completions the cost of a
> duplicate is a wasted generation, so retrying is usually fine; for tool calls with side effects it is not.

## Exponential backoff, jitter and retry limits
<!-- topics: backoff, retry_limits -->

Retrying immediately makes overload worse. **Exponential backoff** waits `base * 2**n` before retry `n` (0, 1, 2...),
capped by `max_delay`. But if 1000 clients all failed together, they all retry together: a **thundering herd**.
**Jitter** randomises the wait to spread them out. **Full jitter** picks uniformly between 0 and the exponential
ceiling.

```python
import random

def backoff(retry, base=0.5, cap=8.0, rng=random.random):
    return rng() * min(cap, base * 2 ** retry)        # full jitter

print([min(8.0, 0.5 * 2 ** n) for n in range(6)])     # prints: [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
print(backoff(3, rng=lambda: 0.5))                    # prints: 2.0
```

Injecting `rng` (and the `sleep` function) makes the policy deterministic in tests. Rules:

* **Server hints beat your formula.** If the response has `Retry-After`, wait at least that long.
* **Cap attempts** (`retry_limits`): 3-5 attempts. Unbounded retries turn an outage into an infinite loop and
  multiply load on a struggling provider.
* **Cap the delay** and ideally set an overall deadline; a user will not wait 5 minutes.
* **Retry only retryable outcomes** (previous section) and only idempotent requests.
* Every layer that retries multiplies attempts (3 retries in the SDK x 3 in your code = 9). Retry in one place.

A complete retry loop with a mock server that fails twice then succeeds:

```python
import asyncio, httpx

calls = []

def handler(request: httpx.Request) -> httpx.Response:
    calls.append(1)
    if len(calls) == 1:
        return httpx.Response(429, headers={"Retry-After": "2"})
    if len(calls) == 2:
        return httpx.Response(503)
    return httpx.Response(200, json={"ok": True})

async def send_with_retries(client, request, max_attempts=4, sleep=asyncio.sleep):
    for attempt in range(max_attempts):
        try:
            resp = await client.send(request)
            if resp.status_code != 429 and resp.status_code < 500:
                return resp
        except httpx.TransportError:
            resp = None
        if attempt == max_attempts - 1:
            break
        hint = float(resp.headers.get("Retry-After", 0)) if resp is not None else 0.0
        delay = max(hint, 0.1 * 2 ** attempt)
        print(f"attempt {attempt + 1} failed, sleeping {delay}")
        await sleep(delay)
    raise RuntimeError("retries exhausted")

async def main():
    async def fake_sleep(_): pass                  # do not really wait in the example
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        r = await send_with_retries(c, c.build_request("GET", "https://api.test/x"), sleep=fake_sleep)
        print(r.status_code, len(calls))

asyncio.run(main())
# prints: attempt 1 failed, sleeping 2.0
# prints: attempt 2 failed, sleeping 0.2
# prints: 200 3
```

The loop sent three requests (429, 503, 200). The server hint (2 s) beat the formula (0.1 s) on the first retry.

## Rate limiting versus concurrency limits
<!-- topics: rate_limiting, concurrency_limits -->

Two different limits that people confuse:

| | Rate limit | Concurrency limit |
|---|---|---|
| Question | how many requests/tokens **per unit time**? | how many requests **in flight at once**? |
| Units | RPM, TPM (requests/tokens per minute) | simultaneous connections or tasks |
| Who enforces | the provider (returns 429) | you (semaphore, pool `Limits`) |
| Tool | token bucket / sliding window | `asyncio.Semaphore` |

With fast responses you can exceed a 60 RPM limit while only 2 requests are in flight; with slow generations you can
have hundreds in flight and stay under the rate limit. Neither implies the other. LLM providers limit **both RPM and
TPM**, so a few huge prompts can trip TPM while RPM is low.

A **concurrency limit** is an `asyncio.Semaphore(n)`: at most n tasks inside the block.

```python
import asyncio

async def main():
    sem = asyncio.Semaphore(3)
    running = peak = 0

    async def call(i):
        nonlocal running, peak
        async with sem:                       # acquire around the request only
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    await asyncio.gather(*(call(i) for i in range(10)))
    print("peak concurrency:", peak)          # prints: peak concurrency: 3

asyncio.run(main())
```

A **rate limit** needs time. The **token bucket**: a bucket holds up to `capacity` tokens and refills at `rate`
tokens/second; each request takes a token (or waits). It allows bursts up to capacity and a long-run average of
`rate`. Refill is computed lazily from an injectable clock, no background task:

```python
class TokenBucket:
    def __init__(self, rate, capacity, clock):
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.tokens, self.last = capacity, clock()

    def try_acquire(self, cost=1):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

t = [0.0]
bucket = TokenBucket(rate=2, capacity=3, clock=lambda: t[0])
print([bucket.try_acquire() for _ in range(4)])    # prints: [True, True, True, False]
t[0] = 1.0                                         # one second later: 2 tokens refilled
print([bucket.try_acquire() for _ in range(3)])    # prints: [True, True, False]
```

An async `acquire()` loops: `await sleep(deficit / rate)`, re-check, with a lock so waiters are served in order. Pass
`cost=prompt_tokens` to respect TPM. The limiter is cooperative; the server's 429 stays the source of truth, so keep
retries.

## Concurrent calls
<!-- topics: concurrent_calls -->

`asyncio.gather(*coros)` runs coroutines concurrently and returns results **in the order you passed them**, not
completion order. That ordering guarantee is what lets you map results back to inputs.

```python
import asyncio, httpx

async def handler(request: httpx.Request) -> httpx.Response:
    n = int(request.url.params["n"])
    await asyncio.sleep(0.03 - n * 0.01)          # later items finish first
    return httpx.Response(200, json={"n": n})

async def main():
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        rs = await asyncio.gather(*(c.get("https://x.test/", params={"n": n}) for n in range(3)))
        print([r.json()["n"] for r in rs])        # prints: [0, 1, 2]  (they finished in order 2, 1, 0)

asyncio.run(main())
```

Alternatives: `asyncio.as_completed` yields as each finishes (progress bars, first useful answer);
`asyncio.TaskGroup` (3.11+) gives structured concurrency but **cancels siblings on the first failure**. Always
combine fan-out with a semaphore: `gather` over 10,000 coroutines creates 10,000 tasks at once.

## Batching
<!-- topics: batching -->

Many APIs accept a list per request: embeddings take up to a few thousand inputs, with limits on item count and
total tokens. Sending 10,000 texts one by one means 10,000 round trips and 10,000 against your RPM limit. Sending
them in batches of 64 means about 157 requests. Batching trades latency per request for throughput and fewer
requests.

```python
texts = [f"t{i}" for i in range(7)]
size = 3
print([(s, len(texts[s:s + size])) for s in range(0, len(texts), size)])   # prints: [(0, 3), (3, 3), (6, 1)]
```

Keep the **start offset** with each batch: the response's per-item `index` is relative to the batch, and providers
may return items out of order, so place each vector at `start + item["index"]`, never rely on arrival order.

Batch size is bounded by the API's max items, max tokens per request (a big batch of long chunks can exceed it), and
the cost of a failure: one failed batch of 2,000 wastes more than one of 64. Typical: 64-256 items.

## Partial failures
<!-- topics: partial_failures -->

With 100 batches, assume some will fail. Plain `await asyncio.gather(*coros)` raises the **first** exception to the
caller while the other tasks keep running unobserved; you lose every result you were waiting for. With
`return_exceptions=True`, exceptions are returned **as values in the result list**, so every task's outcome
is kept.

```python
import asyncio, httpx

def handler(request: httpx.Request) -> httpx.Response:
    n = int(request.url.params["n"])
    if n == 1:
        return httpx.Response(500)
    if n == 2:
        raise httpx.ConnectError("refused", request=request)
    return httpx.Response(200, json={"n": n})

async def fetch(c, n):
    r = await c.get("https://x.test/", params={"n": n})
    r.raise_for_status()
    return r.json()["n"]

async def main():
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        results = await asyncio.gather(*(fetch(c, n) for n in range(4)), return_exceptions=True)
        for i, res in enumerate(results):
            print(i, "failed" if isinstance(res, Exception) else "ok", type(res).__name__)

asyncio.run(main())
# prints: 0 ok int
# prints: 1 failed HTTPStatusError
# prints: 2 failed ConnectError
# prints: 3 ok int
```

Pattern: split `results` into successes and failures, keep failures with their **input indices**, retry just those, and
return a result object (values list with `None` where failed, plus `(index, error)` failures) that makes partial
success explicit.

## Validating responses with Pydantic
<!-- topics: response_validation -->

A 200 status does not mean the body is what you expect. Providers change fields, proxies return HTML error pages
with 200, models return `null` for a field. Validate the body at the boundary with a pydantic model, and treat
"invalid shape" as its own failure category (typically **not** retryable, or retried once).

```python
import httpx
from pydantic import BaseModel, ValidationError

class EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]

class EmbeddingResponse(BaseModel):
    data: list[EmbeddingItem]

def parse(resp: httpx.Response) -> EmbeddingResponse | None:
    try:
        return EmbeddingResponse.model_validate_json(resp.content)
    except ValidationError as e:
        print("invalid response:", e.errors()[0]["type"], e.errors()[0]["loc"])
        return None

html = httpx.Response(200, text="<html>Bad gateway</html>")
wrong = httpx.Response(200, json={"data": [{"index": 0}]})
parse(html)                             # prints: invalid response: json_invalid ()
parse(wrong)                            # prints: invalid response: missing ('data', 0, 'embedding')
```

Beyond shape, check **semantics** the schema cannot: the returned indices are exactly `0..n-1`, the number of
vectors equals the number of inputs, dimensions match your index. A mismatch is a failed batch, not silent data
corruption.

Full picture of a robust client call:

```text
build request -> rate limit (token bucket) -> concurrency limit (semaphore)
  -> send with timeouts -> retry transient failures (backoff + jitter + Retry-After)
  -> raise_for_status / map to typed error -> validate body with pydantic -> result or recorded failure
```

> **AI use** The same flow wraps chat completions: validate `choices[0].message`, check `finish_reason`
> (`"length"` means truncated output), and parse tool-call `arguments` with the tool's pydantic model.

## Quick recap
- HTTP = method + URL + headers + body, status + headers + body. POST is not idempotent; 4xx = your fault, 5xx = theirs.
- Async lets many network waits overlap; it does not speed up a single call. Never block the loop with `requests`.
- httpx: sync + async, 5 s default timeout, no redirects by default, `MockTransport` for offline tests.
- One shared `AsyncClient` per app reuses connections (no repeat DNS/TCP/TLS) and bounds them with `Limits`.
- Four timeouts: connect (short), read (long for LLMs), write, pool. Read is per-gap, not a total deadline.
- httpx does not raise on 4xx/5xx; `raise_for_status()` does. Extract message and request id from the error.
- Retry connect errors, 408/429/5xx; not other 4xx. Retry POST only with an idempotency key.
- Backoff = exponential, capped, full jitter, honour `Retry-After`, limited attempts, retry in one layer only.
- Rate limit = per time (token bucket, RPM/TPM). Concurrency limit = in flight (Semaphore). You need both.
- `gather` preserves input order; `return_exceptions=True` keeps partial results; record failures by input index.
- Batch inputs, keep batch offsets, place results by index; validate every response body with pydantic plus semantic checks.
