import httpx
import pytest

from study.loader import load

ex = load(__file__, "ex05_retry_backoff")

URL = "https://api.example.com/v1/chat/completions"


class Script:
    """MockTransport handler that plays back a list of outcomes (status, Response or exception)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, int):
            return httpx.Response(outcome, json={"status": outcome})
        return outcome


class Sleeps:
    def __init__(self):
        self.calls = []

    async def __call__(self, delay):
        self.calls.append(delay)


def half():
    return 0.5


POLICY = ex.RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=8.0)


async def run(script, method="GET", headers=None, policy=POLICY, rng=half):
    sleeps = Sleeps()
    async with httpx.AsyncClient(transport=httpx.MockTransport(script)) as client:
        request = client.build_request(method, URL, json={"x": 1}, headers=headers)
        response = await ex.send_with_retries(client, request, policy, sleep=sleeps, rng=rng)
    return response, sleeps.calls


@pytest.mark.topic("py.http.backoff")
@pytest.mark.parametrize(
    ("n", "rand", "expected"),
    [
        (0, 0.5, 0.25),
        (1, 0.5, 0.5),
        (2, 0.5, 1.0),
        (3, 1.0, 4.0),
        (5, 0.5, 4.0),
        (9, 0.999, 7.992),
        (0, 0.0, 0.0),
    ],
)
def test_backoff_delay_full_jitter(n, rand, expected):
    assert ex.backoff_delay(n, POLICY, rng=lambda: rand) == pytest.approx(expected)


@pytest.mark.topic("py.http.backoff")
def test_backoff_delay_is_random_within_cap():
    import random

    rng = random.Random(42)
    delays = [ex.backoff_delay(3, POLICY, rng=rng.random) for _ in range(200)]
    assert all(0 <= d <= 4.0 for d in delays)
    assert len(set(delays)) > 150, "jitter: delays must differ between clients/attempts"


@pytest.mark.topic("py.http.rate_limiting", "py.http.status_codes")
@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "3"}, 3.0),
        ({"retry-after": "0.25"}, 0.25),
        ({"retry-after-ms": "1500", "retry-after": "9"}, 1.5),
        ({"retry-after-ms": "junk", "retry-after": "2"}, 2.0),
        ({"retry-after": "-1"}, None),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, None),
        ({}, None),
    ],
)
def test_retry_after_seconds(headers, expected):
    assert ex.retry_after_seconds(httpx.Response(429, headers=headers)) == expected


@pytest.mark.topic("py.http.retries", "py.http.fundamentals")
@pytest.mark.parametrize(
    ("method", "headers", "expected"),
    [
        ("GET", {}, True),
        ("DELETE", {}, True),
        ("PUT", {}, True),
        ("HEAD", {}, True),
        ("POST", {}, False),
        ("PATCH", {}, False),
        ("POST", {"Idempotency-Key": "abc"}, True),
    ],
)
def test_is_idempotent(method, headers, expected):
    assert ex.is_idempotent(httpx.Request(method, URL, headers=headers)) is expected


@pytest.mark.topic("py.http.retries")
async def test_success_first_try():
    script = Script(200)
    response, sleeps = await run(script)
    assert response.status_code == 200
    assert (len(script.requests), sleeps) == (1, [])


@pytest.mark.topic("py.http.retries", "py.http.backoff")
async def test_retries_5xx_with_backoff_then_succeeds():
    script = Script(503, 502, 200)
    response, sleeps = await run(script)
    assert response.status_code == 200
    assert len(script.requests) == 3
    assert sleeps == pytest.approx([0.25, 0.5])


@pytest.mark.topic("py.http.rate_limiting", "py.http.retries")
async def test_honours_retry_after_without_jitter():
    def no_rng():
        raise AssertionError("Retry-After present: don't use jittered backoff")

    script = Script(httpx.Response(429, headers={"retry-after": "3"}), 200)
    response, sleeps = await run(script, rng=no_rng)
    assert response.status_code == 200
    assert sleeps == [3.0]
    script = Script(httpx.Response(429, headers={"retry-after": "120"}), 200)
    _, sleeps = await run(script)
    assert sleeps == [8.0], "Retry-After is capped at max_delay"


@pytest.mark.topic("py.http.status_codes", "py.http.retries")
@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
async def test_client_errors_not_retried(status):
    script = Script(status, 200)
    response, sleeps = await run(script)
    assert response.status_code == status
    assert (len(script.requests), sleeps) == (1, [])


@pytest.mark.topic("py.http.retries", "py.http.error_handling")
async def test_post_500_not_retried_but_429_503_are():
    script = Script(500, 200)
    response, _ = await run(script, method="POST")
    assert response.status_code == 500 and len(script.requests) == 1
    script = Script(429, 503, 200)
    response, _ = await run(script, method="POST")
    assert response.status_code == 200 and len(script.requests) == 3


@pytest.mark.topic("py.http.retries")
async def test_post_with_idempotency_key_retries_500():
    script = Script(500, 504, 200)
    response, _ = await run(script, method="POST", headers={"Idempotency-Key": "k-1"})
    assert response.status_code == 200
    assert [r.headers["idempotency-key"] for r in script.requests] == ["k-1"] * 3


@pytest.mark.topic("py.http.retries", "py.http.connect_timeouts")
@pytest.mark.parametrize(
    "exc", [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow dns"), httpx.PoolTimeout("pool")]
)
async def test_never_sent_errors_retried_even_for_post(exc):
    script = Script(exc, 200)
    response, sleeps = await run(script, method="POST")
    assert response.status_code == 200
    assert len(sleeps) == 1


@pytest.mark.topic("py.http.retries", "py.http.request_timeouts")
async def test_read_timeout_post_not_retried_get_retried():
    timeout = httpx.ReadTimeout("no bytes for 30s")
    with pytest.raises(httpx.ReadTimeout):
        await run(Script(timeout, 200), method="POST")
    script = Script(httpx.ReadTimeout("t"), httpx.RemoteProtocolError("peer closed"), 200)
    response, _ = await run(script, method="GET")
    assert response.status_code == 200 and len(script.requests) == 3


@pytest.mark.topic("py.http.retry_limits")
async def test_gives_up_after_max_attempts_with_response():
    script = Script(503, 503, 503, 503, 200)
    with pytest.raises(ex.RetriesExhausted) as info:
        await run(script)
    assert info.value.attempts == 4
    assert info.value.last_response.status_code == 503
    assert len(script.requests) == 4, "4 attempts in total, not 1 + 4 retries"
    assert len(script.outcomes) == 1


@pytest.mark.topic("py.http.retry_limits", "py.http.error_handling")
async def test_gives_up_after_max_attempts_with_exception():
    errors = [httpx.ConnectError(f"refused {i}") for i in range(3)]
    sleeps = Sleeps()
    async with httpx.AsyncClient(transport=httpx.MockTransport(Script(*errors))) as client:
        request = client.build_request("GET", URL)
        policy = ex.RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=2.0)
        with pytest.raises(ex.RetriesExhausted) as info:
            await ex.send_with_retries(client, request, policy, sleep=sleeps, rng=lambda: 1.0)
    assert info.value.last_response is None
    assert info.value.__cause__ is errors[-1]
    assert sleeps.calls == [1.0, 2.0], "exponential, capped at max_delay, no sleep after last"


@pytest.mark.topic("py.http.retry_limits")
async def test_max_attempts_edge_cases():
    script = Script(503, 200)
    with pytest.raises(ex.RetriesExhausted):
        await run(script, policy=ex.RetryPolicy(max_attempts=1))
    assert len(script.requests) == 1
    with pytest.raises(ValueError):
        await run(Script(200), policy=ex.RetryPolicy(max_attempts=0))


@pytest.mark.topic("py.http.error_handling")
async def test_unexpected_exceptions_propagate():
    script = Script(KeyError("bug in handler"), 200)
    with pytest.raises(KeyError):
        await run(script)
    assert len(script.requests) == 1


@pytest.mark.topic("py.http.backoff", "py.http.retries")
async def test_default_sleep_and_rng_work():
    async with httpx.AsyncClient(transport=httpx.MockTransport(Script(500, 502, 200))) as client:
        request = client.build_request("GET", URL)
        policy = ex.RetryPolicy(base_delay=0.001, max_delay=0.01)
        response = await ex.send_with_retries(client, request, policy)
    assert response.status_code == 200
