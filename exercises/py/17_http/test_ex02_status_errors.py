from datetime import UTC, datetime

import httpx
import pytest

from study.loader import load

ex = load(__file__, "ex02_status_errors")

NOW = datetime(2015, 10, 21, 7, 27, 30, tzinfo=UTC)


@pytest.mark.topic("py.http.status_codes", "py.http.retries")
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2", 2.0),
        ("0.5", 0.5),
        (" 10 ", 10.0),
        ("0", 0.0),
        ("Wed, 21 Oct 2015 07:28:00 GMT", 30.0),
        ("Wed, 21 Oct 2015 07:00:00 GMT", 0.0),  # date in the past
        ("-3", None),
        ("soon", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_retry_after(value, expected):
    assert ex.parse_retry_after(value, now=NOW) == expected


@pytest.mark.topic("py.http.error_handling")
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"json": {"error": {"message": "Invalid model", "type": "invalid_request"}}}, "Invalid model"),
        ({"json": {"error": "quota exceeded"}}, "quota exceeded"),
        ({"json": {"detail": "Not authenticated"}}, "Not authenticated"),
        ({"json": {"message": "boom"}}, "boom"),
        ({"json": {"error": {"code": 7}}}, '{"error":{"code":7}}'),
        ({"text": "  <html>Bad Gateway</html>\n"}, "<html>Bad Gateway</html>"),
        ({"text": "x" * 500}, "x" * 200),
        ({}, "HTTP 502"),
    ],
)
def test_error_message(kwargs, expected):
    assert ex.error_message(httpx.Response(502, **kwargs)) == expected


@pytest.mark.topic("py.http.status_codes", "py.http.error_handling")
@pytest.mark.parametrize(
    ("status", "exc_name"),
    [
        (400, "BadRequestError"),
        (422, "BadRequestError"),
        (401, "AuthenticationError"),
        (403, "PermissionDeniedError"),
        (404, "NotFoundError"),
        (409, "ConflictError"),
        (429, "RateLimitError"),
        (500, "InternalServerError"),
        (503, "InternalServerError"),
        (418, "APIStatusError"),
    ],
)
def test_status_mapping(status, exc_name):
    response = httpx.Response(status, json={"error": {"message": "nope"}}, headers={"x-request-id": "req_1"})
    with pytest.raises(ex.APIStatusError) as info:
        ex.raise_for_api_error(response)
    assert type(info.value).__name__ == exc_name
    assert (info.value.status_code, info.value.request_id, info.value.message) == (status, "req_1", "nope")
    assert str(info.value) == "nope"


@pytest.mark.topic("py.http.status_codes")
@pytest.mark.parametrize("status", [200, 201, 204, 304])
def test_success_passes_through(status):
    response = httpx.Response(status)
    assert ex.raise_for_api_error(response) is response


@pytest.mark.topic("py.http.status_codes", "py.http.rate_limiting")
def test_rate_limit_carries_retry_after():
    response = httpx.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"})
    with pytest.raises(ex.RateLimitError) as info:
        ex.raise_for_api_error(response)
    assert info.value.retry_after == 7.0
    assert info.value.request_id is None
    with pytest.raises(ex.RateLimitError) as info:
        ex.raise_for_api_error(httpx.Response(429))
    assert info.value.retry_after is None


def _client(handler):
    return httpx.Client(base_url="https://api.example.com", transport=httpx.MockTransport(handler))


@pytest.mark.topic("py.http.error_handling", "py.http.fundamentals")
def test_send_success_and_status_error():
    def handler(request):
        if request.url.path == "/ok":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = _client(handler)
    assert ex.send(client, "POST", "/ok", json={"a": 1}).json() == {"ok": True}
    with pytest.raises(ex.AuthenticationError):
        ex.send(client, "GET", "/private")


@pytest.mark.topic("py.http.error_handling", "py.http.request_timeouts")
@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (httpx.ReadTimeout("read timed out"), "APITimeoutError"),
        (httpx.ConnectTimeout("connect timed out"), "APITimeoutError"),
        (httpx.ConnectError("connection refused"), "APIConnectionError"),
        (httpx.RemoteProtocolError("peer closed"), "APIConnectionError"),
    ],
)
def test_send_maps_transport_errors(raised, expected):
    def handler(request):
        raise raised

    with pytest.raises(ex.APIConnectionError) as info:
        ex.send(_client(handler), "GET", "/x")
    assert type(info.value).__name__ == expected
    assert info.value.__cause__ is raised
    assert info.value.status_code is None


@pytest.mark.topic("py.http.retries", "py.http.error_handling")
def test_is_retryable():
    retryable = [
        ex.APIConnectionError("reset"),
        ex.APITimeoutError("slow"),
        ex.RateLimitError("429", status_code=429),
        ex.InternalServerError("502", status_code=502),
        ex.APIStatusError("timeout", status_code=408),
        ex.ConflictError("locked", status_code=409),
    ]
    final = [
        ex.BadRequestError("bad", status_code=400),
        ex.BadRequestError("invalid", status_code=422),
        ex.AuthenticationError("key", status_code=401),
        ex.PermissionDeniedError("no", status_code=403),
        ex.NotFoundError("gone", status_code=404),
        ex.APIStatusError("teapot", status_code=418),
    ]
    assert [ex.is_retryable(e) for e in retryable] == [True] * len(retryable)
    assert [ex.is_retryable(e) for e in final] == [False] * len(final)
