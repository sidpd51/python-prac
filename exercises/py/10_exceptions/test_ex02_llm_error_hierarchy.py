import copy
import pickle

import pytest

from study.loader import load

ex = load(__file__, "ex02_llm_error_hierarchy")


@pytest.mark.topic("py.exceptions.hierarchy", "py.exceptions.custom")
@pytest.mark.parametrize(
    ("status", "cls"),
    [
        (401, "AuthError"),
        (403, "AuthError"),
        (400, "BadRequestError"),
        (404, "BadRequestError"),
        (422, "BadRequestError"),
        (429, "RateLimitError"),
        (408, "APITimeoutError"),
        (500, "ServerError"),
        (503, "ServerError"),
        (418, "LLMError"),
    ],
)
def test_status_mapping(status, cls):
    err = ex.error_from_response(status, {"error": {"message": "nope"}}, provider="openai")
    assert type(err) is getattr(ex, cls)
    assert isinstance(err, ex.LLMError)
    assert err.status == status
    assert err.message == "nope"
    assert err.provider == "openai"


@pytest.mark.topic("py.exceptions.hierarchy")
def test_context_length_is_a_bad_request():
    body = {
        "error": {
            "message": "too long",
            "code": "context_length_exceeded",
            "max_tokens": 8192,
            "requested_tokens": 9000,
        }
    }
    err = ex.error_from_response(400, body)
    assert type(err) is ex.ContextLengthError
    assert isinstance(err, ex.BadRequestError)
    assert (err.max_tokens, err.requested_tokens) == (8192, 9000)
    # generic handlers further up still work
    with pytest.raises(ex.BadRequestError):
        raise err


@pytest.mark.topic("py.exceptions.hierarchy", "py.exceptions.multiple")
def test_timeout_is_also_builtin_timeout():
    err = ex.error_from_response(408)
    assert isinstance(err, TimeoutError)
    with pytest.raises(TimeoutError):  # code that only knows builtins still catches it
        raise err


@pytest.mark.topic("py.exceptions.custom", "py.exceptions.best_practices")
def test_retryable_flags():
    flags = {s: ex.error_from_response(s).retryable for s in (400, 401, 408, 429, 500, 418)}
    assert flags == {400: False, 401: False, 408: True, 429: True, 500: True, 418: False}
    # class attribute, not per-instance state
    assert ex.RateLimitError.retryable is True
    assert ex.LLMError.retryable is False


@pytest.mark.topic("py.exceptions.custom")
@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "7"}, 7.0),
        ({"retry-after": "1.5"}, 1.5),
        ({"Retry-After": "soon"}, None),
        (None, None),
    ],
)
def test_retry_after_header(headers, expected):
    err = ex.error_from_response(429, {}, headers)
    assert err.retry_after == expected


@pytest.mark.topic("py.exceptions.custom")
def test_str_and_default_message():
    assert str(ex.error_from_response(500)) == "[llm 500] HTTP 500"
    assert str(ex.error_from_response(429, {"error": {"message": "slow down"}}, provider="openai")) == (
        "[openai 429] slow down"
    )
    assert str(ex.LLMError("boom", provider="anthropic")) == "[anthropic] boom"
    assert ex.error_from_response(500, {"error": None}).message == "HTTP 500"


@pytest.mark.topic("py.exceptions.custom", "py.exceptions.best_practices")
@pytest.mark.parametrize("dump", [lambda e: pickle.loads(pickle.dumps(e)), copy.copy, copy.deepcopy])
def test_errors_survive_pickle_and_copy(dump):
    rl = ex.RateLimitError("slow", retry_after=3.0, status=429, provider="openai")
    cl = ex.ContextLengthError("long", max_tokens=10, requested_tokens=12, status=400)
    base = ex.LLMError("x", status=1)
    for err in (rl, cl, base, ex.error_from_response(401)):
        clone = dump(err)
        assert type(clone) is type(err)
        assert clone.__dict__ == err.__dict__
        assert str(clone) == str(err)
    assert dump(rl).retry_after == 3.0
