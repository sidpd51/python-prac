"""
Exercise 17.2 - Status codes -> typed API exceptions               level: ★☆☆
topics: http.status_codes, http.error_handling, http.fundamentals, http.retries

Problem
-------
SDKs like openai/anthropic turn HTTP failures into a typed exception hierarchy so callers
can write `except RateLimitError:` instead of inspecting status codes everywhere.
The hierarchy below is given.

1. parse_retry_after(value, now=None) -> float | None
     Retry-After is either delta-seconds ("2", "0.5") or an HTTP-date
     ("Wed, 21 Oct 2015 07:28:00 GMT").
     * seconds -> float(value); negative -> None
     * HTTP-date -> seconds from `now` (default: datetime.now(UTC)) until that date,
       never below 0.0
     * None / empty / unparseable -> None
     (hint: email.utils.parsedate_to_datetime raises on garbage - catch TypeError/ValueError)

2. error_message(response) -> str
     Best human-readable message from an error body:
     * JSON {"error": {"message": "..."}}  -> that message
     * JSON {"error": "..."} or {"detail": "..."} or {"message": "..."} (checked in that
       order, value must be a str)
     * otherwise the stripped response text, cut to 200 characters
     * empty body -> f"HTTP {status_code}"

3. raise_for_api_error(response) -> httpx.Response
     * status < 400 -> return the response unchanged
     * otherwise raise (message=error_message(response), status_code, request_id from the
       "x-request-id" header or None):
         400, 422 -> BadRequestError       401 -> AuthenticationError
         403 -> PermissionDeniedError      404 -> NotFoundError
         409 -> ConflictError              429 -> RateLimitError (with retry_after parsed
                                                  from the Retry-After header)
         >= 500 -> InternalServerError     any other 4xx -> APIStatusError

4. send(client, method, url, **kwargs) -> httpx.Response
     * client.request(...), then raise_for_api_error
     * httpx.TimeoutException -> APITimeoutError; any other httpx.TransportError ->
       APIConnectionError; chain the original (`raise ... from exc`)

5. is_retryable(exc: APIError) -> bool
     * True for connection errors (incl. timeouts), RateLimitError, InternalServerError,
       and APIStatusError with status 408 or 409; False for everything else (400, 401, 403,
       404, 422 will fail again exactly the same way)

Interviewers look for
---------------------
* 4xx = the client's fault (fix the request; don't retry) - except 408/409/429;
  5xx = the server's fault (often transient, retry with backoff).
* 401 (who are you?) vs 403 (I know you, but no); 400 (malformed) vs 422 (well-formed but
  semantically invalid); 429 carries Retry-After.
* Keep the request id in the exception - it's what the provider's support asks for.
* Preserve the cause chain (`from exc`) so tracebacks show the transport error.
"""

from datetime import datetime
from typing import Any

import httpx


class APIError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, request_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id


class APIStatusError(APIError):
    """Any non-2xx response."""


class BadRequestError(APIStatusError): ...


class AuthenticationError(APIStatusError): ...


class PermissionDeniedError(APIStatusError): ...


class NotFoundError(APIStatusError): ...


class ConflictError(APIStatusError): ...


class RateLimitError(APIStatusError):
    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class InternalServerError(APIStatusError): ...


class APIConnectionError(APIError):
    """The request never got a response (DNS, refused, reset, timeout...)."""


class APITimeoutError(APIConnectionError): ...


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    raise NotImplementedError


def error_message(response: httpx.Response) -> str:
    raise NotImplementedError


def raise_for_api_error(response: httpx.Response) -> httpx.Response:
    raise NotImplementedError


def send(client: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response:
    raise NotImplementedError


def is_retryable(exc: APIError) -> bool:
    raise NotImplementedError
