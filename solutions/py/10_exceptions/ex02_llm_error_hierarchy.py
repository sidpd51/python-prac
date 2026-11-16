"""
Exercise 10.2 - Exception hierarchy for an LLM client            level: ★★☆
topics: exceptions.custom, exceptions.hierarchy, exceptions.raising,
        exceptions.best_practices

Problem
-------
Build the error types an LLM SDK exposes, and map HTTP responses onto them.

    LLMError(Exception)                      retryable = False
    ├── AuthError                            401, 403
    ├── BadRequestError                      400, 404, 409, 422
    │   └── ContextLengthError               400 with error code "context_length_exceeded"
    ├── RateLimitError                       429                     retryable = True
    ├── ServerError                          >= 500                  retryable = True
    └── APITimeoutError(LLMError, TimeoutError)   408               retryable = True

1. LLMError.__init__(self, message: str, *, status: int | None = None,
                     provider: str | None = None)
     * attributes .message .status .provider ; class attribute `retryable`
     * str(err): "[openai 429] slow down"  /  without status: "[openai] slow down";
       provider defaults to "llm" in the text:  "[llm 500] boom"
   RateLimitError.__init__(message, *, retry_after: float | None = None, **kw)  -> .retry_after
   ContextLengthError.__init__(message, *, max_tokens: int | None = None,
                               requested_tokens: int | None = None, **kw)
   (the classes with nothing extra to store need no __init__ at all)

2. Errors must survive pickle / copy (they cross process boundaries in
   multiprocessing, concurrent.futures, Celery...). BaseException pickles as
   `cls(*self.args)` + `__dict__`: think about what `self.args` contains for your classes,
   and make pickle.loads(pickle.dumps(err)) give back an equal-looking error of the same
   class with every attribute intact (hint: one `__reduce__` on LLMError is enough).

3. error_from_response(status: int, body: Mapping[str, Any] | None = None,
                       headers: Mapping[str, str] | None = None,
                       provider: str | None = None) -> LLMError
     * RETURNS the error (the caller decides to raise it)
     * body shape: {"error": {"message": str, "code": str, "max_tokens": int,
                              "requested_tokens": int}} - any part may be missing;
       a missing message becomes f"HTTP {status}"
     * 429: retry_after from the "Retry-After" header (header names are
       case-insensitive; seconds as a number; unparsable -> None)
     * unknown 4xx (e.g. 418) -> plain LLMError; status is always stored

Interviewers look for
---------------------
* One base class so callers can `except LLMError`, specific subclasses so they can react
  differently (re-auth vs back off vs truncate the prompt).
* Retryability as data on the class, not a status-code if-chain repeated in every caller.
* Multiple inheritance from a builtin (TimeoutError) so generic code that catches the
  builtin still works.
* Custom exceptions with extra required init args break pickling - a production bug that
  only shows up when errors cross a process boundary.
"""

from collections.abc import Mapping
from typing import Any


class LLMError(Exception):
    retryable: bool = False

    def __init__(self, message: str, *, status: int | None = None, provider: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.provider = provider

    def __str__(self) -> str:
        where = self.provider or "llm"
        if self.status is not None:
            where = f"{where} {self.status}"
        return f"[{where}] {self.message}"

    def __reduce__(self) -> tuple[Any, ...]:
        # args holds only the message; every other attribute lives in __dict__ and is
        # restored by BaseException.__setstate__ - so subclasses need no extra code.
        return (type(self), (self.message,), self.__dict__)


class AuthError(LLMError):
    pass


class BadRequestError(LLMError):
    pass


class ContextLengthError(BadRequestError):
    def __init__(
        self,
        message: str,
        *,
        max_tokens: int | None = None,
        requested_tokens: int | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens


class RateLimitError(LLMError):
    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class ServerError(LLMError):
    retryable = True


class APITimeoutError(LLMError, TimeoutError):
    retryable = True


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None
    lowered = {k.lower(): v for k, v in headers.items()}
    try:
        return float(lowered["retry-after"])
    except (KeyError, ValueError):
        return None


def error_from_response(
    status: int,
    body: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    provider: str | None = None,
) -> LLMError:
    err = (body or {}).get("error") or {}
    message = err.get("message") or f"HTTP {status}"
    kw: dict[str, Any] = {"status": status, "provider": provider}
    if status in (401, 403):
        return AuthError(message, **kw)
    if status == 429:
        return RateLimitError(message, retry_after=_retry_after(headers), **kw)
    if status == 408:
        return APITimeoutError(message, **kw)
    if status == 400 and err.get("code") == "context_length_exceeded":
        return ContextLengthError(
            message,
            max_tokens=err.get("max_tokens"),
            requested_tokens=err.get("requested_tokens"),
            **kw,
        )
    if status in (400, 404, 409, 422):
        return BadRequestError(message, **kw)
    if status >= 500:
        return ServerError(message, **kw)
    return LLMError(message, **kw)
