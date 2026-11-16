"""
Exercise 22.3 - Mocking an LLM client with unittest.mock              level: ★★☆
topics: testing.mocking, testing.unittest_mock, testing.unit

Problem
-------
The code under test is GIVEN below (`LLMClient`, `TransientError`, `summarize`, `ask`).
LLMClient.complete would make a paid network call - your tests must never reach it.

Spec of summarize(client, text, *, model="fast-1", retries=2) -> str
  * empty / whitespace-only text -> ValueError, and the client is NOT called
  * calls client.complete(prompt, model=model, temperature=0.0) - exactly this call shape,
    with prompt = "Summarize in one sentence:\\n" + text
  * returns the reply with surrounding whitespace stripped
  * TransientError -> retry; at most `retries` extra attempts (retries + 1 calls in
    total), then the last TransientError propagates
  * any other exception propagates immediately (no retry)

Spec of ask(question) -> str
  * builds its OWN client: LLMClient() (a hard-wired dependency - see ex06 for the fix)
  * returns LLMClient().complete(f"Q: {question}\\nA:", model="smart-1").strip()

1. check_summarize(impl) -> None
   A mutation-tested suite (like ex01): must pass for the real `summarize` and fail
   (raise - a failing assert or any exception, like a red test) for each broken copy. Use Mock/create_autospec
   with spec=LLMClient, return_value, side_effect (a list -> one item per call; an
   exception instance -> raised), assert_called_once_with, call_count, assert_not_called.
   It must never call the real LLMClient.complete.

2. patched_llm(reply) - context manager. Patch the name `LLMClient` *in this module*
   (where `ask` looks it up - hint: patch(f"{__name__}.LLMClient", autospec=True)) so
   that ask() gets a mock instance whose complete() returns `reply`. Yield the mock
   INSTANCE (so callers can assert on .complete). The real class is restored on exit.

Interviewers look for
---------------------
* "Patch where it's looked up, not where it's defined": `from x import LLMClient` copies
  the name into the importing module; patching x.LLMClient wouldn't affect it.
* spec/autospec: a bare Mock accepts any attribute and any signature, so a typo
  (`client.complet(...)`) or a changed signature still "passes". autospec fails loudly.
* side_effect as a list to script retries: [TransientError(), "ok"].
* Mock the boundary (the network client), not the code under test.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock


class TransientError(Exception):
    """429 / 503 / timeout - worth retrying."""


class LLMClient:
    def complete(self, prompt: str, *, model: str, temperature: float = 0.7) -> str:
        raise RuntimeError("real network call made from a unit test!")


def summarize(client: LLMClient, text: str, *, model: str = "fast-1", retries: int = 2) -> str:
    """Code under test - correct implementation (given)."""
    if not text.strip():
        raise ValueError("empty text")
    prompt = "Summarize in one sentence:\n" + text
    for attempt in range(retries + 1):
        try:
            return client.complete(prompt, model=model, temperature=0.0).strip()
        except TransientError:
            if attempt == retries:
                raise
    raise AssertionError("unreachable")


def ask(question: str) -> str:
    """Code under test - correct implementation (given)."""
    client = LLMClient()
    return client.complete(f"Q: {question}\nA:", model="smart-1").strip()


def check_summarize(impl) -> None:
    raise NotImplementedError


@contextmanager
def patched_llm(reply: str) -> Iterator[Mock]:
    raise NotImplementedError
