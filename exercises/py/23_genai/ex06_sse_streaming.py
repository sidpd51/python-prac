"""
Exercise 23.6 - Streaming LLM responses: parsing Server-Sent Events   level: ★★☆
topics: genai.streaming, genai.async_llm_calls, genai.structured_data

Problem
-------
Chat APIs stream tokens as Server-Sent Events over HTTP. The network hands you arbitrary
byte chunks: a chunk can end in the middle of a line - or in the middle of a multi-byte
UTF-8 character.

    data: {"choices":[{"delta":{"role":"assistant"}}]}
    <blank line>
    data: {"choices":[{"delta":{"content":"Hel"}}]}
    <blank line>
    : keep-alive comment
    data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}
    <blank line>
    data: [DONE]

1. async iter_sse_data(chunks) -> AsyncIterator[str]
   chunks: AsyncIterable[bytes]. Yield the payload of every "data:" line, in order.
   * lines end with "\\n" or "\\r\\n"; a line may be split across chunks; decode UTF-8
     incrementally (codecs.getincrementaldecoder) - never chunk.decode() per chunk
   * payload = text after "data:", with ONE leading space removed if present
   * ignore blank lines, comment lines (starting with ":") and other fields (event:, id:,
     retry:) - one data line = one event in this exercise
   * payload "[DONE]" -> stop immediately: don't yield it, don't pull more chunks
   * a final line without a trailing newline is still processed at end of stream

2. async stream_text(chunks) -> AsyncIterator[str]
   On top of iter_sse_data: json.loads each payload and yield
   payload["choices"][0]["delta"]["content"] when it's a non-empty string (skip role-only
   deltas, null content, events with an empty choices list).
   A payload with an "error" key -> raise StreamError(payload["error"]["message"]).

3. async collect_stream(chunks, on_delta=None) -> StreamResult
   Consume stream_text, call on_delta(delta) for each delta (if given), and return
   StreamResult(text="".join(deltas), finish_reason=<last non-null
   choices[0]["finish_reason"] seen, or None>, n_deltas=<number of deltas>).
   (Hint: finish_reason is only visible at the payload level - share a small helper
   between stream_text and collect_stream that yields (delta, finish_reason) pairs.)

Interviewers look for
---------------------
* Streaming improves time-to-first-token (perceived latency), not total latency.
* Buffer bytes until a newline; decode incrementally - splitting "é" (2 bytes) across
  chunks makes per-chunk .decode() raise UnicodeDecodeError.
* Async generators give back-pressure for free: nothing is read until the consumer asks.
* Stop on [DONE]; surface mid-stream errors (the HTTP status was already 200!).
* Production: also handle client disconnects (cancellation closes the generator) and
  timeouts between chunks.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamResult:
    text: str
    finish_reason: str | None
    n_deltas: int


class StreamError(Exception):
    """The provider reported an error in the middle of the stream."""


async def iter_sse_data(chunks: AsyncIterable[bytes]) -> AsyncIterator[str]:
    raise NotImplementedError
    yield


async def stream_text(chunks: AsyncIterable[bytes]) -> AsyncIterator[str]:
    raise NotImplementedError
    yield


async def collect_stream(
    chunks: AsyncIterable[bytes], on_delta: Callable[[str], object] | None = None
) -> StreamResult:
    raise NotImplementedError
