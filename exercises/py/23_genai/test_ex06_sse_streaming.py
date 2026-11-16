import inspect
import json

import pytest

from study.loader import load

ex = load(__file__, "ex06_sse_streaming")


def event(content=None, *, role=None, finish=None):
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n"


STREAM = (
    event(role="assistant")
    + event("Bon")
    + ": keep-alive\n\n"
    + event("jour, ")
    + "event: message\nid: 7\n"
    + event("café ☕")
    + event(None)  # null/missing content
    + event("!", finish="stop")
    + "data: [DONE]\n\n"
)


class Source:
    """Async byte source that records how many chunks were pulled."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.pulled = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.pulled >= len(self.chunks):
            raise StopAsyncIteration
        self.pulled += 1
        return self.chunks[self.pulled - 1]


def split_every(data: bytes, n: int):
    return [data[i : i + n] for i in range(0, len(data), n)]


async def agen_of(obj):
    """The generated stub of an async generator is a plain coroutine function (no `yield`);
    awaiting it surfaces its NotImplementedError (-> TODO) instead of a confusing TypeError."""
    if inspect.iscoroutine(obj):
        await obj
    return obj


async def collect(agen):
    return [x async for x in await agen_of(agen)]


@pytest.mark.topic("py.genai.streaming")
async def test_iter_sse_data_basic():
    raw = (
        b'data: {"a": 1}\n\ndata:{"b": 2}\n\n: comment\nevent: x\nid: 3\nretry: 100\n\ndata:  two spaces\n\n'
    )
    assert await collect(ex.iter_sse_data(Source([raw]))) == ['{"a": 1}', '{"b": 2}', " two spaces"]


@pytest.mark.topic("py.genai.streaming")
@pytest.mark.parametrize("size", [1, 2, 3, 7, 64, 10_000])
async def test_iter_sse_data_any_chunking(size):
    # chunks split lines AND multi-byte UTF-8 characters at arbitrary points
    raw = STREAM.encode()
    payloads = await collect(ex.iter_sse_data(Source(split_every(raw, size))))
    assert len(payloads) == 6
    assert json.loads(payloads[3])["choices"][0]["delta"]["content"] == "café ☕"


@pytest.mark.topic("py.genai.streaming")
async def test_crlf_and_unterminated_last_line():
    raw = b"data: one\r\n\r\ndata: two\r\n\r\ndata: last"
    assert await collect(ex.iter_sse_data(Source(split_every(raw, 4)))) == ["one", "two", "last"]


@pytest.mark.topic("py.genai.streaming", "py.genai.async_llm_calls")
async def test_done_stops_without_pulling_more():
    chunks = [b"data: a\n\n", b"data: [DONE]\n\ndata: junk-in-same-chunk\n\n", b"data: never\n\n", b"x"]
    src = Source(chunks)
    assert await collect(ex.iter_sse_data(src)) == ["a"]
    assert src.pulled == 2, "must stop reading the network after [DONE]"


@pytest.mark.topic("py.genai.streaming")
async def test_iter_sse_data_is_lazy():
    src = Source([b"data: 1\n\n", b"data: 2\n\n", b"data: 3\n\n"])
    agen = await agen_of(ex.iter_sse_data(src))
    assert src.pulled == 0
    assert await anext(agen) == "1"
    assert src.pulled == 1


@pytest.mark.topic("py.genai.streaming", "py.genai.structured_data")
async def test_stream_text_yields_content_deltas():
    deltas = await collect(ex.stream_text(Source(split_every(STREAM.encode(), 5))))
    assert deltas == ["Bon", "jour, ", "café ☕", "!"]


@pytest.mark.topic("py.genai.streaming")
async def test_stream_text_skips_empty_choices_and_empty_content():
    raw = 'data: {"choices": []}\n\n' + event("") + event("x") + "data: [DONE]\n\n"
    assert await collect(ex.stream_text(Source([raw.encode()]))) == ["x"]


@pytest.mark.topic("py.genai.streaming", "py.genai.async_llm_calls")
async def test_stream_error_mid_stream():
    raw = event("partial ") + 'data: {"error": {"message": "overloaded", "type": "server_error"}}\n\n'
    received = []
    with pytest.raises(ex.StreamError, match="overloaded"):
        async for delta in await agen_of(ex.stream_text(Source([raw.encode()]))):
            received.append(delta)
    assert received == ["partial "], "deltas before the error are still delivered"


@pytest.mark.topic("py.genai.streaming")
async def test_collect_stream():
    seen = []
    result = await ex.collect_stream(Source(split_every(STREAM.encode(), 3)), on_delta=seen.append)
    assert result == ex.StreamResult(text="Bonjour, café ☕!", finish_reason="stop", n_deltas=4)
    assert seen == ["Bon", "jour, ", "café ☕", "!"]


@pytest.mark.topic("py.genai.streaming")
async def test_collect_stream_without_finish_reason_or_callback():
    raw = event("a") + event("b")  # connection closed without [DONE]
    result = await ex.collect_stream(Source([raw.encode()]))
    assert result == ex.StreamResult(text="ab", finish_reason=None, n_deltas=2)


@pytest.mark.topic("py.genai.streaming")
async def test_finish_reason_on_contentless_event():
    raw = event("hi") + event(None, finish="length") + "data: [DONE]\n\n"
    result = await ex.collect_stream(Source([raw.encode()]))
    assert result.finish_reason == "length" and result.text == "hi"
