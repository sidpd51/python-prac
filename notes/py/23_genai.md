# Python for Generative AI Applications

Most GenAI engineering is ordinary Python applied to awkward inputs and unreliable services: messy
text, JSON that is almost JSON, models limited by tokens rather than characters, APIs that rate-limit
and time out, and secrets that must never leak. This note collects the building blocks of an LLM/RAG
pipeline in the order data flows through it: clean, extract, tokenise, chunk, batch, call concurrently,
retry, validate, stream, configure, observe. Interviews test whether you can build these *robustly*,
so every section ends on the failure modes.

## JSON and structured data
<!-- topics: structured_data -->

JSON is the wire format of every LLM API, tool call and JSONL dataset. `json.loads` parses a string
into dicts/lists/str/int/float/bool/None; `json.dumps` goes back.

```python
import json

print(json.dumps({"a": "é", "b": None}))                     # prints: {"a": "\u00e9", "b": null}
print(json.dumps({"a": "é"}, ensure_ascii=False))            # prints: {"a": "é"}
print(json.dumps({"s": {1, 2}}, default=sorted))             # prints: {"s": [1, 2]}
print(json.loads('{"a": 1, "a": 2}'))                        # prints: {'a': 2}  (last duplicate wins)
try:
    json.loads("{'a': 1}")                                   # single quotes are not JSON
except json.JSONDecodeError as e:
    print(e)   # prints: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
```

* `dumps` fails on `set`, `datetime`, `Decimal`, dataclasses (pass `default=`); JSON has no tuples or int keys.
* **JSONL** (one object per line) is the standard for big datasets: stream it line by line, and one bad line
  (`json.JSONDecodeError`, with `.msg`) does not kill the file. Skip blank lines, count and log failures.

> **Gotcha** `json.loads` returns plain dicts: wrong types surface far from the cause. Validate at the boundary (see structured output).

## Text normalisation and cleaning
<!-- topics: text_processing, normalization -->

The same visible text can have different code points, so equality, hashing, dedup and search silently
fail. `é` can be one code point (U+00E9) or `e` plus a combining accent (U+0301).

```python
import unicodedata

a, b = "caf\u00e9", "cafe\u0301"
print(a == b, len(a), len(b))                                  # prints: False 4 5
print(unicodedata.normalize("NFC", b) == a)                    # prints: True
print(unicodedata.normalize("NFKC", "\ufb01ne \u2460 \uff21\uff11\u00a0x"))      # prints: fine 1 A1 x
```

| form | does | use for |
|---|---|---|
| NFC | compose (canonical) | storing text, comparing |
| NFD | decompose | stripping accents |
| NFKC | compose + replace *compatibility* characters (ligatures, full-width, circled digits, nbsp) | search keys, dedup, preprocessing before embedding |

NFKC is lossy (`①` becomes `1`), so keep the original text and normalise a copy for keys and matching.

Other tools: `casefold()` (not `lower()`: `"Straße".casefold()` is `"strasse"`) for case-insensitive matching;
NFD plus dropping `unicodedata.combining` characters strips accents (`café` becomes `cafe`).

A typical clean-up order: decode bytes as UTF-8 (`errors="replace"` for dirty data), NFKC, remove
zero-width/control characters, map curly quotes and dashes, collapse whitespace, then strip.

> **Gotcha** Cleaning shifts offsets: keep the original text (or a mapping) if you need citations or highlights.

## Regular expressions for extraction
<!-- topics: regex -->

Regex is the fast, deterministic tool for patterns with structure: ids, emails, dates, citations,
code fences, PII. Know the few features that matter.

```python
import re

text = "Mail ada@example.com or bob.smith@corp.io; see [1] and [12]. Paid $1,299.50 on 2024-03-05."
print(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text))   # prints: ['ada@example.com', 'bob.smith@corp.io']
print(re.findall(r"\[(\d+)\]", text))                  # prints: ['1', '12']   (findall returns the group)
m = re.search(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})", text)
print(m.groupdict())                                   # prints: {'y': '2024', 'm': '03', 'd': '05'}
print(re.search(r"<.+>", "<a><b>").group())            # prints: <a><b>     (greedy)
print(re.search(r"<.+?>", "<a><b>").group())           # prints: <a>        (lazy)
print(re.sub(r"\b\d{3}-\d{4}\b", "<PHONE>", "call 555-1234 now"))   # prints: call <PHONE> now
```

* `match` anchors at the start, `search` finds anywhere, `fullmatch` needs the whole string; `findall` returns
  the groups if the pattern has any; `finditer` gives spans. Use raw strings, named groups, `re.compile`, flags `re.I`/`re.S`/`re.M`.
* Greedy `.+` takes as much as possible, lazy `.+?` as little: most "grabbed too much" bugs.
* To pull JSON out of a chatty reply, match a code fence: `` re.search(r"```(?:json)?\s*(.*?)```", reply, re.S) ``.

> **Gotcha** Nested structures (balanced braces, HTML) are not regular: use a real parser. Beware catastrophic backtracking (`(a+)+$`) on untrusted input.

## Tokenization fundamentals
<!-- topics: tokenization -->

LLMs do not read characters or words; they read **tokens**: integer ids for pieces of text from a
fixed vocabulary (typically 30k-200k entries). Context windows, prices and rate limits are all in
tokens. A token is usually a common word, a word fragment, punctuation, or even a single byte for rare
characters.

**BPE (byte-pair encoding) intuition.** Start with single characters (or bytes). Repeatedly find the most
frequent adjacent pair in the training corpus and merge it into a new symbol. Frequent words end up as
one token, rare words split into known pieces, and any text can be encoded because the base is bytes.

```python
from collections import Counter

words = {tuple("low"): 5, tuple("lower"): 2, tuple("newest"): 6}   # word -> frequency, as characters
pairs = Counter()
for w, c in words.items():
    for pair in zip(w, w[1:]):
        pairs[pair] += c
print(pairs.most_common(2))   # prints: [(('w', 'e'), 8), (('l', 'o'), 7)]
```

Merge the top pair into a new symbol (`we`), re-count, repeat until the vocabulary is full.

Real tokenizers (tiktoken, sentencepiece) store the learned merges and are **model-specific**: token counts differ per model.

**Why tokens are not words:**

* English averages roughly 4 characters or 0.75 words per token, a rule of thumb only. Code, numbers,
  non-English text and emoji cost far more tokens per character.
* Bytes matter: `"日本語"` is 3 characters but 9 UTF-8 bytes, so often several tokens.
* Leading spaces are part of tokens (` the` differs from `the`).

```python
print(len("héllo"), len("héllo".encode()), len("日本語"), len("日本語".encode()))   # prints: 5 6 3 9
```

Practical rules: **count with the target model's tokenizer** before sending, reserve room for the output
(`context = prompt + completion`), truncate or summarise history to a token budget, and use a cheap
estimate (`len(text) / 4`) only for rough guards, never for hard limits.

## Document processing and chunking
<!-- topics: document_processing, chunking -->

RAG retrieves *chunks*, not documents: documents are too large for the context window and one embedding
per whole document is too blurry. Chunking splits text into pieces small enough to embed and precise
enough to retrieve. Document processing is everything before that: load (PDF, HTML, Markdown),
extract text, normalise, keep structure (titles, pages) as metadata.

| strategy | idea | trade-off |
|---|---|---|
| fixed-size + overlap | every N chars/tokens, overlapping | simplest; cuts sentences, ignores structure |
| recursive | split on `\n\n`, then `\n`, then sentences, then words until it fits | respects structure, the usual default |
| structure-aware | split by headings/sections/code blocks | best coherence; needs a parser |
| semantic | split where embedding similarity drops | adapts to meaning; extra cost, less predictable |

Small chunks give precise matches but lose context; large chunks keep context but dilute the embedding and burn prompt budget. **Overlap** (10-15%) stops an answer being cut at a boundary, at the cost of duplicate storage.

```python
def fixed(text, size, overlap):
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]

print(fixed("Alpha beta gamma. Delta epsilon zeta. Eta theta iota.", 20, 5))
# prints: ['Alpha beta gamma. De', 'a. Delta epsilon zet', 'n zeta. Eta theta io', 'ta iota.']
```

Fixed windows split `Delta` and `zeta` mid-word. A **recursive** splitter tries the coarsest separator first
(`"\n\n"`), packs pieces up to the size limit, and only falls back to `"\n"`, `". "`, then `" "` for pieces
still too big, so cuts land on paragraph and sentence boundaries. Always measure size in **tokens** when the
limit is a token limit, and attach metadata (doc id, source, page, offsets) to each chunk for citations and filters.

> **Interview** "How do you choose a chunking strategy?" Start with recursive 300-800 tokens and ~10%
> overlap, build a golden question set, and compare retrieval recall@k and answer quality across settings.
> Tables, code and PDFs usually need structure-aware handling.

## Batching
<!-- topics: batching -->

Embedding and LLM APIs accept many inputs per request, and per-request overhead (latency, rate-limit
cost) dominates small calls. Batching groups items so you make fewer, fuller requests. Python 3.12+
has `itertools.batched`.

```python
from itertools import batched

print(list(batched(range(7), 3)))      # prints: [(0, 1, 2), (3, 4, 5), (6,)]

def batch_by_chars(texts, max_items, max_chars):
    cur, size = [], 0
    for t in texts:
        if cur and (len(cur) == max_items or size + len(t) > max_chars):
            yield cur
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    if cur:
        yield cur

print(list(batch_by_chars(["aaa", "bb", "cccc", "d", "eeeee"], 3, 6)))
# prints: [['aaa', 'bb'], ['cccc', 'd'], ['eeeee']]
```

Batch on **both** limits (items and total tokens), keep the original index so results map back, flush the
last partial batch (classic off-by-one), and send one oversized item alone rather than dropping it. `batched`
returns tuples and needs `n >= 1`.

## Async LLM calls and concurrent embeddings
<!-- topics: async_llm_calls, concurrent_embeddings -->

LLM calls are I/O-bound waits of 1-30 seconds. Sequentially, 100 calls take 100 times as long; with
`asyncio` they overlap. But unlimited concurrency triggers 429s and exhausts connections, so you bound it
with a **semaphore**.

```python
import asyncio

async def fake_embed(batch):                    # stands in for an HTTP call
    await asyncio.sleep(0.01)
    return [[float(len(t))] for t in batch]

async def embed_all(texts, batch_size=2, limit=2):
    sem = asyncio.Semaphore(limit)
    peak = inflight = 0

    async def one(batch):
        nonlocal inflight, peak
        async with sem:                         # at most `limit` inside at once
            inflight += 1
            peak = max(peak, inflight)
            try:
                return await fake_embed(batch)
            finally:
                inflight -= 1

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(*(one(b) for b in batches))   # order preserved
    return [v for r in results for v in r], peak

vectors, peak = asyncio.run(embed_all(["a", "bb", "ccc", "dddd", "eeeee"]))
print(vectors, peak)   # prints: [[1.0], [2.0], [3.0], [4.0], [5.0]] 2
```

Key points:

* `asyncio.gather` returns results **in input order**, regardless of completion order, so vectors stay
  aligned with texts.
* Share one `httpx.AsyncClient` (connection pooling). `gather(..., return_exceptions=True)` stops one failure
  from cancelling the rest; you then inspect each result.
* Blocking calls (`requests.get`, `time.sleep`, a CPU-heavy tokenizer) freeze the whole loop: use async
  clients, `asyncio.to_thread`, or a process pool.
* A semaphore bounds concurrency, not rate: 10 concurrent calls can still exceed 100 requests/minute.

> **Gotcha** 10,000 tasks at once means 10,000 live coroutines and responses: process in windows or use a queue with N workers.

## Rate limits and retry with backoff
<!-- topics: rate_limits, retry_backoff -->

Providers enforce limits in requests per minute, tokens per minute and concurrent requests. Exceeding
them returns HTTP **429** (sometimes with a `Retry-After` header); server trouble returns 500/502/503/529;
networks time out. Transient errors deserve a retry; permanent ones (400 bad request, 401 auth, 404) never do.

**Exponential backoff with jitter:** wait `base * 2**attempt`, capped, with randomness, so many clients
that failed together do not retry in lock-step ("thundering herd").

```python
import asyncio, random

class RateLimit(Exception):
    def __init__(self, retry_after=None):
        self.retry_after = retry_after

async def retry(fn, *, attempts=4, base=0.5, cap=8.0, sleep=asyncio.sleep, rng=random.random):
    for i in range(attempts):
        try:
            return await fn()
        except RateLimit as e:
            if i == attempts - 1:
                raise                                   # out of attempts: surface it
            delay = e.retry_after if e.retry_after is not None else rng() * min(cap, base * 2**i)
            print(f"attempt {i + 1} failed, sleeping {delay:.2f}")
            await sleep(delay)

calls = 0
async def flaky():
    global calls
    calls += 1
    if calls < 3:
        raise RateLimit()
    return "ok"

async def no_sleep(delay):
    pass

print(asyncio.run(retry(flaky, sleep=no_sleep, rng=lambda: 1.0)))
# prints: attempt 1 failed, sleeping 0.50
# prints: attempt 2 failed, sleeping 1.00
# prints: ok
```

Backoff schedule: 0.5, 1, 2, 4, 8, 8 ... seconds (capped), each multiplied by a random factor in real use.

Rules: retry only retryable errors with a cap on attempts, honour `Retry-After`, make `sleep` and `rng`
injectable (instant, deterministic tests), retry only idempotent calls, add a client-side token bucket to
stay *under* the limit, and do not retry at several layers at once (attempts multiply).

## Structured output validation and repair
<!-- topics: structured_output -->

Prompting for JSON is a request, not a guarantee. Output can be fenced, wrapped in prose, truncated,
have the wrong types, or invent fields. The robust pattern is **extract, parse, validate, repair or retry**,
with pydantic (v2) as the validator.

```python
from pydantic import BaseModel, Field, ValidationError

class Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    tags: list[str] = []

print(Person.model_validate_json('{"name":"Ada","age":"36"}'))   # prints: name='Ada' age=36 tags=[]
try:
    Person.model_validate_json('{"name":"Ada","age":-1}')
except ValidationError as e:
    err = e.errors()[0]
    print(e.error_count(), err["loc"], err["type"])   # prints: 1 ('age',) greater_than_equal
print(sorted(Person.model_json_schema()["required"]))  # prints: ['age', 'name']
```

Notes: pydantic **coerces** in the default (lax) mode (`"36"` becomes `36`); use `strict=True` or
`ConfigDict(strict=True)` when you want a type error instead. `model_json_schema()` gives the JSON Schema you
can pass to providers that support schema-constrained output or tool/function definitions.

The **repair loop** feeds the validation error back to the model:

```python
import json
from pydantic import BaseModel, ValidationError

class Score(BaseModel):
    score: int
    reason: str

def extract_json(raw):                  # tolerate fences and chatter around the object
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end + 1] if start != -1 and end > start else raw

def get_structured(llm, prompt, max_repairs=2):
    messages = prompt
    for attempt in range(max_repairs + 1):
        raw = llm(messages)
        try:
            return Score.model_validate_json(extract_json(raw))
        except ValidationError as e:
            messages = f"{prompt}\nYour last answer was invalid: {e.errors()[0]['msg']}. Return only JSON."
    raise ValueError("no valid output after repairs")

replies = iter(['Here: {"score": "high"}', '```json\n{"score": 4, "reason": "clear"}\n```'])
print(get_structured(lambda m: next(replies), "Rate this."))
# prints: score=4 reason='clear'
```

Cap repairs (they cost tokens and latency), log failure types, prefer provider-native structured output or
tool calling where available, and raise `max_tokens` for truncated JSON instead of repairing it.

## Streaming responses and SSE
<!-- topics: streaming -->

Streaming returns tokens as they are generated, so users see output immediately (lower
*time-to-first-token*). Chat APIs use **Server-Sent Events**: a long-lived HTTP response with
`Content-Type: text/event-stream` whose body is `data: {...}` events separated by a blank line (`\n\n`), ending with `data: [DONE]` on many APIs.

The trap: network chunks arrive at **arbitrary boundaries**, in the middle of a line or even an event. You
must buffer and split on the `\n\n` terminator, not assume one chunk equals one event.

```python
import json

def sse_events(chunks):
    buf = ""
    for chunk in chunks:
        buf += chunk
        while "\n\n" in buf:
            raw, buf = buf.split("\n\n", 1)
            data = [ln[5:].lstrip() for ln in raw.split("\n") if ln.startswith("data:")]
            if data:                                   # ignore comments / empty events
                yield "\n".join(data)                  # multi data: lines are joined by newline

stream = ['data: {"t":"He"}\n\nda', 'ta: {"t":"llo"}\n', '\ndata: [DONE]\n\n']   # split mid-event
out = []
for d in sse_events(stream):
    if d == "[DONE]":
        break
    out.append(json.loads(d)["t"])
print("".join(out))   # prints: Hello
```

With `httpx` you iterate lines from a streaming response without loading it all. `MockTransport` lets you
test it offline:

```python
import httpx

def handler(request):
    body = 'data: {"t":"A"}\n\ndata: [DONE]\n\n'
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

with httpx.Client(transport=httpx.MockTransport(handler)) as client:
    with client.stream("POST", "http://llm/v1/chat") as resp:
        print([ln for ln in resp.iter_lines() if ln])   # prints: ['data: {"t":"A"}', 'data: [DONE]']
```

(`httpx.AsyncClient` with `aiter_lines()` is the async twin.) Production: stream with generators, keep partial
text if the connection drops, set read timeouts, stop on the terminal event (usage often arrives last).

## Environment variables, secrets and configuration
<!-- topics: env_vars, secrets, config -->

**Twelve-factor config**: settings that differ between environments (keys, URLs, model names, timeouts)
come from environment variables, not from code. Read them with `os.environ`:

```python
import os

os.environ["MODEL"] = "small"
print(os.environ.get("NOPE", "dflt"), os.getenv("NOPE"))   # prints: dflt None
try:
    os.environ["NOPE"]
except KeyError as e:
    print("KeyError", e)                                   # prints: KeyError 'NOPE'
```

```

Env vars are always **strings**: parse numbers and booleans explicitly (`bool("0")` is `True`). Locally, a `.env` file (never
committed; listed in `.gitignore`) loaded by `python-dotenv` or pydantic-settings fills the environment.
In production the platform injects them from a secret store (cloud secret manager, Kubernetes secrets).

**pydantic-settings** gives typed, validated config that **fails at startup** if something is missing,
instead of at the first request:

```python
import os
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_")
    api_key: SecretStr
    timeout_s: float = 30.0
    debug: bool = False

os.environ.update(APP_API_KEY="sk-secret-123", APP_TIMEOUT_S="5", APP_DEBUG="true")
st = Settings()
print(st)                                  # prints: api_key=SecretStr('**********') timeout_s=5.0 debug=True
print(st.api_key.get_secret_value()[:3])   # prints: sk-
```

Without `APP_API_KEY` set, `Settings()` raises a `ValidationError` (type `missing`) at startup.

```

**Secrets rules** (a favourite interview topic):

* Never hard-code or commit keys; if one leaks into git history, **rotate it**.
* Wrap in `SecretStr` so `print`/`repr`/logs show `**********`; call `get_secret_value()` only at the point of use.
* Never put keys in URLs or error messages; send them in headers. One least-privilege key per service and environment.
* Load settings once and inject them (cache with `functools.lru_cache`) so tests can override them.

## Logging and error tracking in pipelines
<!-- topics: logging_tracking -->

A pipeline that processes 100,000 documents needs to tell you *which* ones failed and *why*, without
stopping and without leaking data. Use the `logging` module (levels, `getLogger(__name__)`), plus structure.

* Log **per item context** (doc id, stage, attempt, latency, tokens, model) and end-of-run counters
  (`processed=99872 failed=8`). Catch per item, record, continue; fail the run above a threshold.
* `logger.exception(...)` keeps the traceback; error trackers (Sentry) group them; add a request id.
* **Redact** secrets and PII with a logging `Filter`: the cheapest leak is a debug log of a request.

```python
import logging, re

class Redact(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r"sk-[A-Za-z0-9]+", "sk-***", str(record.msg))
        return True

rec = logging.LogRecord("p", 30, "f", 1, "key sk-abc123 failed", None, None)
Redact().filter(rec)
print(rec.msg)   # prints: key sk-*** failed
```

> **Gotcha** The filter is a safety net; the real fix is never putting the secret in the message.

## Memory-efficient pipelines and reusable packages
<!-- topics: pipelines, packages -->

A corpus rarely fits in memory, so build the pipeline from generators: each stage pulls one record from the
previous, so memory stays flat (O(batch), not O(corpus)) and you can stop early.

```python
import json, pathlib, tempfile
from itertools import batched

path = pathlib.Path(tempfile.mkdtemp()) / "a.jsonl"
path.write_text("\n".join(json.dumps({"id": i, "text": f"doc {i} " * 3}) for i in range(5)) + "\n", encoding="utf-8")

def read(p):                                    # stage 1: stream lines, never read_text()
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def embed(batches):                             # stage 2: one (fake) API call per batch
    for b in batches:
        yield [(r["id"], len(r["text"])) for r in b]

for out in embed(batched(read(path), 2)):
    print(out)
# prints: [(0, 18), (1, 18)]
# prints: [(2, 18), (3, 18)]
# prints: [(4, 18)]
```

Habits: iterate files line by line, write results incrementally so a crash loses little, make jobs
**resumable** (skip done ids), and never build a list just to count.

**Reusable packages.** Once code is shared between notebooks, services and teammates, make it a package:

* **src layout** (`src/mylib/`, `pyproject.toml`, `tests/`, `py.typed`) makes tests run against the installed
  package; develop with `pip install -e .`. `__init__.py`/`__all__` define the public API.
* Libraries take collaborators (client, settings) as arguments and never configure logging or read env at import time.
* Semver, version *ranges* for libraries, heavy vendor SDKs behind extras (`pip install mylib[openai]`).

## Quick recap
- JSON: `loads/dumps`, `ensure_ascii=False`, `default=` for custom types; JSONL streams line by line with per-line error handling.
- Normalise text (NFKC for keys, `casefold`, strip zero-width chars) but keep the original for offsets and citations.
- Regex: greedy vs lazy, `findall` with groups, named groups, `re.S` for fences; use a parser for nested structures.
- LLMs see tokens (BPE merges over bytes), not words; counts are model-specific; count with the real tokenizer and budget the output.
- Chunking: recursive by structure, sized in tokens, ~10% overlap, with metadata; tune on a golden set.
- Batch by item count and token budget; flush the last batch; keep indices.
- Async: `gather` + `Semaphore` bound concurrency, shared client, never block the loop; semaphore is not a rate limiter.
- Retry only transient errors, exponential backoff with jitter, honour `Retry-After`, cap attempts, inject `sleep`.
- Structured output: extract, parse, validate with pydantic, feed the error back for a capped repair loop; never trust or `eval` output.
- SSE: buffer across chunk boundaries, split on blank lines, handle `[DONE]`, mid-stream failures and timeouts.
- Config from env (strings!), validated at startup with pydantic-settings; `SecretStr`; never log or commit secrets; rotate leaks.
- Log per-item context and counters, redact secrets, continue past item failures.
- Pipelines are chains of generators with incremental, resumable writes; packages use src layout, a public `__init__`, extras.
