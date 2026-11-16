# Asynchronous programming (asyncio)

asyncio lets one thread juggle many tasks that spend most of their time *waiting*: an LLM call, an
embedding request, a vector-DB query, a token stream. While one coroutine waits, the event loop runs
another, so 50 calls of 200 ms finish in about 200 ms instead of 10 s. It is the backbone of FastAPI
services, agent frameworks and every "fan out N calls" pipeline. Interviewers probe whether you know
*what actually runs when*, how cancellation and timeouts behave, and which mistakes silently stall a service.
It builds directly on generators (see `08_generators.md`): a coroutine is a pausable frame that a loop resumes.

## Sync vs async, blocking, I/O-bound
<!-- topics: sync_vs_async, blocking, io_bound -->

**Synchronous** code does one thing at a time and the thread sits idle while waiting for the network.
**Asynchronous** code lets the thread switch to other work during those waits. This is *concurrency on one
thread*, not parallelism: at any instant exactly one piece of Python is executing.

```text
sync:   |--call A (waiting)--|--call B (waiting)--|--call C (waiting)--|        total = A+B+C
async:  |--A wait-----------|
         |--B wait----------|
          |--C wait---------|                                                    total = max(A,B,C)
```

A **blocking** operation holds the thread and does not let anyone else run (`time.sleep`, `requests.get`,
a sync DB driver, `open().read()` on a big file, a CPU loop). A **non-blocking** one says "wake me when ready"
and hands control back (`await asyncio.sleep`, `await httpx_client.get`).

| Workload | Examples | asyncio helps? |
|---|---|---|
| **I/O-bound** (time spent waiting) | LLM/embedding APIs, DB queries, HTTP, streaming | Yes, a lot |
| **CPU-bound** (time spent computing) | tokenizing a corpus, local inference, parsing, hashing | No. Use processes or native code that releases the GIL |

```python
import asyncio, time

async def fake_call(i):
    await asyncio.sleep(0.1)       # stands in for a 100 ms API call
    return i

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(*(fake_call(i) for i in range(20)))
    took = time.perf_counter() - start
    print(results[:3], took < 0.5)  # prints: [0, 1, 2] True   (20 calls, ~0.1 s, not 2 s)

asyncio.run(main())
```

> **Remember** async makes waiting overlap. It does nothing for computation. One blocking call freezes
> *every* coroutine in the process.

> **Interview** "When does async not help?" CPU-bound work, a single sequential call chain with nothing to overlap,
> and any code path where a blocking library call sits on the hot path (it cancels out the benefit).

## async def, await, coroutine functions and objects
<!-- topics: async_keyword, await, coroutine_functions, coroutine_objects -->

Three different things, often confused:

| Thing | What it is | Runs code? |
|---|---|---|
| **coroutine function** | `async def f(): ...` (the function) | no |
| **coroutine object** | what `f()` returns | **not yet**: paused before line 1 |
| **task** | a coroutine wrapped and scheduled on the loop | yes, soon, on its own |

Calling an `async def` function never runs its body. It builds a coroutine object, exactly like calling a
generator function builds a generator.

```python
import asyncio, inspect

async def fetch():
    print("body runs")
    return 42

print(inspect.iscoroutinefunction(fetch))        # prints: True
coro = fetch()                                   # nothing printed
print(type(coro).__name__, inspect.getcoroutinestate(coro))  # prints: coroutine CORO_CREATED

async def main():
    print(await coro)                            # prints: body runs, then 42

asyncio.run(main())
print(inspect.getcoroutinestate(coro))           # prints: CORO_CLOSED
```

`await x` does two things: it **runs the awaitable `x` to completion** (inline, in the current task), and if
`x` has to wait, it **suspends the current coroutine and hands control back to the event loop**. Only
the keyword `await` may suspend; a plain function call never does. `await` is legal only inside `async def`.

A coroutine really is a generator-like object. You can drive one by hand, which is what the loop does:

```python
async def f():
    return 7

c = f()
try:
    c.send(None)          # run it to its first suspension (here: to the end)
except StopIteration as e:
    print(e.value)        # prints: 7   (the return value rides on StopIteration, as with generators)
```

> **Gotcha** `await coro_fn()` is *not* concurrent. It runs the callee to completion before the next line.
> Concurrency needs a task (`create_task`, `gather`, `TaskGroup`).

<details markdown="1"><summary>Check yourself: `x = fetch()` on a line by itself. What happens?</summary>

A coroutine object is created and nothing else: the body never runs. If you drop it, Python warns
`coroutine 'fetch' was never awaited`.

</details>

## The event loop, asyncio.run, suspension and resumption
<!-- topics: event_loop, asyncio_run, suspension -->

The **event loop** is a single-threaded scheduler: a queue of ready callbacks plus a selector (epoll) watching
sockets and timers. It repeats: pop a ready task, run it *until its next `await` that cannot complete
immediately*, then pick the next. Nothing preempts a task; tasks cooperate by awaiting.

```text
loop:  run ready tasks -> poll sockets/timers -> mark waiters ready -> run ready tasks -> ...
task:  run ... await (suspend) ......................... resume here ... await (suspend) ...
```

`asyncio.run(main())` is the entry point: create a fresh loop, run `main()` as the top task, then cancel leftover
tasks, shut down async generators and the default executor, and close the loop. Call it **once**, at the top
of a program. Calling it inside a running loop raises `RuntimeError` (in Jupyter use `await` directly).

```python
import asyncio

async def worker(name):
    for i in range(2):
        print(name, i)
        await asyncio.sleep(0)    # suspend once, let others run, resume next iteration

async def main():
    await asyncio.gather(worker("A"), worker("B"))

asyncio.run(main())
# prints: A 0
# prints: B 0
# prints: A 1
# prints: B 1
```

Each `await` that suspends is a **suspension point**; the line after it is the **resumption**. Local variables
live in the paused frame and survive, like generator locals. Between two awaits your code runs atomically
with respect to other tasks, which is why many races cannot happen in asyncio (but see Lock below).

```python
import asyncio

async def other():
    return 1

async def main():
    c = other()
    try:
        asyncio.run(c)            # a loop is already running in this thread
    except RuntimeError as e:
        print(e)                  # prints: asyncio.run() cannot be called from a running event loop
    c.close()

asyncio.run(main())
```

> **Remember** `asyncio.run` once at the top; everywhere else just `await`. Never nest it.

## asyncio.sleep vs time.sleep
<!-- topics: asyncio_sleep, sleep_vs_sleep -->

`await asyncio.sleep(d)` registers a timer and suspends the task; the loop runs others and wakes it after `d`
seconds. `time.sleep(d)` freezes the **whole thread**, so the loop, and every task on it, stops. `asyncio.sleep(0)`
is the idiom for "yield to the loop once".

A background heartbeat makes the difference visible:

```python
import asyncio, time

async def ticker(log):
    while True:
        log.append("tick")
        await asyncio.sleep(0.05)

async def run(sleeper):
    log = []
    t = asyncio.create_task(ticker(log))
    await asyncio.sleep(0)       # let the ticker start: log == ["tick"]
    sleeper()                    # pretend to "wait" for 0.18 s
    n_blocking = len(log)
    t.cancel()
    return n_blocking

async def main():
    # blocking: the ticker cannot run at all while time.sleep holds the thread
    print("time.sleep ticks:", await run(lambda: time.sleep(0.18)))   # prints: time.sleep ticks: 1

    log = []
    t = asyncio.create_task(ticker(log))
    await asyncio.sleep(0.18)    # non-blocking: the ticker keeps ticking (0, .05, .10, .15)
    t.cancel()
    print("asyncio.sleep ticks:", len(log))                           # prints: asyncio.sleep ticks: 4

asyncio.run(main())
```

> **Gotcha** `time.sleep`, `requests.get`, `sqlite3`, `subprocess.run`, heavy `json.loads` and CPU loops inside
> `async def` all block the loop. Symptoms: every request in the service gets slow together, timeouts misfire,
> health checks fail. Run `asyncio.run(main(), debug=True)` to get "Executing ... took 0.3 seconds" warnings.

## Sequential vs concurrent, create_task and scheduling
<!-- topics: sequential_vs_concurrent, create_task, scheduling -->

Awaiting coroutines one after another is **sequential** even though it is async. For concurrency you must create
**tasks**: `asyncio.create_task(coro)` wraps the coroutine in a `Task` and schedules it on the loop right away. It
starts running the next time the current task suspends.

```python
import asyncio

async def call(name, d):
    print("start", name)
    await asyncio.sleep(d)
    print("done", name)

async def main():
    print("-- sequential")
    await call("a", 0.05)
    await call("b", 0.02)
    print("-- concurrent")
    ta = asyncio.create_task(call("a", 0.05))
    tb = asyncio.create_task(call("b", 0.02))
    await ta
    await tb

asyncio.run(main())
# prints: -- sequential / start a / done a / start b / done b
# prints: -- concurrent / start a / start b / done b / done a
```

**Scheduling order.** `create_task` does *not* run the task immediately. The task is put on the ready queue and
runs when the creator yields to the loop:

```python
import asyncio

async def job():
    print("job runs")
    await asyncio.sleep(0.01)
    print("job ends")

async def main():
    t = asyncio.create_task(job())
    print("created")            # job has NOT started yet
    await asyncio.sleep(0)      # yield: the ready queue runs job up to its first real wait
    print("main resumed")
    await t

asyncio.run(main())
# prints: created / job runs / main resumed / job ends
```

Tasks are scheduled FIFO. Without a suspension point, a task never gives anyone else a turn.

> **Gotcha** The loop keeps only a *weak* reference to tasks. If you drop the reference to a fire-and-forget
> task it can be garbage-collected mid-flight and silently vanish. Keep a strong reference.

```python
import asyncio

background: set[asyncio.Task] = set()

def spawn(coro):
    t = asyncio.create_task(coro)
    background.add(t)                       # strong reference while it runs
    t.add_done_callback(background.discard) # and let go when it finishes
    return t

async def main():
    async def audit_log():
        await asyncio.sleep(0.01)
        print("logged")
    spawn(audit_log())
    await asyncio.sleep(0.05)
    print(len(background))                  # prints: logged / 0

asyncio.run(main())
```

> **AI use** Fire-and-forget tasks (usage logging, tracing, cache write-back after the response is sent) need
> this pattern, plus error handling inside the coroutine: an unhandled exception in a task nobody awaits only
> surfaces as a "Task exception was never retrieved" log line.

## gather: results, ordering, exceptions
<!-- topics: gather, gather_ordering, gather_exceptions, return_exceptions -->

`await asyncio.gather(*aws)` runs awaitables concurrently (wrapping coroutines in tasks) and returns a list of
results **in argument order, not completion order**.

```python
import asyncio

async def embed(text, delay):
    await asyncio.sleep(delay)
    print("finished", text)
    return len(text)

async def main():
    out = await asyncio.gather(embed("slow", 0.06), embed("fast", 0.01), embed("mid", 0.03))
    print(out)

asyncio.run(main())
# prints: finished fast / finished mid / finished slow
# prints: [4, 4, 3]
```

**Exceptions.** By default the *first* exception raised by any child propagates out of `gather`, but the **other
children are not cancelled**: they keep running in the background.

```python
import asyncio

async def ok(n):
    await asyncio.sleep(0.05)
    print("finished", n)
    return n

async def bad():
    await asyncio.sleep(0.01)
    raise ValueError("boom")

async def main():
    try:
        await asyncio.gather(ok(1), bad())
    except ValueError as e:
        print("caught", e)
    print("after gather")
    await asyncio.sleep(0.1)     # give the orphan time to finish

asyncio.run(main())
# prints: caught boom / after gather / finished 1
```

With `return_exceptions=True` nothing propagates; failures come back **as values** in their slot, so one bad
item does not lose the rest:

```python
import asyncio

async def embed(i):
    if i == 1:
        raise ValueError("bad input")
    return i * 10

async def main():
    results = await asyncio.gather(*(embed(i) for i in range(3)), return_exceptions=True)
    print(results)                        # prints: [0, ValueError('bad input'), 20]
    ok = [r for r in results if not isinstance(r, BaseException)]
    print(ok)                             # prints: [0, 20]

asyncio.run(main())
```

> **Gotcha** With `return_exceptions=True`, a child's `CancelledError` also comes back as a value (it is a
> `BaseException`). Check `isinstance(r, Exception)` for item errors and re-raise anything else; never report
> cancellation as "one item failed". Also: if the task *running gather* is cancelled, the children are cancelled.

> **Interview** "gather vs TaskGroup?" gather: results as a list, first error propagates, siblings keep running,
> no scope. TaskGroup: structured scope, first error cancels siblings, all errors arrive as an `ExceptionGroup`.

<details markdown="1"><summary>Check yourself: why do you still want `return_exceptions=True` for 100 embedding calls?</summary>

Without it one rate-limit error aborts your view of the batch and you lose 99 good results (they still run,
but you cannot collect them). With it you get a result-or-exception per item and can retry only the failures.

</details>

## Concurrent execution with dependencies
<!-- topics: dependencies -->

Real work is a DAG: some steps are independent, others need earlier results. Rule: run independent steps
concurrently, `await` a result only when you need it.

```python
import asyncio

async def get_user():
    await asyncio.sleep(0.03); return "ada"

async def get_docs():
    await asyncio.sleep(0.05); return ["d1", "d2"]

async def llm(prompt):
    await asyncio.sleep(0.01); return f"answer({prompt})"

async def main():
    user, docs = await asyncio.gather(get_user(), get_docs())     # independent: together
    print(await llm(f"{user}:{len(docs)}"))                        # depends on both: after
asyncio.run(main())
# prints: answer(ada:2)
```

For irregular graphs, make each node a task that awaits the tasks it depends on; the loop sorts out the order:

```python
import asyncio

async def node(name, delay, *deps):
    await asyncio.gather(*deps)           # wait for dependency tasks
    await asyncio.sleep(delay)
    print("done", name)
    return name

async def main():
    a = asyncio.create_task(node("a", 0.05))
    b = asyncio.create_task(node("b", 0.05, a))        # b needs a
    c = asyncio.create_task(node("c", 0.12))           # c is independent
    d = asyncio.create_task(node("d", 0.05, b, c))     # d needs b and c
    print(await d)

asyncio.run(main())
# prints: done a / done b / done c / done d / d
```

Here `c` runs alongside `a` and `b` instead of waiting for them. (For failure handling prefer a TaskGroup; see below.)

## Cancellation: cancel(), CancelledError, finally, done() and cancelled()
<!-- topics: cancellation, task_cancel, cancellederror, cleanup_finally, task_done, task_cancelled -->

`task.cancel()` does not kill a task. It *requests* cancellation: at the task's next suspension point the loop
raises `asyncio.CancelledError` **inside** it. The task can run `finally` blocks, then the exception unwinds and
the task ends in the *cancelled* state.

```python
import asyncio

async def worker():
    try:
        print("working")
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("cancelled: cleaning up")
        raise                                   # MUST re-raise
    finally:
        print("finally runs")

async def main():
    t = asyncio.create_task(worker())
    await asyncio.sleep(0)                      # let it start
    print("done before cancel:", t.done())      # prints: done before cancel: False
    print("cancel() ->", t.cancel())            # prints: cancel() -> True (request accepted)
    try:
        await t
    except asyncio.CancelledError:
        print("main saw CancelledError")
    print(t.done(), t.cancelled())              # prints: True True

asyncio.run(main())
# prints: working / done before cancel: False / cancel() -> True / cancelled: cleaning up
# prints: finally runs / main saw CancelledError / True True
```

State methods: `t.done()` is True when finished *for any reason* (result, exception, cancelled);
`t.cancelled()` is True only when it ended by cancellation. `t.result()` / `t.exception()` on a cancelled task
raise `CancelledError`. `cancel()` returns `False` if the task is already done.

`CancelledError` derives from `BaseException` (not `Exception`) precisely so that a lazy `except Exception`
does not eat it. **Swallowing it is the classic bug**: the task then finishes "normally" and the canceller's
intent is lost.

```python
import asyncio

async def swallower():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("swallowed")                      # no re-raise
    return "finished normally"

async def main():
    t = asyncio.create_task(swallower())
    await asyncio.sleep(0)
    t.cancel()
    print(await t, t.cancelled())               # prints: swallowed / finished normally False

asyncio.run(main())
```

Rules: catch `CancelledError` only to clean up, then **re-raise**. Put cleanup in `finally` so it runs on success,
error and cancellation alike. Awaiting inside cleanup is allowed, but a second cancellation can interrupt it;
wrap must-finish cleanup in `asyncio.shield`.

**shield** protects an inner task from the *outer* cancellation: the outer await is cancelled, the inner keeps going.

```python
import asyncio

async def save():
    await asyncio.sleep(0.05)
    return "saved"

async def main():
    inner = asyncio.create_task(save())
    try:
        await asyncio.wait_for(asyncio.shield(inner), 0.01)
    except TimeoutError:
        print("caller gave up, inner done:", inner.done())   # prints: caller gave up, inner done: False
    print(await inner)                                       # prints: saved

asyncio.run(main())
```

> **Gotcha** Cancelling a task that never started means its body never runs (so its `finally` never runs
> either). Cancelling does not interrupt blocking code or a thread started by `to_thread`; it only acts at an `await`.

> **Interview** "What happens on `task.cancel()`?" CancelledError is thrown into the coroutine at its
> current await; `finally` runs; if re-raised the task is `cancelled()`; if swallowed the task continues. Cancel is a
> request, and cancelling then not awaiting the task leaves its cleanup unfinished.

<details markdown="1"><summary>Check yourself: after `t.cancel()` and `await t` in a try/except, why is it important to await?</summary>

`cancel()` only schedules the CancelledError. The task's cleanup runs when the loop next runs it; awaiting the task
(it raises CancelledError) guarantees the cleanup finished before you move on.

</details>

## Timeouts: wait_for, asyncio.timeout, and cancellation on timeout
<!-- topics: wait_for, timeouts, timeout_cancellation, asyncio_timeout -->

Every network call needs a timeout. Both tools implement a timeout the same way: after the deadline they
**cancel** the inner work, wait for it to finish unwinding, and raise `TimeoutError`
(`asyncio.TimeoutError` is just an alias of the builtin since 3.11).

`asyncio.wait_for(aw, timeout)` bounds a single awaitable:

```python
import asyncio

async def slow_llm():
    try:
        await asyncio.sleep(1)
        return "text"
    finally:
        print("slow_llm cancelled")

async def main():
    try:
        await asyncio.wait_for(slow_llm(), timeout=0.05)
    except TimeoutError:
        print("timed out")

asyncio.run(main())
# prints: slow_llm cancelled / timed out
```

`async with asyncio.timeout(seconds)` bounds a whole **block**, so several steps share one budget. That is what
you want for "this RAG request must answer within 2 s", where retrieval and generation each take part of it.

```python
import asyncio

async def step(n):
    await asyncio.sleep(0.04)
    print("step", n)

async def main():
    try:
        async with asyncio.timeout(0.1):
            for i in (1, 2, 3):
                await step(i)
    except TimeoutError:
        print("budget exhausted")

asyncio.run(main())
# prints: step 1 / step 2 / budget exhausted
```

How it works: a timer calls `task.cancel()` on the current task; on leaving the block the context manager turns
that `CancelledError` into `TimeoutError`. Corollaries:

* `except Exception` inside the block is safe (CancelledError is not an Exception); a bare `except:` or
  `except BaseException` that does not re-raise **breaks timeouts**: the work continues and no `TimeoutError` is raised.
* Catch `TimeoutError` *outside* the block, never inside.
* `cm.expired()` tells you whether the deadline fired; `cm.reschedule(new_deadline)` moves it.

```python
import asyncio

async def main():
    try:
        async with asyncio.timeout(0.05) as cm:
            await asyncio.sleep(1)
    except TimeoutError:
        print("expired:", cm.expired())          # prints: expired: True

asyncio.run(main())
```

| | `wait_for(aw, t)` | `timeout(t)` |
|---|---|---|
| Scope | one awaitable | a block of awaits |
| Shared budget | no | yes |
| Python | all | 3.11+ |

> **Gotcha** A timeout does not stop *non-cooperative* work: a thread behind `to_thread` or a blocking call
> keeps running; you merely stop waiting. Also, a timeout is a cancel, so the callee's `finally` blocks must be fast.

> **AI use** Timeouts on LLM calls: set a per-attempt timeout with `wait_for`, a total budget with `timeout()`,
> and a fallback (cached answer, smaller model) on `TimeoutError`. Retry only idempotent calls, with backoff.

## asyncio.wait: FIRST_COMPLETED, ALL_COMPLETED, FIRST_EXCEPTION, done and pending
<!-- topics: wait, all_completed, first_completed, first_exception, done_pending -->

`asyncio.wait(tasks, return_when=..., timeout=...)` is the low-level tool when you need to react to *progress*.
It returns two sets, `(done, pending)`, and:

* takes **tasks/futures only** (passing bare coroutines is a `TypeError` since 3.11),
* **never cancels** the pending ones and **never raises** the tasks' exceptions: you inspect them,
* on `timeout` simply returns what has finished so far (no `TimeoutError`),
* returns **unordered sets**.

| `return_when` | Returns when |
|---|---|
| `ALL_COMPLETED` (default) | every task is done |
| `FIRST_COMPLETED` | any task finishes (success, error or cancelled) |
| `FIRST_EXCEPTION` | any task raises (or all done if none raises) |

```python
import asyncio

async def replica(name, d):
    await asyncio.sleep(d)
    return name

async def main():
    tasks = {asyncio.create_task(replica(n, d), name=n) for n, d in [("a", 0.1), ("b", 0.02), ("c", 0.2)]}
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    print([t.result() for t in done])            # prints: ['b']
    print(sorted(t.get_name() for t in pending)) # prints: ['a', 'c']
    for t in pending:
        t.cancel()                               # losers are NOT cancelled for you
    await asyncio.gather(*pending, return_exceptions=True)   # wait for their cleanup

asyncio.run(main())
```

```python
import asyncio

async def boom():
    await asyncio.sleep(0.01)
    raise RuntimeError("fail fast")

async def long():
    await asyncio.sleep(0.1)

async def main():
    t1, t2 = asyncio.create_task(boom()), asyncio.create_task(long())
    done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_EXCEPTION)
    print(done == {t1}, pending == {t2})         # prints: True True
    print(repr(t1.exception()))                  # prints: RuntimeError('fail fast')
    t2.cancel()
    await asyncio.gather(t2, return_exceptions=True)

asyncio.run(main())
```

Typical uses: racing replicas, the "first answer wins" pattern, deadline collection
(`wait(..., timeout=d)` then cancel `pending`), and a manual loop of FIRST_COMPLETED to process results as they land.

> **Gotcha** If you never retrieve a failed task's exception you get "Task exception was never retrieved" at GC time.
> Look at `t.exception()` / `t.result()` for every task in `done`.

Hedged request (tail-latency hedging): if the first attempt has not answered after a delay, start a second
and take whichever finishes first.

```python
import asyncio

async def call_replica(i):
    await asyncio.sleep(0.2 if i == 0 else 0.02)   # replica 0 is having a bad day
    return f"replica {i}"

async def hedged(hedge_after):
    first = asyncio.create_task(call_replica(0))
    done, _ = await asyncio.wait({first}, timeout=hedge_after)
    if done:
        return first.result()
    second = asyncio.create_task(call_replica(1))
    done, pending = await asyncio.wait({first, second}, return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return done.pop().result()

print(asyncio.run(hedged(0.05)))      # prints: replica 1
```

## as_completed and completion order
<!-- topics: as_completed, completion_order -->

`gather` gives results in **input order** but only after *all* finish. `as_completed` yields each result in
**completion order**, as soon as it is ready: the right tool for streaming partial results to a user, early exit, or
processing a big batch while the rest is in flight.

```python
import asyncio

async def work(i, d):
    await asyncio.sleep(d)
    return i

async def main():
    coros = [work(0, 0.08), work(1, 0.02), work(2, 0.05)]
    for next_done in asyncio.as_completed(coros):
        print("got", await next_done)

asyncio.run(main())
# prints: got 1 / got 2 / got 0
```

The loop variable is an awaitable, not the result, and you do not know *which input* it came from. Tag the result
yourself. Since 3.13 the object also supports `async for` (it yields the finished futures, which you still `await`):

```python
import asyncio

async def tagged(i, d):
    await asyncio.sleep(d)
    return i, f"vec{i}"

async def main():
    coros = [tagged(0, 0.08), tagged(1, 0.02), tagged(2, 0.05)]
    async for fut in asyncio.as_completed(coros):         # 3.13+; on 3.12 use the plain for loop
        i, vec = await fut
        print(i, vec)

asyncio.run(main())
# prints: 1 vec1 / 2 vec2 / 0 vec0
```

`as_completed(coros, timeout=t)` raises `TimeoutError` when the deadline passes with results missing, and does
**not** cancel the rest (cancel them yourself). Exceptions surface at `await next_done` for the failing item.

> **Remember** completion order is for latency and streaming; input order (gather) is for aligning results with inputs.
> Use gather when you must zip results back to inputs, as_completed (with tags) when you want the earliest ones first.

## Semaphore and Lock
<!-- topics: semaphore, lock -->

**`asyncio.Semaphore(n)`** allows at most `n` holders at once. It is how you cap **concurrency**: provider
rate limits, DB pool size, file descriptors. Without it, `gather` over 10 000 items opens 10 000 requests at once.

```python
import asyncio

active = 0
peak = 0

async def call(i, sem):
    global active, peak
    async with sem:                      # at most 3 inside
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
    return i

async def main():
    sem = asyncio.Semaphore(3)
    out = await asyncio.gather(*(call(i, sem) for i in range(10)))
    print(out)                           # prints: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("peak in flight:", peak)       # prints: peak in flight: 3

asyncio.run(main())
```

`BoundedSemaphore` additionally raises if you release more than you acquired. Always use `async with sem:` so the
slot is freed on error or cancellation. A semaphore limits *running*, not *created* tasks: 100 000 tasks all waiting
on a semaphore still cost memory. For huge inputs use a bounded queue with N workers (next sections).

**`asyncio.Lock`** gives mutual exclusion across awaits. Single-threaded does not mean race-free: any shared
read-modify-write that spans an `await` can interleave.

```python
import asyncio

state = {"n": 0}

async def inc(lock=None):
    async def body():
        v = state["n"]
        await asyncio.sleep(0)           # another task runs here
        state["n"] = v + 1
    if lock:
        async with lock:
            await body()
    else:
        await body()

async def main():
    await asyncio.gather(*(inc() for _ in range(10)))
    print("no lock:", state["n"])        # prints: no lock: 1   (lost updates)
    state["n"] = 0
    lock = asyncio.Lock()
    await asyncio.gather(*(inc(lock) for _ in range(10)))
    print("lock:", state["n"])           # prints: lock: 10

asyncio.run(main())
```

> **Gotcha** A Lock is not reentrant (acquiring twice in one task deadlocks), and not thread-safe; it is for
> coroutines on one loop. Waiters are served FIFO. Keep critical sections short and never `await` slow I/O while holding one unless you must.

> **AI use** Semaphore = "at most 8 concurrent embedding calls". Lock = "only one coroutine refreshes the auth
> token / rebuilds the cache; the rest wait". Token-bucket rate limiters combine a Lock with `asyncio.sleep`.

## Queue: producers, consumers, backpressure
<!-- topics: queue -->

`asyncio.Queue(maxsize)` hands work between coroutines. `await q.put(x)` suspends when the queue is full and
`await q.get()` suspends when empty, so a bounded queue gives **backpressure**: a fast producer is slowed to the
consumers' pace and memory stays bounded.

```python
import asyncio

async def producer(q):
    for i in range(4):
        await q.put(i)
        print("put", i)
    await q.put(None)                    # sentinel: no more work

async def consumer(q):
    while (item := await q.get()) is not None:
        print("got", item)
        await asyncio.sleep(0.01)

async def main():
    q = asyncio.Queue(maxsize=2)
    await asyncio.gather(producer(q), consumer(q))

asyncio.run(main())
```

```text
put 0
put 1
got 0
put 2
got 1
put 3
got 2
got 3
```

The producer fills the queue (2 slots), then `put 2` blocks until the consumer's `get` frees a slot. The producer
can never be more than `maxsize` items ahead.

For worker pools use `task_done()` and `join()`: each `get()` must be paired with a `task_done()`; `await q.join()`
returns when every item put has been marked done.

```python
import asyncio

async def worker(name, q, results):
    while True:
        item = await q.get()
        try:
            results.append(item * 2)
        finally:
            q.task_done()               # always, even on error, or join() hangs

async def main():
    q, results = asyncio.Queue(), []
    workers = [asyncio.create_task(worker(f"w{i}", q, results)) for i in range(3)]
    for i in range(6):
        q.put_nowait(i)
    await q.join()                      # all 6 processed
    for w in workers:
        w.cancel()                      # workers loop forever: cancel them
    await asyncio.gather(*workers, return_exceptions=True)
    print(sorted(results))              # prints: [0, 2, 4, 6, 8, 10]

asyncio.run(main())
```

Also: `get_nowait()` raises `QueueEmpty`, `put_nowait` raises `QueueFull`; `PriorityQueue` and `LifoQueue` exist.
Python 3.13 added `q.shutdown()` (consumers get `QueueShutDown` once drained), a built-in alternative to sentinels.

> **Gotcha** Forgetting `task_done()` makes `join()` wait forever. An unbounded queue (default `maxsize=0`) removes
> backpressure and can eat all memory. Sentinel-per-worker is needed if several workers must each be told to stop.

> **AI use** Ingestion pipeline: a bounded queue between a reader (streaming documents) and N embedding workers
> keeps memory flat and lets you cap concurrent API calls without creating a task per document.

## Event and Condition
<!-- topics: event, condition -->

**`asyncio.Event`** is a one-bit flag coroutines can wait for. `await ev.wait()` suspends until some other code
calls `ev.set()`, which wakes *all* waiters; `clear()` resets. Use it instead of polling a flag with `sleep`.

```python
import asyncio

async def worker(n, ready):
    await ready.wait()                  # no busy loop
    print("worker", n, "go")

async def main():
    ready = asyncio.Event()
    tasks = [asyncio.create_task(worker(i, ready)) for i in range(2)]
    await asyncio.sleep(0.01)
    print("model loaded")
    ready.set()
    await asyncio.gather(*tasks)

asyncio.run(main())
# prints: model loaded / worker 0 go / worker 1 go
```

**`asyncio.Condition`** combines a lock with "wait until some state becomes true". The pattern: hold the
lock, check a predicate, `wait()` (releases the lock while suspended, re-acquires on wake), and mutators
call `notify()` / `notify_all()` after changing state. `wait_for(predicate)` does the check loop for you.

```python
import asyncio

async def consumer(cond, items):
    async with cond:
        await cond.wait_for(lambda: len(items) >= 2)   # releases lock while waiting
        print("batch ready:", items)

async def producer(cond, items):
    for i in range(3):
        await asyncio.sleep(0.01)
        async with cond:
            items.append(i)
            cond.notify_all()

async def main():
    cond, items = asyncio.Condition(), []
    await asyncio.gather(consumer(cond, items), producer(cond, items))

asyncio.run(main())
# prints: batch ready: [0, 1]
```

> **Gotcha** Always re-check the condition in a loop (or use `wait_for`): a wakeup does not guarantee the state is still
> true. `wait()` and `notify()` must be called while holding the condition's lock, else `RuntimeError`.

> **AI use** A micro-batcher: callers append their text to a pending list and `notify()`; one flusher task waits on
> the condition until `len(pending) >= max_batch` or `max_wait` elapses, then sends **one** batched embedding call
> and resolves a `Future` per caller. Batch calls are far cheaper per text than single calls.

## asyncio.run shutdown and graceful shutdown
<!-- topics: run_shutdown, graceful_shutdown -->

When `main()` returns, `asyncio.run` does not just close the loop. It: (1) cancels every task still pending and
waits for them (so their `finally` blocks run), (2) shuts down async generators, (3) shuts down the default
thread executor, (4) closes the loop.

```python
import asyncio

async def background():
    try:
        await asyncio.sleep(10)
    finally:
        print("background cleaned up")

async def main():
    t = asyncio.create_task(background())
    await asyncio.sleep(0.01)
    print("main returns")

asyncio.run(main())
# prints: main returns / background cleaned up
```

That is a safety net, not a design. **Graceful shutdown** means: stop accepting new work, let in-flight work finish
(bounded by a deadline), release resources, then exit. The usual shape:

```text
signal (SIGTERM) -> set stop flag / close intake -> drain queue -> wait for workers (with timeout)
                 -> cancel stragglers -> await them -> close clients
```

```python
import asyncio

async def worker(q, done):
    while (item := await q.get()) is not None:
        await asyncio.sleep(0.005)
        done.append(item)

async def main():
    q, done = asyncio.Queue(), []
    workers = [asyncio.create_task(worker(q, done)) for _ in range(2)]
    for i in range(6):
        q.put_nowait(i)                       # work accepted before shutdown
    for _ in workers:
        q.put_nowait(None)                    # stop intake: one sentinel per worker (after the real items)
    try:
        await asyncio.wait_for(asyncio.gather(*workers), timeout=1)   # drain with a deadline
    except TimeoutError:
        for w in workers:
            w.cancel()                        # stragglers: force
        await asyncio.gather(*workers, return_exceptions=True)
    print(sorted(done))                       # prints: [0, 1, 2, 3, 4, 5]

asyncio.run(main())
```

Wiring a signal to this on Linux/macOS:

```python
# nocheck
import asyncio, signal

async def serve():
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)   # not available on Windows
    ...start workers...
    await stop.wait()
    ...drain, cancel, close clients...
```

> **Gotcha** Ctrl-C under `asyncio.run` cancels the main task, so a `finally` in `main` still runs; but cleanup that
> itself awaits can be cancelled again, so shield what must finish. Never `sys.exit()` from inside a task.

> **Interview** "How do you shut an LLM worker down on SIGTERM?" Stop intake, drain with a deadline (Kubernetes
> gives ~30 s), cancel what is left and *await* it, flush buffers and close HTTP clients. Idempotent, bounded in time.

## TaskGroup and ExceptionGroup (structured concurrency)
<!-- topics: taskgroup -->

`asyncio.TaskGroup` (3.11+) is **structured concurrency**: tasks are children of an `async with` block, and
the block does not exit until all of them finish. No orphan tasks, no lost exceptions.

```python
import asyncio

async def fetch(name, d):
    await asyncio.sleep(d)
    return name

async def main():
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(fetch("a", 0.03))
        t2 = tg.create_task(fetch("b", 0.01))
    print(t1.result(), t2.result())       # prints: a b   (block exits only when both are done)

asyncio.run(main())
```

Failure semantics differ from `gather`: when any child fails, the group **cancels all siblings**, waits for them,
then raises an **`ExceptionGroup`** bundling every failure. Catch it with `except*`, which matches by type
inside the group:

```python
import asyncio

async def ok():
    try:
        await asyncio.sleep(1)
    finally:
        print("sibling cancelled")

async def bad(msg, d):
    await asyncio.sleep(d)
    raise ValueError(msg)

async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(ok())
            tg.create_task(bad("first", 0.01))
            tg.create_task(bad("second", 0.01))
    except* ValueError as eg:
        print(type(eg).__name__, [str(e) for e in eg.exceptions])

asyncio.run(main())
# prints: sibling cancelled
# prints: ExceptionGroup ['first', 'second']
```

> **Gotcha** A single failing child still arrives wrapped: `except ValueError:` does **not** catch it; you need
> `except* ValueError:` (or catch `ExceptionGroup`). Exceptions inside a child that you want handled per item must be
> caught inside the child. Cancelling the parent cancels the whole group.

| | `gather` | `TaskGroup` |
|---|---|---|
| Result | list in order | read `task.result()` after the block |
| First failure | propagates; siblings keep running | siblings **cancelled** |
| Multiple failures | only first propagates (rest lost) | all in `ExceptionGroup` |
| Scope | none | tasks cannot outlive the block |
| Partial results | `return_exceptions=True` | catch inside children |

> **Interview** "Why prefer TaskGroup in new code?" Structured lifetimes, automatic sibling cancellation on
> failure, no swallowed errors. Use gather(return_exceptions=True) when you explicitly want partial results.

## Async generators, iterators, async for and comprehensions
<!-- topics: async_generators, async_iterators, async_for, async_comprehensions -->

An **async iterator** implements `__aiter__` and `__anext__` (an awaitable that raises `StopAsyncIteration` at the end).
An **async generator** is an `async def` containing `yield`: the easy way to write one. `async for` awaits `__anext__`
each round, so the loop can suspend between items. This is exactly the shape of a streamed LLM response.

```python
import asyncio

async def tokens(text):
    for word in text.split():
        await asyncio.sleep(0.005)       # waiting on the network
        yield word

async def main():
    async for tok in tokens("the quick brown fox"):
        print(tok, end=" ")
    print()
    # prints: the quick brown fox

asyncio.run(main())
```

A hand-written async iterator, e.g. a paginated API where each page is an awaited fetch:

```python
import asyncio

class Pages:
    def __init__(self, n):
        self.i, self.n = 0, n
    def __aiter__(self):
        return self
    async def __anext__(self):
        if self.i >= self.n:
            raise StopAsyncIteration
        await asyncio.sleep(0.001)       # fetch next page
        self.i += 1
        return f"page{self.i}"

async def main():
    print([p async for p in Pages(3)])   # prints: ['page1', 'page2', 'page3']

asyncio.run(main())
```

**Async comprehensions**: `[x async for x in aiter]` and `{k: v async for ...}` work only inside `async def`.
`await` is allowed inside the expression too, but it runs **sequentially**:

```python
import asyncio

async def embed(t):
    await asyncio.sleep(0.01)
    return len(t)

async def main():
    texts = ["a", "bb", "ccc"]
    seq = [await embed(t) for t in texts]                  # one after another (~3 x 0.01)
    con = await asyncio.gather(*(embed(t) for t in texts)) # together (~0.01)
    print(seq, con)                                        # prints: [1, 2, 3] [1, 2, 3]

asyncio.run(main())
```

**Closing early.** If you `break` out of an async generator, its `finally` has not necessarily run yet; it runs when
the loop finalises the generator later. Close deterministically with `aclose()` or `contextlib.aclosing`:

```python
import asyncio
from contextlib import aclosing

async def stream():
    try:
        for w in "a b c d".split():
            await asyncio.sleep(0.001)
            yield w
    finally:
        print("stream closed")           # release the HTTP response here

async def main():
    async with aclosing(stream()) as s:
        async for w in s:
            print(w)
            if w == "b":                 # e.g. a stop sequence matched
                break
    print("after block")

asyncio.run(main())
# prints: a / b / stream closed / after block
```

> **Gotcha** You cannot `yield` and `return value` in an async generator, and `async for` is a `SyntaxError`
> outside `async def`. An async generator that is garbage-collected un-exhausted relies on the loop's finalizer; do not count on timing.

> **AI use** Streaming completions: `async for chunk in client.stream(...)` yields deltas; forward them to the user
> via SSE, check stop sequences, and `aclose()` the stream on client disconnect so the provider stops generating (and billing).

## Async context managers, async with and resource management
<!-- topics: async_context_managers, async_with, resource_management -->

`async with` is `with` where entering and leaving may await: implement `__aenter__` / `__aexit__` (both `async def`).
Use it for anything whose setup or teardown does I/O: opening a connection, acquiring a pool slot, a lock, a
transaction, an HTTP client session.

```python
import asyncio

class Conn:
    async def __aenter__(self):
        await asyncio.sleep(0.001)       # connect
        print("open")
        return self
    async def __aexit__(self, exc_type, exc, tb):
        await asyncio.sleep(0.001)       # close
        print("close", exc_type.__name__ if exc_type else None)
        return False                     # do not swallow exceptions

async def main():
    try:
        async with Conn():
            print("using")
            raise ValueError("oops")
    except ValueError:
        print("caught")

asyncio.run(main())
# prints: open / using / close ValueError / caught
```

The generator-based shortcut is `@asynccontextmanager`; teardown goes in `finally`:

```python
import asyncio
from contextlib import asynccontextmanager, AsyncExitStack

@asynccontextmanager
async def client(name):
    print("open", name)
    try:
        yield name.upper()
    finally:
        await asyncio.sleep(0)
        print("close", name)

async def main():
    async with AsyncExitStack() as stack:          # a dynamic number of resources
        llm = await stack.enter_async_context(client("llm"))
        db = await stack.enter_async_context(client("db"))
        print("using", llm, db)

asyncio.run(main())
# prints: open llm / open db / using LLM DB / close db / close llm
```

Resource rules: acquire in `async with`, never in a bare `await open()` you might forget to close; resources
close in reverse order (LIFO); create **one** long-lived client (HTTP connection pool) per process and reuse it
instead of one per request; make `__aexit__` quick and robust because it also runs on cancellation, and shield
cleanup that must complete.

> **Gotcha** Using `with` on an async resource (or `async with` on a sync one) is a `TypeError`/`AttributeError`.
> `asyncio.Lock`, `Semaphore`, `TaskGroup`, `timeout()` are all async context managers.

## Blocking code: to_thread and run_in_executor
<!-- topics: blocking -->

You cannot always avoid blocking code (a sync SDK, `pandas`, file I/O, a CPU-heavy tokenizer). Move it off the
loop's thread: `await asyncio.to_thread(fn, *args)` runs `fn` in the default thread pool and suspends only the
caller. The loop stays responsive.

```python
import asyncio, threading, time

def blocking_embed(text):
    time.sleep(0.05)                     # sync SDK call, would freeze the loop
    return threading.get_ident()

async def main():
    log = []
    async def heartbeat():
        await asyncio.sleep(0.01)
        log.append("tick")               # runs while the thread is busy
    hb = asyncio.create_task(heartbeat())
    worker_thread = await asyncio.to_thread(blocking_embed, "hi")
    await hb
    print(worker_thread != threading.get_ident(), log)   # prints: True ['tick']

asyncio.run(main())
```

| Work | Tool |
|---|---|
| sync I/O (legacy SDK, files) | `asyncio.to_thread` / `loop.run_in_executor(None, fn)` |
| CPU-bound pure Python | `loop.run_in_executor(ProcessPoolExecutor(), fn)` (threads do not help: the GIL) |
| C code releasing the GIL (numpy, hashing, some tokenizers) | threads are enough |

> **Gotcha** Threads cannot be cancelled: cancelling the awaiting task abandons the wait but the thread runs to the
> end. The default pool is small (`min(32, cpu+4)`), so thousands of `to_thread` calls queue up. Code in the thread
> must not touch the loop or asyncio objects (use `asyncio.run_coroutine_threadsafe` to call back). On free-threaded
> Python 3.14 builds threads can run Python in parallel, but the same loop rules apply.

## Traps and best practices
<!-- topics: best_practices -->

**Forgotten `await`** is the most common bug: you get a coroutine object, the work never happens, and a truthy value
sneaks into conditions.

```python
import asyncio

async def is_ready():
    return False

async def main():
    c = is_ready()                 # forgot await
    print(type(c).__name__, bool(c))   # prints: coroutine True   (an `if is_ready():` is always taken!)
    c.close()                      # silence "never awaited"
    print(await is_ready())        # prints: False

asyncio.run(main())
```

Checklist of traps and what to do:

| Trap | Effect | Fix |
|---|---|---|
| missing `await` | work skipped, `RuntimeWarning: coroutine ... never awaited` | await it; lint (ruff `RUF006`, `ASYNC` rules) |
| dropped task reference | task GC'd mid-run | hold a set of tasks, or use a TaskGroup |
| blocking call in `async def` | whole service stalls | async library or `to_thread`; debug mode shows slow callbacks |
| unbounded fan-out (`gather` of 100k) | rate limits, OOM, thousands of sockets | Semaphore / worker pool on a bounded Queue |
| swallowing `CancelledError` | cancel and timeout stop working | re-raise; `except Exception` is fine |
| `asyncio.run` inside a loop | `RuntimeError` | `await` instead |
| unretrieved task exception | lost errors | await tasks, or `add_done_callback` that logs |
| timeout without cancel safety | half-written state | idempotent steps, cleanup in `finally` |
| shared state across awaits | lost updates | Lock, or make updates await-free |

Best-practice summary for cancellation and timeouts:

* Every external call has a timeout; use `timeout()` for a shared budget, `wait_for` for one call.
* Own your tasks: await them, or put them in a TaskGroup; after `cancel()` always await the task.
* Let `CancelledError` propagate; do cleanup in `finally` and keep it short and bounded.
* Bound concurrency (semaphore or workers) and queues (`maxsize`) so overload turns into waiting, not a crash.
* Do not share one client across event loops; create long-lived clients inside the running loop.
* Retry only idempotent operations, with a per-attempt timeout and jittered exponential backoff, and a total budget.

> **Interview** "Your async FastAPI endpoint is slow under load though CPU is idle. Debug?" Look for blocking calls
> on the loop (sync SDK, `requests`, big `json`/pandas, `time.sleep`): use debug mode / `loop.slow_callback_duration`,
> py-spy, then offload with `to_thread` or switch to an async client. Check for unbounded fan-out hitting provider rate limits.

## Quick recap
- Async = concurrency on one thread by overlapping *waiting*; useless for CPU-bound work, and one blocking call freezes everything.
- `async def` call -> coroutine object (nothing ran); `await` runs it inline and may suspend; a *task* runs on its own.
- `await coro()` is sequential; use `create_task` / `gather` / `TaskGroup` for concurrency. Tasks start when the creator next yields.
- `asyncio.run` once at the top; it cancels leftovers and closes the loop. `time.sleep` blocks, `asyncio.sleep` yields.
- `gather`: results in input order; first exception propagates, siblings keep running; `return_exceptions=True` returns errors as values.
- `cancel()` throws `CancelledError` at the next await: clean up in `finally`, re-raise, await the task; `shield` protects inner work.
- `wait_for` = one awaitable; `timeout()` = shared budget for a block; both cancel on expiry and raise `TimeoutError`.
- `asyncio.wait` takes tasks, returns `(done, pending)` sets, never cancels or raises; modes ALL_COMPLETED / FIRST_COMPLETED / FIRST_EXCEPTION.
- `as_completed` = completion order (tag results); `gather` = input order.
- Semaphore caps concurrency, Lock guards read-modify-write across awaits, Queue gives backpressure (`task_done` + `join`), Event = flag, Condition = wait-for-state.
- TaskGroup: structured scope, siblings cancelled on failure, errors as `ExceptionGroup` (`except*`).
- Async generators/iterators power streaming; close them with `aclosing`. `await` inside a comprehension is sequential.
- `async with` for async setup/teardown; `AsyncExitStack` for dynamic sets. Offload blocking code with `to_thread`.
- Traps: forgotten await, dropped task refs, swallowed cancel, unbounded fan-out, nested `asyncio.run`.

## Interview angles
- "Coroutine vs task vs future?" Coroutine: a pausable function call. Task: a coroutine scheduled on the loop. Future: a low-level placeholder for a result that a task (or callback code) completes.
- "Is asyncio multi-threaded / parallel?" No: one thread, cooperative switching at `await`. Parallelism needs processes or GIL-free code.
- "Walk me through what happens when two tasks `await asyncio.sleep`." Each registers a timer and yields; the loop's selector sleeps until the earliest timer, then resumes that task.
- "How do you cap concurrent LLM calls, and limit requests per second?" Semaphore for in-flight count; token bucket (Lock + sleep) for rate; both inside a retry-with-backoff wrapper.
- "Design a hedged request / fallback." `wait(FIRST_COMPLETED)` with a delayed second attempt, cancel and await the loser; or `wait_for` with a cached fallback.
- "gather vs TaskGroup vs wait vs as_completed?" Ordered all-results; scoped fail-fast; low-level sets and modes; stream in completion order.
