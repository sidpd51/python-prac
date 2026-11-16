# Internals, concurrency and performance

"Python is slow" and "Python has a GIL" are the two statements you must be able to unpack at 3-4 years of
experience. This note covers what CPython actually does with your code (bytecode on a stack machine), why
only one thread runs Python at a time, and which tool to reach for: threads, processes or asyncio. It ends
with the part interviewers care most about: how to find the real bottleneck with `cProfile` and `timeit`
and fix it honestly. In AI engineering this decides how you fan out LLM calls, tokenise or parse in
parallel, and why a "faster" rewrite sometimes changes nothing.

## CPython fundamentals
<!-- topics: cpython -->

**Python** is a language; **CPython** is the reference implementation written in C that you run when you type
`python`. Others exist (PyPy has a JIT, Jython/IronPython target the JVM/.NET), but behaviour such as refcounting
and the GIL is CPython-specific.

```text
source .py --> parser --> AST --> compiler --> bytecode (code object)
                                                   |
                                      cached in __pycache__/*.pyc (skips compile next import)
                                                   v
                                   evaluation loop (a C `switch` over opcodes)
                                   executes on a stack machine, objects are C structs
```

Key facts:

* Python is **compiled to bytecode, then interpreted**. "Interpreted vs compiled" is a false dichotomy.
* Every operation on a Python object goes through dynamic dispatch: `a + b` must look at the types of `a` and
  `b` at runtime. That, plus boxing every int/float as a heap object with a refcount, is the main cost.
* A `.pyc` file is just cached bytecode for imports; the main script is compiled in memory each run.
* Since 3.11 the "specializing adaptive interpreter" rewrites hot opcodes in place for the types it keeps seeing
  (e.g. `BINARY_OP` on two ints). 3.13 ships an experimental copy-and-patch JIT (off by default, build flag); 3.14
  keeps it experimental. Don't promise users it makes anything faster.

```python
import sys
print(sys.implementation.name)   # prints: cpython
print(sys.version_info >= (3, 12))   # prints: True
```

> **Interview** "Is Python interpreted or compiled?" Both: compiled to bytecode, interpreted by a VM. Follow with
> "which is why the speed fix is usually fewer bytecodes executed (vectorise, use C-implemented builtins), not
> micro-tuning syntax."

## Bytecode
<!-- topics: bytecode -->

A function's compiled form lives in `func.__code__`. Bytecode is a sequence of tiny instructions that push and
pop values on a per-frame **evaluation stack**. Local variables live in numbered slots (`LOAD_FAST`), constants in a
table (`LOAD_CONST`).

```text
c = a + b           LOAD_FAST a ; LOAD_FAST b ; BINARY_OP + ; STORE_FAST c
                     stack: [a]      [a, b]       [a+b]        []
```

```python
def f(a, b):
    c = a + b
    return c * 2

code = f.__code__
print(code.co_varnames)   # prints: ('a', 'b', 'c')
print(code.co_argcount)   # prints: 2
print(type(code.co_code).__name__)  # prints: bytes
```

Why you care: more instructions executed means more time; local lookups (`LOAD_FAST`) are cheaper than
global/attribute lookups (`LOAD_GLOBAL`, `LOAD_ATTR`); a comprehension runs its loop in one tight bytecode
loop (`LIST_APPEND`) instead of calling `.append` for every item.

> **Gotcha** Bytecode is an implementation detail and **changes between minor versions**. The same function
> disassembles with `LOAD_FAST_LOAD_FAST` on 3.13 and `LOAD_FAST_BORROW_LOAD_FAST_BORROW` plus `LOAD_SMALL_INT` on 3.14.
> Never write code or tests that depend on exact opcode sequences; check for the *presence* of an opcode family.

## The dis module
<!-- topics: dis -->

`dis.dis(func)` prints a human-readable listing; `dis.get_instructions(func)` yields `Instruction` objects you can
analyse (`opname`, `argrepr`, `starts_line`).

```python
import dis

def uses_global():
    return len([1, 2])      # LOAD_GLOBAL for the builtin `len`

def uses_local(n):
    return n * 2

def opnames(fn):
    return {ins.opname for ins in dis.get_instructions(fn)}

print("LOAD_GLOBAL" in opnames(uses_global))   # prints: True
print("LOAD_GLOBAL" in opnames(uses_local))    # prints: False
print("RETURN_VALUE" in opnames(uses_local))   # prints: True
```

Typical uses: confirm a constant was folded (`2 * 3` is computed at compile time, so no multiply is left in the bytecode), see why a closure variable
is slower (`LOAD_DEREF`), see where a `with` or `try` adds work, and compare two implementations of the same
thing. For ordinary performance questions prefer measuring (profiling sections below) over reading bytecode.

```python
import dis

def folded():
    return 2 * 3 * 700

print(4200 in folded.__code__.co_consts)   # prints: True  (computed at compile time)
print("BINARY_OP" in {i.opname for i in dis.get_instructions(folded)})   # prints: False
```

<details markdown="1"><summary>Check yourself: why is a `for` loop calling `lst.append(x)` slower than a list comprehension?</summary>

The loop does a global/attribute lookup (`lst.append`) and a real method call for every item through generic
bytecode. The comprehension uses the dedicated `LIST_APPEND` opcode inside one loop, so far fewer instructions run.

</details>

## The GIL
<!-- topics: gil -->

The **Global Interpreter Lock** is a mutex in CPython: **only one thread executes Python bytecode at a time** per
process. Why does it exist? Reference counts (`ob_refcnt += 1`) are updated constantly and are not atomic. Rather
than lock every object, CPython uses one lock around the interpreter. That makes single-threaded code fast and C
extensions simple, at the cost of CPU parallelism across threads.

```text
Thread A: |run----|        |run----|
Thread B:          |run----|        |run---|     only one holds the GIL at any instant
                   ^ switch (every ~5 ms, or when a thread blocks on I/O)
```

What the GIL does and does not do:

* **CPU-bound Python in 2 threads takes about as long as running sequentially** (measured: two 3M-iteration
  loops, 0.15 s sequential vs 0.15 s in two threads).
* **I/O-bound threads overlap well**: a thread waiting on a socket, `time.sleep`, or file read **releases** the GIL.
* C extensions can release it for long computations; **NumPy, hashing, compression, many tokenizers** do, so
  threads *can* speed up those.
* The GIL does **not** make your code thread-safe: `counter += 1` is several bytecodes (load, add, store), and a
  thread switch can happen between them. Compound operations need a `threading.Lock`.

```python
import sys
print(sys._is_gil_enabled())   # prints: True   (a normal build, 3.13 and 3.14)
```

**Free-threaded builds.** PEP 703 made the GIL optional. In 3.13 an experimental separate build (`python3.13t`) can
run with it off; in 3.14 free-threading is officially supported (PEP 779) but still a separate build and not the
default. There, threads really run Python in parallel, refcounts use atomic/biased counting, and single-thread code is
somewhat slower. `sysconfig.get_config_var("Py_GIL_DISABLED")` is `1` on such a build (here: `0`). Your code
must still use locks for shared mutable state, and some C extensions re-enable the GIL on import until updated.

> **Interview** "How do you parallelise CPU-bound Python?" Processes (`multiprocessing` /
> `ProcessPoolExecutor`), or move the loop into C-backed code (NumPy, Rust, Cython), or (new) a free-threaded build.
> "Threads" is the wrong answer for pure-Python CPU work, the right one for blocking I/O.

## Threading
<!-- topics: threading -->

`threading.Thread` runs a function concurrently in the same process, sharing memory. Sharing is the benefit
(cheap, no serialisation) and the danger (races).

```python
import threading

counter = 0
lock = threading.Lock()

def work():
    global counter
    for _ in range(100_000):
        with lock:            # without the lock, updates can be lost
            counter += 1

threads = [threading.Thread(target=work) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)   # prints: 400000
```

Tools: `Lock`, `RLock` (re-entrant), `Semaphore` (limit concurrency, e.g. max 5 in-flight API calls), `Event`
(signal), `Queue` (thread-safe producer/consumer, the preferred way to share work), `threading.local` (per-thread
state). `daemon=True` threads die when the main thread exits without cleanup.

> **Gotcha** A race without a lock *sometimes* gives the right answer, so tests pass and production corrupts.
> Prefer message passing (`queue.Queue`) or immutable data over shared mutable state. Always `join()` or use an
> executor so exceptions and shutdown are handled; an exception in a bare thread just prints and the thread dies.

## concurrent.futures and ThreadPoolExecutor
<!-- topics: futures, threadpool -->

`concurrent.futures` is the high-level API: submit callables to a pool, get back **`Future`** objects (a promise of a
result). `ThreadPoolExecutor` and `ProcessPoolExecutor` share the same interface, so switching is a one-line change.

```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch(i):
    time.sleep(0.1 * (3 - i))          # stand-in for a blocking HTTP call
    if i == 1:
        raise ValueError("bad 1")
    return i * 10

with ThreadPoolExecutor(max_workers=3) as ex:
    futures = {ex.submit(fetch, i): i for i in range(3)}
    for fut in as_completed(futures):       # yields in completion order
        i = futures[fut]
        try:
            print(i, fut.result())          # .result() re-raises the worker's exception
        except ValueError as e:
            print(i, "failed:", e)
# prints: 2 20
# prints: 1 failed: bad 1
# prints: 0 0
```

| API | Order | Errors |
|---|---|---|
| `ex.submit(fn, *a)` -> `Future` | you decide | `future.result()` / `.exception()` |
| `ex.map(fn, items)` | **input order** | raised when you iterate to that item |
| `as_completed(futs)` | **completion order** | per future |

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(3) as ex:
    print(list(ex.map(lambda n: n * n, range(5))))   # prints: [0, 1, 4, 9, 16]
```

The `with` block waits for all tasks and shuts the pool down. `max_workers` is your concurrency cap, the
natural place to respect an API rate limit. `future.result(timeout=...)` bounds waiting, but you **cannot kill a
running thread**; `cancel()` only works on tasks not yet started.

> **Gotcha** `ex.map` hides a failure until you reach that element in the iteration, and stops yielding there.
> If you want "collect every success and every failure", use `submit` + `as_completed` and catch per future.

> **AI use** Calling a sync LLM or embedding SDK for 500 prompts: a pool of 8-16 threads cuts wall time roughly
> by the pool size until you hit the provider's rate limit. Keep result ordering by index, retry per task.

## Multiprocessing and ProcessPoolExecutor
<!-- topics: multiprocessing, processpool -->

A separate **process** has its own interpreter, own GIL and own memory, so CPU-bound Python runs truly parallel
on multiple cores. The price: everything crossing the boundary is **pickled** (serialised) and copied, and
starting processes costs tens of milliseconds.

```text
parent  --pickle(args)-->  worker 1 (own GIL)  --pickle(result)-->  parent
        --pickle(args)-->  worker 2 (own GIL)  --pickle(result)-->
```

```python
# norun
# run as a script file; guard is mandatory
from concurrent.futures import ProcessPoolExecutor

def cpu(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as ex:
        print(list(ex.map(cpu, [1000, 2000], chunksize=1)))
# prints: [332833500, 2664667000]
```

(Verified on 3.13 and 3.14; the example is `norun` because pool start-up depends on how the checker launches code.)

Rules that bite:

* **`if __name__ == "__main__":` is required** wherever the start method is `spawn` or `forkserver` (workers re-import
  your module; without the guard they spawn more workers).
* The function and its arguments must be **picklable**: top-level functions yes; lambdas, nested functions, open
  files, locks, DB connections no (`PicklingError`).
* Start methods: `fork` (copy of the parent, fast, unsafe with threads), `spawn` (fresh interpreter, default on
  macOS/Windows), `forkserver`. Linux default is `fork` on 3.13 and **`forkserver` from 3.14**: code that
  silently relied on inherited globals via fork can break on 3.14.
* Use `chunksize` with `map` for many small tasks to amortise pickling and IPC; a task that takes less time than
  being sent through the pipe makes the pool slower than a loop.
* Pass **data in, data out**; mutating a global in a worker changes only that worker's copy.

`multiprocessing` offers lower-level pieces (`Process`, `Queue`, `Pipe`, `Pool`, shared memory via
`multiprocessing.shared_memory`) but for most work `ProcessPoolExecutor` is enough.

> **Interview** "Why is my multiprocessing code slower than the loop?" Tasks too small (overhead dominates),
> large arguments pickled per call, or the work was I/O or NumPy-bound (already releasing the GIL).

## CPU-bound vs I/O-bound, and choosing a model
<!-- topics: cpu_vs_io, concurrency_models -->

First ask: **where does the time go?** If the program is mostly *waiting* (network, disk, a model API), it is
I/O-bound; if it is mostly *computing in Python*, it is CPU-bound. Waiting can overlap, computing needs more cores.

| Workload | Best tool | Why |
|---|---|---|
| Many blocking calls, sync libraries (requests, DB driver) | `ThreadPoolExecutor` | waits release the GIL; easy |
| Thousands of concurrent connections, async libraries (httpx async, aiohttp) | `asyncio` | one thread, tiny per-task cost, cancellation and timeouts |
| Pure-Python CPU work (parsing, scoring, feature loops) | `ProcessPoolExecutor` | separate GILs; pays pickling |
| Numeric work on arrays | NumPy/BLAS, maybe threads | C code releases the GIL; no Python loop |
| Mixed: async app calling CPU work | `loop.run_in_executor(process_pool)` | keep the event loop responsive |

```text
asyncio:  1 thread, tasks voluntarily yield at `await`      cheap, thousands of tasks, needs async libs
threads:  N threads, OS switches them (preemptive)          simple with blocking code, races possible
processes: N interpreters, isolated memory                  true CPU parallelism, serialisation cost
```

A rule of thumb for LLM systems: calling hosted models is I/O-bound (use asyncio or threads, cap with a
semaphore, respect rate limits); local embedding/inference is compute-bound but runs in native code (batch it, let
the library use cores/GPU); pure-Python post-processing at scale is where processes pay off.

> **Interview** Give the decision table, then add the caveat: "I'd measure first. Async only helps if the
> libraries are async; threads are the quick win for legacy sync SDKs; processes only when profiling shows
> Python-level CPU time."

<details markdown="1"><summary>Check yourself: you must hit an API 1000 times with a sync client. Threads or processes?</summary>

Threads. The time is spent waiting on the network, which releases the GIL. Processes would add pickling and
start-up cost for no gain.

</details>

## Profiling: measure, don't guess
<!-- topics: profiling, cprofile -->

**Profiling** finds where time (or memory) goes so you optimise the 5% that matters. The workflow: make it
correct, **measure** with realistic input, find the hot spot, change one thing, measure again.

`cProfile` is a deterministic profiler (C implementation, low overhead) that records every function call:
how many times, time inside the function itself (`tottime`) and including callees (`cumtime`).

```python
import cProfile
import pstats

def slow():
    return sum(i * i for i in range(1000))

def main():
    for _ in range(50):
        slow()

prof = cProfile.Profile()
prof.enable()
main()
prof.disable()
stats = pstats.Stats(prof)
calls = {name: row[1] for (_, _, name), row in stats.stats.items() if name in ("main", "slow")}
print(calls)   # prints: {'main': 1, 'slow': 50}
stats.sort_stats("cumulative")   # then stats.print_stats(5) prints the table below (timings vary)
```

```text
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.007    0.007 script.py:32(main)
       50    0.000    0.000    0.007    0.000 script.py:31(slow)
       50    0.004    0.000    0.007    0.000 {built-in method builtins.sum}
    50050    0.003    0.000    0.003    0.000 script.py:31(<genexpr>)
```

Reading it: sort by `cumulative` to find the slow *path*, by `tottime` to find the slow *function*. High `ncalls`
with tiny per-call time means call overhead (inline or vectorise); high `tottime` in one function means that code is the
hot spot. From a shell: `python -m cProfile -s cumtime script.py`. Visualise with `snakeviz`.

Limits: profiling adds overhead and distorts tiny functions; it shows Python-level calls, not time inside a C
extension's internals; for production, a **sampling** profiler (`py-spy`) attaches to a live process without
code changes. For line-level detail use `line_profiler`. Memory is `tracemalloc` (see the memory note).

> **AI use** A RAG request takes 3 s. Profile or just time each stage (embed, search, rerank, LLM call, parse).
> Almost always the model call dominates and "optimising" your Python loop is wasted effort.

## timeit: micro-benchmarks done right
<!-- topics: timeit -->

`timeit` times a small snippet many times, disables the garbage collector during timing, and uses a high-resolution clock.
One measurement is noise; take **the minimum of several repeats**, because other processes only ever make a run slower.

```python
import timeit

setup = "data = list(range(1000))"
t_list = min(timeit.repeat("999 in data", setup, number=2000, repeat=5))
t_set = min(timeit.repeat("999 in s", setup + "; s = set(data)", number=2000, repeat=5))
print(t_set < t_list)   # prints: True   (hash lookup vs linear scan)
```

CLI form: `python -m timeit -s "setup" "stmt"` picks the loop count automatically. In notebooks use `%timeit`.

> **Gotcha** Benchmarking pitfalls: timing the setup together with the statement, measuring once, tiny inputs that
> hide the O(n) vs O(1) difference, and results the optimiser caches (constant folding turns `2*3` into a constant, so it
> measures nothing). Use realistic sizes and compare more than one.

Use `time.perf_counter()` for a one-off stopwatch around real code; use `timeit` for comparing snippets.

## Optimization techniques
<!-- topics: optimization -->

Honest optimisation order: (1) is it correct and is there a test? (2) measure and find the hot spot; (3) fix the
**algorithm / data structure** first; (4) then reduce Python-level work; (5) then change the execution model; (6) re-measure and keep
only changes that win on realistic data.

**1. Complexity beats micro-tuning.** `x in list` is O(n); `x in set` is O(1). A nested loop that compares
every pair is O(n^2); a dict index makes it O(n).

```python
def dedupe_slow(items):
    out = []
    for x in items:
        if x not in out:         # O(n) scan inside an O(n) loop: O(n^2)
            out.append(x)
    return out

def dedupe_fast(items):
    return list(dict.fromkeys(items))   # O(n), keeps first-seen order

xs = [3, 1, 3, 2, 1]
print(dedupe_slow(xs), dedupe_fast(xs))   # prints: [3, 1, 2] [3, 1, 2]
```

**2. Use the built-ins and the right structure.** `sum`, `sorted`, `max(key=...)`, `str.join`, `collections.Counter`,
`heapq.nlargest`, `bisect` are C-speed. Build strings with `"".join(parts)`, not `s += part` in a loop. Use a
`deque` for queues (`list.pop(0)` is O(n)).

**3. Do work once.** Hoist invariants out of loops, cache pure results (`functools.lru_cache`), compile a regex
once, avoid repeated attribute lookups in a hot loop (`append = out.append`), read each file once.

```python
from functools import lru_cache

calls = 0

@lru_cache(maxsize=None)
def embed(text):
    global calls
    calls += 1
    return len(text)       # stand-in for an expensive embedding call

for t in ["a", "bb", "a", "a", "bb"]:
    embed(t)
print(calls, embed.cache_info().hits)   # prints: 2 3
```

**4. Leave the Python loop.** Vectorise with NumPy/pandas (the loop runs in C), or use generators to avoid
materialising. **5. Change the model:** threads/asyncio for I/O, processes for CPU. **6. Last resort:** Cython, Numba, Rust
(PyO3), or just a faster algorithm elsewhere.

> **Remember** Premature optimisation is guessing. A 10x faster function that is 2% of the runtime gives a
> 2% win. Profile first, and keep a correctness test next to every optimisation.

> **Interview** "How would you speed up slow Python?" Profile, then algorithm/data structure, then
> vectorisation or built-ins, then the right concurrency model, then compiled extensions. Mention that you re-measure.

## Quick recap
- CPython compiles source to bytecode and interprets it on a stack VM; `.pyc` caches imports; 3.11+ specializes hot opcodes.
- `__code__` and `dis` expose bytecode; opcodes change between versions, so test for presence, not exact sequences.
- The GIL lets one thread run bytecode at a time because refcounting is not atomic; I/O and many C extensions release it.
- The GIL does not make compound operations atomic: use locks or queues. Free-threaded builds (3.13 experimental, 3.14 supported, not default) remove it.
- Threads: shared memory, races, great for blocking I/O. Processes: own GIL, pickling, great for CPU.
- `concurrent.futures`: `submit` + `as_completed` (per-task errors) vs `map` (input order, error on iteration); `result()` re-raises.
- Process pools need the `__main__` guard, picklable callables, chunking; Linux default start method is fork on 3.13 and forkserver on 3.14.
- Decision: blocking I/O -> threads, many async connections -> asyncio, Python CPU -> processes, array math -> NumPy.
- `cProfile`: `ncalls`, `tottime`, `cumtime`; sort by cumulative for paths, tottime for hot functions; `py-spy` for live processes.
- `timeit`: repeat, take the minimum, use realistic sizes. Fix complexity first, then built-ins, caching, vectorising; always re-measure.
