"""
Exercise 19.4 - Processes for CPU: chunking, pickling, the GIL        level: ★★★
topics: internals.processpool, internals.multiprocessing, internals.gil, internals.cpu_vs_io,
        internals.concurrency_models

Problem
-------
Pure-Python CPU-bound work doesn't speed up with threads on the default CPython build
(the GIL lets one thread execute bytecode at a time). Processes each have their own
interpreter and GIL - but everything crossing the process boundary is PICKLED.

1. gil_status() -> {"free_threaded_build": bool, "gil_enabled": bool}
     free_threaded_build - interpreter was built with --disable-gil (sysconfig config var
                           "Py_GIL_DISABLED" == 1)
     gil_enabled         - the GIL is active right now: sys._is_gil_enabled() where it
                           exists (3.13+), else True. (A free-threaded build can re-enable
                           the GIL, e.g. PYTHON_GIL=1 or an incompatible C extension.)

2. split_ranges(n, parts) -> list[range]
   Split range(n) into at most `parts` contiguous, non-empty ranges whose lengths differ
   by at most 1 (longer ones first), covering 0..n-1 in order.
       split_ranges(10, 3) == [range(0, 4), range(4, 7), range(7, 10)]
       split_ranges(2, 5)  == [range(0, 1), range(1, 2)]
       split_ranges(0, 3)  == []
   parts < 1 or n < 0 -> ValueError.

3. parallel_sum_of_squares(n, workers=4) -> int
   sum(i*i for i in range(n)) computed with a ProcessPoolExecutor: one task per range from
   split_ranges(n, workers). The worker function must be picklable AND importable by a
   child process - use the builtin math.sumprod(r, r) (3.12+) on each range (ranges
   pickle cheaply: start/stop/step, not a list of numbers).

4. run_cpu_bound(func, items, max_workers=4, chunksize=1) -> list
   Map func over items in a ProcessPoolExecutor, results in input order.
   BEFORE starting any process, verify func can be pickled (pickle.dumps); if not, raise
   TypeError("... not picklable ...") chained from the original error. Lambdas and
   functions defined inside other functions can't be pickled. Exceptions raised inside a
   worker propagate to the caller with their original type.

Note: child processes find a function by importing its module by name. With the "spawn"
start method (macOS/Windows) and "forkserver" (Linux default since 3.14) a function from a
module loaded by file path isn't importable in the child - stick to stdlib/builtin workers
here. In real code, put worker functions at module top level and guard entry points with
`if __name__ == "__main__":`.

Interviewers look for
---------------------
* CPU-bound -> processes (or C extensions / NumPy that release the GIL); I/O-bound ->
  threads or asyncio.
* Process overhead: startup, pickling args/results both ways. Send small descriptions of
  work (a range, a file path), not big data; use chunksize for many tiny tasks.
* Free-threaded CPython (PEP 703): experimental in 3.13 (python3.13t), officially supported
  but still optional in 3.14 (PEP 779) - the default build still has a GIL.
"""

from collections.abc import Callable, Iterable
from typing import Any


def gil_status() -> dict[str, bool]:
    raise NotImplementedError


def split_ranges(n: int, parts: int) -> list[range]:
    raise NotImplementedError


def parallel_sum_of_squares(n: int, workers: int = 4) -> int:
    raise NotImplementedError


def run_cpu_bound(
    func: Callable[[Any], Any], items: Iterable[Any], max_workers: int = 4, chunksize: int = 1
) -> list[Any]:
    raise NotImplementedError
