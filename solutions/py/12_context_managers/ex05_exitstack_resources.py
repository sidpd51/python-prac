"""
Exercise 12.5 - ExitStack: N resources, partial failure, handoff  level: ★★★
topics: context_managers.exitstack, context_managers.resources,
        context_managers.cleanup, context_managers.custom

Problem
-------
You can't write `with open(a), open(b), ...` when the number of resources is only known
at runtime (N output shards, N model clients). contextlib.ExitStack solves that.

1. open_shards(directory, n, *, prefix="shard") - context manager
     * opens n files directory/f"{prefix}-{i:05d}.jsonl" for writing (UTF-8) and yields
       the list of file objects
     * if opening file k fails, files 0..k-1 are closed and the error propagates
     * all files are closed when the block exits (normally or with an exception)
     * n < 1 -> ValueError

2. class ShardedWriter(directory, n)
     * `with ShardedWriter(d, 4) as w:` opens the shards (reuse open_shards or an
       ExitStack directly); w.write(record: dict) appends one JSON line, round-robin
       over the shards: record 0 -> shard 0, record 1 -> shard 1, ...
     * w.counts -> list[int] records written per shard
     * __exit__ closes everything and never swallows exceptions
     * if __enter__ fails half-way, nothing stays open

3. acquire_all(factories) -> list
     * calls each factory() to create a client (objects with .close())
     * if factory k raises: close the clients 0..k-1 in REVERSE order, re-raise
     * on success return all clients OPEN - ownership passes to the caller
       (hint: ExitStack.callback + pop_all())

Interviewers look for
---------------------
* ExitStack.enter_context / callback / push, LIFO unwind, and pop_all() to transfer
  ownership out of a "construct or clean up" block.
* Partial-failure cleanup: the bug where file #7 fails to open and #0-#6 leak.
* Implementing __enter__/__exit__ on a class by delegating to an internal ExitStack.
"""

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Protocol, TypeVar


class Closeable(Protocol):
    def close(self) -> object: ...


C = TypeVar("C", bound=Closeable)


@contextmanager
def open_shards(directory: str | Path, n: int, *, prefix: str = "shard") -> Iterator[list[IO[str]]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    directory = Path(directory)
    with ExitStack() as stack:
        files: list[IO[str]] = [
            stack.enter_context(open(directory / f"{prefix}-{i:05d}.jsonl", "w", encoding="utf-8"))
            for i in range(n)
        ]
        yield files


class ShardedWriter:
    def __init__(self, directory: str | Path, n: int) -> None:
        self.directory = Path(directory)
        self.n = n
        self.counts: list[int] = [0] * n
        self._files: list[IO[str]] = []
        self._stack: ExitStack | None = None
        self._next = 0

    def __enter__(self) -> "ShardedWriter":
        with ExitStack() as stack:
            self._files = stack.enter_context(open_shards(self.directory, self.n))
            self._stack = stack.pop_all()  # success: keep everything open
        self.counts = [0] * self.n
        self._next = 0
        return self

    def write(self, record: dict[str, Any]) -> None:
        i = self._next
        self._files[i].write(json.dumps(record, ensure_ascii=False) + "\n")
        self.counts[i] += 1
        self._next = (i + 1) % self.n

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        stack, self._stack = self._stack, None
        if stack is not None:
            stack.close()


def acquire_all(factories: Sequence[Callable[[], C]]) -> list[C]:
    with ExitStack() as stack:
        clients: list[C] = []
        for factory in factories:
            client = factory()
            stack.callback(client.close)
            clients.append(client)
        stack.pop_all()  # nothing failed: caller now owns the clients
    return clients
