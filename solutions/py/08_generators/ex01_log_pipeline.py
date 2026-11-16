"""
Exercise 08.1 - Lazy log-processing pipeline                     level: ★☆☆
topics: generators.pipelines, generators.lazy_evaluation, generators.chaining

Problem
-------
Build three small generator stages that chain into a pipeline:

    parse(lines)            -> yields dicts {"level": str, "msg": str} from lines like
                               "ERROR  disk full". Skip blank lines and lines starting with "#".
                               Level is the first whitespace-separated word, upper-cased;
                               msg is the rest of the line, stripped.
    only(records, *levels)  -> yields records whose level is in `levels`
    take(n, iterable)       -> yields at most n items, then STOPS pulling from the source

    take(2, only(parse(lines), "ERROR"))

The whole point: the pipeline must be lazy. Pulling 2 errors from a 10 GB log
must only read as many lines as needed to find those 2 errors.

Rules: no lists that hold the whole input. Don't use itertools.islice in `take`
(write it yourself once; use islice in real life).

Interviewers look for
---------------------
* Why pull-based pipelines use O(1) memory regardless of input size.
* That `take` must not call next() one extra time (that would consume an item
  from a shared iterator).
"""

from collections.abc import Iterable, Iterator


def parse(lines: Iterable[str]) -> Iterator[dict[str, str]]:
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        level, _, msg = line.partition(" ")
        yield {"level": level.upper(), "msg": msg.strip()}


def only(records: Iterable[dict[str, str]], *levels: str) -> Iterator[dict[str, str]]:
    wanted = {lv.upper() for lv in levels}
    return (r for r in records if r["level"] in wanted)


def take(n: int, iterable: Iterable):
    if n <= 0:
        return
    for i, item in enumerate(iterable, 1):
        yield item
        if i >= n:
            return
