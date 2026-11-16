"""
Exercise 08.5 - Batching & sliding windows for embedding jobs       level: ★★☆
topics: generators.memory_efficiency, generators.list_vs_generator, genai.batching

Problem
-------
Embedding APIs accept at most N inputs per request, and your corpus may not fit in RAM.

1. batched(iterable, n) -> yields tuples of up to n items (last may be shorter).
   n < 1 -> ValueError raised at call time. Must work on infinite iterators.
   Don't use itertools.batched (write it with itertools.islice or a manual loop).

2. sliding_window(iterable, n) -> yields tuples of n consecutive items:
       list(sliding_window("abcd", 2)) == [("a","b"), ("b","c"), ("c","d")]
   If the input has fewer than n items, yield nothing. Use collections.deque(maxlen=n)
   so memory is O(n), not O(len(input)).

3. token_budget_batches(texts, max_tokens, count_tokens=len)
   -> yields lists of texts whose summed count_tokens(...) <= max_tokens, preserving
   order, greedily packing. A single text over budget is yielded alone in its own
   batch (the caller decides what to do with it). Empty input -> nothing.

Interviewers look for
---------------------
* O(batch) memory regardless of corpus size.
* Batch by token budget, not just count - real LLM/embedding APIs limit both.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    raise NotImplementedError


def sliding_window(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    raise NotImplementedError


def token_budget_batches(
    texts: Iterable[str], max_tokens: int, count_tokens: Callable[[str], int] = len
) -> Iterator[list[str]]:
    raise NotImplementedError
