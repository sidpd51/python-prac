"""
Exercise 02.1 - Token statistics and deduplication                  level: ★☆☆
topics: data_structures.counter, data_structures.dedup, data_structures.sets,
        data_structures.set_operations, data_structures.lists

Problem
-------
Tokenisation in this exercise is simply `text.lower().split()`.

1. top_k_tokens(docs, k, stopwords=()) -> list[tuple[str, int]]
     * the k most frequent tokens across all docs, excluding stopwords (compare lowercased)
     * sorted by count descending, ties broken alphabetically (deterministic!)
     * k <= 0 -> []; fewer than k distinct tokens -> all of them

2. jaccard(a, b) -> float
     * Jaccard similarity of the token SETS of two texts: |A & B| / |A | B|
     * both texts without tokens -> 1.0 (two empty sets are identical)

3. dedup(items) -> list
     * remove duplicates, keep the FIRST occurrence, preserve order; items are hashable;
       works on any iterable (including a generator), O(n)

4. dedup_by(items, key) -> list
     * like dedup but two items are duplicates when key(item) is equal; keep the first
     * e.g. drop retrieved chunks whose whitespace-normalised text was already seen
       (results arrive sorted by score, so "first" == best)

Interviewers look for
---------------------
* Counter.most_common(k) breaks ties by insertion order - fine for a demo, not for a
  deterministic API/test. Sort by (-count, token) yourself.
* list(set(xs)) deduplicates but LOSES order; dict.fromkeys(xs) or a `seen` set keeps it.
* `x in some_list` inside a loop makes dedup O(n^2); a set makes membership O(1) average.
* Sets need hashable elements: dedup a list of dicts by a key, not by the dict.
"""

from collections import Counter
from collections.abc import Callable, Hashable, Iterable


def top_k_tokens(
    docs: Iterable[str], k: int, stopwords: Iterable[str] = ()
) -> list[tuple[str, int]]:
    if k <= 0:
        return []
    stop = {w.lower() for w in stopwords}
    counts = Counter(tok for doc in docs for tok in doc.lower().split() if tok not in stop)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def dedup[T: Hashable](items: Iterable[T]) -> list[T]:
    return list(dict.fromkeys(items))


def dedup_by[T](items: Iterable[T], key: Callable[[T], Hashable]) -> list[T]:
    seen: set[Hashable] = set()
    out: list[T] = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out
