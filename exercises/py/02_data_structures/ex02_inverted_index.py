"""
Exercise 02.2 - Inverted index for keyword (BM25-style) retrieval   level: ★★☆
topics: data_structures.defaultdict, data_structures.namedtuple, data_structures.tuples,
        data_structures.dict_lookup, data_structures.list_vs_dict, data_structures.set_operations

Problem
-------
Hybrid RAG systems pair vector search with a keyword index. Build a tiny one.
Tokenisation: `text.lower().split()`.

`Hit` is given: a namedtuple("Hit", ["doc_id", "score"]).

1. build_index(docs: Mapping[str, str]) -> dict[str, dict[str, int]]
     * docs maps doc_id -> text; result maps term -> {doc_id: term_frequency}
     * return PLAIN dicts (outer and inner), not defaultdicts - see "Interviewers look for"

2. search(index, query, mode="and") -> list[Hit]
     * query terms = distinct tokens of the query
     * mode "and": docs containing ALL terms; mode "or": docs containing ANY term
     * score = sum of the doc's term frequencies over the query terms
     * sorted by score descending, then doc_id ascending
     * empty query -> []; unknown mode -> ValueError
     * must NOT modify the index (searching for an unknown term must not add it)

3. group_by(records: Iterable[dict], key: str) -> dict[Any, list[dict]]
     * group records by record[key] (records without the key go under None),
       keeping input order inside each group; returns a plain dict

Interviewers look for
---------------------
* defaultdict is great while BUILDING; returning it leaks a trap: `index["typo"]` silently
  inserts an empty entry (a read that writes). Convert to dict at the boundary, and use
  `.get()` / `in` for lookups.
* Term lookup is O(1) average in a dict vs O(n) scanning a list of (term, postings) pairs -
  the whole point of an inverted index.
* AND = set intersection of posting key-sets, OR = union; dict keys views support & and |.
  Intersect starting from the smallest posting list.
* namedtuples are tuples: immutable, unpackable, comparable, light-weight, readable fields.
"""

from collections import namedtuple
from collections.abc import Iterable, Mapping
from typing import Any

Hit = namedtuple("Hit", ["doc_id", "score"])


def build_index(docs: Mapping[str, str]) -> dict[str, dict[str, int]]:
    raise NotImplementedError


def search(index: Mapping[str, Mapping[str, int]], query: str, mode: str = "and") -> list[Hit]:
    raise NotImplementedError


def group_by(records: Iterable[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    raise NotImplementedError
