"""
Exercise 01.2 - Text normalisation for an LLM pipeline              level: ★☆☆
topics: fundamentals.strings, fundamentals.list_comprehensions, fundamentals.dict_comprehensions,
        fundamentals.set_comprehensions, fundamentals.slicing

Problem
-------
Prefer comprehensions where they stay readable; a plain loop is fine where it is clearer.

1. normalize(text: str) -> str
     * casefold, split on any whitespace, strip punctuation (string.punctuation) from the
       *edges* of each token, drop tokens that become empty, join with a single space
     * "  Hello,   WORLD!!  -- it's  " -> "hello world it's"   (inner apostrophe kept)

2. vocabulary(docs: list[str], min_len: int = 1) -> set[str]
     * the set of distinct normalized tokens across all docs with len(token) >= min_len

3. first_positions(tokens: list[str]) -> dict[str, int]
     * token -> index of its FIRST occurrence; keys in first-occurrence order
     * ["a", "b", "a", "c"] -> {"a": 0, "b": 1, "c": 3}
     * careful: the obvious {t: i for i, t in enumerate(tokens)} gives the LAST occurrence

4. ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]
     * all contiguous n-grams in order; fewer than n tokens -> []
     * n <= 0 -> ValueError

5. strip_code_fence(text: str) -> str
     * LLMs often wrap JSON in a markdown fence: a "```json" line, the payload, a "```" line
     * if the stripped text starts with ``` : drop the first line (``` or ```json ...),
       drop the last line if it is exactly ```, return the remainder stripped
     * otherwise return text.strip()

Interviewers look for
---------------------
* Duplicate keys in a dict comprehension: the last write wins (iterate reversed, or use
  dict.setdefault in a loop, to keep the first).
* Strings are immutable: every strip/lower/replace returns a new string; build with
  " ".join(...), never += in a loop (quadratic in the worst case).
* casefold() vs lower() (German "ß" -> "ss") for case-insensitive matching.
* Slicing never raises IndexError: tokens[i:i+n] near the end just returns fewer items -
  which is why the range bound for ngrams matters.
"""


def normalize(text: str) -> str:
    raise NotImplementedError


def vocabulary(docs: list[str], min_len: int = 1) -> set[str]:
    raise NotImplementedError


def first_positions(tokens: list[str]) -> dict[str, int]:
    raise NotImplementedError


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    raise NotImplementedError


def strip_code_fence(text: str) -> str:
    raise NotImplementedError
