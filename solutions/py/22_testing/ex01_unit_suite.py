"""
Exercise 22.1 - Table-driven unit tests for an LLM-judge score parser   level: ★☆☆
topics: testing.unit, testing.parametrize, testing.pytest

Problem
-------
This exercise is inverted: the code under test is GIVEN (`extract_score`, below - don't
edit it). YOU write the tests. We grade them by *mutation testing*: we run your suite
against the correct implementation (it must pass) and against several subtly broken
copies ("mutants") - your suite must FAIL on every one of them.

Spec of extract_score(text) -> int | None   (an LLM judge replies in free text)
  * finds "score: N" where N is a non-negative integer and "score" is a whole word
  * case-insensitive; any whitespace (including none) around the colon:
    "Score:7", "SCORE :  7", "score: 7/10" are all 7
  * several scores in the text -> the LAST one wins (judges revise themselves)
  * valid range is 0..10 inclusive; out of range -> None
  * no score at all -> None   (None, not 0: "no verdict" is not "terrible")

1. cases() -> list[tuple[str, int | None]]
   A table of (judge_reply, expected) rows - exactly what you would feed to
   @pytest.mark.parametrize. At least 8 rows, no duplicate inputs, at least two rows
   expecting None, and every row must be correct for the real extract_score.
   The table alone must be strong enough to expose every mutant.

2. run_suite(impl) -> None
   Check that `impl` (same signature as extract_score) meets the spec, e.g. by looping
   over cases(). Fail with a plain `assert` (pytest.raises / pytest.fail, or any other
   exception, also count - just like a red pytest test).
   Must not raise for the correct implementation.

Interviewers look for
---------------------
* Equivalence classes + boundary values: the bugs live at 0, 10, 11, "no match" and
  "more than one match". One row per class, not twenty rows of the happy path.
* Assert the exact value: `assert impl(x) == 0` - `assert not impl(x)` can't tell
  0 from None (the classic truthiness bug).
* Good failure messages / parametrize ids: a red test should say WHICH case broke.

Hints (only after 15 honest minutes)
------------------------------------
1. For every rule in the spec, ask "what one-character bug would break only this rule?"
   and write the row that catches it.
"""

import re

__stub_keep__ = ["extract_score"]

SCORE_RE = re.compile(r"\bscore\s*:\s*(\d+)", re.IGNORECASE)


def extract_score(text: str) -> int | None:
    """Code under test - correct implementation (given)."""
    matches = SCORE_RE.findall(text)
    if not matches:
        return None
    value = int(matches[-1])
    return value if 0 <= value <= 10 else None


def cases() -> list[tuple[str, int | None]]:
    return [
        ("Score: 7", 7),
        ("score:7/10", 7),  # no space
        ("SCORE :   4", 4),  # casing + extra spaces
        ("Reasoning... Score: 0", 0),  # lower boundary (truthiness bug)
        ("Score: 10/10 - perfect", 10),  # upper boundary
        ("Score: 11", None),  # just out of range
        ("Score: 3. On reflection, score: 8", 8),  # last one wins
        ("The answer is fine, I'd give it 9.", None),  # no score
        ("", None),
    ]


def run_suite(impl) -> None:
    for text, expected in cases():
        got = impl(text)
        assert got == expected, f"impl({text!r}) == {got!r}, expected {expected!r}"
