import re

import pytest

from study.loader import load

ex = load(__file__, "ex01_unit_suite")


def assert_suite_fails(run_suite, *args):
    """Like a red pytest run: ANY exception counts as "the suite caught the bug".
    NotImplementedError is re-raised so an unsolved stub shows up as TODO, not as a pass."""
    try:
        run_suite(*args)
    except NotImplementedError:
        raise
    except (Exception, pytest.fail.Exception):
        return
    raise AssertionError("your suite PASSED on a buggy implementation - add a check for it")


def _make(*, pattern=r"\bscore\s*:\s*(\d+)", flags=re.IGNORECASE, pick=-1, check=None, missing=None):
    rx = re.compile(pattern, flags)

    def impl(text: str) -> int | None:
        matches = rx.findall(text)
        if not matches:
            return missing
        value = int(matches[pick])
        ok = check(value) if check else 0 <= value <= 10
        return value if ok else None

    return impl


MUTANTS = {
    "first_match_wins": _make(pick=0),
    "case_sensitive": _make(flags=0),
    "requires_space_after_colon": _make(pattern=r"\bscore\s*: (\d+)"),
    "no_upper_bound": _make(check=lambda v: v >= 0),
    "rejects_10": _make(check=lambda v: 0 <= v < 10),
    "rejects_0_truthiness": _make(check=lambda v: bool(v) and v <= 10),
    "missing_returns_0": _make(missing=0),
}


@pytest.mark.topic("py.testing.unit", "py.testing.pytest")
def test_suite_passes_on_correct_impl():
    ex.run_suite(ex.extract_score)


@pytest.mark.topic("py.testing.unit")
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_suite_catches_mutant(name):
    assert_suite_fails(ex.run_suite, MUTANTS[name])


@pytest.mark.topic("py.testing.parametrize")
def test_cases_table_is_correct_and_rich():
    rows = ex.cases()
    assert len(rows) >= 8
    inputs = [text for text, _ in rows]
    assert len(set(inputs)) == len(inputs), "duplicate inputs add no information"
    assert sum(expected is None for _, expected in rows) >= 2
    for text, expected in rows:
        assert ex.extract_score(text) == expected, f"row {text!r} is wrong"


@pytest.mark.topic("py.testing.parametrize")
def test_each_mutant_is_caught_by_some_row():
    # the table itself (not extra hand-written asserts) should kill every mutant
    rows = ex.cases()
    for name, mutant in MUTANTS.items():
        assert any(mutant(text) != expected for text, expected in rows), (
            f"no row in cases() distinguishes mutant {name!r}"
        )
