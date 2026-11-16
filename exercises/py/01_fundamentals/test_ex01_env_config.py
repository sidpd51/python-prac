import pytest

from study.loader import load

ex = load(__file__, "ex01_env_config")


@pytest.mark.topic("py.fundamentals.type_conversion", "py.fundamentals.variables")
@pytest.mark.parametrize(
    ("raw", "expected", "typ"),
    [
        ("42", 42, int),
        ("-3", -3, int),
        ("+7", 7, int),
        ("1_000", 1000, int),
        ("0.7", 0.7, float),
        ("1e-4", 1e-4, float),
        ("3.0", 3.0, float),
        ("gpt-4o-mini", "gpt-4o-mini", str),
        ("  spaced  ", "spaced", str),
    ],
)
def test_coerce_numbers_and_strings(raw, expected, typ):
    out = ex.coerce(raw)
    assert out == expected
    assert type(out) is typ


@pytest.mark.topic("py.fundamentals.type_conversion", "py.fundamentals.truthiness")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("TRUE", True), ("yes", True), ("On", True),
     ("false", False), ("False", False), ("no", False), ("OFF", False)],
)
def test_coerce_booleans(raw, expected):
    assert ex.coerce(raw) is expected


@pytest.mark.topic("py.fundamentals.none")
@pytest.mark.parametrize("raw", ["", "   ", "none", "None", "NULL"])
def test_coerce_none(raw):
    assert ex.coerce(raw) is None


@pytest.mark.topic("py.fundamentals.strings")
@pytest.mark.parametrize(
    ("raw", "expected"), [('"42"', "42"), ("'true'", "true"), ('""', ""), ("'a b'", "a b")]
)
def test_coerce_quoted_values_stay_strings(raw, expected):
    assert ex.coerce(raw) == expected
    assert isinstance(ex.coerce(raw), str)


@pytest.mark.topic("py.fundamentals.loops", "py.fundamentals.loop_control", "py.fundamentals.strings")
def test_parse_env_full_file():
    text = """
# LLM settings
export MODEL=claude-small   # cheap one
TEMPERATURE = 0.2
MAX_TOKENS=1024

STREAM=false
DSN=postgres://h/db?sslmode=require
SYSTEM_PROMPT="You are #1 at helping"
EMPTY=
MAX_TOKENS=2048
"""
    assert ex.parse_env(text) == {
        "MODEL": "claude-small",
        "TEMPERATURE": 0.2,
        "MAX_TOKENS": 2048,  # later duplicate wins
        "STREAM": False,
        "DSN": "postgres://h/db?sslmode=require",
        "SYSTEM_PROMPT": "You are #1 at helping",
        "EMPTY": None,
    }


@pytest.mark.topic("py.fundamentals.conditionals")
def test_parse_env_empty():
    assert ex.parse_env("") == {}
    assert ex.parse_env("\n   \n# only comments\n") == {}


@pytest.mark.topic("py.fundamentals.conditionals", "py.fundamentals.loops")
@pytest.mark.parametrize(
    "text",
    ["A=1\nB=2\nnot a pair", "A=1\nB=2\n1BAD=3", "A=1\n\n=3"],
)
def test_parse_env_errors_report_line(text):
    lineno = len(text.splitlines())
    assert ex.parse_env("A=1") == {"A": 1}
    with pytest.raises(ValueError, match=f"line {lineno}"):
        ex.parse_env(text)


@pytest.mark.topic("py.fundamentals.truthiness", "py.fundamentals.none")
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((None, 0, 5), 0),
        ((None, False, True), False),
        ((None, "", "x"), ""),
        ((None, 0.0), 0.0),
        ((None, None, "env"), "env"),
    ],
)
def test_first_set_keeps_falsy_values(values, expected):
    out = ex.first_set(*values, default="DEFAULT")
    assert out == expected
    assert type(out) is type(expected)


@pytest.mark.topic("py.fundamentals.none")
def test_first_set_default():
    assert ex.first_set(None, None, default=7) == 7
    assert ex.first_set() is None
