import pytest

from study.loader import load

ex = load(__file__, "ex01_text_cleaning")


@pytest.mark.topic("py.genai.normalization")
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ＡＩ ｅｎｇｉｎｅｅｒ", "AI engineer"),  # full-width
        ("ﬁnancial ﬂow", "financial flow"),  # ligatures
        ("a b", "a b"),  # NBSP
        ("zero​width‌‍⁠join﻿", "zerowidthjoin"),
        ("“smart” ‘quotes’ „low‟ ‚x‛", "\"smart\" 'quotes' \"low\" 'x'"),
        ("2020–2024 — done", "2020-2024 - done"),
        ("  many \t\n\n spaces\r\nhere  ", "many spaces here"),
        ("", ""),
    ],
)
def test_normalize_text(raw, expected):
    assert ex.normalize_text(raw) == expected


@pytest.mark.topic("py.genai.normalization")
def test_normalize_casefold():
    assert ex.normalize_text("Straße ＡＢＣ", casefold=True) == "strasse abc"
    assert ex.normalize_text("Straße") == "Straße", "no casefold by default"


@pytest.mark.topic("py.genai.normalization")
def test_zero_width_removed_before_whitespace_collapse():
    assert ex.normalize_text("a ​ b") == "a b"
    assert ex.normalize_text("﻿Title") == "Title"  # BOM at start of a file


@pytest.mark.topic("py.genai.text_processing", "py.genai.normalization")
def test_dedupe_texts_keeps_first_original():
    texts = [
        "The model’s answer",
        "the MODEL's   answer",
        "Ｔｈｅ model's answer​",
        "Another one",
        "   ",
        "​",
        "another ONE",
    ]
    assert ex.dedupe_texts(texts) == ["The model’s answer", "Another one"]


@pytest.mark.topic("py.genai.text_processing")
def test_dedupe_texts_accepts_any_iterable():
    assert ex.dedupe_texts(t for t in ["x", "X", "y"]) == ["x", "y"]
    assert ex.dedupe_texts([]) == []


@pytest.mark.topic("py.genai.regex", "py.genai.text_processing")
def test_redact_emails():
    text = "Contact jane.doe+rag@mail.example.co.uk or bob_99@corp-x.io now."
    assert ex.redact_pii(text) == "Contact [EMAIL] or [EMAIL] now."
    assert ex.redact_pii("user@localhost is not an email here") == "user@localhost is not an email here"


@pytest.mark.topic("py.genai.regex")
@pytest.mark.parametrize(
    "phone", ["+1 415-555-0132", "(415) 555-0132", "415.555.0132", "4155550132", "+44 415 555 0132"]
)
def test_redact_phones(phone):
    assert ex.redact_pii(f"call {phone} today") == "call [PHONE] today"


@pytest.mark.topic("py.genai.regex")
@pytest.mark.parametrize(
    "not_phone", ["2024-01-15", "order 12345678901234", "v3.14.0", "id x4155550132", "415-555-01"]
)
def test_redact_leaves_non_phones(not_phone):
    assert ex.redact_pii(not_phone) == not_phone


@pytest.mark.topic("py.genai.regex")
def test_redact_email_with_digits_first():
    assert ex.redact_pii("4155550132@sms.example.com") == "[EMAIL]"


@pytest.mark.topic("py.genai.regex", "py.genai.structured_data")
def test_extract_citations():
    assert ex.extract_citations("A [2]. B [1, 3-4]. C [2][5]") == [2, 1, 3, 4, 5]
    assert ex.extract_citations("ranges [7–9] and [ 10 , 11 ]") == [7, 8, 9, 10, 11]


@pytest.mark.topic("py.genai.regex")
def test_extract_citations_ignores_non_citations():
    text = "See [a], [], [1a], [link](http://x.io), arr[i] and [6-4]; real one [3]."
    assert ex.extract_citations(text) == [3]
    assert ex.extract_citations("no citations") == []
