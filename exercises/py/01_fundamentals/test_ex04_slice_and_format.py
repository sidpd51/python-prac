import pytest

from study.loader import load

ex = load(__file__, "ex04_slice_and_format")


@pytest.mark.topic("py.fundamentals.slicing")
@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, []), (1, ["t4"]), (2, ["t3", "t4"]), (4, ["t1", "t2", "t3", "t4"]), (10, ["t1", "t2", "t3", "t4"])],
)
def test_last_turns(n, expected):
    history = ["t1", "t2", "t3", "t4"]
    out = ex.last_turns(history, n)
    assert out == expected
    assert out is not history
    assert history == ["t1", "t2", "t3", "t4"]


@pytest.mark.topic("py.fundamentals.slicing")
def test_last_turns_edges():
    assert ex.last_turns([], 3) == []
    with pytest.raises(ValueError):
        ex.last_turns([1, 2], -1)


@pytest.mark.topic("py.fundamentals.slicing", "py.fundamentals.operators")
@pytest.mark.parametrize(
    ("text", "max_len", "marker", "expected"),
    [
        ("abcdefghij", 7, "...", "ab...ij"),
        ("abcdefghij", 6, "...", "ab...j"),  # head gets the extra char
        ("abcdefghij", 4, "...", "a..."),  # tail is empty - beware text[-0:]
        ("abcdefghij", 3, "...", "..."),
        ("abcdefghij", 10, "...", "abcdefghij"),
        ("abcdefghij", 50, "...", "abcdefghij"),
        ("abcdefghij", 5, "~", "ab~ij"),
        ("", 0, "", ""),
    ],
)
def test_truncate_middle(text, max_len, marker, expected):
    out = ex.truncate_middle(text, max_len, marker)
    assert out == expected
    assert len(out) <= max_len


@pytest.mark.topic("py.fundamentals.slicing")
def test_truncate_middle_long_text_exact_length():
    text = "".join(str(i % 10) for i in range(10_000))
    out = ex.truncate_middle(text, 101)
    assert len(out) == 101
    assert out.startswith(text[:49]) and out.endswith(text[-49:])
    with pytest.raises(ValueError):
        ex.truncate_middle(text, 2)


@pytest.mark.topic("py.fundamentals.fstrings")
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((1234, 56, 0.0123), "prompt=1,234 completion=56 total=1,290 cost=$0.0123"),
        ((0, 0, 0), "prompt=0 completion=0 total=0 cost=$0.0000"),
        ((1_000_000, 250_000, 12.5), "prompt=1,000,000 completion=250,000 total=1,250,000 cost=$12.5000"),
        ((10, 5, 0.00004), "prompt=10 completion=5 total=15 cost=$0.0000"),
    ],
)
def test_format_usage(args, expected):
    assert ex.format_usage(*args) == expected


@pytest.mark.topic("py.fundamentals.fstrings", "py.fundamentals.operators")
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((5, 20, 10), "[##........]  25% (5/20)"),
        ((0, 3, 5), "[.....]   0% (0/3)"),
        ((3, 3, 5), "[#####] 100% (3/3)"),
        ((1, 3, 10), "[###.......]  33% (1/3)"),
        ((2, 3, 10), "[######....]  67% (2/3)"),
    ],
)
def test_progress_bar(args, expected):
    assert ex.progress_bar(*args) == expected


@pytest.mark.topic("py.fundamentals.conditionals")
def test_progress_bar_default_width_and_validation():
    assert ex.progress_bar(10, 20) == "[" + "#" * 10 + "." * 10 + "]  50% (10/20)"
    for bad in [(1, 0, 10), (-1, 5, 10), (6, 5, 10), (1, 5, 0)]:
        with pytest.raises(ValueError):
            ex.progress_bar(*bad)
