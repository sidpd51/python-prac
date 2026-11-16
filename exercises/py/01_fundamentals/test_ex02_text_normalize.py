import pytest

from study.loader import load

ex = load(__file__, "ex02_text_normalize")


@pytest.mark.topic("py.fundamentals.strings", "py.fundamentals.list_comprehensions")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Hello,   WORLD!!  -- it's  ", "hello world it's"),
        ("RAG\tpipelines\n(are) fun.", "rag pipelines are fun"),
        ("", ""),
        ("!!! ... ---", ""),
        ("Straße", "strasse"),  # casefold, not lower
        ("e-mail co-pilot", "e-mail co-pilot"),  # inner punctuation kept
    ],
)
def test_normalize(text, expected):
    assert ex.normalize(text) == expected


@pytest.mark.topic("py.fundamentals.set_comprehensions")
def test_vocabulary():
    docs = ["The cat sat.", "the DOG sat!", "a cat"]
    assert ex.vocabulary(docs) == {"the", "cat", "sat", "dog", "a"}
    assert ex.vocabulary(docs, min_len=3) == {"the", "cat", "sat", "dog"}
    assert ex.vocabulary([]) == set()


@pytest.mark.topic("py.fundamentals.dict_comprehensions")
def test_first_positions_keeps_first_occurrence_and_order():
    out = ex.first_positions(["a", "b", "a", "c", "b"])
    assert out == {"a": 0, "b": 1, "c": 3}
    assert list(out) == ["a", "b", "c"]
    assert ex.first_positions([]) == {}


@pytest.mark.topic("py.fundamentals.slicing", "py.fundamentals.list_comprehensions")
def test_ngrams():
    toks = ["the", "cat", "sat", "down"]
    assert ex.ngrams(toks, 2) == [("the", "cat"), ("cat", "sat"), ("sat", "down")]
    assert ex.ngrams(toks, 4) == [tuple(toks)]
    assert ex.ngrams(toks, 5) == []
    assert ex.ngrams(toks, 1) == [("the",), ("cat",), ("sat",), ("down",)]
    assert ex.ngrams([], 1) == []


@pytest.mark.topic("py.fundamentals.slicing")
@pytest.mark.parametrize("n", [0, -1])
def test_ngrams_invalid(n):
    with pytest.raises(ValueError):
        ex.ngrams(["a"], n)


@pytest.mark.topic("py.fundamentals.strings")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('  ```\n{"a": 1}\n```  \n', '{"a": 1}'),
        ('```json\n{"a": 1,\n "b": 2}\n```', '{"a": 1,\n "b": 2}'),
        ('```json\n{"a": 1}', '{"a": 1}'),  # truncated response: no closing fence
        ('  {"a": 1}  ', '{"a": 1}'),
        ("plain answer", "plain answer"),
    ],
)
def test_strip_code_fence(text, expected):
    assert ex.strip_code_fence(text) == expected
