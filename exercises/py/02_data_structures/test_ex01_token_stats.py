import pytest

from study.loader import load

ex = load(__file__, "ex01_token_stats")

DOCS = [
    "the model retrieves the chunks",
    "The chunks are embedded",
    "embedded chunks feed the model",
]


@pytest.mark.topic("py.data_structures.counter")
def test_top_k_tokens():
    assert ex.top_k_tokens(DOCS, 3) == [("the", 4), ("chunks", 3), ("embedded", 2)]


@pytest.mark.topic("py.data_structures.counter", "py.data_structures.sets")
def test_top_k_tokens_stopwords_and_ties():
    out = ex.top_k_tokens(DOCS, 4, stopwords=["THE", "are"])
    # embedded and model both have 2 -> alphabetical
    assert out == [("chunks", 3), ("embedded", 2), ("model", 2), ("feed", 1)]


@pytest.mark.topic("py.data_structures.counter")
def test_top_k_tokens_edges():
    assert ex.top_k_tokens(DOCS, 0) == []
    assert ex.top_k_tokens(DOCS, -2) == []
    assert ex.top_k_tokens([], 5) == []
    assert ex.top_k_tokens(["b a b"], 10) == [("b", 2), ("a", 1)]
    assert ex.top_k_tokens((d for d in DOCS), 1) == [("the", 4)]  # any iterable


@pytest.mark.topic("py.data_structures.set_operations")
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("the cat sat", "the cat ran", 2 / 4),
        ("a b", "c d", 0.0),
        ("A b b", "b a", 1.0),
        ("", "", 1.0),
        ("x", "", 0.0),
    ],
)
def test_jaccard(a, b, expected):
    assert ex.jaccard(a, b) == pytest.approx(expected)


@pytest.mark.topic("py.data_structures.dedup", "py.data_structures.lists")
def test_dedup_preserves_first_occurrence_order():
    assert ex.dedup(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert ex.dedup([]) == []
    assert ex.dedup(x % 3 for x in range(10)) == [0, 1, 2]
    assert ex.dedup([1, 1.0, True]) == [1]  # equal and same hash -> duplicates


@pytest.mark.topic("py.data_structures.dedup", "py.data_structures.complexity")
def test_dedup_is_linear():
    items = list(range(50_000)) * 2
    assert ex.dedup(items) == list(range(50_000))  # an O(n^2) version takes far too long


@pytest.mark.topic("py.data_structures.dedup", "py.data_structures.sets")
def test_dedup_by_key():
    chunks = [
        {"id": 1, "text": "RAG  is great", "score": 0.9},
        {"id": 2, "text": "rag is great", "score": 0.8},
        {"id": 3, "text": "agents", "score": 0.7},
        {"id": 4, "text": "RAG is   great", "score": 0.6},
    ]
    out = ex.dedup_by(chunks, key=lambda c: " ".join(c["text"].lower().split()))
    assert [c["id"] for c in out] == [1, 3]
    assert out[0] is chunks[0]
    assert ex.dedup_by([], key=len) == []
