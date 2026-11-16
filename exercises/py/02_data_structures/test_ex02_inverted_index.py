import pytest

from study.loader import load

ex = load(__file__, "ex02_inverted_index")

DOCS = {
    "d1": "vector search with embeddings",
    "d2": "keyword search search search",
    "d3": "hybrid search combines keyword and vector search",
    "d4": "agents call tools",
}


@pytest.mark.topic("py.data_structures.defaultdict", "py.data_structures.dicts")
def test_build_index():
    index = ex.build_index(DOCS)
    assert index["search"] == {"d1": 1, "d2": 3, "d3": 2}
    assert index["vector"] == {"d1": 1, "d3": 1}
    assert index["agents"] == {"d4": 1}
    assert len(index) == len({t for text in DOCS.values() for t in text.split()})


@pytest.mark.topic("py.data_structures.defaultdict", "py.data_structures.dict_lookup")
def test_build_index_returns_plain_dicts():
    index = ex.build_index(DOCS)
    assert type(index) is dict
    assert all(type(p) is dict for p in index.values())
    with pytest.raises(KeyError):
        index["not-a-term"]
    assert ex.build_index({}) == {}


@pytest.mark.topic("py.data_structures.set_operations", "py.data_structures.namedtuple")
def test_search_and():
    index = ex.build_index(DOCS)
    hits = ex.search(index, "vector search")
    assert hits == [ex.Hit("d3", 3), ex.Hit("d1", 2)]
    assert hits[0].doc_id == "d3" and hits[0].score == 3


@pytest.mark.topic("py.data_structures.set_operations", "py.data_structures.tuples")
def test_search_or_and_tie_break():
    index = ex.build_index(DOCS)
    hits = ex.search(index, "keyword vector", mode="or")
    # d3 has both (2), d1 and d2 one each -> tie broken by doc_id
    assert hits == [("d3", 2), ("d1", 1), ("d2", 1)]


@pytest.mark.topic("py.data_structures.dict_lookup", "py.data_structures.set_operations")
def test_search_unknown_terms_and_duplicates():
    index = ex.build_index(DOCS)
    before = {t: dict(p) for t, p in index.items()}
    assert ex.search(index, "vector zebra") == []  # AND with a missing term
    assert ex.search(index, "zebra", mode="or") == []
    assert ex.search(index, "") == []
    assert ex.search(index, "AGENTS agents") == [ex.Hit("d4", 1)]  # distinct, lowercased
    assert index == before, "search must not modify the index"
    assert "zebra" not in index


@pytest.mark.topic("py.data_structures.dict_lookup")
def test_search_bad_mode():
    index = ex.build_index(DOCS)
    assert ex.search(index, "tools") == [ex.Hit("d4", 1)]
    with pytest.raises(ValueError):
        ex.search(index, "tools", mode="xor")


@pytest.mark.topic("py.data_structures.namedtuple", "py.data_structures.tuples")
def test_hit_is_a_real_namedtuple():
    index = ex.build_index(DOCS)
    hit = ex.search(index, "tools")[0]
    doc_id, score = hit
    assert (doc_id, score) == ("d4", 1)
    assert isinstance(hit, tuple) and hit._fields == ("doc_id", "score")
    with pytest.raises(AttributeError):
        hit.score = 5


@pytest.mark.topic("py.data_structures.defaultdict", "py.data_structures.list_vs_dict")
def test_group_by():
    recs = [
        {"id": 1, "source": "a.pdf"},
        {"id": 2, "source": "b.md"},
        {"id": 3},
        {"id": 4, "source": "a.pdf"},
    ]
    groups = ex.group_by(recs, "source")
    assert groups == {"a.pdf": [recs[0], recs[3]], "b.md": [recs[1]], None: [recs[2]]}
    assert type(groups) is dict
    assert list(groups) == ["a.pdf", "b.md", None]
    assert ex.group_by([], "source") == {}
