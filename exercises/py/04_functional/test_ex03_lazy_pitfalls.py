import pytest

from study.loader import load

ex = load(__file__, "ex03_lazy_pitfalls")


def one_shot(items, pulled=None):
    for x in items:
        if pulled is not None:
            pulled.append(x)
        yield x


@pytest.mark.topic("py.functional.iterator_consumption", "py.functional.filter")
def test_embed_valid_with_one_shot_iterator():
    calls = []

    def embed(text):
        calls.append(text)
        return [float(len(text))]

    count, vectors = ex.embed_valid(one_shot(["rag", "  ", "", "agents"]), embed)
    assert count == 2
    assert vectors == [[3.0], [6.0]]
    assert calls == ["rag", "agents"], "embed must run exactly once per valid doc"


@pytest.mark.topic("py.functional.iterator_consumption")
def test_embed_valid_empty_and_list_input():
    assert ex.embed_valid([], len) == (0, [])
    assert ex.embed_valid(["a", "bb"], len) == (2, [1, 2])


@pytest.mark.topic("py.functional.side_effects", "py.functional.map", "py.functional.lazy_evaluation")
def test_warm_cache_is_eager():
    loaded = []
    n = ex.warm_cache(["k1", "k2", "k3"], loaded.append)
    assert loaded == ["k1", "k2", "k3"], "nothing was loaded - lazy map() never ran?"
    assert n == 3
    assert ex.warm_cache(iter([]), loaded.append) == 0


@pytest.mark.topic("py.functional.iterator_consumption")
@pytest.mark.parametrize(
    ("scores", "expected"),
    [([0.3, 0.9, 0.1, 0.5], (0.1, 0.9)), ([2.0], (2.0, 2.0)), ([5, 4, 3], (3, 5)), ([1, 2, 3], (1, 3))],
)
def test_min_max_single_pass(scores, expected):
    assert ex.min_max(scores) == expected
    assert ex.min_max(one_shot(scores)) == expected  # generator: only one pass possible


@pytest.mark.topic("py.functional.iterator_consumption")
def test_min_max_empty():
    assert ex.min_max([1]) == (1, 1)
    with pytest.raises(ValueError):
        ex.min_max(one_shot([]))


@pytest.mark.topic("py.functional.lazy_evaluation", "py.functional.iterator_consumption")
def test_peek_pulls_one_and_keeps_everything():
    pulled = []
    first, it = ex.peek(one_shot([{"v": 1}, {"v": 2}, {"v": 3}], pulled))
    assert first == {"v": 1}
    assert len(pulled) == 1, "peek must only pull the first item"
    assert next(it) == {"v": 1}
    assert len(pulled) == 1, "the returned iterator must stay lazy"
    assert list(it) == [{"v": 2}, {"v": 3}]
    assert list(it) == []  # one-shot, like any iterator


@pytest.mark.topic("py.functional.lazy_evaluation")
def test_peek_infinite_and_empty():
    def naturals():
        n = 0
        while True:
            yield n
            n += 1

    first, it = ex.peek(naturals())
    assert first == 0
    assert [next(it) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(ValueError):
        ex.peek([])
