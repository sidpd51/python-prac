import math

import pytest

from study.loader import load

ex = load(__file__, "ex03_embedding")


def emb(values, model="m"):
    return ex.Embedding(values, model=model)


@pytest.mark.topic("py.oop.property", "py.oop.encapsulation")
def test_construction_and_read_only_properties():
    e = emb([1, 2, 3])
    assert e.values == (1.0, 2.0, 3.0)
    assert all(isinstance(v, float) for v in e.values)
    assert e.model == "m"
    assert ex.Embedding([0.5]).model == ""
    with pytest.raises(AttributeError):
        e.values = (9.0,)
    with pytest.raises(AttributeError):
        e.model = "other"
    with pytest.raises(ValueError):
        ex.Embedding([])


@pytest.mark.topic("py.oop.slots")
def test_slots_no_instance_dict():
    e = emb([1.0])
    assert not hasattr(e, "__dict__")
    with pytest.raises(AttributeError):
        e.extra = 1
    assert set(ex.Embedding.__slots__) == {"_values", "_model"}


@pytest.mark.topic("py.oop.dunder")
def test_repr():
    assert repr(emb([0.1, 0.2])) == "Embedding([0.1, 0.2], model='m')"
    assert repr(emb([1, 2, 3, 4])) == "Embedding([1.0, 2.0, 3.0, 4.0], model='m')"
    long = ex.Embedding([i / 10 for i in range(1, 11)], model="te3")
    assert repr(long) == "Embedding([0.1, 0.2, ..., 1.0], dim=10, model='te3')"


@pytest.mark.topic("py.oop.dunder")
def test_sequence_protocol():
    e = emb([1, 2, 3, 4])
    assert len(e) == 4
    assert list(e) == [1.0, 2.0, 3.0, 4.0]
    assert e[0] == 1.0
    assert e[-1] == 4.0
    head = e[:2]
    assert isinstance(head, ex.Embedding)
    assert head == emb([1, 2])
    with pytest.raises(IndexError):
        e[10]


@pytest.mark.topic("py.oop.dunder")
def test_eq_and_hash_consistent():
    a, b = emb([1, 2]), emb([1.0, 2.0])
    assert a == b
    assert hash(a) == hash(b)
    assert a != emb([1, 2], model="other")
    assert a != [1.0, 2.0]
    assert (a == "x") is False
    assert len({a, b, emb([2, 1])}) == 2
    cache = {a: "doc-1"}
    assert cache[b] == "doc-1"


@pytest.mark.topic("py.oop.dunder")
def test_arithmetic():
    a, b = emb([1, 2, 3]), emb([0.5, 0.5, 0.5])
    assert a + b == emb([1.5, 2.5, 3.5])
    assert a - b == emb([0.5, 1.5, 2.5])
    assert a * 2 == emb([2, 4, 6])
    assert 2 * a == emb([2, 4, 6])
    assert (a + b).model == "m"
    with pytest.raises(ValueError):
        a + emb([1, 2])
    with pytest.raises(ValueError):
        a + emb([1, 2, 3], model="other")
    with pytest.raises(TypeError):
        a + [1, 2, 3]
    with pytest.raises(TypeError):
        a * "2"
    with pytest.raises(TypeError):
        a * True


@pytest.mark.topic("py.oop.dunder")
def test_dot_norm_bool_cosine():
    a, b = emb([1, 0]), emb([0, 1])
    assert a @ b == 0.0
    assert emb([1, 2, 3]) @ emb([4, 5, 6]) == 32.0
    assert abs(emb([3, 4])) == 5.0
    assert bool(emb([0, 0])) is False
    assert bool(emb([0, 0.1])) is True
    assert a.cosine(b) == pytest.approx(0.0)
    assert emb([1, 1]).cosine(emb([2, 2])) == pytest.approx(1.0)
    assert emb([1, 0]).cosine(emb([-1, 0])) == pytest.approx(-1.0)
    assert emb([1, 2]).cosine(emb([2, 1])) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        emb([0, 0]).cosine(a)
    with pytest.raises(ValueError):
        emb([1, 2]) @ emb([1, 2, 3])
    assert math.isclose(abs(emb([1e-3] * 4)), 2e-3)
