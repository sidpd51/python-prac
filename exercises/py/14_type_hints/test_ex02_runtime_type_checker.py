from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired, Optional, TypedDict, TypeVar, Union

import pytest

from study.loader import load

ex = load(__file__, "ex02_runtime_type_checker")

type Embedding = list[float]
type Scores = dict[str, Embedding | None]


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, Any]
    id: NotRequired[str]


class Doc:
    pass


def gen():
    yield 1


CASES = [
    # (value, type, expected)
    (1, Any, True),
    (None, object, True),
    (None, None, True),
    (0, None, False),
    (3, int, True),
    (True, int, True),  # bool IS an int in Python
    ("3", int, False),
    (3, float, True),  # numeric promotion
    (3.5, int, False),
    (Doc(), Doc, True),
    (None, int | None, True),
    ("x", Optional[int], False),  # noqa: UP045 - testing the old spelling
    ("x", Union[int, str], True),  # noqa: UP007 - testing the old spelling
    ("user", Literal["user", "assistant"], True),
    ("robot", Literal["user", "assistant"], False),
    (True, Literal[1], False),
    (1, Literal[1], True),
    ([1, 2], list[int], True),
    ([1, "2"], list[int], False),
    ((1, 2), list[int], False),
    ([], list[int], True),
    ({1, 2}, set[int], True),
    (frozenset({"a"}), frozenset[str], True),
    ((1, "a"), tuple[int, str], True),
    ((1, "a", 3), tuple[int, str], False),
    ((1, 2, 3), tuple[int, ...], True),
    ((), tuple[()], True),
    ((1,), tuple[()], False),
    ({"a": [1.0]}, dict[str, list[float]], True),
    ({1: [1.0]}, dict[str, list[float]], False),
    ({"a": ["x"]}, dict[str, list[float]], False),
    ("abc", Sequence[str], True),
    ((1, 2), Sequence[int], True),
    ({"k": 1}, Mapping[str, int], True),
    (gen(), Iterator[int], True),
    ([1], Iterator[int], False),
    (iter(["not", "checked"]), Iterable[int], True),
    (len, Callable[..., int], True),
    (42, Callable[[int], int], False),
    ([1.0, 2.0], Embedding, True),
    ({"q1": [0.1], "q2": None}, Scores, True),
    ({"q1": ["x"]}, Scores, False),
    (5, Annotated[int, "tokens"], True),
    ([1], list, True),
    ({"a": 1}, dict, True),
]


@pytest.mark.topic("py.type_hints.union", "py.type_hints.collections", "py.type_hints.generics")
@pytest.mark.parametrize(("value", "tp", "expected"), CASES, ids=[repr(c[1]) for c in CASES])
def test_check_type(value, tp, expected):
    assert ex.check_type(value, tp) is expected


@pytest.mark.topic("py.type_hints.any_vs_object", "py.type_hints.optional")
def test_any_object_none():
    for v in (None, 0, "", object(), [1]):
        assert ex.check_type(v, Any) and ex.check_type(v, object)
    assert ex.check_type(None, type(None))


@pytest.mark.topic("py.type_hints.typeddict")
def test_typeddict():
    ok = {"name": "search", "arguments": {"q": "rag"}}
    assert ex.check_type(ok, ToolCall)
    assert ex.check_type({**ok, "id": "call_1"}, ToolCall)
    assert not ex.check_type({"name": "search"}, ToolCall)  # missing required key
    assert not ex.check_type({**ok, "id": 7}, ToolCall)  # wrong type for optional key
    assert not ex.check_type({**ok, "extra": 1}, ToolCall)  # unknown key
    assert not ex.check_type([ok], ToolCall)
    assert ex.check_type([ok, ok], list[ToolCall])


@pytest.mark.topic("py.type_hints.aliases", "py.type_hints.typevar")
def test_unsupported_raises():
    T = TypeVar("T")
    with pytest.raises(TypeError):
        ex.check_type(1, T)
    assert ex.check_type([1.0], Embedding)


def _make_validated():
    @ex.validate_call
    def score(query: str, top_k: int = 3, *weights: float, **filters: str) -> list[float]:
        """Fake retrieval scorer."""
        return [1.0] * top_k

    return score


@pytest.mark.topic("py.type_hints.functions", "py.type_hints.callable")
def test_validate_call_accepts_valid_calls():
    score = _make_validated()
    assert score("rag") == [1.0, 1.0, 1.0]
    assert score("rag", 2, 0.5, 1, lang="en") == [1.0, 1.0]
    assert score.__name__ == "score"
    assert score.__doc__ == "Fake retrieval scorer."
    assert score.__wrapped__.__name__ == "score", "use functools.wraps"


@pytest.mark.topic("py.type_hints.functions")
@pytest.mark.parametrize(
    ("args", "kwargs", "param"),
    [
        ((1,), {}, "query"),
        (("q", "3"), {}, "top_k"),
        (("q", 3, 0.5, "heavy"), {}, "weights"),
        (("q",), {"lang": 1}, "filters"),
    ],
)
def test_validate_call_rejects_bad_arguments(args, kwargs, param):
    score = _make_validated()
    with pytest.raises(TypeError, match=param):
        score(*args, **kwargs)


@pytest.mark.topic("py.type_hints.returns", "py.type_hints.functions")
def test_validate_call_checks_return_and_string_annotations():
    @ex.validate_call
    def bad() -> "list[int]":
        return ["not", "ints"]

    with pytest.raises(TypeError, match="return"):
        bad()

    @ex.validate_call
    def unannotated(x):
        return x

    assert unannotated("anything") == "anything"
