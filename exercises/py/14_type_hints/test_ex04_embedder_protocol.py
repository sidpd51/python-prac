import hashlib
import math
import re
import shutil
import subprocess
import sys

import pytest

from study.loader import load

ex = load(__file__, "ex04_embedder_protocol")


class FakeEmbedder:
    """Structural Embedder used as a test double: records every batch."""

    def __init__(self, dim=3):
        self.dim = dim
        self.batches = []

    def embed(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t))] * self.dim for t in texts]


@pytest.mark.topic("py.type_hints.protocol")
def test_hashing_embedder_satisfies_protocol_structurally():
    emb = ex.HashingEmbedder(16)
    assert isinstance(emb, ex.Embedder)
    assert ex.Embedder not in type(emb).__mro__, "don't inherit - satisfy it structurally"
    assert isinstance(FakeEmbedder(), ex.Embedder)
    with pytest.raises(ValueError):
        ex.HashingEmbedder(0)


@pytest.mark.topic("py.type_hints.protocol", "py.type_hints.collections")
def test_hashing_embedder_values():
    emb = ex.HashingEmbedder(8)
    v_hello, v_empty, v_rep = emb.embed(("Hello world", "", "rag rag"))
    idx = [int(hashlib.sha256(t.encode()).hexdigest(), 16) % 8 for t in ("hello", "world")]
    expected = [0.0] * 8
    for i in idx:
        expected[i] += 1.0
    norm = math.sqrt(sum(x * x for x in expected))
    assert v_hello == pytest.approx([x / norm for x in expected])
    assert v_empty == [0.0] * 8
    assert math.isclose(sum(x * x for x in v_rep), 1.0)
    assert emb.embed(["Hello WORLD"]) == [v_hello], "deterministic + lower-cased"


@pytest.mark.topic("py.type_hints.protocol")
def test_caching_embedder_dedups_and_batches_misses():
    inner = FakeEmbedder()
    cached = ex.CachingEmbedder(inner)
    assert isinstance(cached, ex.Embedder)
    assert cached.dim == 3
    out = cached.embed(["a", "bb", "a", "ccc"])
    assert out == [[1.0] * 3, [2.0] * 3, [1.0] * 3, [3.0] * 3]
    assert inner.batches == [["a", "bb", "ccc"]]
    cached.embed(["bb", "dddd", "dddd", "a"])
    assert inner.batches[-1] == ["dddd"]
    cached.embed(["a", "bb"])
    assert len(inner.batches) == 2, "fully cached -> no call"
    assert cached.calls == 2


@pytest.mark.topic("py.type_hints.protocol", "py.type_hints.any_vs_object")
def test_embed_in_batches():
    fake = FakeEmbedder(dim=2)
    texts = [f"t{i}" for i in range(7)]
    vecs = ex.embed_in_batches(fake, texts, batch_size=3)
    assert len(vecs) == 7
    assert fake.batches == [texts[0:3], texts[3:6], texts[6:7]]
    assert ex.embed_in_batches(fake, [], batch_size=3) == []
    with pytest.raises(ValueError):
        ex.embed_in_batches(fake, texts, batch_size=0)


@pytest.mark.topic("py.type_hints.protocol")
def test_runtime_checkable_limits():
    class NoDim:
        def embed(self, texts):
            return []

    with pytest.raises(TypeError):
        ex.embed_in_batches(NoDim(), ["x"])
    with pytest.raises(TypeError):
        ex.embed_in_batches(object(), ["x"])

    class WrongShape:  # passes isinstance (members exist) but lies about dim
        dim = 4

        def embed(self, texts):
            return [[1.0] for _ in texts]

    assert isinstance(WrongShape(), ex.Embedder)
    with pytest.raises(ValueError):
        ex.embed_in_batches(WrongShape(), ["x"])


@pytest.mark.topic("py.type_hints.collections")
def test_cosine():
    assert ex.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert ex.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert ex.cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)
    assert ex.cosine((0.0, 0.0), (1.0, 2.0)) == 0.0
    with pytest.raises(ValueError):
        ex.cosine([1], [1, 2])
    emb = ex.HashingEmbedder(64)
    a, b, c = emb.embed(["vector database", "database vector", "banana bread"])
    assert ex.cosine(a, b) == pytest.approx(1.0)
    assert ex.cosine(a, c) < 0.5


SNIPPET = """\
from collections.abc import Sequence
from typed_mod import CachingEmbedder, HashingEmbedder, embed_in_batches


class GoodLocalModel:
    dim: int = 384

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


class BadModel:
    dim: int = 384

    def embed(self, text: str) -> list[float]:  # single text, flat vector
        return [0.0] * 384


embed_in_batches(GoodLocalModel(), ["a"])  # OK1
embed_in_batches(HashingEmbedder(8), ("a", "b"))  # OK2
embed_in_batches(CachingEmbedder(GoodLocalModel()), ["a"])  # OK3
embed_in_batches(BadModel(), ["a"])  # E1
"""


@pytest.mark.topic("py.type_hints.static_checking", "py.type_hints.protocol")
def test_mypy_checks_protocol_signatures(tmp_path):
    ex.HashingEmbedder(2).embed(["warm up"])  # unsolved stub -> TODO before running mypy
    pytest.importorskip("mypy")
    shutil.copy(ex.__file__, tmp_path / "typed_mod.py")
    (tmp_path / "snippet.py").write_text(SNIPPET, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "snippet.py", "--strict", "--follow-imports=silent",
         "--no-error-summary", "--no-pretty", "--cache-dir", str(tmp_path / ".mypy_cache")],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    error_lines = {int(n) for n in re.findall(r"^snippet\.py:(\d+): error", proc.stdout, re.M)}
    bad_line = SNIPPET.splitlines().index('embed_in_batches(BadModel(), ["a"])  # E1') + 1
    assert error_lines == {bad_line}, proc.stdout
