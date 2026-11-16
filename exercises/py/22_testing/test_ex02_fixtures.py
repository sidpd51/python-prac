import os

import pytest

from study.loader import load

ex = load(__file__, "ex02_fixtures")

# register YOUR fixture in this test module (pytest discovers fixtures by module attribute)
chunk_factory = ex.chunk_factory

VAR_SET = "STUDY_T22_EXISTING"
VAR_NEW = "STUDY_T22_NEW"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setenv(VAR_SET, "original")
    monkeypatch.delenv(VAR_NEW, raising=False)


@pytest.mark.topic("py.testing.fixtures")
def test_factory_defaults_and_sequence():
    make = ex.make_chunk_factory()
    a, b = make(), make()
    assert a == {"id": "chunk-1", "doc_id": "doc-1", "text": "lorem ipsum", "score": 0.0, "metadata": {}}
    assert b["id"] == "chunk-2"


@pytest.mark.topic("py.testing.fixtures")
def test_factory_overrides_and_counter_per_factory():
    make1, make2 = ex.make_chunk_factory(), ex.make_chunk_factory()
    c = make1(text="hello", score=0.9, id="custom")
    assert c["text"] == "hello" and c["score"] == 0.9 and c["id"] == "custom"
    assert make1()["id"] == "chunk-2", "an explicit id still advances the counter"
    assert make2()["id"] == "chunk-1", "factories must not share a counter"


@pytest.mark.topic("py.testing.fixtures", "py.testing.clean_code")
def test_factory_rejects_unknown_fields():
    make = ex.make_chunk_factory()
    with pytest.raises(TypeError):
        make(txt="typo")


@pytest.mark.topic("py.testing.fixtures")
def test_factory_no_shared_mutable_state():
    make = ex.make_chunk_factory()
    a, b = make(), make()
    a["metadata"]["page"] = 3
    assert b["metadata"] == {}
    assert make()["metadata"] == {}


@pytest.mark.topic("py.testing.fixtures")
def test_temp_env_sets_and_restores():
    with ex.temp_env(**{VAR_SET: "patched", VAR_NEW: "added"}):
        assert os.environ[VAR_SET] == "patched"
        assert os.environ[VAR_NEW] == "added"
    assert os.environ[VAR_SET] == "original"
    assert VAR_NEW not in os.environ


@pytest.mark.topic("py.testing.fixtures")
def test_temp_env_none_deletes_and_restores():
    with ex.temp_env(**{VAR_SET: None, VAR_NEW: None}):
        assert VAR_SET not in os.environ
        assert VAR_NEW not in os.environ
    assert os.environ[VAR_SET] == "original"
    assert VAR_NEW not in os.environ


@pytest.mark.topic("py.testing.fixtures")
def test_temp_env_restores_on_exception():
    with pytest.raises(KeyError), ex.temp_env(**{VAR_SET: "patched", VAR_NEW: "x"}):
        raise KeyError("boom")
    assert os.environ[VAR_SET] == "original"
    assert VAR_NEW not in os.environ


@pytest.mark.topic("py.testing.fixtures", "py.testing.pytest")
def test_fixture_gives_fresh_factory_a(chunk_factory):
    assert chunk_factory()["id"] == "chunk-1"
    assert chunk_factory()["id"] == "chunk-2"


@pytest.mark.topic("py.testing.fixtures", "py.testing.pytest")
def test_fixture_gives_fresh_factory_b(chunk_factory):
    # function scope: a new factory per test, so the counter starts over
    assert chunk_factory()["id"] == "chunk-1"
