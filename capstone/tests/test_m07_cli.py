import pytest
from docmind import cli, config


def _clear_settings_cache():
    # get_settings should be cached (functools.lru_cache/cache); tolerate it not being yet
    getattr(config.get_settings, "cache_clear", lambda: None)()


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "rag.md").write_text(
        "Retrieval augmented generation grounds answers in documents.", encoding="utf-8"
    )
    (data / "cake.txt").write_text("Chocolate cake recipe with cocoa and sugar.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOCMIND_DATA_DIR", str(data))
    monkeypatch.setenv("DOCMIND_INDEX_DIR", str(tmp_path / "idx"))
    monkeypatch.setenv("DOCMIND_CHUNK_SIZE", "200")
    monkeypatch.setenv("DOCMIND_CHUNK_OVERLAP", "0")
    _clear_settings_cache()
    yield tmp_path
    _clear_settings_cache()


@pytest.mark.topic("py.packaging.main_guard", "py.genai.packages", "py.async.asyncio_run")
def test_ingest_then_search(env, capsys):
    assert cli.main(["search", "anything"]) == 1
    assert "docmind ingest" in capsys.readouterr().err

    assert cli.main(["ingest"]) == 0
    assert "ingested 2 documents, 2 chunks" in capsys.readouterr().out
    assert (env / "idx" / "vectors.npy").exists()

    assert cli.main(["search", "retrieval augmented generation", "-k", "1"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "rag.md#0" in out[0]

    assert cli.main(["search", "retrieval augmented generation", "--source", "cake.txt"]) == 0
    assert "cake.txt#0" in capsys.readouterr().out

    # re-ingest is idempotent (documents replaced, not duplicated)
    cli.main(["ingest"])
    capsys.readouterr()
    cli.main(["search", "cake", "-k", "10"])
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
