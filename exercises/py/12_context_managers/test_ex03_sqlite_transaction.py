import contextlib
import sqlite3

import pytest

from study.loader import load

ex = load(__file__, "ex03_sqlite_transaction")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.execute("CREATE TABLE kv (k TEXT PRIMARY KEY, v TEXT)")
    yield c
    c.close()


def keys(conn):
    return [k for (k,) in conn.execute("SELECT k FROM kv ORDER BY k")]


@pytest.mark.topic("py.context_managers.resources", "py.context_managers.cleanup")
def test_connect_closes(tmp_path):
    with ex.connect(tmp_path / "db.sqlite") as c:
        assert isinstance(c, sqlite3.Connection)
        assert c.isolation_level is None
        c.execute("CREATE TABLE t (x)")
    with pytest.raises(sqlite3.ProgrammingError):
        c.execute("SELECT 1")  # closed


@pytest.mark.topic("py.context_managers.cleanup")
def test_connect_closes_on_error(tmp_path):
    with pytest.raises(ZeroDivisionError), ex.connect(tmp_path / "db.sqlite") as c:
        _ = 1 / 0
    with pytest.raises(sqlite3.ProgrammingError):
        c.execute("SELECT 1")


@pytest.mark.topic("py.context_managers.contextmanager")
def test_transaction_commits(conn):
    with ex.transaction(conn) as c:
        assert c is conn
        assert conn.in_transaction
        conn.execute("INSERT INTO kv VALUES ('a', '1')")
    assert not conn.in_transaction
    assert keys(conn) == ["a"]


@pytest.mark.topic("py.context_managers.exit", "py.context_managers.cleanup")
def test_transaction_rolls_back_and_reraises(conn):
    err = ConnectionError("embedding API down")
    with pytest.raises(ConnectionError) as info, ex.transaction(conn):
        conn.execute("INSERT INTO kv VALUES ('a', '1')")
        raise err
    assert info.value is err
    assert not conn.in_transaction
    assert keys(conn) == []


@pytest.mark.topic("py.context_managers.contextmanager")
def test_nested_savepoint_inner_failure_keeps_outer_work(conn):
    with ex.transaction(conn):
        conn.execute("INSERT INTO kv VALUES ('outer', '1')")
        try:
            with ex.transaction(conn):
                conn.execute("INSERT INTO kv VALUES ('inner', '1')")
                raise ValueError("inner failed")
        except ValueError:
            pass
        with ex.transaction(conn):
            conn.execute("INSERT INTO kv VALUES ('inner2', '1')")
        assert conn.in_transaction
    assert keys(conn) == ["inner2", "outer"]


@pytest.mark.topic("py.context_managers.contextmanager")
def test_nested_outer_failure_undoes_everything(conn):
    with pytest.raises(KeyError), ex.transaction(conn):
        with ex.transaction(conn), ex.transaction(conn):
            conn.execute("INSERT INTO kv VALUES ('deep', '1')")
        raise KeyError("outer")
    assert keys(conn) == []
    assert not conn.in_transaction


@pytest.mark.topic("py.context_managers.resources")
def test_document_store_atomic_add(conn):
    store = ex.DocumentStore(conn)
    store.add_document("d1", "Intro to RAG", ["chunk a", "chunk b"])
    assert store.counts() == (1, 2)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_document("d2", "Broken", ["ok", None, "never"])
    assert store.counts() == (1, 2), "partial document left behind"
    with pytest.raises(sqlite3.IntegrityError):
        store.add_document("d1", "Duplicate", ["x"])
    assert store.counts() == (1, 2)
    assert not conn.in_transaction
    ex.DocumentStore(conn)  # tables already exist: must not fail
    store.add_document("d3", "Later", [])
    assert store.counts() == (2, 2)


@pytest.mark.topic("py.context_managers.cleanup")
def test_document_store_inside_outer_transaction(conn):
    store = ex.DocumentStore(conn)
    with ex.transaction(conn):
        store.add_document("d1", "A", ["x"])
        with contextlib.suppress(sqlite3.IntegrityError):
            store.add_document("d2", "B", [None])
    assert store.counts() == (1, 1)
