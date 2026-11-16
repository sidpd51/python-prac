"""
Exercise 12.3 - Transactions with rollback (sqlite3)             level: ★★☆
topics: context_managers.cleanup, context_managers.resources,
        context_managers.contextmanager, context_managers.exit

Problem
-------
A tiny document store for a RAG app. Adding a document means inserting the document row
AND all of its chunk rows - all or nothing.

1. connect(path) - context manager yielding a sqlite3.Connection opened with
   isolation_level=None (autocommit: we control transactions explicitly) and
   CLOSED on exit, even on error.
   (Note: `with sqlite3.connect(...) as conn:` does NOT close the connection - it only
    commits/rolls back. That surprises many people.)

2. transaction(conn) - context manager
     * outermost use: "BEGIN" on enter; "COMMIT" on success; "ROLLBACK" if the block
       raises, then re-raise the same exception
     * nested use (conn.in_transaction is already True): use a SAVEPOINT instead
       ("SAVEPOINT name" / "RELEASE name" / "ROLLBACK TO name" then "RELEASE name"),
       so an inner failure that the caller catches only undoes the inner work.
       Savepoint names must be unique (e.g. sp_1, sp_2, ... from a counter).

3. class DocumentStore(conn)
     * __init__ creates (if missing) tables
           docs(id TEXT PRIMARY KEY, title TEXT NOT NULL)
           chunks(doc_id TEXT NOT NULL, idx INTEGER NOT NULL, text TEXT NOT NULL)
     * add_document(doc_id, title, chunks: list[str]) - atomic (use transaction());
       a duplicate id or a None chunk raises sqlite3.IntegrityError and leaves nothing
       of that document behind
     * counts() -> tuple[int, int]   (number of docs, number of chunks)

Interviewers look for
---------------------
* "Release the resource no matter what" (close) vs "decide the outcome by success or
  failure" (commit/rollback) - __exit__ gets the exception info to make that decision.
* Re-raising after rollback: a transaction CM must not swallow errors.
* Savepoints for nesting; knowing that sqlite3's implicit transaction handling
  (isolation_level != None) can surprise you.
"""

import itertools
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        yield conn
    finally:
        conn.close()


_savepoint_ids = itertools.count(1)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    if not conn.in_transaction:
        conn.execute("BEGIN")
        try:
            yield conn
        except BaseException:  # also roll back on Ctrl-C / cancellation, then re-raise
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return

    name = f"sp_{next(_savepoint_ids)}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield conn
    except BaseException:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    conn.execute(f"RELEASE {name}")


class DocumentStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(doc_id TEXT NOT NULL, idx INTEGER NOT NULL, text TEXT NOT NULL)"
        )

    def add_document(self, doc_id: str, title: str, chunks: list[str]) -> None:
        with transaction(self.conn):
            self.conn.execute("INSERT INTO docs (id, title) VALUES (?, ?)", (doc_id, title))
            self.conn.executemany(
                "INSERT INTO chunks (doc_id, idx, text) VALUES (?, ?, ?)",
                [(doc_id, i, text) for i, text in enumerate(chunks)],
            )

    def counts(self) -> tuple[int, int]:
        (docs,) = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()
        (chunks,) = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return docs, chunks
