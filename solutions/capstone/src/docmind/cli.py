"""
Milestone M0.7 - command-line interface
topics: py.packaging.main_guard, py.packaging.pyproject, py.genai.packages, py.async.asyncio_run

Build
-----
main(argv=None) -> int     (exposed as the `docmind` console script in capstone/pyproject.toml)

    docmind ingest [DIR]              DIR defaults to settings.data_dir
    docmind search QUERY [-k 5] [--source PATH]

  * settings from get_settings(); logging via configure_logging(settings.log_level)
  * embedder: HashingEmbedder(settings.embedding_dim) (HTTP embedder wiring comes in Phase 2)
  * ingest: loads the existing index from settings.index_dir if present, ingests with
    RecursiveChunker(settings.chunk_size), saves the index, prints
    "ingested <documents> documents, <chunks> chunks" and returns 0
  * search: loads the index (missing index -> print an error to stderr, return 1), prints one
    line per hit: "<score:.3f>  <chunk id>  <first 80 chars of text, newlines -> spaces>"
    --source filters on metadata "source"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import configure_logging, get_settings
from .embeddings.hashing import HashingEmbedder
from .ingest.chunking import RecursiveChunker
from .ingest.loaders import load_documents
from .pipeline import ingest, search
from .store.memory import InMemoryVectorStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docmind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("dir", nargs="?")
    p_s = sub.add_parser("search")
    p_s.add_argument("query")
    p_s.add_argument("-k", type=int, default=5)
    p_s.add_argument("--source")
    args = ap.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    embedder = HashingEmbedder(settings.embedding_dim)
    index = settings.index_dir
    exists = (index / "vectors.npy").exists()

    if args.cmd == "ingest":
        store = InMemoryVectorStore.load(index) if exists else InMemoryVectorStore(embedder.dim)
        docs = load_documents(args.dir or settings.data_dir)
        report = asyncio.run(ingest(docs, RecursiveChunker(settings.chunk_size), embedder, store))
        store.save(index)
        print(f"ingested {report.documents} documents, {report.chunks} chunks")
        return 0

    if not exists:
        print(f"no index at {index} - run `docmind ingest` first", file=sys.stderr)
        return 1
    store = InMemoryVectorStore.load(index)
    where = {"source": args.source} if args.source else None
    for hit in asyncio.run(search(args.query, embedder, store, k=args.k, where=where)):
        preview = hit.chunk.text[:80].replace("\n", " ")
        print(f"{hit.score:.3f}  {hit.chunk.id}  {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
