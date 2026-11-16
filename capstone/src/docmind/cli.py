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


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
