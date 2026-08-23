"""mla — offline, venue-complete ML literature search over the ML Anthology corpus.

The corpus (``data/papers/*.json.gz`` + ``data/legacy/*.jsonl.gz`` in this repo) is
~217k papers across 44 ML venues, 1969-2026, with ~99% abstract coverage. This package
compiles it into a SQLite FTS5 index and exposes it three ways:

  ``mla`` (CLI)        — the primary surface: JSON on stdout, made for a research harness
                         that shells out and records what it got.
  ``mla-mcp``          — a secondary MCP server over the same library, for MCP-native seats.
  ``mla.query``        — the library, for in-process callers.

Nothing here reaches the network. A search is a local SQLite read, so it cannot 429,
cannot rate-limit a sweep, and answers in single-digit milliseconds.
"""

__version__ = "0.1.0"

CORPUS_NAME = "mlanthology"
