"""The SQLite FTS5 index: schema, build, and location resolution.

The index is DERIVED — it is rebuilt from ``data/`` and never edited, so it is
gitignored and safe to delete. It is built into a temp file and atomically renamed,
which means a half-written index can never be searched: either the old one answers,
or the new one does.

Ranking is BM25 with the title weighted well above the abstract. The corpus carries no
citation counts, so relevance here is purely lexical; pair it with a citation-graph
provider when you need impact ordering.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import time
from pathlib import Path

from mla import __version__
from mla.corpus import Paper, corpus_sha, read, source_files
from mla.text import fold

SCHEMA_VERSION = 1
DEFAULT_INDEX_RELPATH = Path("anthology-search") / "index" / "mla.sqlite3"
BATCH = 5_000

# Title matches beat abstract matches decisively; an author-name hit sits between them.
# (bm25() in SQLite returns a NEGATIVE score, lower = better; the CLI flips the sign.)
BM25_WEIGHTS = (12.0, 1.0, 3.0)

_COLUMNS = (
    "citekey", "title", "abstract", "authors", "authors_fold", "author_slugs",
    "year", "venue", "venue_name", "venue_type", "doi", "doi_fold", "pdf_url",
    "venue_url", "openreview_url", "code_url", "source", "volume", "pages", "number",
)

DDL = f"""
CREATE TABLE papers (
    rowid          INTEGER PRIMARY KEY,
    citekey        TEXT NOT NULL UNIQUE,
    title          TEXT NOT NULL,
    abstract       TEXT NOT NULL DEFAULT '',
    authors        TEXT NOT NULL DEFAULT '',
    authors_fold   TEXT NOT NULL DEFAULT '',
    author_slugs   TEXT NOT NULL DEFAULT '',
    year           INTEGER,
    venue          TEXT NOT NULL DEFAULT '',
    venue_name     TEXT NOT NULL DEFAULT '',
    venue_type     TEXT NOT NULL DEFAULT '',
    doi            TEXT NOT NULL DEFAULT '',
    doi_fold       TEXT NOT NULL DEFAULT '',
    pdf_url        TEXT NOT NULL DEFAULT '',
    venue_url      TEXT NOT NULL DEFAULT '',
    openreview_url TEXT NOT NULL DEFAULT '',
    code_url       TEXT NOT NULL DEFAULT '',
    source         TEXT NOT NULL DEFAULT '',
    volume         TEXT NOT NULL DEFAULT '',
    pages          TEXT NOT NULL DEFAULT '',
    number         TEXT NOT NULL DEFAULT ''
);
CREATE VIRTUAL TABLE papers_fts USING fts5(
    title, abstract, authors,
    content='papers', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TABLE meta   (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE venues (
    venue      TEXT PRIMARY KEY,
    venue_name TEXT NOT NULL DEFAULT '',
    venue_type TEXT NOT NULL DEFAULT '',
    papers     INTEGER NOT NULL,
    year_min   INTEGER,
    year_max   INTEGER,
    abstracts  INTEGER NOT NULL
);
"""

POST_DDL = """
CREATE INDEX papers_venue_year ON papers(venue, year);
CREATE INDEX papers_year       ON papers(year);
CREATE INDEX papers_doi        ON papers(doi_fold) WHERE doi_fold <> '';
"""


class IndexUnavailable(RuntimeError):
    """The index is missing, stale, or unreadable — fail closed."""


def repo_root(start: Path | None = None) -> Path:
    """Walk up for the ML Anthology checkout (the directory that owns ``data/``)."""
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "data" / "papers").is_dir():
            return candidate
    raise IndexUnavailable(
        "MLA-INDEX: no ML Anthology checkout found above "
        f"{here} (looked for a data/papers/ directory)"
    )


def default_index_path() -> Path:
    """``$MLA_INDEX`` wins; otherwise the index lives beside the corpus it came from."""
    if override := os.environ.get("MLA_INDEX", "").strip():
        return Path(override).expanduser()
    return repo_root() / DEFAULT_INDEX_RELPATH


def default_data_dir() -> Path:
    if override := os.environ.get("MLA_DATA", "").strip():
        return Path(override).expanduser()
    return repo_root() / "data"


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the index read-only. A missing index is an error with the fix in it."""
    target = Path(path) if path else default_index_path()
    if not target.is_file():
        raise IndexUnavailable(
            f"MLA-INDEX: no index at {target}. Build it with `mla index` "
            "(about a minute; roughly 700 MB)."
        )
    connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _git_commit(root: Path) -> str:
    """The corpus commit, when the checkout is a git repo — else "" (never fatal)."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return done.stdout.strip() if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _corpus_commit(data_dir: Path) -> str:
    """Best-effort: a corpus built from a bare directory has no commit, and that is fine."""
    try:
        return _git_commit(repo_root(data_dir))
    except IndexUnavailable:
        return ""


def _row(paper: Paper) -> tuple:
    authors = "; ".join(paper.authors)
    slugs = " ".join(paper.author_slugs)
    return (
        paper.citekey, paper.title, paper.abstract, authors, fold(authors),
        f" {slugs} " if slugs else "", paper.year, paper.venue, paper.venue_name,
        paper.venue_type, paper.doi, fold(paper.doi), paper.pdf_url, paper.venue_url,
        paper.openreview_url, paper.code_url, paper.source, paper.volume,
        paper.pages, paper.number,
    )


def build(data_dir: Path | None = None, index_path: Path | None = None,
          progress=None) -> dict:
    """Compile ``data/`` into a fresh FTS5 index. Returns the meta dict it recorded."""
    data = Path(data_dir) if data_dir else default_data_dir()
    target = Path(index_path) if index_path else default_index_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".building")
    staging.unlink(missing_ok=True)

    files = source_files(data)
    started = time.time()

    connection = sqlite3.connect(staging)
    try:
        connection.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
        connection.executescript(DDL)

        placeholders = ", ".join("?" * len(_COLUMNS))
        insert = (f"INSERT OR IGNORE INTO papers ({', '.join(_COLUMNS)}) "
                  f"VALUES ({placeholders})")

        batch: list[tuple] = []
        count = 0
        for paper in read(data):
            batch.append(_row(paper))
            if len(batch) >= BATCH:
                connection.executemany(insert, batch)
                count += len(batch)
                batch.clear()
                if progress:
                    progress(count)
        if batch:
            connection.executemany(insert, batch)
            count += len(batch)
            if progress:
                progress(count)

        connection.executescript(POST_DDL)
        connection.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
        connection.execute("""
            INSERT INTO venues (venue, venue_name, venue_type, papers,
                                year_min, year_max, abstracts)
            SELECT venue,
                   COALESCE(MAX(NULLIF(venue_name, '')), ''),
                   COALESCE(MAX(NULLIF(venue_type, '')), ''),
                   COUNT(*), MIN(year), MAX(year),
                   SUM(CASE WHEN abstract <> '' THEN 1 ELSE 0 END)
              FROM papers GROUP BY venue
        """)

        stored = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        span = connection.execute("SELECT MIN(year), MAX(year) FROM papers").fetchone()
        venues = connection.execute("SELECT COUNT(*) FROM venues").fetchone()[0]
        abstracts = connection.execute(
            "SELECT COUNT(*) FROM papers WHERE abstract <> ''").fetchone()[0]

        meta = {
            "schema_version": str(SCHEMA_VERSION),
            "mla_version": __version__,
            "corpus_sha": corpus_sha(files),
            "corpus_commit": _corpus_commit(data),
            "data_dir": str(data),
            "source_files": str(len(files)),
            "papers": str(stored),
            "papers_read": str(count),
            "abstracts": str(abstracts),
            "venues": str(venues),
            "year_min": str(span[0] or ""),
            "year_max": str(span[1] or ""),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build_seconds": f"{time.time() - started:.1f}",
        }
        connection.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)", sorted(meta.items()))

        connection.execute("INSERT INTO papers_fts(papers_fts) VALUES('optimize')")
        connection.commit()
        connection.execute("VACUUM")
        connection.commit()
    finally:
        connection.close()

    staging.replace(target)                      # atomic: never search a partial index
    meta["index_path"] = str(target)
    meta["index_bytes"] = str(target.stat().st_size)
    return meta


def meta(connection: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM meta")}


def venues(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute(
        "SELECT * FROM venues ORDER BY papers DESC, venue ASC").fetchall()
    return [dict(row) for row in rows]


def stale(connection: sqlite3.Connection, data_dir: Path | None = None) -> bool:
    """True when ``data/`` has moved since the index was built (a fresh `git pull`)."""
    data = Path(data_dir) if data_dir else default_data_dir()
    try:
        return meta(connection).get("corpus_sha", "") != corpus_sha(source_files(data))
    except Exception:
        return False
