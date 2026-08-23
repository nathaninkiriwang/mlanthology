"""The search API over the FTS5 index.

Every result set names the corpus that produced it (`sha`, `commit`, `built_at`) so a
hit can be traced back to exact bytes, and reports `total` alongside the returned page
so a caller can tell "the field is thin" from "my limit was 20".

Filters compose: venue, year span, venue type, author, and the has-code / has-pdf
predicates all narrow the same MATCH. With an empty query they become a browse — every
NeurIPS 2024 paper, say — which is the cheapest way to establish what a venue-year
actually contains before searching inside it.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field

from mla.index import BM25_WEIGHTS, meta as index_meta
from mla.text import compile_match, fold
from mla.venues import canonical_name

SITE_URL = os.environ.get("MLA_SITE_URL", "https://mlanthology.org").rstrip("/")
SEARCHABLE = ("title", "abstract", "authors")
VENUE_TYPES = ("conference", "journal", "workshop")
SNIPPET_TOKENS = 24
MAX_LIMIT = 500

# BM25 barely separates "Attention is All you Need" from "Attention Is All You Need But
# You Don't Need All Of It" — both titles carry every term. These bonuses break that tie
# toward the paper the caller is obviously naming, which is what `--purpose
# targeted-lookup` means. They are additive on the flipped BM25 score and deterministic.
TITLE_PHRASE_BONUS = 10.0     # the whole query appears verbatim in the title
TITLE_TERMS_BONUS = 4.0       # every query term appears in the title, any order
RERANK_DEPTH = 5              # candidates pulled per requested hit before re-ranking
RERANK_CAP = 400

_SELECT = """
    p.citekey, p.title, p.authors, p.year, p.venue, p.venue_name, p.venue_type,
    p.doi, p.pdf_url, p.venue_url, p.openreview_url, p.code_url, p.source,
    p.volume, p.pages, p.number, p.abstract
"""


class QueryError(ValueError):
    """A search the index cannot run as asked — the caller's fault, stated plainly."""


@dataclass
class Filters:
    venues: tuple[str, ...] = ()
    year_min: int | None = None
    year_max: int | None = None
    venue_types: tuple[str, ...] = ()
    author: str = ""
    has_code: bool = False
    has_pdf: bool = False

    def as_dict(self) -> dict:
        out: dict = {}
        if self.venues:
            out["venues"] = list(self.venues)
        if self.year_min is not None:
            out["year_min"] = self.year_min
        if self.year_max is not None:
            out["year_max"] = self.year_max
        if self.venue_types:
            out["venue_types"] = list(self.venue_types)
        if self.author:
            out["author"] = self.author
        if self.has_code:
            out["has_code"] = True
        if self.has_pdf:
            out["has_pdf"] = True
        return out

    def sql(self) -> tuple[list[str], list]:
        clauses: list[str] = []
        params: list = []
        if self.venues:
            clauses.append(f"p.venue IN ({', '.join('?' * len(self.venues))})")
            params.extend(v.lower() for v in self.venues)
        if self.year_min is not None:
            clauses.append("p.year >= ?")
            params.append(self.year_min)
        if self.year_max is not None:
            clauses.append("p.year <= ?")
            params.append(self.year_max)
        if self.venue_types:
            clauses.append(f"p.venue_type IN ({', '.join('?' * len(self.venue_types))})")
            params.extend(self.venue_types)
        if self.author:
            # Slug form ("he-kaiming") matches the author index exactly; anything else
            # is a folded substring, so "schölkopf" and "scholkopf" behave the same.
            needle = fold(self.author.strip())
            if "-" in needle and " " not in needle:
                clauses.append("(p.author_slugs LIKE ? OR p.authors_fold LIKE ?)")
                params.extend([f"% {needle} %", f"%{needle.replace('-', ' ')}%"])
            else:
                clauses.append("p.authors_fold LIKE ?")
                params.append(f"%{needle}%")
        if self.has_code:
            clauses.append("p.code_url <> ''")
        if self.has_pdf:
            clauses.append("p.pdf_url <> ''")
        return clauses, params


@dataclass
class Result:
    hits: list[dict] = field(default_factory=list)
    total: int = 0
    match: str = ""
    dropped_terms: list[str] = field(default_factory=list)
    # Set only when filters shut a non-empty match down to nothing: how many papers the
    # same query finds with the filters lifted. It separates "the field is thin" from
    # "my venue/year box was wrong", which are opposite next moves.
    unfiltered_total: int | None = None


def _title_bonus(query: str, title: str) -> float:
    """A deterministic nudge for titles that literally contain what was asked for."""
    folded_query = " ".join(fold(query).split())
    folded_title = " ".join(fold(title).split())
    if not folded_query:
        return 0.0
    if folded_query in folded_title:
        # Scaled by how much of the title the query covers: naming a paper's whole title
        # is the strongest targeted-lookup signal there is, while matching five words of
        # a sixty-character title is mostly coincidence.
        return TITLE_PHRASE_BONUS * (len(folded_query) / max(len(folded_title), 1))
    terms = [t for t in folded_query.split() if len(t) > 1]
    if terms and all(term in folded_title for term in terms):
        return TITLE_TERMS_BONUS
    return 0.0


def anthology_url(citekey: str, venue: str, year: int | None) -> str:
    """The deterministic permalink. Computable from metadata alone, per the URL scheme."""
    if not (venue and year):
        return ""
    return f"{SITE_URL}/{venue}/{year}/{citekey}"


def _hit(row: sqlite3.Row, *, score: float | None = None, snippet: str = "",
         abstracts: bool = False) -> dict:
    authors = [name for name in (row["authors"] or "").split("; ") if name]
    permalink = anthology_url(row["citekey"], row["venue"], row["year"])
    hit = {
        "paper_id": row["citekey"],           # the id every other surface keys on
        "citekey": row["citekey"],
        "title": row["title"],
        "authors": authors,
        "year": row["year"],
        "venue": row["venue"],
        # Always the canonical name, never the per-volume string: the corpus stores
        # "Proceedings of the 42nd ICML" for 2025 and "" for 1988, and a consumer that
        # keys on venue_name should not see it flicker between a slug and a sentence.
        "venue_name": canonical_name(row["venue"], row["venue_name"]),
        "venue_type": row["venue_type"],
        "doi": row["doi"],
        # `url` is what an importer should follow: the publisher record first, the PDF
        # next, the anthology page last. `anthology_url` stays available separately.
        "url": row["venue_url"] or row["pdf_url"] or permalink,
        "anthology_url": permalink,
        "pdf_url": row["pdf_url"],
        "openreview_url": row["openreview_url"],
        "code_url": row["code_url"],
        "source": "mla",
        "upstream_source": row["source"],
    }
    if snippet:
        hit["snippet"] = snippet
    if score is not None:
        hit["score"] = round(score, 4)
    if abstracts:
        hit["abstract"] = row["abstract"]
    return hit


def _column_filter(expression: str, fields: tuple[str, ...]) -> str:
    """Restrict a MATCH to some columns: ``{title} : (expr)``."""
    if not expression or set(fields) == set(SEARCHABLE):
        return expression
    unknown = sorted(set(fields) - set(SEARCHABLE))
    if unknown:
        raise QueryError(
            f"MLA-QUERY: unknown search field(s) {', '.join(unknown)}; "
            f"expected any of {', '.join(SEARCHABLE)}")
    return "{" + " ".join(fields) + "} : (" + expression + ")"


def search(connection: sqlite3.Connection, query: str, *, limit: int = 20,
           offset: int = 0, filters: Filters | None = None, mode: str = "all",
           fields: tuple[str, ...] = SEARCHABLE, abstracts: bool = False) -> Result:
    """Run one search. An empty query with filters is a browse, ordered newest-first."""
    if limit < 1 or limit > MAX_LIMIT:
        raise QueryError(f"MLA-QUERY: limit must be 1..{MAX_LIMIT}, got {limit}")
    if offset < 0:
        raise QueryError(f"MLA-QUERY: offset must be >= 0, got {offset}")
    filters = filters or Filters()
    for kind in filters.venue_types:
        if kind not in VENUE_TYPES:
            raise QueryError(
                f"MLA-QUERY: unknown venue type {kind!r}; "
                f"expected any of {', '.join(VENUE_TYPES)}")

    expression, dropped = compile_match(query, mode)
    where, params = filters.sql()

    if not expression:
        if not where:
            raise QueryError(
                "MLA-QUERY: an empty query needs at least one filter "
                "(--venue / --year / --author) — refusing to return the whole corpus")
        clause = " WHERE " + " AND ".join(where)
        total = connection.execute(
            f"SELECT COUNT(*) FROM papers p{clause}", params).fetchone()[0]
        rows = connection.execute(
            f"SELECT {_SELECT} FROM papers p{clause} "
            "ORDER BY p.year DESC, p.citekey ASC LIMIT ? OFFSET ?",
            [*params, limit, offset]).fetchall()
        return Result(
            hits=[_hit(row, abstracts=abstracts) for row in rows],
            total=total, match="", dropped_terms=dropped)

    expression = _column_filter(expression, fields)
    clause = " AND " + " AND ".join(where) if where else ""
    count_sql = ("SELECT COUNT(*) FROM papers_fts JOIN papers p "
                 "ON p.rowid = papers_fts.rowid WHERE papers_fts MATCH ?")
    # Pull deeper than asked, re-rank, then page — so a title match sitting just past
    # the BM25 cut still surfaces.
    depth = min((offset + limit) * RERANK_DEPTH, RERANK_CAP)
    try:
        total = connection.execute(count_sql + clause, [expression, *params]).fetchone()[0]
        rows = connection.execute(
            f"SELECT {_SELECT}, bm25(papers_fts, ?, ?, ?) AS score, "
            f"snippet(papers_fts, 1, '', '', '…', {SNIPPET_TOKENS}) AS snip "
            "FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid "
            f"WHERE papers_fts MATCH ?{clause} "
            "ORDER BY score ASC, p.year DESC LIMIT ?",
            [*BM25_WEIGHTS, expression, *params, depth]).fetchall()
    except sqlite3.OperationalError as exc:
        raise QueryError(f"MLA-QUERY: {exc} (compiled match: {expression})") from exc

    # bm25() is negative with better matches more negative; flip it so bigger is better.
    scored = [(-row["score"] + _title_bonus(query, row["title"]), row) for row in rows]
    scored.sort(key=lambda pair: (-pair[0], -(pair[1]["year"] or 0)))
    page = scored[offset:offset + limit]

    unfiltered = None
    if total == 0 and where:
        unfiltered = connection.execute(count_sql, [expression]).fetchone()[0]

    return Result(
        hits=[_hit(row, score=score, snippet=row["snip"], abstracts=abstracts)
              for score, row in page],
        total=total, match=expression, dropped_terms=dropped, unfiltered_total=unfiltered)


def get(connection: sqlite3.Connection, citekey: str, abstracts: bool = True) -> dict | None:
    row = connection.execute(
        f"SELECT {_SELECT} FROM papers p WHERE p.citekey = ?", [citekey.strip()]).fetchone()
    return _hit(row, abstracts=abstracts) if row else None


def by_doi(connection: sqlite3.Connection, doi: str, abstracts: bool = True) -> dict | None:
    """Resolve a DOI to a paper — the cheapest possible dedup check against this corpus."""
    needle = fold(doi.strip()).removeprefix("https://doi.org/").removeprefix("doi:")
    row = connection.execute(
        f"SELECT {_SELECT} FROM papers p WHERE p.doi_fold = ?", [needle]).fetchone()
    return _hit(row, abstracts=abstracts) if row else None


def corpus_stamp(connection: sqlite3.Connection) -> dict:
    """The provenance block every response carries."""
    info = index_meta(connection)
    return {
        "name": "mlanthology",
        "sha": info.get("corpus_sha", ""),
        "commit": info.get("corpus_commit", ""),
        "papers": int(info.get("papers", 0) or 0),
        "venues": int(info.get("venues", 0) or 0),
        "years": [info.get("year_min", ""), info.get("year_max", "")],
        "built_at": info.get("built_at", ""),
        "schema_version": int(info.get("schema_version", 0) or 0),
    }
