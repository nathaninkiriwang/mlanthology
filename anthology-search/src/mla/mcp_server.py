"""``mla-mcp`` — an MCP server over the same index, for MCP-native seats.

This is the SECONDARY surface. The CLI is primary, because a harness that records what
it searched needs a subprocess boundary it can log, an exit code it can branch on, and a
JSON payload it can attach to a record. An MCP tool call gives none of those to the
caller's ledger.

Use this when the consumer is an agent runtime that mounts MCP servers and has no shell
(a Claude Code seat with Bash denied, say). It exposes the same four operations, and
every tool returns a STRING — a JSON blob on success, a plain sentinel on a bad input —
so a malformed argument can never crash the seat's tool surface.

    mla-mcp                      # stdio server
"""

from __future__ import annotations

import json
import sys

from mla.index import IndexUnavailable, connect
from mla.query import Filters, QueryError, SEARCHABLE, by_doi, corpus_stamp, get, search
from mla.venues import FAMILIES, FAMILY_OF, canonical_name, expand_families

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:                     # library mode: importable without `mcp`
    class FastMCP:                              # installed, so the CLI never depends on it
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def decorate(fn):
                return fn
            return decorate

        def run(self, *args, **kwargs):
            raise RuntimeError(
                "mla-mcp needs the `mcp` package: install this project with the "
                "[mcp] extra (`uv sync --extra mcp`).")

mcp = FastMCP("mla")

_connection = None


def _db():
    global _connection
    if _connection is None:
        _connection = connect()
    return _connection


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def mla_search(query: str, limit: int = 20, venues: str = "", family: str = "",
               year_min: int | None = None, year_max: int | None = None,
               venue_types: str = "", author: str = "", any_terms: bool = False,
               abstracts: bool = False) -> str:
    """Search ~217k ML papers (44 venues, 1969-2026) by title, abstract and author.

    Offline and exhaustive within its venues: it cannot rate-limit, and a venue-year it
    covers, it covers completely. It carries no citation counts, so ranking is lexical.

    venues:      comma-separated slugs, e.g. "neurips,icml" (see mla_venues)
    family:      comma-separated topical families instead of naming venues:
                 core-ml, theory, vision, ai, robotics, health, graphs, causal
    venue_types: comma-separated subset of conference, journal, workshop
    author:      an author slug ("he-kaiming") or a name fragment
    any_terms:   OR the terms rather than AND — the widening rung when results are thin
    """
    picked = [v.strip().lower() for v in venues.split(",") if v.strip()]
    if family:
        expanded, unknown = expand_families(
            tuple(f.strip() for f in family.split(",") if f.strip()))
        if unknown:
            return (f"MLA-FAMILY: unknown family/families {', '.join(unknown)}. "
                    f"Known: {', '.join(sorted(FAMILIES))}.")
        picked.extend(v for v in expanded if v not in picked)
    try:
        connection = _db()
        result = search(
            connection, query, limit=max(1, min(limit, 100)),
            filters=Filters(
                venues=tuple(picked), year_min=year_min, year_max=year_max,
                venue_types=tuple(t.strip() for t in venue_types.split(",") if t.strip()),
                author=author),
            mode="any" if any_terms else "all", abstracts=abstracts)
    except (QueryError, IndexUnavailable) as exc:
        return str(exc)
    payload = {
        "provider": "mla", "query": query, "match": result.match,
        "total": result.total, "returned": len(result.hits),
        "corpus": corpus_stamp(connection), "hits": result.hits,
    }
    if result.dropped_terms:
        payload["dropped_terms"] = result.dropped_terms
    if result.unfiltered_total is not None:
        payload["unfiltered_total"] = result.unfiltered_total
        payload["note"] = "0 hits with filters, more without — widen the filters, not the query."
    return _json(payload)


@mcp.tool()
def mla_get(citekey: str) -> str:
    """Fetch one paper by its ML Anthology citekey (e.g. `vaswani2017neurips-attention`)."""
    try:
        paper = get(_db(), citekey)
    except IndexUnavailable as exc:
        return str(exc)
    return _json(paper) if paper else f"MLA-GET: no paper with citekey {citekey!r}"


@mcp.tool()
def mla_resolve_doi(doi: str) -> str:
    """Resolve a DOI against the corpus — the cheapest dedup check before importing."""
    try:
        paper = by_doi(_db(), doi)
    except IndexUnavailable as exc:
        return str(exc)
    return _json(paper) if paper else f"MLA-DOI: {doi} is not in this corpus"


@mcp.tool()
def mla_venues() -> str:
    """List the 44 indexed venues with paper counts, year spans and topical families."""
    try:
        connection = _db()
        rows = [
            {"venue": row["venue"],
             "name": canonical_name(row["venue"], row["venue_name"]),
             "family": FAMILY_OF.get(row["venue"], ""),
             "venue_type": row["venue_type"],
             "papers": row["papers"],
             "years": [row["year_min"], row["year_max"]]}
            for row in connection.execute(
                "SELECT * FROM venues ORDER BY papers DESC").fetchall()
        ]
    except IndexUnavailable as exc:
        return str(exc)
    return _json({"provider": "mla", "corpus": corpus_stamp(connection),
                  "families": {k: list(v) for k, v in FAMILIES.items()}, "venues": rows})


def main() -> int:
    try:
        connect().close()
    except IndexUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 4
    mcp.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
