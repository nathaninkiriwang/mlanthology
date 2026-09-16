"""``mla`` — the command line surface.

Built for a harness that shells out, parses JSON, and writes down what it got: every
machine command prints one JSON object to stdout and nothing else, diagnostics go to
stderr, and the exit code says which kind of thing went wrong.

    0  ok
    2  the caller asked for something malformed (bad flag, bad limit, empty query)
    3  the caller asked for something refused (unknown venue, unknown venue type)
    4  the index is missing or unreadable — build it with `mla index`; likewise a hybrid
       lane that is not built or the `hybrid` extra that is not installed (0.2.0)

An unknown venue is a REFUSAL, not an empty result. `--venue nips` is a typo for
`neurips`, and a sweep that silently returns zero for a typo teaches the searcher the
wrong lesson about the field.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mla import __version__
from mla.index import (
    IndexUnavailable, build, connect, default_data_dir, default_index_path,
    meta as index_meta, stale, venues as index_venues,
)
from mla.query import (
    Filters, QueryError, SEARCHABLE, VENUE_TYPES, by_doi, corpus_stamp, get, search,
)
from mla.venues import FAMILIES, FAMILY_OF, canonical_name, expand_families

# The hybrid lanes (0.2.0, D-107) import numpy/bm25s/torch lazily inside mla.hybrid; this
# module only ever touches them behind `--mode`, `bm25`, `vec`, `batch` and `rerank`.
from mla.hybrid import (
    BACKENDS, DEFAULT_RERANKER, LaneUnavailable, MODELS, MODES, RERANKERS, Engine, Reranker, build_bm25,
    build_vec, default_dense, lanes_status, rerank_pool, set_default_dense,
)

EXIT_OK, EXIT_USAGE, EXIT_REFUSED, EXIT_NO_INDEX = 0, 2, 3, 4


class Refused(Exception):
    """Something the tool will not do, named so the caller can fix it."""


def _emit(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=False)
    sys.stdout.write("\n")


def _known_venues(connection) -> dict[str, dict]:
    return {row["venue"]: row for row in index_venues(connection)}


def _check_venues(connection, requested: tuple[str, ...]) -> tuple[str, ...]:
    if not requested:
        return ()
    known = _known_venues(connection)
    wanted = tuple(v.strip().lower() for v in requested if v.strip())
    if unknown := [v for v in wanted if v not in known]:
        raise Refused(
            f"MLA-VENUE: unknown venue(s) {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}. (Run `mla venues` for counts and spans.)")
    return wanted


FILTER_KEYS = ("venues", "families", "year", "since", "until", "types", "no_workshops",
               "author", "has_code", "has_pdf")


def filters_from(spec: dict, connection) -> tuple[Filters, list[str]]:
    """The one place filters are validated — for the flags and for `mla batch` alike.

    `spec` keys: venues, families, year, since, until, types (lists), no_workshops, author,
    has_code, has_pdf. Returns the Filters and the family members this index lacks."""
    if unknown_keys := sorted(set(spec) - set(FILTER_KEYS)):
        raise Refused(f"MLA-FILTER: unknown filter key(s) {', '.join(unknown_keys)}. "
                      f"Known: {', '.join(FILTER_KEYS)}.")
    types = tuple(spec.get("types") or ())
    if unknown := [t for t in types if t not in VENUE_TYPES]:
        raise Refused(f"MLA-TYPE: unknown venue type(s) {', '.join(unknown)}. "
                      f"Known: {', '.join(VENUE_TYPES)}.")
    if spec.get("no_workshops"):
        types = tuple(t for t in VENUE_TYPES if t != "workshop") if not types else \
            tuple(t for t in types if t != "workshop")
    year_min, year_max = spec.get("since"), spec.get("until")
    if spec.get("year") is not None:
        year_min = year_max = spec["year"]
    for label, value in (("year", spec.get("year")), ("since", year_min), ("until", year_max)):
        if value is not None and not isinstance(value, int):
            raise QueryError(f"MLA-YEAR: {label} must be an integer, got {value!r}")
    if year_min is not None and year_max is not None and year_min > year_max:
        raise QueryError(f"MLA-YEAR: --since {year_min} is after --until {year_max}")
    # A hand-named --venue is a claim that venue exists, so a typo is refused. A --family
    # is a claim about a topic: members this index happens not to carry are skipped and
    # reported, because losing ISIPTA should not cancel a theory sweep.
    requested = _check_venues(connection, tuple(spec.get("venues") or ()))
    absent: list[str] = []
    families = tuple(spec.get("families") or ())
    if families:
        expanded, unknown = expand_families(families)
        if unknown:
            raise Refused(f"MLA-FAMILY: unknown family/families {', '.join(unknown)}. "
                          f"Known: {', '.join(sorted(FAMILIES))}.")
        known = _known_venues(connection)
        absent = [v for v in expanded if v not in known]
        requested = tuple(requested) + tuple(
            v for v in expanded if v in known and v not in requested)
        if not requested:
            raise Refused(
                f"MLA-FAMILY: this index carries none of {', '.join(families)} "
                f"({', '.join(expanded)}). Run `mla venues` to see what it holds.")
    filters = Filters(
        venues=tuple(requested),
        year_min=year_min, year_max=year_max, venue_types=types,
        author=str(spec.get("author") or ""), has_code=bool(spec.get("has_code")),
        has_pdf=bool(spec.get("has_pdf")),
    )
    return filters, absent


def _filters(args, connection) -> Filters:
    filters, absent = filters_from({
        "venues": list(args.venue or ()), "families": list(args.family or ()),
        "year": args.year, "since": args.since, "until": args.until,
        "types": list(args.type or ()), "no_workshops": bool(args.no_workshops),
        "author": args.author or "", "has_code": bool(args.has_code),
        "has_pdf": bool(args.has_pdf),
    }, connection)
    args.family_absent = absent
    return filters


def _lane_payload(args, connection, filters: Filters, answer: dict) -> dict:
    """The non-FTS modes' payload: the same top-level keys the harness parses (`provider`,
    `query`, `match`, `total`, `returned`, `corpus`, `hits`, the diagnostics) plus the lanes'
    own — additive, never renamed."""
    payload = {
        "provider": "mla",
        "query": args.query,
        "match": "",
        "mode": answer["mode"],
        "dense": answer["dense"],
        "fields": ["title", "abstract"],
        "filters": filters.as_dict(),
        "total": answer["total"],
        "totals": answer["totals"],
        "returned": len(answer["hits"]),
        "offset": args.offset,
        "depth": answer["depth"],
        "corpus": corpus_stamp(connection),
        "hits": answer["hits"],
    }
    if answer.get("unknown_terms"):
        payload["unknown_terms"] = answer["unknown_terms"]
    if answer.get("reranker"):
        payload["reranker"] = answer["reranker"]
    if absent := getattr(args, "family_absent", None):
        payload["family_venues_absent"] = absent
    return payload


def _rerank_hits(connection, hits: list[dict], query: str, depth: int, model: str | None) -> dict:
    """Re-order the top `depth` of a hit list once; the fused rank is kept on every hit."""
    reranker = Reranker(model or DEFAULT_RERANKER)
    head, tail = hits[:depth], hits[depth:]
    for rank, hit in enumerate(hits, start=1):
        hit["fused_rank"] = rank
    verdict = rerank_pool(connection, reranker, query,
                          [{"id": hit["citekey"], "citekey": hit["citekey"]} for hit in head])
    score_of = {row["id"]: row["score"] for row in verdict["ranked"]}
    head.sort(key=lambda hit: -score_of.get(hit["citekey"], float("-inf")))
    for hit in head:
        hit["rerank_score"] = score_of.get(hit["citekey"])
    hits[:] = head + tail
    return {"model": verdict["model"], "depth": len(head), "seconds": verdict["seconds"],
            "device": verdict["device"]}


def cmd_search(args) -> int:
    connection = connect(args.index)
    filters = _filters(args, connection)
    if args.mode != "fts":
        if args.rerank < 0:
            raise QueryError(f"MLA-QUERY: --rerank must be >= 0, got {args.rerank}")
        if not args.query.strip():
            raise QueryError("MLA-QUERY: the hybrid lanes need a non-empty query")
        engine = Engine(connection, Path(args.index or default_index_path()), dense=args.dense)
        answer = engine.search(args.query, mode=args.mode, limit=args.limit, offset=args.offset,
                               filters=filters, abstracts=args.abstracts)
        if args.rerank:
            answer["reranker"] = _rerank_hits(connection, answer["hits"], args.query, args.rerank,
                                              args.reranker)
        _emit(_lane_payload(args, connection, filters, answer))
        return EXIT_OK
    fields = tuple(f.strip() for f in (args.field or "").split(",") if f.strip()) or SEARCHABLE
    result = search(
        connection, args.query, limit=args.limit, offset=args.offset, filters=filters,
        mode="any" if getattr(args, "any") else "all", fields=fields,
        abstracts=args.abstracts,
    )
    payload = {
        "provider": "mla",
        "query": args.query,
        "match": result.match,
        "mode": "any" if getattr(args, "any") else "all",
        "fields": list(fields),
        "filters": filters.as_dict(),
        "total": result.total,
        "returned": len(result.hits),
        "offset": args.offset,
        "corpus": corpus_stamp(connection),
        "hits": result.hits,
    }
    if result.dropped_terms:
        payload["dropped_terms"] = result.dropped_terms
    if absent := getattr(args, "family_absent", None):
        payload["family_venues_absent"] = absent
    if result.unfiltered_total is not None:
        payload["unfiltered_total"] = result.unfiltered_total
        # The count decides which advice is true. Saying "the filters cut this" when the
        # query finds nothing anywhere sends the searcher to widen a box that was never
        # the problem, and costs them a whole rung.
        payload["note"] = (
            f"0 hits with filters, {result.unfiltered_total} without — the filters cut "
            "this, not the query. Widen the venue/year box before rewording."
            if result.unfiltered_total
            else "0 hits with or without the filters — the terms, not the box. Try "
                 "synonyms or --any before widening scope.")
    _emit(payload)
    return EXIT_OK


def cmd_get(args) -> int:
    connection = connect(args.index)
    paper = get(connection, args.citekey)
    if paper is None:
        _emit({"provider": "mla", "citekey": args.citekey, "found": False,
               "corpus": corpus_stamp(connection)})
        return EXIT_OK
    _emit({"provider": "mla", "found": True, "corpus": corpus_stamp(connection),
           "paper": paper})
    return EXIT_OK


def cmd_doi(args) -> int:
    connection = connect(args.index)
    paper = by_doi(connection, args.doi)
    _emit({"provider": "mla", "doi": args.doi, "found": paper is not None,
           "corpus": corpus_stamp(connection),
           **({"paper": paper} if paper else {})})
    return EXIT_OK


def _bibtex(paper: dict) -> str:
    kind = "article" if paper["venue_type"] == "journal" else "inproceedings"
    container = canonical_name(paper["venue"], paper["venue_name"])
    fields: list[tuple[str, str]] = [
        ("title", "{" + paper["title"] + "}"),
        ("author", " and ".join(paper["authors"])),
        ("year", str(paper["year"] or "")),
        ("journal" if kind == "article" else "booktitle", container),
    ]
    for name in ("volume", "number", "pages", "doi"):
        if value := paper.get(name):
            fields.append((name, value))
    # `anthology_url` is COMPUTED from venue/year/citekey, not verified, so it 404s for
    # any paper the mlanthology site does not host. ACL Anthology imports live at
    # aclanthology.org (measured: the computed permalink returns 404, the stored landing
    # page 200), so for those the stored `url` is canonical. Every other venue keeps the
    # site permalink it had.
    site_hosts_it = paper.get("upstream_source") != "acl-anthology"
    if url := ((paper.get("anthology_url") if site_hosts_it else "") or paper.get("url")):
        fields.append(("url", url))
    body = ",\n".join(f"  {k:<9} = {{{v}}}" for k, v in fields if v)
    return f"@{kind}{{{paper['citekey']},\n{body}\n}}"


def cmd_bibtex(args) -> int:
    connection = connect(args.index)
    missing: list[str] = []
    entries: list[str] = []
    for citekey in args.citekeys:
        if paper := get(connection, citekey):
            entries.append(_bibtex(paper))
        else:
            missing.append(citekey)
    if entries:
        sys.stdout.write("\n\n".join(entries) + "\n")
    for citekey in missing:
        print(f"MLA-BIBTEX: no paper with citekey {citekey!r}", file=sys.stderr)
    return EXIT_REFUSED if missing and not entries else EXIT_OK


def cmd_venues(args) -> int:
    connection = connect(args.index)
    rows = [
        {**row,
         "venue_name": canonical_name(row["venue"], row["venue_name"]),
         "family": FAMILY_OF.get(row["venue"], "")}
        for row in index_venues(connection)
    ]
    if args.family_only:
        if args.family_only not in FAMILIES:
            raise Refused(f"MLA-FAMILY: unknown family {args.family_only!r}. "
                          f"Known: {', '.join(sorted(FAMILIES))}.")
        rows = [r for r in rows if r["family"] == args.family_only]
    if args.text:
        print(f"{'venue':<10} {'papers':>7} {'years':>11} {'abs%':>5}  "
              f"{'type':<11}{'family':<9} name")
        for row in rows:
            span = f"{row['year_min']}-{row['year_max']}"
            pct = round(100.0 * row["abstracts"] / max(row["papers"], 1))
            print(f"{row['venue']:<10} {row['papers']:>7} {span:>11} {pct:>4}%  "
                  f"{row['venue_type'] or '-':<11}{row['family']:<9} {row['venue_name'][:46]}")
        return EXIT_OK
    _emit({"provider": "mla", "corpus": corpus_stamp(connection),
           "families": {name: list(members) for name, members in FAMILIES.items()},
           "venues": rows, "count": len(rows)})
    return EXIT_OK


def cmd_stats(args) -> int:
    connection = connect(args.index)
    total = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    coverage = {
        name: connection.execute(
            f"SELECT COUNT(*) FROM papers WHERE {name} <> ''").fetchone()[0]
        for name in ("abstract", "doi", "pdf_url", "code_url", "openreview_url")
    }
    _emit({
        "provider": "mla",
        "index": {"path": str(args.index or default_index_path()),
                  **index_meta(connection)},
        "corpus": corpus_stamp(connection),
        "papers": total,
        # Coverage is published because a filter over a sparse field returns near-zero
        # and reads like an empty field. `code_url` in particular sits around 3%.
        "field_coverage": {
            name: {"papers": count, "pct": round(100.0 * count / max(total, 1), 1)}
            for name, count in coverage.items()
        },
        "stale": stale(connection, args.data),
        "lanes": lanes_status(connection, Path(args.index or default_index_path())),
    })
    return EXIT_OK


def cmd_index(args) -> int:
    target = args.out or default_index_path()
    if target.exists() and not args.force:
        connection = connect(target)
        try:
            if not stale(connection, args.data):
                _emit({"provider": "mla", "built": False, "reason": "up-to-date",
                       "index": str(target), "corpus": corpus_stamp(connection)})
                return EXIT_OK
        finally:
            connection.close()
        print("MLA-INDEX: data/ has changed since the last build; rebuilding.",
              file=sys.stderr)

    def progress(count: int) -> None:
        if count % 50_000 == 0:
            print(f"MLA-INDEX: {count:,} papers…", file=sys.stderr)

    meta = build(data_dir=args.data, index_path=target,
                 progress=None if args.quiet else progress)
    _emit({"provider": "mla", "built": True, **meta})
    return EXIT_OK


def cmd_doctor(args) -> int:
    """Is this installation ready to answer a search? One JSON verdict, honest exit code."""
    path = args.index or default_index_path()
    report: dict = {"provider": "mla", "version": __version__, "index": str(path)}
    if not Path(path).is_file():
        report.update(ok=False, reason="index-missing", fix="mla index")
        _emit(report)
        return EXIT_NO_INDEX
    connection = connect(path)
    is_stale = stale(connection, args.data)
    # Probe with a word taken from the index's own first title. A hardcoded term can be
    # absent from a small or future corpus; this cannot, so a zero here means the FTS
    # table and the content table have genuinely drifted apart.
    probe = None
    for row in connection.execute("SELECT title FROM papers LIMIT 5"):
        for word in row[0].split():
            try:
                probe = search(connection, word, limit=1)
            except QueryError:
                continue        # an unindexable first word ("ℓ∞ bounds") is not a fault
            if probe.total:
                break
        if probe is not None and probe.total:
            break
    report.update(
        ok=not is_stale and probe is not None and probe.total > 0,
        stale=is_stale,
        probe_hits=probe.total if probe else 0,
        corpus=corpus_stamp(connection),
        **({"fix": "mla index --force"} if is_stale else {}),
        # The lanes never move `ok`: the FTS contract is what the harness's doctor leg reads,
        # and a hybrid lane that is missing is reported here, with its own fix, not as a
        # broken index.
        lanes=lanes_status(connection, Path(path)),
    )
    _emit(report)
    return EXIT_OK if report["ok"] else EXIT_NO_INDEX


# --- the hybrid lanes (0.2.0, D-107) -----------------------------------------------------------

def _progress_line(done: int, total: int, elapsed: float) -> None:
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("nan")
    print(f"MLA-VEC: {done:,}/{total:,} embedded, {rate:.1f} docs/s, "
          f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left", file=sys.stderr)


def cmd_bm25_build(args) -> int:
    connection = connect(args.index)
    index_path = Path(args.index or default_index_path())
    meta = build_bm25(connection, index_path)
    _emit({"provider": "mla", "lane": "bm25", "built": True, "corpus": corpus_stamp(connection), **meta})
    return EXIT_OK


def cmd_vec_build(args) -> int:
    connection = connect(args.index)
    index_path = Path(args.index or default_index_path())
    if args.model not in MODELS:
        raise Refused(f"MLA-VEC: unknown model {args.model!r}. Known: {', '.join(sorted(MODELS))}.")
    meta = build_vec(connection, index_path, args.model, backend=args.backend, batch=args.batch,
                     limit=args.limit, fp16=args.fp16, bf16=args.bf16, max_seq=args.max_seq,
                     progress=None if args.quiet else _progress_line)
    marker = None
    if args.default:
        marker = str(set_default_dense(index_path, args.model))
    _emit({"provider": "mla", "lane": "dense", "built": True, "corpus": corpus_stamp(connection),
           **meta, **({"default_marker": marker} if marker else {})})
    return EXIT_OK


def cmd_vec_default(args) -> int:
    connection = connect(args.index)
    index_path = Path(args.index or default_index_path())
    marker = set_default_dense(index_path, args.model)
    _emit({"provider": "mla", "default": args.model, "marker": str(marker),
           "corpus": corpus_stamp(connection)})
    return EXIT_OK


def cmd_vec_status(args) -> int:
    connection = connect(args.index)
    index_path = Path(args.index or default_index_path())
    _emit({"provider": "mla", "corpus": corpus_stamp(connection),
           "lanes": lanes_status(connection, index_path), "models": sorted(MODELS)})
    return EXIT_OK


def _read_json_input(path: str | None) -> dict:
    raw = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    try:
        payload = json.loads(raw or "{}")
    except ValueError as exc:
        raise QueryError(f"MLA-BATCH: malformed JSON input: {exc}") from exc
    if not isinstance(payload, dict):
        raise QueryError("MLA-BATCH: expected a JSON object")
    return payload


def cmd_batch(args) -> int:
    """Many queries, one process, the models loaded once — the sweep's ladder in one call.

    Input: {"queries": [{"id", "query", "mode"?, "limit"?, "offset"?, "filters"?, "abstracts"?}],
    "dense"?: name}. Every query's filters are validated BEFORE any lane loads, so a typo'd
    venue refuses the whole batch (exit 3) instead of answering half a ladder."""
    connection = connect(args.index)
    index_path = Path(args.index or default_index_path())
    spec = _read_json_input(args.file)
    queries = spec.get("queries")
    if not isinstance(queries, list) or not queries:
        raise QueryError("MLA-BATCH: `queries` must be a non-empty list")
    if len(queries) > args.max_queries:
        raise QueryError(f"MLA-BATCH: {len(queries)} queries exceeds --max-queries {args.max_queries}")
    prepared = []
    for position, entry in enumerate(queries):
        if not isinstance(entry, dict) or not str(entry.get("query") or "").strip():
            raise QueryError(f"MLA-BATCH: queries[{position}] needs a non-empty `query`")
        mode = str(entry.get("mode") or args.mode)
        if mode not in MODES or mode == "fts":
            raise QueryError(f"MLA-BATCH: queries[{position}] mode must be one of "
                             f"{', '.join(m for m in MODES if m != 'fts')}, got {mode!r}")
        limit = int(entry.get("limit") or args.limit)
        offset = int(entry.get("offset") or 0)
        if limit < 1 or limit > 500 or offset < 0:
            raise QueryError(f"MLA-BATCH: queries[{position}] limit must be 1..500 and offset >= 0")
        filters, absent = filters_from(dict(entry.get("filters") or {}), connection)
        prepared.append((str(entry.get("id") or f"q{position + 1}"), str(entry["query"]), mode,
                         limit, offset, filters, absent, bool(entry.get("abstracts"))))
    engine = Engine(connection, index_path, dense=spec.get("dense") or args.dense)
    results = []
    started = time.time()
    for entry_id, query, mode, limit, offset, filters, absent, abstracts in prepared:
        t0 = time.time()
        answer = engine.search(query, mode=mode, limit=limit, offset=offset, filters=filters,
                               abstracts=abstracts)
        results.append({
            "id": entry_id, "query": query, "mode": mode, "filters": filters.as_dict(),
            "total": answer["total"], "totals": answer["totals"],
            "unknown_terms": answer["unknown_terms"], "returned": len(answer["hits"]),
            "ms": round((time.time() - t0) * 1000, 1),
            **({"family_venues_absent": absent} if absent else {}),
            "hits": answer["hits"],
        })
    _emit({"provider": "mla", "batch": True, "queries": len(results),
           "dense": engine.dense, "seconds": round(time.time() - started, 2),
           "corpus": corpus_stamp(connection), "results": results})
    return EXIT_OK


def cmd_rerank(args) -> int:
    """Score a pool once against one query. Input: {"query"?, "model"?, "pool": [{"id",
    "citekey"?, "text"?}]}; `--query` on the command line wins over the JSON's."""
    connection = connect(args.index)
    spec = _read_json_input(args.file)
    query = args.query or str(spec.get("query") or "")
    pool = spec.get("pool")
    if not isinstance(pool, list) or not pool:
        raise QueryError("MLA-RERANK: `pool` must be a non-empty list of {id, citekey|text}")
    if len(pool) > args.max_pool:
        raise QueryError(f"MLA-RERANK: pool of {len(pool)} exceeds --max-pool {args.max_pool}")
    if not query.strip():
        raise QueryError("MLA-RERANK: a non-empty --query is required")
    name = args.model or spec.get("model") or DEFAULT_RERANKER
    if name not in RERANKERS:
        raise Refused(f"MLA-RERANK: unknown reranker {name!r}. Known: {', '.join(sorted(RERANKERS))}.")
    verdict = rerank_pool(connection, Reranker(name), query, pool)
    _emit({"provider": "mla", "corpus": corpus_stamp(connection), **verdict})
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mla",
        description="Offline venue-complete ML literature search over the ML Anthology corpus.")
    parser.add_argument("--version", action="version", version=f"mla {__version__}")
    parser.add_argument("--index", type=Path, default=None,
                        help="index file (default: $MLA_INDEX or <repo>/anthology-search/index/)")
    parser.add_argument("--data", type=Path, default=None,
                        help="corpus directory (default: $MLA_DATA or <repo>/data/)")
    sub = parser.add_subparsers(dest="command", required=True)

    search_cmd = sub.add_parser("search", help="search titles, abstracts and authors")
    search_cmd.add_argument("query", help="free text; \"quoted phrases\", AND/OR/NOT and term* work")
    search_cmd.add_argument("-n", "--limit", type=int, default=20)
    search_cmd.add_argument("--offset", type=int, default=0)
    search_cmd.add_argument("--venue", action="append", metavar="V",
                            help="restrict to a venue slug; repeatable")
    search_cmd.add_argument("--family", action="append", metavar="F",
                            help=f"add every venue in a topical family ({', '.join(sorted(FAMILIES))}); "
                                 "repeatable")
    search_cmd.add_argument("--year", type=int, help="exactly this year")
    search_cmd.add_argument("--since", type=int, metavar="Y")
    search_cmd.add_argument("--until", type=int, metavar="Y")
    search_cmd.add_argument("--type", action="append", choices=VENUE_TYPES,
                            help="conference | journal | workshop; repeatable")
    search_cmd.add_argument("--no-workshops", action="store_true",
                            help="drop workshop papers (29k of the corpus)")
    search_cmd.add_argument("--author", metavar="NAME",
                            help="author slug (he-kaiming) or a name fragment")
    search_cmd.add_argument("--has-code", action="store_true",
                            help="only papers with a code link (~3%% of the corpus)")
    search_cmd.add_argument("--has-pdf", action="store_true")
    search_cmd.add_argument("--any", action="store_true",
                            help="OR the terms instead of AND — the widening rung")
    search_cmd.add_argument("--field", metavar="F,F",
                            help=f"restrict matching to some of: {', '.join(SEARCHABLE)}")
    search_cmd.add_argument("--abstracts", action="store_true",
                            help="include full abstracts (large output)")
    search_cmd.add_argument("--mode", choices=MODES, default="fts",
                            help="fts (default: FTS5, AND, no stemming — the exact-term lane) | "
                                 "bm25 (stemmed) | dense (vectors) | hybrid (guarded fusion of "
                                 "bm25 + dense; the measured default for sentence-shaped queries)")
    search_cmd.add_argument("--dense", metavar="MODEL", default=None,
                            help="the dense lane to use (default: index/vec/DEFAULT or $MLA_DENSE)")
    search_cmd.add_argument("--rerank", type=int, default=0, metavar="N",
                            help="re-order the top N of a lane result with the reranker (0 = off)")
    search_cmd.add_argument("--reranker", metavar="MODEL", default=None,
                            help=f"which reranker ({', '.join(sorted(RERANKERS))}); default qwen3-0.6b")
    search_cmd.set_defaults(func=cmd_search)

    get_cmd = sub.add_parser("get", help="one paper by citekey")
    get_cmd.add_argument("citekey")
    get_cmd.set_defaults(func=cmd_get)

    doi_cmd = sub.add_parser("doi", help="resolve a DOI against the corpus")
    doi_cmd.add_argument("doi")
    doi_cmd.set_defaults(func=cmd_doi)

    bib_cmd = sub.add_parser("bibtex", help="BibTeX entries for one or more citekeys")
    bib_cmd.add_argument("citekeys", nargs="+")
    bib_cmd.set_defaults(func=cmd_bibtex)

    venues_cmd = sub.add_parser("venues", help="venue coverage table")
    venues_cmd.add_argument("--text", action="store_true", help="human table instead of JSON")
    venues_cmd.add_argument("--family", dest="family_only", metavar="F",
                            help="only venues in this family")
    venues_cmd.set_defaults(func=cmd_venues)

    stats_cmd = sub.add_parser("stats", help="index provenance and field coverage")
    stats_cmd.set_defaults(func=cmd_stats)

    index_cmd = sub.add_parser("index", help="build or refresh the index from data/")
    index_cmd.add_argument("--out", type=Path, default=None)
    index_cmd.add_argument("--force", action="store_true", help="rebuild even if current")
    index_cmd.add_argument("--quiet", action="store_true")
    index_cmd.set_defaults(func=cmd_index)

    doctor_cmd = sub.add_parser("doctor", help="is the index present, fresh and answering?")
    doctor_cmd.set_defaults(func=cmd_doctor)

    bm25_cmd = sub.add_parser("bm25", help="the stemmed BM25 lane (hybrid extra)")
    bm25_sub = bm25_cmd.add_subparsers(dest="bm25_command", required=True)
    bm25_build = bm25_sub.add_parser("build", help="index title+abstract with Porter stemming (~20 s)")
    bm25_build.set_defaults(func=cmd_bm25_build)

    vec_cmd = sub.add_parser("vec", help="the dense lanes: one vector per paper per model (hybrid extra)")
    vec_sub = vec_cmd.add_subparsers(dest="vec_command", required=True)
    vec_build = vec_sub.add_parser("build", help="embed the papers a lane lacks (incremental by citekey)")
    vec_build.add_argument("--model", required=True, help=f"one of: {', '.join(sorted(MODELS))}")
    vec_build.add_argument("--backend", choices=BACKENDS, default="hf",
                           help="hf (sentence-transformers) | ollama (the served GGUF) | hashed (tests)")
    vec_build.add_argument("--batch", type=int, default=32)
    vec_build.add_argument("--limit", type=int, default=None, help="only the first N papers (smoke)")
    vec_build.add_argument("--max-seq", type=int, default=512)
    vec_build.add_argument("--fp16", action="store_true")
    vec_build.add_argument("--bf16", action="store_true")
    vec_build.add_argument("--default", action="store_true", help="mark this lane as the default")
    vec_build.add_argument("--quiet", action="store_true")
    vec_build.set_defaults(func=cmd_vec_build)
    vec_default = vec_sub.add_parser("default", help="mark a built lane as the one hybrid searches use")
    vec_default.add_argument("model")
    vec_default.set_defaults(func=cmd_vec_default)
    vec_status = vec_sub.add_parser("status", help="which lanes exist, how many papers each covers")
    vec_status.set_defaults(func=cmd_vec_status)

    batch_cmd = sub.add_parser("batch", help="answer a JSON list of queries in one process (the sweep's ladder)")
    batch_cmd.add_argument("--file", metavar="PATH", default=None, help="JSON spec (default: stdin)")
    batch_cmd.add_argument("--mode", choices=[m for m in MODES if m != "fts"], default="hybrid")
    batch_cmd.add_argument("--dense", metavar="MODEL", default=None)
    batch_cmd.add_argument("-n", "--limit", type=int, default=50, help="per-query default limit")
    batch_cmd.add_argument("--max-queries", type=int, default=64)
    batch_cmd.set_defaults(func=cmd_batch)

    rerank_cmd = sub.add_parser("rerank", help="score a JSON pool of papers against one query, once")
    rerank_cmd.add_argument("--query", default=None)
    rerank_cmd.add_argument("--model", default=None, help=f"one of: {', '.join(sorted(RERANKERS))}")
    rerank_cmd.add_argument("--file", metavar="PATH", default=None, help="JSON pool (default: stdin)")
    rerank_cmd.add_argument("--max-pool", type=int, default=1000)
    rerank_cmd.set_defaults(func=cmd_rerank)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED
    except QueryError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    except (IndexUnavailable, LaneUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NO_INDEX
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
