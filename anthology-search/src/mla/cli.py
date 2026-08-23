"""``mla`` — the command line surface.

Built for a harness that shells out, parses JSON, and writes down what it got: every
machine command prints one JSON object to stdout and nothing else, diagnostics go to
stderr, and the exit code says which kind of thing went wrong.

    0  ok
    2  the caller asked for something malformed (bad flag, bad limit, empty query)
    3  the caller asked for something refused (unknown venue, unknown venue type)
    4  the index is missing or unreadable — build it with `mla index`

An unknown venue is a REFUSAL, not an empty result. `--venue nips` is a typo for
`neurips`, and a sweep that silently returns zero for a typo teaches the searcher the
wrong lesson about the field.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _filters(args, connection) -> Filters:
    types = tuple(args.type or ())
    if unknown := [t for t in types if t not in VENUE_TYPES]:
        raise Refused(f"MLA-TYPE: unknown venue type(s) {', '.join(unknown)}. "
                      f"Known: {', '.join(VENUE_TYPES)}.")
    if args.no_workshops:
        types = tuple(t for t in VENUE_TYPES if t != "workshop") if not types else \
            tuple(t for t in types if t != "workshop")
    year_min, year_max = args.since, args.until
    if args.year is not None:
        year_min = year_max = args.year
    if year_min is not None and year_max is not None and year_min > year_max:
        raise QueryError(f"MLA-YEAR: --since {year_min} is after --until {year_max}")
    # A hand-named --venue is a claim that venue exists, so a typo is refused. A --family
    # is a claim about a topic: members this index happens not to carry are skipped and
    # reported, because losing ISIPTA should not cancel a theory sweep.
    requested = _check_venues(connection, tuple(args.venue or ()))
    absent: list[str] = []
    if args.family:
        expanded, unknown = expand_families(tuple(args.family))
        if unknown:
            raise Refused(f"MLA-FAMILY: unknown family/families {', '.join(unknown)}. "
                          f"Known: {', '.join(sorted(FAMILIES))}.")
        known = _known_venues(connection)
        absent = [v for v in expanded if v not in known]
        requested = tuple(requested) + tuple(
            v for v in expanded if v in known and v not in requested)
        if not requested:
            raise Refused(
                f"MLA-FAMILY: this index carries none of {', '.join(args.family)} "
                f"({', '.join(expanded)}). Run `mla venues` to see what it holds.")
    args.family_absent = absent
    return Filters(
        venues=tuple(requested),
        year_min=year_min, year_max=year_max, venue_types=types,
        author=args.author or "", has_code=args.has_code, has_pdf=args.has_pdf,
    )


def cmd_search(args) -> int:
    connection = connect(args.index)
    filters = _filters(args, connection)
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
    if url := (paper.get("anthology_url") or paper.get("url")):
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
    )
    _emit(report)
    return EXIT_OK if report["ok"] else EXIT_NO_INDEX


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
    except IndexUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_NO_INDEX
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
