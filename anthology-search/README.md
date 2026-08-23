# anthology-search — `mla`

Offline, venue-complete ML literature search over the [ML Anthology](https://mlanthology.org)
corpus, built for an agentic research harness.

The upstream project is a Hugo site whose search is Pagefind — client-side, no API. The
valuable part for a machine is `data/`: **217,346 papers, 44 venues, 1969–2026, ~99% with
abstracts**, in one uniform schema. This subproject compiles that into a SQLite FTS5 index
and puts three surfaces on it.

```bash
uv run mla index                       # ~15 s, ~640 MB, gitignored and rebuildable
uv run mla search "entrywise eigenvector perturbation" --family theory --since 2015
```

## Why local

| | external providers | `mla` |
|---|---|---|
| rate limits | 429s throttle a sweep | none — it never leaves the machine |
| latency | 0.5–5 s per call | 1–15 ms |
| coverage claim | ranked; a zero is a cutoff artifact | complete within a venue-year, so a zero is **evidence** |
| reproducibility | the index moves under you | every response names a `corpus_sha` over the exact bytes searched |
| ranking | citations, recency, vendor secret sauce | BM25 + a title-exactness boost, and nothing hidden |

What it does **not** hold: arXiv preprints, ACL/EMNLP, KDD/WWW/SIGIR/ICASSP, and **citation
counts**. It cannot rank by impact and cannot see the unpublished frontier. It is a recall
instrument for the ML proceedings — pair it with a citation-graph provider, don't replace one.

## Surfaces

**CLI (primary).** JSON on stdout, diagnostics on stderr, meaning in the exit code —
`0` ok · `2` malformed request · `3` refused (unknown venue/family) · `4` index missing.

```bash
mla search "<query>" [-n N] [--venue V]... [--family F]... [--year Y | --since Y --until Y]
                     [--type conference|journal|workshop] [--no-workshops] [--author NAME]
                     [--has-code] [--has-pdf] [--any] [--field title,abstract] [--abstracts]
mla get <citekey>            # one paper
mla doi <doi>                # resolve a DOI against the corpus (a cheap dedup probe)
mla bibtex <citekey>...      # BibTeX with canonical venue names
mla venues [--text] [--family F]
mla stats                    # provenance + field coverage
mla index [--force]          # build or refresh
mla doctor                   # present, fresh, and answering?
```

**MCP (secondary).** `uv run --extra mcp mla-mcp` serves `mla_search`, `mla_get`,
`mla_resolve_doi`, `mla_venues` over stdio, for seats that mount MCP servers and have no
shell. The CLI is primary because a harness that records what it searched needs a
subprocess boundary it can log and an exit code it can branch on.

**Library.** `from mla.query import search` against `mla.index.connect()`.

## Search behaviour worth knowing

- **No stemming, deliberately.** `learning` does not match `learnable`. A stemmer would
  quietly widen a query past what was recorded. Widen explicitly instead: `eigen*`,
  `--any`, `"quoted phrase"`.
- **Hyphens and diacritics fold on their own.** `entry-wise` = `entrywise`; `--author
  scholkopf` finds Schölkopf; `lukasz` finds Łukasz.
- **A typo'd venue is refused, not answered empty** (exit 3, with the known list). A silent
  zero teaches you something false about the field.
- **Diagnostics come back with the results**: `total` (matches before your limit — the
  flooded-vs-thin signal), `dropped_terms` (terms the tokenizer could not index: `ℓ∞`
  becomes nothing, spell it `linf`), `unfiltered_total` (present only when your *filters*
  produced the zero, so you widen the box rather than reword), and `family_venues_absent`.
- **Families** save you remembering who publishes where: `core-ml`, `theory`, `vision`,
  `ai`, `robotics`, `health`, `graphs`, `causal`. `--family theory` sweeps COLT, ALT, UAI,
  ISIPTA, PGM and FTML in one call.

## Staying current

Upstream refreshes `data/` quarterly via a scheduled Action that commits back, so:

```bash
make refresh     # git fetch upstream && merge --ff-only && mla index
```

`mla doctor` reports `stale: true` whenever `data/` has moved since the index was built.

## Using it from a harness

The CLI is the surface built for this. It is safe to shell out to: it never touches the
network, it emits exactly one JSON object on stdout, it puts diagnostics on stderr, and
its exit code distinguishes *malformed request* (2) from *refused* (3) from *index
missing* (4) — so a caller can branch instead of parsing prose. Isolate it with
`uv run --directory <clone> mla …` and nothing crosses the boundary but JSON.

Two properties matter if you log what you searched: `corpus.sha` names the exact bytes
that answered, and hit keys are readable anthology citekeys rather than opaque ids.

## Layout

```
src/mla/
  corpus.py      reading data/papers/*.json.gz + data/legacy/*.jsonl.gz
  index.py       schema, build, corpus_sha, atomic install
  query.py       filters, BM25 + title boost, diagnostics
  text.py        folding and the FTS5 query compiler
  venues.py      canonical names and topical families
  cli.py         the primary surface
  mcp_server.py  the secondary surface
tests/           69 hermetic tests against a synthetic corpus
```

`make test` · `make index` · `make doctor` · `make venues` · `make help`
