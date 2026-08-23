"""Reading the ML Anthology data files into one normalized record stream.

Two on-disk shapes exist upstream and both are read here:

  ``data/papers/<venue>-<year>.json.gz``  — ``{"venue":…, "year":…, "papers":[record,…]}``
  ``data/legacy/<venue>-legacy.jsonl.gz`` — one record per line

The record schema is identical between them (verified across all 217,351 rows), so the
only difference is the container. Records carry no citation counts, which is why ranking
downstream is lexical: this corpus is a recall instrument, not an impact one.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PAPERS_GLOB = "papers/*.json.gz"
LEGACY_GLOB = "legacy/*.jsonl.gz"

# Fields the upstream schema guarantees on every row.
_TEXT_FIELDS = (
    "bibtex_key", "title", "year", "venue", "venue_type", "volume", "pages",
    "abstract", "pdf_url", "venue_url", "doi", "openreview_url", "code_url",
    "source", "source_id",
)


class CorpusError(RuntimeError):
    """The data directory is missing or unreadable — fail closed, never search a stub."""


@dataclass(frozen=True)
class Paper:
    citekey: str
    title: str
    authors: tuple[str, ...]          # "Given Family", display order
    author_slugs: tuple[str, ...]     # "family-given", the upstream slug
    year: int | None
    venue: str
    venue_name: str
    venue_type: str
    abstract: str
    doi: str
    pdf_url: str
    venue_url: str
    openreview_url: str
    code_url: str
    source: str
    volume: str
    pages: str
    number: str


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _year(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def normalize(raw: dict) -> Paper | None:
    """One upstream record -> one Paper. Returns None for a row with no citekey or title."""
    citekey = _clean(raw.get("bibtex_key"))
    title = _clean(raw.get("title"))
    if not citekey or not title:
        return None

    names: list[str] = []
    slugs: list[str] = []
    for entry in raw.get("authors") or []:
        if not isinstance(entry, dict):
            # A few legacy rows carry bare strings rather than {given, family}.
            if name := _clean(entry):
                names.append(name)
            continue
        given, family = _clean(entry.get("given")), _clean(entry.get("family"))
        if full := " ".join(part for part in (given, family) if part):
            names.append(full)
        if slug := _clean(entry.get("slug")):
            slugs.append(slug)

    return Paper(
        citekey=citekey,
        title=title,
        authors=tuple(names),
        author_slugs=tuple(slugs),
        year=_year(raw.get("year")),
        venue=_clean(raw.get("venue")).lower(),
        venue_name=_clean(raw.get("venue_name")),
        venue_type=_clean(raw.get("venue_type")),
        abstract=_clean(raw.get("abstract")),
        doi=_clean(raw.get("doi")),
        pdf_url=_clean(raw.get("pdf_url")),
        venue_url=_clean(raw.get("venue_url")),
        openreview_url=_clean(raw.get("openreview_url")),
        code_url=_clean(raw.get("code_url")),
        source=_clean(raw.get("source")),
        volume=_clean(raw.get("volume")),
        pages=_clean(raw.get("pages")),
        number=_clean(raw.get("number")),
    )


def source_files(data_dir: Path) -> list[Path]:
    """Every corpus file, in a stable order so a rebuild is byte-comparable."""
    if not data_dir.is_dir():
        raise CorpusError(f"MLA-CORPUS: no data directory at {data_dir}")
    files = sorted(data_dir.glob(PAPERS_GLOB)) + sorted(data_dir.glob(LEGACY_GLOB))
    if not files:
        raise CorpusError(
            f"MLA-CORPUS: {data_dir} holds no {PAPERS_GLOB} or {LEGACY_GLOB} files"
        )
    return files


def corpus_sha(files: list[Path]) -> str:
    """A single digest over the exact bytes searched.

    Recorded in the index and echoed on every result set, so a downstream ledger can pin
    which corpus snapshot answered a query — a search whose corpus cannot be named is not
    reproducible evidence.
    """
    rolling = hashlib.sha256()
    for path in files:
        rolling.update(path.name.encode("utf-8"))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        rolling.update(digest.digest())
    return rolling.hexdigest()


def _read_file(path: Path) -> Iterator[dict]:
    if path.match(PAPERS_GLOB) or path.parent.name == "papers":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from (payload.get("papers") or []) if isinstance(payload, dict) else payload
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line := line.strip():
                yield json.loads(line)


def read(data_dir: Path) -> Iterator[Paper]:
    """Stream every paper in the corpus, `data/papers/` first.

    Order matters on exactly one point: `papers/` (2013+, near-total abstract coverage)
    is loaded ahead of `legacy/`, so if the two ever disagree on a citekey the richer
    row wins. Upstream currently has zero overlap between them; this keeps that a
    property of the data rather than an assumption of the loader.
    """
    for path in source_files(data_dir):
        for raw in _read_file(path):
            if isinstance(raw, dict) and (paper := normalize(raw)) is not None:
                yield paper
