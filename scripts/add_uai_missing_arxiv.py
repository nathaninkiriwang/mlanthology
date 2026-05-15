#!/usr/bin/env python3
"""Add UAI 1988, 1989, 1991 to legacy data from arXiv.

These three years were missing from data/legacy/uai-legacy.jsonl.gz.  AUAI
bulk-uploaded the proceedings to arXiv in April 2013; each year has an index
paper that lists every paper with its arXiv ID.  This script:

  1. Fetches the three index pages and parses out (arxiv_id, title) pairs.
  2. Pulls full metadata (title, authors, abstract) per arXiv ID via the
     export.arxiv.org API in small batches.
  3. Builds legacy-schema records and appends them to uai-legacy.jsonl.gz.

Usage::

    python scripts/add_uai_missing_arxiv.py
    python scripts/add_uai_missing_arxiv.py --dry-run
"""

import argparse
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from adapters.common import (
    make_bibtex_key,
    normalize_paper,
    parse_author_name,
    resolve_bibtex_collisions,
)
from scripts.enrich_uai_arxiv import _parse_index_page
from scripts.utils import LEGACY_DIR, make_session, read_legacy, write_legacy

logger = logging.getLogger(__name__)

# year -> arXiv ID of the index paper that lists all papers for that year
YEAR_TO_INDEX = {
    "1988": "1304.3856",  # Fourth UAI, Minneapolis MN
    "1989": "1304.3855",  # Fifth UAI, Windsor ON
    "1991": "1304.3853",  # Seventh UAI, Los Angeles CA
}

ARXIV_HTML_BASE = "https://arxiv.org/html/"
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf/"
ARXIV_ABS_BASE = "https://arxiv.org/abs/"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}

VENUE_NAME = "Conference on Uncertainty in Artificial Intelligence"
UAI_LEGACY_FILE = LEGACY_DIR / "uai-legacy.jsonl.gz"

# arXiv asks for ≥3s between API requests
API_DELAY = 3.5
# Conservative batch size — keeps each API call cheap and avoids huge responses
API_BATCH = 25


def fetch_arxiv_ids(session, year: str, index_id: str) -> list[str]:
    """Return the list of arXiv IDs in the order they appear on the index page."""
    url = f"{ARXIV_HTML_BASE}{index_id}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    pairs = _parse_index_page(resp.text)
    logger.info("  %s: %d papers from index %s", year, len(pairs), index_id)
    return [arxiv_id for arxiv_id, _title in pairs]


def fetch_arxiv_metadata(session, arxiv_ids: list[str]) -> dict[str, dict]:
    """Fetch title/authors/abstract for each arXiv ID via the export API.

    Returns {arxiv_id: {title, authors, abstract}}.
    """
    result: dict[str, dict] = {}
    for batch_start in range(0, len(arxiv_ids), API_BATCH):
        batch = arxiv_ids[batch_start : batch_start + API_BATCH]
        url = f"{ARXIV_API}?id_list={','.join(batch)}&max_results={len(batch)}"
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for entry in root.findall("a:entry", ATOM_NS):
            id_url = (entry.find("a:id", ATOM_NS).text or "").strip()
            # http://arxiv.org/abs/1304.2339v1 -> 1304.2339
            arxiv_id = id_url.rsplit("/", 1)[-1].split("v")[0]
            title = (entry.find("a:title", ATOM_NS).text or "").strip()
            summary = (entry.find("a:summary", ATOM_NS).text or "").strip()
            authors = [
                (a.find("a:name", ATOM_NS).text or "").strip()
                for a in entry.findall("a:author", ATOM_NS)
            ]
            result[arxiv_id] = {
                "title": " ".join(title.split()),
                "authors": [a for a in authors if a],
                "abstract": " ".join(summary.split()),
            }
        logger.info(
            "  fetched metadata %d/%d",
            min(batch_start + API_BATCH, len(arxiv_ids)),
            len(arxiv_ids),
        )
        if batch_start + API_BATCH < len(arxiv_ids):
            time.sleep(API_DELAY)
    return result


def build_paper(arxiv_id: str, meta: dict, year: str) -> dict:
    """Build one legacy-schema paper dict from arXiv metadata."""
    authors = [parse_author_name(name) for name in meta["authors"]]
    first_family = authors[0]["family"] if authors else "anon"
    return {
        "bibtex_key": make_bibtex_key(first_family, year, "uai", meta["title"]),
        "title": meta["title"],
        "authors": authors,
        "year": year,
        "venue": "uai",
        "venue_name": VENUE_NAME,
        "venue_type": "conference",
        "volume": "",
        "pages": "",
        "abstract": meta["abstract"],
        "pdf_url": f"{ARXIV_PDF_BASE}{arxiv_id}",
        "venue_url": f"{ARXIV_ABS_BASE}{arxiv_id}",
        "doi": "",
        "openreview_url": "",
        "code_url": "",
        "source": "arxiv",
        "source_id": f"arXiv:{arxiv_id}",
    }


def add_missing(*, dry_run: bool = False) -> None:
    existing = read_legacy(UAI_LEGACY_FILE)
    logger.info("Loaded %d existing UAI papers from %s", len(existing), UAI_LEGACY_FILE.name)
    existing_keys = {p["bibtex_key"] for p in existing}

    session = make_session(retries=3, backoff_factor=2.0)
    session.headers.update({"User-Agent": "mlanthology-arxiv/1.0 (research bot)"})

    all_new: list[dict] = []
    for i, (year, index_id) in enumerate(YEAR_TO_INDEX.items()):
        logger.info("=== UAI %s ===", year)
        arxiv_ids = fetch_arxiv_ids(session, year, index_id)
        if i < len(YEAR_TO_INDEX) - 1:
            time.sleep(API_DELAY)
        meta = fetch_arxiv_metadata(session, arxiv_ids)
        if len(meta) != len(arxiv_ids):
            missing = [a for a in arxiv_ids if a not in meta]
            logger.warning("  %s: missing metadata for %d IDs: %s", year, len(missing), missing)

        # Preserve index ordering
        year_papers = [build_paper(aid, meta[aid], year) for aid in arxiv_ids if aid in meta]
        # Resolve bibtex collisions inside this year's batch
        keys = resolve_bibtex_collisions([p["bibtex_key"] for p in year_papers])
        for paper, key in zip(year_papers, keys):
            paper["bibtex_key"] = key
        all_new.extend(year_papers)
        logger.info("  %s: built %d papers", year, len(year_papers))
        time.sleep(API_DELAY)

    # Filter out any collision with already-stored bibtex_keys (shouldn't happen for missing years)
    collisions = [p for p in all_new if p["bibtex_key"] in existing_keys]
    if collisions:
        logger.warning("  %d bibtex_key collisions with existing papers — suffixing", len(collisions))
        for p in collisions:
            base = p["bibtex_key"]
            n = 1
            while f"{base}-x{n}" in existing_keys:
                n += 1
            p["bibtex_key"] = f"{base}-x{n}"

    normalized = [normalize_paper(p) for p in all_new]
    logger.info("Total new papers built: %d", len(normalized))

    if dry_run:
        logger.info("Dry-run — not writing file. Sample record:")
        if normalized:
            import json
            sample = {k: v for k, v in normalized[0].items() if k != "abstract"}
            sample["abstract"] = normalized[0]["abstract"][:120] + "..."
            logger.info(json.dumps(sample, indent=2))
        return

    combined = existing + normalized
    write_legacy(UAI_LEGACY_FILE, combined, atomic=True)
    logger.info("Wrote %d total papers to %s", len(combined), UAI_LEGACY_FILE.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't write the file")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    add_missing(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
