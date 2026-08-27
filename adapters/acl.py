"""ACL Anthology adapter.

The ACL Anthology publishes its canonical metadata as XML in a public git
repository — one file per *collection* (e.g. ``2024.acl.xml``, ``P19.xml``).
This adapter imports the MAIN-conference and Findings volumes and skips
workshops, tutorials and shared tasks.

Two upstream naming schemes coexist and both are handled:

* modern  ``YYYY.<venue>``  with volumes named ``long`` / ``short`` / ``main``
* legacy  ``<Letter><YY>``  with volumes numbered ``1`` / ``2`` / ``3``

Workshop content CANNOT be identified from the volume id alone.  EMNLP 2019
(``D19``) files sixteen co-located workshops as numeric volumes 50-66, which
are indistinguishable by id from the main proceedings in volume 1.  The
volume ``<booktitle>`` is therefore the discriminator, not the id.

Acquisition is a blobless sparse clone of ``data/xml`` (~180 MB, one network
round trip) rather than ~280 individual HTTP fetches.  The clone yields an
atomic snapshot and a commit sha, so an import is reproducible: the sha names
the exact bytes that were parsed.
"""

import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .cache import mark_fetched, should_fetch
from .common import (
    make_bibtex_key,
    normalize_paper,
    resolve_bibtex_collisions,
    write_venue_json,
)

logger = logging.getLogger(__name__)

ANTHOLOGY_REPO = "https://github.com/acl-org/acl-anthology.git"
ANTHOLOGY_BRANCH = "master"
BASE_URL = "https://aclanthology.org"

# modern collection suffix -> canonical venue slug
MODERN_VENUES = {
    "acl": "acl",
    "emnlp": "emnlp",
    "naacl": "naacl",
    "eacl": "eacl",
    "aacl": "aacl",
    "coling": "coling",
    "lrec-coling": "lrec-coling",
    "conll": "conll",
    "lrec": "lrec",
    "ijcnlp": "ijcnlp",
    "anlp": "anlp",
    "hlt": "hlt",
    "tacl": "tacl",
    "cl": "cl",
    "findings": "findings",  # volume id names the host conference
}

# legacy single-letter collection prefix -> canonical venue slug
LEGACY_VENUES = {
    "P": "acl",
    "D": "emnlp",
    "N": "naacl",
    "E": "eacl",
    "C": "coling",
    "K": "conll",
    "J": "cl",
    "Q": "tacl",
    "L": "lrec",
    "I": "ijcnlp",
    "A": "anlp",
    "H": "hlt",
}

# human-readable names, also used to seed hugo/data/venues.yaml
VENUE_NAMES = {
    "acl": ("ACL", "Annual Meeting of the Association for Computational Linguistics"),
    "emnlp": ("EMNLP", "Conference on Empirical Methods in Natural Language Processing"),
    "naacl": ("NAACL", "North American Chapter of the Association for Computational Linguistics"),
    "eacl": ("EACL", "European Chapter of the Association for Computational Linguistics"),
    "aacl": ("AACL", "Asia-Pacific Chapter of the Association for Computational Linguistics"),
    "coling": ("COLING", "International Conference on Computational Linguistics"),
    "lrec-coling": ("LREC-COLING", "Joint International Conference on Computational Linguistics, Language Resources and Evaluation"),
    "conll": ("CoNLL", "Conference on Computational Natural Language Learning"),
    "lrec": ("LREC", "International Conference on Language Resources and Evaluation"),
    "ijcnlp": ("IJCNLP", "International Joint Conference on Natural Language Processing"),
    "anlp": ("ANLP", "Conference on Applied Natural Language Processing"),
    "hlt": ("HLT", "Human Language Technology Conference"),
    "tacl": ("TACL", "Transactions of the Association for Computational Linguistics"),
    "cl": ("CL", "Computational Linguistics"),
    "findings-acl": ("Findings of ACL", "Findings of the Association for Computational Linguistics: ACL"),
    "findings-emnlp": ("Findings of EMNLP", "Findings of the Association for Computational Linguistics: EMNLP"),
    "findings-naacl": ("Findings of NAACL", "Findings of the Association for Computational Linguistics: NAACL"),
    "findings-eacl": ("Findings of EACL", "Findings of the Association for Computational Linguistics: EACL"),
    "findings-ijcnlp": ("Findings of IJCNLP", "Findings of the Association for Computational Linguistics: IJCNLP"),
    "findings-aacl": ("Findings of AACL", "Findings of the Association for Computational Linguistics: AACL"),
}

JOURNAL_VENUES = frozenset({"cl", "tacl"})

# A volume whose booktitle matches this is not main-conference content.
# Matched against the booktitle because ids are unreliable (see module docstring).
SKIP_VOLUME_RE = re.compile(r"workshop|tutorial|shared task", re.I)

_LEGACY_RE = re.compile(r"^([A-Z])(\d{2})$")
_MODERN_RE = re.compile(r"^(\d{4})\.(.+)$")


def _flatten(elem: Optional[ET.Element]) -> str:
    """Flatten mixed-content markup to plain text.

    Anthology titles and abstracts carry inline markup — ``<fixed-case>``,
    ``<tex-math>``, ``<b>``, ``<i>``, ``<url>`` — whose text is part of the
    string.  ``.text`` alone would truncate at the first child element.
    """
    if elem is None:
        return ""
    return " ".join("".join(elem.itertext()).split())


def _legacy_year(two_digit: str) -> str:
    """Expand a legacy two-digit year. 79 -> 1979, 00 -> 2000."""
    n = int(two_digit)
    return str(1900 + n) if n >= 60 else str(2000 + n)


def collection_venue(stem: str) -> Optional[str]:
    """Return the base venue slug for a collection file stem, or None to skip."""
    m = _MODERN_RE.match(stem)
    if m:
        return MODERN_VENUES.get(m.group(2))
    m = _LEGACY_RE.match(stem)
    if m:
        return LEGACY_VENUES.get(m.group(1))
    return None


def sync_repo(workdir: Path) -> tuple[Path, str]:
    """Clone or update the Anthology repo. Returns (xml_dir, commit_sha)."""
    workdir = Path(workdir)
    git = ["git", "-C", str(workdir)]
    if (workdir / ".git").exists():
        logger.info(f"Updating ACL Anthology clone at {workdir}")
        subprocess.run(git + ["fetch", "--depth", "1", "origin", ANTHOLOGY_BRANCH],
                       check=True, capture_output=True)
        subprocess.run(git + ["checkout", "-f", "FETCH_HEAD"],
                       check=True, capture_output=True)
    else:
        logger.info(f"Cloning ACL Anthology into {workdir} (blobless, sparse)")
        workdir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "--branch", ANTHOLOGY_BRANCH, ANTHOLOGY_REPO, str(workdir)],
            check=True, capture_output=True,
        )
        subprocess.run(git + ["sparse-checkout", "set", "data/xml"],
                       check=True, capture_output=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    return workdir / "data" / "xml", sha


def _parse_paper(paper: ET.Element, venue: str, year: str, collection: str,
                 volume_id: str, booktitle: str) -> Optional[dict]:
    """Convert one <paper> element to a pre-normalization paper dict."""
    title = _flatten(paper.find("title"))
    if not title:
        return None

    authors = []
    for a in paper.findall("author"):
        given = _flatten(a.find("first"))
        family = _flatten(a.find("last"))
        if given or family:
            authors.append({"given": given, "family": family})
    if not authors:
        return None

    # <url> carries the canonical Anthology id for both naming schemes.
    raw_url = _flatten(paper.find("url"))
    if raw_url and not raw_url.startswith("http"):
        anthology_id = raw_url
    elif _LEGACY_RE.match(collection):
        anthology_id = f"{collection}-{paper.get('id', '')}"
    else:
        anthology_id = f"{collection}-{volume_id}.{paper.get('id', '')}"

    short, long_name = VENUE_NAMES.get(venue, (venue.upper(), booktitle))
    return {
        "bibtex_key": make_bibtex_key(authors[0]["family"], year, venue, title),
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "venue_name": f"{long_name} {year}" if venue not in JOURNAL_VENUES else long_name,
        "venue_type": "journal" if venue in JOURNAL_VENUES else "conference",
        "volume": volume_id,
        "pages": _flatten(paper.find("pages")),
        "abstract": _flatten(paper.find("abstract")),
        "pdf_url": f"{BASE_URL}/{anthology_id}.pdf" if anthology_id else "",
        "venue_url": f"{BASE_URL}/{anthology_id}/" if anthology_id else "",
        "doi": _flatten(paper.find("doi")),
        "source": "acl-anthology",
        "source_id": anthology_id,
    }


def parse_collection(path: Path, base_venue: str) -> tuple[dict, dict]:
    """Parse one collection file.

    Returns ({(venue, year): [papers]}, {"kept": n, "skipped": n}).
    """
    stem = path.stem
    root = ET.parse(path).getroot()
    out: dict[tuple[str, str], list] = defaultdict(list)
    stats = {"kept": 0, "skipped": 0}

    for volume in root.findall("volume"):
        booktitle = _flatten(volume.find("meta/booktitle"))
        if SKIP_VOLUME_RE.search(booktitle):
            stats["skipped"] += len(volume.findall("paper"))
            continue

        volume_id = volume.get("id", "")
        year = _flatten(volume.find("meta/year"))
        if not year:
            m = _MODERN_RE.match(stem)
            year = m.group(1) if m else _legacy_year(_LEGACY_RE.match(stem).group(2))

        # In a findings collection each volume is a different host conference.
        venue = f"findings-{volume_id}" if base_venue == "findings" else base_venue

        for paper in volume.findall("paper"):
            rec = _parse_paper(paper, venue, year, stem, volume_id, booktitle)
            if rec:
                out[(venue, year)].append(rec)
                stats["kept"] += 1

    return out, stats


def _quick_subset(stems: list[str]) -> list[str]:
    """Latest collection per modern venue — used by --quick smoke runs."""
    best: dict[str, tuple[str, str]] = {}
    for stem in stems:
        m = _MODERN_RE.match(stem)
        if not m:
            continue
        year, suffix = m.group(1), m.group(2)
        if suffix in MODERN_VENUES and (suffix not in best or year > best[suffix][0]):
            best[suffix] = (year, stem)
    return sorted(stem for _, stem in best.values())


def fetch_all(
    output_dir: Optional[Path] = None,
    cache: Optional[dict] = None,
    workdir: Optional[Path] = None,
    collections: Optional[list[str]] = None,
    quick: bool = False,
) -> dict[str, list[dict]]:
    """Import ACL Anthology main + Findings papers into the canonical schema."""
    if output_dir is None:
        output_dir = Path("data/papers")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if workdir is None:
        workdir = output_dir.parent / ".acl-anthology"

    xml_dir, sha = sync_repo(workdir)
    logger.info(f"ACL Anthology at commit {sha[:12]}")

    stems = sorted(p.stem for p in xml_dir.glob("*.xml"))
    if collections:
        wanted = set(collections)
        stems = [s for s in stems if s in wanted]
    elif quick:
        stems = _quick_subset(stems)
        logger.info(f"Quick mode: {len(stems)} latest collections")

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    totals = {"kept": 0, "skipped": 0}
    for stem in stems:
        base_venue = collection_venue(stem)
        if base_venue is None:
            continue
        try:
            parsed, stats = parse_collection(xml_dir / f"{stem}.xml", base_venue)
        except ET.ParseError as exc:
            logger.warning(f"  Skipping malformed {stem}.xml: {exc}")
            continue
        for key, papers in parsed.items():
            grouped[key].extend(papers)
        totals["kept"] += stats["kept"]
        totals["skipped"] += stats["skipped"]

    logger.info(
        f"Parsed {totals['kept']:,} main/Findings papers "
        f"({totals['skipped']:,} workshop/tutorial/shared-task papers skipped)"
    )

    all_papers = {}
    for (venue, year), papers in sorted(grouped.items()):
        cache_key = f"aclanth-{venue}-{year}"
        if cache is not None and not should_fetch(cache, cache_key, year):
            logger.info(f"Skipping {venue} {year} — cached")
            continue

        keys = resolve_bibtex_collisions([p["bibtex_key"] for p in papers])
        for paper, key in zip(papers, keys):
            paper["bibtex_key"] = key

        write_venue_json(venue, year, [normalize_paper(p) for p in papers], output_dir)
        all_papers[f"{venue}-{year}"] = papers
        if cache is not None:
            mark_fetched(cache, cache_key)

    return all_papers
