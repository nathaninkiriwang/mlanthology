"""Import NeurIPS accepted papers from the conference's virtual-site data file.

Used when the year's proceedings are not yet on proceedings.neurips.cc and OpenReview answers
automated clients with ChallengeRequiredError (2026-11-18). The file carries titles, authors and
OpenReview/PDF links but no abstracts, so records carry an empty abstract; the proceedings fetch
replaces this file once the year is published.

    python -m scripts.import_neurips_virtual --year 2026 \
        --source https://neurips.cc/static/virtual/data/neurips-2026-orals-posters.json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from adapters.common import (
    make_bibtex_key,
    normalize_paper,
    parse_author_name,
    resolve_bibtex_collisions,
    write_venue_json,
)

VENUE_NAME = "Advances in Neural Information Processing Systems"


def load(source: str) -> list[dict]:
    if source.startswith("http"):
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    return data.get("results", data) if isinstance(data, dict) else data


def openreview_id(url: str) -> str:
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url or "")
    return m.group(1) if m else ""


def convert(items: list[dict], year: str) -> list[dict]:
    seen: dict[str, dict] = {}
    for item in items:
        if not str(item.get("decision") or "").lower().startswith("accept"):
            continue
        key = openreview_id(item.get("paper_url") or "") or str(item.get("uid") or item.get("id"))
        # posters and orals list the same paper; keep one record per paper, preferring the poster
        if key in seen and str(seen[key].get("eventtype")) == "Poster":
            continue
        seen[key] = item
    papers = []
    for key, item in seen.items():
        authors = [parse_author_name(str(a.get("fullname") or "")) for a in item.get("authors") or []]
        authors = [a for a in authors if a.get("given") or a.get("family")]
        title = str(item.get("name") or "").strip()
        if not title or not authors:
            continue
        papers.append(
            {
                "bibtex_key": make_bibtex_key(authors[0].get("family") or "", year, "neurips", title),
                "title": title,
                "authors": authors,
                "year": year,
                "venue": "neurips",
                "venue_name": VENUE_NAME,
                "abstract": "",
                "pdf_url": item.get("paper_pdf_url") or "",
                "venue_url": item.get("paper_url") or "",
                "openreview_url": item.get("paper_url") or "",
                "source": "neurips-virtual",
                "source_id": key,
            }
        )
    keys = resolve_bibtex_collisions([p["bibtex_key"] for p in papers])
    for paper, k in zip(papers, keys):
        paper["bibtex_key"] = k
    return [normalize_paper(p) for p in papers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", required=True)
    parser.add_argument("--source", required=True, help="URL or path of the virtual-site JSON")
    parser.add_argument("--output", default="data/papers")
    args = parser.parse_args()
    papers = convert(load(args.source), args.year)
    out = write_venue_json("neurips", args.year, papers, Path(args.output))
    print(f"wrote {len(papers)} NeurIPS {args.year} papers to {out}")


if __name__ == "__main__":
    main()
