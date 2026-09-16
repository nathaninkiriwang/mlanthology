"""Stratified known-item sample of the corpus for the retrieval evaluation (D-106).

Deterministic (seed 1063). Cells: family × era × abstract/title-only, plus the lab-field slice.
Writes eval/papers.jsonl — one line per sampled paper with its strata tags.
"""
import json, random, sqlite3, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "index" / "mla.sqlite3"
SEED = 1063
PER_CELL = 6
LAB_N = 60
FAMILY_OF = {
    **{v: "core-ml" for v in "neurips icml iclr aistats tmlr jmlr mlj neco dmlr acml ecml ecmlpkdd cpal collas automl distill mloss neuripsw icmlw iclrw".split()},
    **{v: "theory" for v in "colt alt uai isipta pgm ftml".split()},
    **{v: "vision" for v in "cvpr iccv eccv wacv cvprw iccvw eccvw".split()},
    **{v: "ai" for v in "aaai ijcai jair".split()},
    **{v: "nlp" for v in "acl emnlp naacl eacl aacl coling conll lrec ijcnlp anlp hlt tacl cl findings-acl findings-emnlp findings-naacl findings-eacl findings-aacl findings-ijcnlp".split()},
}
def family(venue: str) -> str:
    v = venue.lower()
    if v in FAMILY_OF: return FAMILY_OF[v]
    for k, f in FAMILY_OF.items():
        if v.startswith(k): return f
    return "other"
def era(year):
    if year is None: return "unknown"
    return "old" if year <= 2005 else "mid" if year <= 2017 else "recent"

LAB_MATCH = ('eigenvector OR eigenvectors OR "davis kahan" OR "spectral norm" OR "leave one out" OR "leave-one-out" '
             'OR entrywise OR "entry-wise" OR elementwise OR "sup norm" OR "infinity norm" OR "matrix perturbation" '
             'OR "singular vector" OR "principal component" OR "spectral method" OR "random matrix"')

def main():
    rng = random.Random(SEED)
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    rows = con.execute("select rowid, citekey, title, abstract, year, venue, venue_type, authors from papers").fetchall()
    cells: dict[tuple, list] = {}
    for r in rows:
        key = (family(r["venue"]), era(r["year"]), "abstract" if r["abstract"] else "title-only")
        cells.setdefault(key, []).append(r)
    lab_ids = {r[0] for r in con.execute(f"select rowid from papers_fts where papers_fts match ?", (LAB_MATCH,))}
    picked, seen = [], set()
    def take(r, tags):
        if r["citekey"] in seen: return
        seen.add(r["citekey"])
        picked.append({"citekey": r["citekey"], "title": r["title"], "abstract": r["abstract"], "year": r["year"],
                       "venue": r["venue"], "venue_type": r["venue_type"], "authors": r["authors"], **tags})
    for key in sorted(cells):
        fam, er, ab = key
        if er == "unknown": continue
        pool = sorted(cells[key], key=lambda r: r["citekey"]); rng.shuffle(pool)
        for r in pool[:PER_CELL]:
            take(r, {"family": fam, "era": er, "text": ab, "slice": "generic", "lab": r["rowid"] in lab_ids})
    lab_rows = sorted((r for r in rows if r["rowid"] in lab_ids and r["abstract"]), key=lambda r: r["citekey"]); rng.shuffle(lab_rows)
    entry = [r for r in lab_rows if any(w in (r["title"] + " " + r["abstract"]).lower() for w in ("entrywise", "entry-wise", "elementwise", "sup norm", "infinity norm", "ℓ∞", "linf"))]
    for r in entry[:20]:
        take(r, {"family": family(r["venue"]), "era": era(r["year"]), "text": "abstract", "slice": "lab-entrywise", "lab": True})
    for r in lab_rows:
        if sum(1 for p in picked if p["slice"].startswith("lab")) >= LAB_N: break
        take(r, {"family": family(r["venue"]), "era": era(r["year"]), "text": "abstract", "slice": "lab", "lab": True})
    out = HERE / "papers.jsonl"
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picked), encoding="utf-8")
    from collections import Counter
    print(f"sampled {len(picked)} papers -> {out}")
    print("by slice:", dict(Counter(p["slice"] for p in picked)))
    print("by family:", dict(Counter(p["family"] for p in picked)))
    print("by era:", dict(Counter(p["era"] for p in picked)), "| title-only:", sum(p["text"] == "title-only" for p in picked))
if __name__ == "__main__":
    main()
