"""Generate realistic scout queries per sampled paper, by style, with Claude through the harness
profile (D-106). Batches of 8 papers per call; validated; writes eval/queries.jsonl.

Styles: S1 keywords · S2 paraphrase (the paper's distinctive terms FORBIDDEN) · S3 problem-only ·
S4 adjacent-field vocabulary (theory / lab papers) · S5 exact rare term or notation · S7 a
two-sentence campaign brief. Title-only papers get S1/S2/S3 from the title alone (slice S6).
"""
import json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_HOME = "/Users/nathan/research/kernel"
SHIM = f"{KERNEL_HOME}/runtimes/claude-code/node_modules/.bin-wrapper/kclaude"
BATCH = 8

RULES = """You are simulating research scouts who have NOT read these papers and are searching a 286k-paper
ML/NLP corpus (titles + abstracts) for them. For EACH paper produce queries in these styles:
S1 keywords: 2-5 keywords a scout would type; the paper's own terms are allowed; NOT the title copied.
S2 paraphrase: ONE sentence describing the contribution using ONLY plain words and synonyms - none of the
   distinctive noun phrases, coined names, acronyms or notation from the title or abstract may appear.
S3 problem: ONE short sentence stating the problem or goal the paper addresses, with NO method names.
S4 adjacent (only when the paper is theoretical/mathematical/statistical): the same idea phrased the way a
   neighbouring field would say it (probability, statistics, econometrics, optimization, signal processing).
S5 exact: 2-4 tokens that include ONE rare exact term, notation, acronym, dataset or coined name that appears in
   the abstract (e.g. an estimator's name, "2->infinity norm", an acronym) - the kind of thing BM25 loves.
S7 brief: two sentences like a campaign brief ("We are looking for work that ... Ideally it ...").
For papers marked TITLE-ONLY (no abstract) produce S1, S2, S3 from the title alone; omit S4/S5/S7.
Return STRICT JSON: {"queries": [{"citekey": "...", "S1": "...", "S2": "...", "S3": "...", "S4": "..."|null,
"S5": "..."|null, "S7": "..."|null}, ...]} - one object per paper, nothing else."""

def call_claude(prompt: str) -> dict:
    env = {**os.environ, "KERNEL_HOME": KERNEL_HOME}
    with tempfile.TemporaryDirectory(prefix="mla-eval-") as scratch:
        proc = subprocess.run([SHIM, "-p", "--output-format", "json", prompt], capture_output=True, text=True,
                              env=env, cwd=scratch, stdin=subprocess.DEVNULL, timeout=600)
    out = proc.stdout
    payload = json.loads(out[out.index("{"):])
    text = payload.get("result", "")
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))

STOP = set("""a an the of for and or in on to with from by via into over under between within without
using based towards toward learning model models method methods approach approaches network networks neural deep
data analysis system systems problem problems task tasks paper study results new novel general efficient effective
improved improving improve large small fast robust simple unified adaptive scalable end-to-end towards multi single
via""".split())

def title_terms(title: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", title.lower()) if w not in STOP}

def validate(paper: dict, style: str, q: str) -> tuple[bool, str]:
    if not q or not isinstance(q, str): return False, "empty"
    q = q.strip(); ql = q.lower()
    title = paper["title"].lower()
    if style not in ("S1", "S5"):
        tw = title.split()
        for i in range(len(tw) - 3):
            if " ".join(tw[i:i + 4]) in ql: return False, "verbatim title span"
    if style == "S2":
        leaked = [w for w in title_terms(paper["title"]) if re.search(r"\b" + re.escape(w) + r"\b", ql)]
        if len(leaked) >= 2: return False, f"title terms leaked: {leaked[:3]}"
    n = len(q.split())
    if style == "S1":
        items = [s for s in re.split(r",|;", q) if s.strip()]
        if not (2 <= len(items) <= 6) or n > 16: return False, f"S1 shape: {len(items)} items, {n} words"
        return True, ""
    caps = {"S2": (6, 45), "S3": (4, 35), "S4": (4, 45), "S5": (2, 8), "S7": (12, 90)}
    lo, hi = caps[style]
    if not (lo <= n <= hi): return False, f"length {n} outside {lo}-{hi}"
    return True, ""

def main():
    papers = [json.loads(l) for l in (HERE / "papers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    out_path = HERE / "queries.jsonl"; log = HERE / "generate.log"
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip(): done.add(json.loads(l)["citekey"])
    todo = [p for p in papers if p["citekey"] not in done]
    print(f"{len(papers)} papers, {len(done)} already done, {len(todo)} to go", flush=True)
    rejected = 0; qid = sum(1 for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()) if out_path.exists() else 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        parts = []
        for p in batch:
            kind = "TITLE-ONLY" if not p["abstract"] else ("THEORETICAL" if (p["lab"] or p["family"] == "theory") else "paper")
            parts.append(f"### {p['citekey']} [{kind}] ({p['venue']} {p['year']})\nTITLE: {p['title']}\nABSTRACT: {p['abstract'][:2500] or '(none)'}")
        prompt = RULES + "\n\n" + "\n\n".join(parts)
        t0 = time.time()
        try:
            data = call_claude(prompt)
        except Exception as exc:
            log.open("a").write(f"batch {i}: FAILED {exc}\n"); print(f"batch {i}: failed {exc}", flush=True); continue
        by_key = {q.get("citekey"): q for q in data.get("queries", [])}
        with out_path.open("a", encoding="utf-8") as fh:
            for p in batch:
                q = by_key.get(p["citekey"]) or {}
                for style in ("S1", "S2", "S3", "S4", "S5", "S7"):
                    text = q.get(style)
                    if text is None: continue
                    ok, why = validate(p, style, text)
                    if not ok:
                        rejected += 1; log.open("a").write(f"{p['citekey']} {style} rejected: {why}: {text!r}\n"); continue
                    qid += 1
                    fh.write(json.dumps({"qid": f"q{qid:05d}", "citekey": p["citekey"], "style": style if p["abstract"] else f"S6-{style}",
                                         "query": text.strip(), "slice": p["slice"], "family": p["family"], "era": p["era"],
                                         "text": p["text"], "lab": p["lab"]}, ensure_ascii=False) + "\n")
        print(f"batch {i // BATCH + 1}: {len(batch)} papers in {time.time() - t0:.0f}s; rejected so far {rejected}", flush=True)
    print("done; queries:", qid, "rejected:", rejected)
if __name__ == "__main__":
    main()
