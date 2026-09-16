"""Top up the S2 (paraphrase) slice for abstract papers that lost theirs to the length cap (D-106)."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_queries import BATCH, call_claude, title_terms, HERE  # noqa: E402

RULES = """You are simulating research scouts who have NOT read these papers and are searching a 286k-paper
ML/NLP corpus (titles + abstracts) for them. For EACH paper write ONE paraphrase query (style S2): a single
sentence of at most 40 words describing the contribution using ONLY plain words and synonyms - none of the
distinctive noun phrases, coined names, acronyms or notation from the title or abstract may appear.
Return STRICT JSON: {"queries": [{"citekey": "...", "S2": "..."}, ...]} - one object per paper, nothing else."""

def main():
    papers = {json.loads(l)["citekey"]: json.loads(l) for l in (HERE / "papers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    out = HERE / "queries.jsonl"
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    have = {r["citekey"] for r in rows if r["style"] == "S2"}
    todo = [p for k, p in papers.items() if p["abstract"] and k not in have]
    qid = len(rows)
    print(f"{len(todo)} abstract papers lack an S2", flush=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        prompt = RULES + "\n\n" + "\n\n".join(f"### {p['citekey']}\nTITLE: {p['title']}\nABSTRACT: {p['abstract'][:2500]}" for p in batch)
        try:
            data = call_claude(prompt)
        except Exception as exc:
            print(f"batch {i}: failed {exc}", flush=True); continue
        by = {q.get("citekey"): q.get("S2") for q in data.get("queries", [])}
        with out.open("a", encoding="utf-8") as fh:
            for p in batch:
                q = (by.get(p["citekey"]) or "").strip(); ql = q.lower()
                if not q or not (6 <= len(q.split()) <= 70): continue
                tw = p["title"].lower().split()
                if any(" ".join(tw[j:j + 4]) in ql for j in range(len(tw) - 3)): continue
                if len([w for w in title_terms(p["title"]) if re.search(r"\b" + re.escape(w) + r"\b", ql)]) >= 2: continue
                qid += 1
                fh.write(json.dumps({"qid": f"q{qid:05d}", "citekey": p["citekey"], "style": "S2", "query": q, "slice": p["slice"], "family": p["family"], "era": p["era"], "text": p["text"], "lab": p["lab"]}, ensure_ascii=False) + "\n")
        print(f"batch {i // BATCH + 1} done", flush=True)
    print("topup done", flush=True)
if __name__ == "__main__":
    main()
