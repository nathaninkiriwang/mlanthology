"""The judged slice (D-106): ~40 stratified queries, top-20 pooled across every run in eval/runs,
graded 0/1/2 by Claude from title + abstract → eval/judgments.jsonl. Incremental: re-running after new
runs land grades only the (query, paper) pairs the pool gained. The known-item paper is
relevant by construction (floored at 2). evaluate.py --judged then reports nDCG@10 / P@10.
"""
import json, os, random, re, sqlite3, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "index" / "mla.sqlite3"
KERNEL_HOME = "/Users/nathan/research/kernel"
SHIM = f"{KERNEL_HOME}/runtimes/claude-code/node_modules/.bin-wrapper/kclaude"
PER_STYLE = {"S1": 8, "S2": 10, "S3": 8, "S5": 6, "S7": 6, "S6-S2": 4}
POOL_DEPTH = 20

def call_claude(prompt):
    env = {**os.environ, "KERNEL_HOME": KERNEL_HOME}
    with tempfile.TemporaryDirectory(prefix="mla-judge-") as scratch:
        proc = subprocess.run([SHIM, "-p", "--output-format", "json", prompt], capture_output=True, text=True, env=env, cwd=scratch, stdin=subprocess.DEVNULL, timeout=600)
    out = proc.stdout; payload = json.loads(out[out.index("{"):]); text = payload.get("result", "")
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))

def main():
    rng = random.Random(1063)
    qs = [json.loads(l) for l in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_style = {}
    for q in qs: by_style.setdefault(q["style"], []).append(q)
    chosen = []
    for st, n in PER_STYLE.items():
        pool = sorted(by_style.get(st, []), key=lambda q: q["qid"]); rng.shuffle(pool); chosen += pool[:n]
    runs = {p.stem: {json.loads(l)["qid"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()} for p in (HERE / "runs").glob("*.jsonl")}
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    out = HERE / "judgments.jsonl"; done = set()
    if out.exists():
        done = {(j["qid"], j["citekey"]) for j in (json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip())}
    print(f"{len(chosen)} judged queries; pooling top-{POOL_DEPTH} from {len(runs)} runs; {len(done)} (query, paper) pairs already graded", flush=True)
    for q in chosen:
        pool = []
        for run in runs.values():
            r = run.get(q["qid"])
            if r:
                for c in r["ranked"][:POOL_DEPTH]:
                    if c not in pool: pool.append(c)
        if q["citekey"] not in pool: pool.append(q["citekey"])
        pool = [c for c in pool if (q["qid"], c) not in done]   # incremental: only papers not yet graded for this query
        if not pool: continue
        docs = []
        for c in pool:
            row = con.execute("select citekey, title, abstract from papers where citekey = ?", (c,)).fetchone()
            if row: docs.append(row)
        prompt = ("You are grading search results for a research query. Grade each paper 0 (not relevant), 1 (partially / "
                  "adjacent), or 2 (directly relevant: this paper is what the searcher wants or a paper they would need to cite).\n"
                  f"QUERY: {q['query']}\n\nPAPERS:\n" + "\n\n".join(f"[{i}] {d['citekey']}\nTITLE: {d['title']}\nABSTRACT: {(d['abstract'] or '(none)')[:1200]}" for i, d in enumerate(docs)) +
                  '\n\nReturn STRICT JSON: {"grades": {"<citekey>": 0|1|2, ...}} for EVERY paper listed.')
        t0 = time.time()
        try:
            data = call_claude(prompt)
        except Exception as exc:
            print(f"{q['qid']}: judge failed {exc}", flush=True); continue
        grades = data.get("grades", {})
        with out.open("a", encoding="utf-8") as fh:
            for d in docs:
                g = grades.get(d["citekey"])
                try: g = int(g)
                except (TypeError, ValueError): g = 0
                if d["citekey"] == q["citekey"]: g = max(g, 2)
                fh.write(json.dumps({"qid": q["qid"], "citekey": d["citekey"], "grade": max(0, min(2, g)), "style": q["style"]}) + "\n")
        print(f"{q['qid']} ({q['style']}): {len(docs)} docs graded in {time.time() - t0:.0f}s; relevant(2)={sum(1 for d in docs if int(grades.get(d['citekey'], 0) or 0) == 2)}", flush=True)
    print("judging done", flush=True)

if __name__ == "__main__":
    main()
