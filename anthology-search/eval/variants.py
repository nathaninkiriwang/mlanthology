"""LLM-written query variants as a rung (PaSa / LitLLM / Query2doc shape), measured on a stratified
240-query subset of the D-106 fixture, answered by the PRODUCTION engine (`mla batch`, hybrid),
so the measurement is of the sweep's own lane.

Stage 1 (`generate`): Claude (the harness's kclaude shim) writes, per query, three variants — a
keyword form, a sentence paraphrase, an adjacent-field phrasing — and a Query2doc-style
pseudo-abstract of the paper being sought. Saved to eval/variants.jsonl (resumable).
Stage 2 (`run`): one `mla batch` over original + variants + (original ⊕ pseudo-abstract); then
  var_hyb_orig   the original query alone (the baseline, same engine)
  var_mq_rrf     plain RRF over original + 3 variants
  var_mq_guard   guarded RRF (seats 5) over original + 3 variants     ← "rung 0.5"
  var_q2d        original ⊕ pseudo-abstract as one query
  var_all        guarded RRF over all five lists
Leakage check (Findings ACL 2025, "Hypothetical documents or knowledge leakage?"): R@10 is
reported separately for target papers published ≥ 2025 and < 2025.
"""
import json, os, random, re, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parent
KERNEL_HOME = "/Users/nathan/research/kernel"
SHIM = f"{KERNEL_HOME}/runtimes/claude-code/node_modules/.bin-wrapper/kclaude"
PER_STYLE = {"S1": 30, "S2": 30, "S3": 30, "S4": 30, "S5": 30, "S7": 30, "S6-S1": 20, "S6-S2": 20, "S6-S3": 20}
OUT = HERE / "variants.jsonl"
RUNS = HERE / "runs"
K = 100
BATCH = 8


def call_claude(prompt):
    env = {**os.environ, "KERNEL_HOME": KERNEL_HOME}
    with tempfile.TemporaryDirectory(prefix="mla-variants-") as scratch:
        proc = subprocess.run([SHIM, "-p", "--output-format", "json", prompt], capture_output=True, text=True, env=env, cwd=scratch, stdin=subprocess.DEVNULL, timeout=900)
    out = proc.stdout; payload = json.loads(out[out.index("{"):]); text = payload.get("result", "")
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def subset():
    rng = random.Random(1063)
    qs = [json.loads(l) for l in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_style = {}
    for q in qs: by_style.setdefault(q["style"], []).append(q)
    chosen = []
    for st, n in PER_STYLE.items():
        pool = sorted(by_style.get(st, []), key=lambda q: q["qid"]); rng.shuffle(pool); chosen += pool[:n]
    return chosen


def generate():
    chosen = subset()
    done = {}
    if OUT.exists():
        done = {json.loads(l)["qid"]: json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [q for q in chosen if q["qid"] not in done]
    print(f"{len(chosen)} in the subset; {len(done)} generated; {len(todo)} to do", flush=True)
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        prompt = (
            "You help a researcher search a corpus of ML and NLP papers (titles + abstracts). For EACH query below, "
            "write (1) three alternative queries a different researcher might type for the SAME information need: "
            "a keyword form (3-6 terms), a one-sentence paraphrase using different vocabulary, and a phrasing in the "
            "vocabulary of a neighbouring field; and (2) a plausible 60-90 word abstract of the paper being sought "
            "(Query2doc style: what such a paper would say, not a rewording of the query). Do NOT quote the query "
            "verbatim in the variants.\n\nQUERIES:\n"
            + "\n".join(f"[{q['qid']}] ({q['style']}) {q['query']}" for q in batch)
            + '\n\nReturn STRICT JSON only: {"<qid>": {"variants": ["...", "...", "..."], "pseudo": "..."}, ...} for EVERY qid.'
        )
        t0 = time.time()
        try:
            data = call_claude(prompt)
        except Exception as exc:
            print(f"batch at {start}: failed {exc}", flush=True); continue
        with OUT.open("a", encoding="utf-8") as fh:
            for q in batch:
                row = data.get(q["qid"]) or {}
                variants = [str(v).strip() for v in (row.get("variants") or []) if str(v).strip()][:3]
                pseudo = str(row.get("pseudo") or "").strip()
                if len(variants) < 3 or not pseudo:
                    print(f"{q['qid']}: incomplete ({len(variants)} variants, pseudo={bool(pseudo)})", flush=True)
                    continue
                fh.write(json.dumps({**q, "variants": variants, "pseudo": pseudo}) + "\n")
        print(f"batch {start // BATCH + 1}: {len(batch)} queries in {time.time() - t0:.0f}s", flush=True)
    print("generation done", flush=True)


def guarded(lists, seats=5, k=60):
    scores = {}
    for ranked in lists:
        for rank, d in enumerate(ranked):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank + 1)
    seated, seen = [], set()
    for depth in range(seats):
        for ranked in lists:
            if depth < len(ranked) and ranked[depth] not in seen:
                seen.add(ranked[depth]); seated.append(ranked[depth])
    return seated + sorted((d for d in scores if d not in seen), key=lambda d: -scores[d])


def plain_rrf(lists, k=60):
    scores = {}
    for ranked in lists:
        for rank, d in enumerate(ranked):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])


def run():
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    spec = {"queries": []}
    for r in rows:
        spec["queries"].append({"id": f"{r['qid']}:o", "query": r["query"], "limit": K})
        for i, v in enumerate(r["variants"], 1):
            spec["queries"].append({"id": f"{r['qid']}:v{i}", "query": v, "limit": K})
        spec["queries"].append({"id": f"{r['qid']}:p", "query": (r["query"] + " " + r["pseudo"])[:2000], "limit": K})
    spec_path = HERE / "variants.batch.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    print(f"batch of {len(spec['queries'])} hybrid queries for {len(rows)} originals", flush=True)
    t0 = time.time()
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.setdefault("MLA_DEVICE", "cpu")
    done = subprocess.run(["uv", "run", "--quiet", "--directory", str(A), "mla", "batch", "--file", str(spec_path), "--max-queries", str(len(spec["queries"]) + 1)],
                          capture_output=True, text=True, env=env, timeout=7200)
    if done.returncode != 0:
        sys.exit(f"mla batch failed: {done.stderr[-500:]}")
    payload = json.loads(done.stdout)
    print(f"batch answered in {time.time() - t0:.0f}s (dense {payload.get('dense')})", flush=True)
    lists = {res["id"]: [h["citekey"] for h in res["hits"]] for res in payload["results"]}
    systems = {"var_hyb_orig": [], "var_mq_rrf": [], "var_mq_guard": [], "var_q2d": [], "var_all": []}
    for r in rows:
        q = r["qid"]
        o = lists.get(f"{q}:o", []); vs = [lists.get(f"{q}:v{i}", []) for i in (1, 2, 3)]; p = lists.get(f"{q}:p", [])
        systems["var_hyb_orig"].append({"qid": q, "ranked": o[:K], "ms": 0})
        systems["var_mq_rrf"].append({"qid": q, "ranked": plain_rrf([o, *vs])[:K], "ms": 0})
        systems["var_mq_guard"].append({"qid": q, "ranked": guarded([o, *vs])[:K], "ms": 0})
        systems["var_q2d"].append({"qid": q, "ranked": p[:K], "ms": 0})
        systems["var_all"].append({"qid": q, "ranked": guarded([o, *vs, p])[:K], "ms": 0})
    RUNS.mkdir(exist_ok=True)
    for name, out in systems.items():
        (RUNS / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
        (RUNS / f"{name}.meta.json").write_text(json.dumps({"system": name, "subset": len(rows), "engine": "mla batch hybrid (production)"}, indent=1), encoding="utf-8")
        print(f"[{name}] {len(out)} queries", flush=True)
    # leakage split: R@10 by the target paper's year
    import sqlite3
    con = sqlite3.connect(A / "index" / "mla.sqlite3")
    year = {ck: y for ck, y in con.execute("select citekey, year from papers")}
    print("\nR@10 by target year (≥2025 = at or past the writer's likely knowledge; <2025 = older)")
    for name, out in systems.items():
        new = [x for x in out if (year.get(next(r["citekey"] for r in rows if r["qid"] == x["qid"])) or 0) >= 2025]
        old = [x for x in out if x not in new]
        def r10(group):
            hits = 0
            for x in group:
                target = next(r["citekey"] for r in rows if r["qid"] == x["qid"])
                hits += target in x["ranked"][:10]
            return hits / max(len(group), 1)
        print(f"  {name:14s} ≥2025 {r10(new):.3f} (n={len(new)})   <2025 {r10(old):.3f} (n={len(old)})")
    print("runs done", flush=True)


if __name__ == "__main__":
    {"generate": generate, "run": run}[sys.argv[1]]()
