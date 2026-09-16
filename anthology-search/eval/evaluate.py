"""Score run files against the known-item fixture (D-106).

A run is eval/runs/<system>.jsonl: {"qid", "ranked": [citekey, ...] (top-100), "ms"}; an optional
eval/runs/<system>.meta.json may list "components" for the hybrid admissibility check.
Metrics: recall@{1,5,10,20,50,100}, MRR@10 — overall and per style / slice / family / era /
text; paired bootstrap CIs of the recall@10/@20 difference against a baseline; latency.
"""
import argparse, json, random, statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = (1, 5, 10, 20, 50, 100)

def load_queries():
    return {json.loads(l)["qid"]: json.loads(l) for l in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

def load_run(name):
    path = HERE / "runs" / f"{name}.jsonl"
    rows = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); rows[r["qid"]] = r
    meta = HERE / "runs" / f"{name}.meta.json"
    return rows, (json.loads(meta.read_text()) if meta.exists() else {})

def rank_of(target, ranked):
    try: return ranked.index(target) + 1
    except ValueError: return None

def per_query(queries, run):
    out = {}
    for qid, q in queries.items():
        r = run.get(qid)
        if r is None: continue
        pos = rank_of(q["citekey"], r["ranked"])
        out[qid] = {"pos": pos, "ms": r.get("ms")}
    return out

def metrics(pq, qids):
    n = len(qids)
    if n == 0: return {}
    m = {f"R@{k}": sum(1 for q in qids if pq[q]["pos"] and pq[q]["pos"] <= k) / n for k in KS}
    m["MRR@10"] = sum((1 / pq[q]["pos"]) for q in qids if pq[q]["pos"] and pq[q]["pos"] <= 10) / n
    lat = [pq[q]["ms"] for q in qids if pq[q].get("ms") is not None]
    m["ms_p50"] = statistics.median(lat) if lat else None
    m["n"] = n
    return m

def bootstrap_diff(pq_a, pq_b, qids, k, B=1000, seed=1063):
    rng = random.Random(seed)
    def hit(pq, q): return 1 if pq[q]["pos"] and pq[q]["pos"] <= k else 0
    diffs = [hit(pq_a, q) - hit(pq_b, q) for q in qids]
    if not diffs: return None
    point = sum(diffs) / len(diffs)
    boots = []
    for _ in range(B):
        s = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(s) / len(s))
    boots.sort()
    return point, boots[int(0.025 * B)], boots[int(0.975 * B)]

def admissibility(hybrid_run, components, queries):
    """top-20 of the hybrid must contain each component's top-5 (dedup by citekey)."""
    viol = total = 0
    for qid in queries:
        h = hybrid_run.get(qid)
        if not h: continue
        top20 = set(h["ranked"][:20])
        for comp in components:
            c = comp.get(qid)
            if not c: continue
            total += 1
            if not set(c["ranked"][:5]) <= top20: viol += 1
    return viol, total

def judged_metrics(run, judgments, k=10):
    """nDCG@k and P@k over the judged queries (unjudged docs count as 0)."""
    import math
    by_q = defaultdict(dict)
    for j in judgments: by_q[j["qid"]][j["citekey"]] = j["grade"]
    nd, pr, n = 0.0, 0.0, 0
    for qid, rel in by_q.items():
        r = run.get(qid)
        if not r: continue
        ranked = r["ranked"][:k]
        dcg = sum((2 ** rel.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(ranked))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        nd += (dcg / idcg) if idcg else 0.0
        pr += sum(1 for c in ranked if rel.get(c, 0) == 2) / k
        n += 1
    return {"nDCG@10": nd / n if n else None, "P@10": pr / n if n else None, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+")
    ap.add_argument("--baseline", default="fts_and")
    ap.add_argument("--json", default=str(HERE / "results.json"))
    ap.add_argument("--judged", action="store_true", help="also score nDCG@10 / P@10 on eval/judgments.jsonl")
    a = ap.parse_args()
    queries = load_queries()
    runs = {s: load_run(s) for s in a.systems}
    pqs = {s: per_query(queries, r) for s, (r, _) in runs.items()}
    common = sorted(set.intersection(*(set(pq) for pq in pqs.values())))
    groups = {"style": lambda q: q["style"], "slice": lambda q: q["slice"], "family": lambda q: q["family"], "era": lambda q: q["era"], "text": lambda q: q["text"]}
    results = {"n_queries": len(common), "systems": {}}
    print(f"queries scored (present in every run): {len(common)}\n")
    header = f"{'system':22s} " + " ".join(f"{('R@'+str(k)):>6s}" for k in KS) + f" {'MRR@10':>7s} {'p50ms':>6s}"
    print(header)
    for s in a.systems:
        m = metrics(pqs[s], common); results["systems"][s] = {"overall": m}
        print(f"{s:22s} " + " ".join(f"{m['R@'+str(k)]:6.3f}" for k in KS) + f" {m['MRR@10']:7.3f} {str(round(m['ms_p50']) if m['ms_p50'] is not None else '-'):>6s}")
    for gname, keyf in groups.items():
        print(f"\n--- by {gname}: R@10 (n) ---")
        buckets = defaultdict(list)
        for q in common: buckets[keyf(queries[q])].append(q)
        print(f"{'':22s} " + " ".join(f"{b[:11]:>12s}" for b in sorted(buckets)))
        for s in a.systems:
            cells = []
            for b in sorted(buckets):
                m = metrics(pqs[s], buckets[b]); results["systems"][s].setdefault(gname, {})[b] = m
                cells.append(f"{m['R@10']:5.3f} ({m['n']:3d})")
            print(f"{s:22s} " + " ".join(f"{c:>12s}" for c in cells))
    if a.baseline in pqs:
        print(f"\n--- paired bootstrap vs {a.baseline}: Δ recall (95% CI), overall and by style ---")
        style_b = defaultdict(list)
        for q in common: style_b[queries[q]["style"]].append(q)
        for s in a.systems:
            if s == a.baseline: continue
            line = []
            for k in (10, 20):
                d = bootstrap_diff(pqs[s], pqs[a.baseline], common, k)
                line.append(f"Δ@{k} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
                results["systems"][s].setdefault("vs_baseline", {})[f"R@{k}"] = d
            per = []
            for st in sorted(style_b):
                d = bootstrap_diff(pqs[s], pqs[a.baseline], style_b[st], 10)
                per.append(f"{st}:{d[0]:+.2f}{'*' if d[1] > 0 or d[2] < 0 else ''}")
                results["systems"][s]["vs_baseline"][f"R@10:{st}"] = d
            print(f"{s:22s} " + "  ".join(line) + "   | " + " ".join(per))
        print("(* = the CI excludes zero)")
    for s, (run, meta) in runs.items():
        comps = meta.get("components") or []
        if comps:
            viol, total = admissibility(run, [runs[c][0] for c in comps if c in runs], {q: None for q in common})
            results["systems"][s]["admissibility"] = {"violations": viol, "checks": total}
            print(f"\nadmissibility {s}: {viol}/{total} component top-5 sets not within the hybrid top-20")
    if a.judged and (HERE / "judgments.jsonl").exists():
        judgments = [json.loads(l) for l in (HERE / "judgments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n--- judged slice: nDCG@10 / P@10 (n queries) ---")
        for s, (run, _) in runs.items():
            jm = judged_metrics(run, judgments); results["systems"][s]["judged"] = jm
            print(f"{s:22s} nDCG@10 {jm['nDCG@10'] if jm['nDCG@10'] is None else round(jm['nDCG@10'], 3)}  P@10 {jm['P@10'] if jm['P@10'] is None else round(jm['P@10'], 3)}  (n={jm['n']})")
    Path(a.json).write_text(json.dumps(results, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
