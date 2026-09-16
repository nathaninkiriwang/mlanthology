"""Run one retrieval system over eval/queries.jsonl → eval/runs/<system>.jsonl (D-106).

Systems:  fts_and | fts_or            the corpus's own `mla search` path (title bonus, rerank depth), AND / OR
          bm25s                       proper BM25 with Porter stemming + English stopwords (bm25s), title+abstract
          dense:<name>                cosine over index/vec/<name>.f16.npy (see embed.py for names/prompts)
          hybrid:<sysA>+<sysB>        reciprocal-rank fusion (k=60) of two EXISTING runs
          rerank:<model>:<base>       cross-encoder rerank of an existing run's top-100
Each run row: {"qid", "ranked": [citekey...] (≤100), "ms"}; a meta.json beside it records components.
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
sys.path.insert(0, str(HERE))
K = 100

def queries():
    return [json.loads(l) for l in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

def write_run(name, rows, meta=None):
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if meta: (RUNS / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[{name}] {len(rows)} queries → {RUNS / (name + '.jsonl')}", flush=True)

def load_run(name):
    return {json.loads(l)["qid"]: json.loads(l) for l in (RUNS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

# --- the corpus's own FTS path ---------------------------------------------------------------
def run_fts(mode):
    from mla.index import connect
    from mla.query import QueryError, search
    con = connect()
    rows = []
    for q in queries():
        t0 = time.perf_counter()
        try:
            res = search(con, q["query"], limit=K, mode=mode)
            ranked = [h["citekey"] for h in res.hits]
        except QueryError:
            ranked = []
        rows.append({"qid": q["qid"], "ranked": ranked, "ms": (time.perf_counter() - t0) * 1000})
    write_run(f"fts_{'and' if mode == 'all' else 'or'}", rows, {"system": f"mla.query.search mode={mode}"})

# --- proper BM25 -------------------------------------------------------------------------------
def run_bm25s():
    import bm25s, Stemmer
    from embed import corpus
    ids, texts = corpus()
    stemmer = Stemmer.Stemmer("english")
    t0 = time.time()
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
    bm = bm25s.BM25(); bm.index(tokens, show_progress=False)
    print(f"[bm25s] indexed {len(ids)} docs in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for q in queries():
        t0 = time.perf_counter()
        qt = bm25s.tokenize([q["query"]], stopwords="en", stemmer=stemmer, show_progress=False)
        docs, scores = bm.retrieve(qt, k=K, show_progress=False)
        ranked = [ids[i] for i, s in zip(docs[0], scores[0]) if s > 0]
        rows.append({"qid": q["qid"], "ranked": ranked, "ms": (time.perf_counter() - t0) * 1000})
    write_run("bm25s", rows, {"system": "bm25s Porter+stopwords over title+abstract"})

# --- dense --------------------------------------------------------------------------------------
def run_dense(name, fp16=False):
    import numpy as np
    from embed import PROMPTS, VEC, load_model
    vec = np.load(VEC / f"{name}.f16.npy").astype(np.float32)
    ids = json.loads((VEC / f"{name}.ids.json").read_text(encoding="utf-8"))
    meta = json.loads((VEC / f"{name}.meta.json").read_text(encoding="utf-8"))
    model, device = load_model(name, fp16, meta.get("backend", "hf"), meta.get("bf16", False))
    _, qprefix = PROMPTS[name]
    qs = queries()
    t0 = time.time()
    qemb = model.encode([qprefix + q["query"] for q in qs], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    enc_ms = (time.time() - t0) * 1000 / len(qs)
    rows = []
    for q, e in zip(qs, qemb):
        t0 = time.perf_counter()
        scores = vec @ e
        top = np.argpartition(-scores, K)[:K]; top = top[np.argsort(-scores[top])]
        rows.append({"qid": q["qid"], "ranked": [ids[i] for i in top], "ms": enc_ms + (time.perf_counter() - t0) * 1000})
    write_run(f"dense_{name}", rows, {"system": f"dense {name}", "docs": len(ids), "device": device, "query_prefix": qprefix})

# --- hybrid (RRF) -------------------------------------------------------------------------------
def run_hybrid(a, b, k=60):
    ra, rb = load_run(a), load_run(b)
    rows = []
    for qid in ra:
        if qid not in rb: continue
        score = {}
        for run in (ra[qid], rb[qid]):
            for rank, key in enumerate(run["ranked"][:K], 1):
                score[key] = score.get(key, 0.0) + 1.0 / (k + rank)
        ranked = [key for key, _ in sorted(score.items(), key=lambda kv: -kv[1])][:K]
        rows.append({"qid": qid, "ranked": ranked, "ms": (ra[qid].get("ms") or 0) + (rb[qid].get("ms") or 0)})
    write_run(f"hybrid_{a}+{b}", rows, {"system": f"RRF(k={k})", "components": [a, b]})

# --- guarded hybrid: each source's top-G is guaranteed a seat, then RRF fills (D-106 admissibility) ---
def run_hybrid_guard(*names, guard=5, k=60):
    """N-way: each source's top-G seated first (interleaved by rank), then RRF over the rest."""
    runs = [load_run(n) for n in names]
    rows = []
    for qid in runs[0]:
        if any(qid not in r for r in runs): continue
        lists = [r[qid] for r in runs]
        head, seen = [], set()
        for i in range(guard):
            for run in lists:
                if i < len(run["ranked"]) and run["ranked"][i] not in seen:
                    seen.add(run["ranked"][i]); head.append(run["ranked"][i])
        score = {}
        for run in lists:
            for rank, key in enumerate(run["ranked"][:K], 1):
                score[key] = score.get(key, 0.0) + 1.0 / (k + rank)
        tail = [key for key, _ in sorted(score.items(), key=lambda kv: -kv[1]) if key not in seen]
        rows.append({"qid": qid, "ranked": (head + tail)[:K], "ms": sum(run.get("ms") or 0 for run in lists)})
    write_run("guard_" + "+".join(names), rows, {"system": f"top-{guard} of each source interleaved, then RRF(k={k})", "components": list(names)})

# --- rerank -------------------------------------------------------------------------------------
RERANKERS = {"bge-m3": "BAAI/bge-reranker-v2-m3", "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B", "mxbai-v2": "mixedbread-ai/mxbai-rerank-large-v2"}

def run_rerank(rname, base, depth=100, resume=False):
    import torch
    from embed import corpus
    ids, texts = corpus(); text_of = dict(zip(ids, texts))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    base_run = load_run(base); qs = {q["qid"]: q for q in queries()}
    if rname == "qwen3-0.6b":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(RERANKERS[rname], padding_side="left")
        mdl = AutoModelForCausalLM.from_pretrained(RERANKERS[rname], torch_dtype=torch.float16).to(device).eval()
        yes_id, no_id = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instr = "Given a research query, judge whether the paper is relevant."
        def score(query, docs):
            out = []
            for s in range(0, len(docs), 16):
                batch = [f"{prefix}<Instruct>: {instr}\n<Query>: {query}\n<Document>: {d[:1500]}{suffix}" for d in docs[s:s + 16]]
                enc = tok(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = mdl(**enc).logits[:, -1, :]
                    lp = torch.log_softmax(torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float(), dim=1)[:, 1]
                out.extend(lp.cpu().tolist())
            return out
    else:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(RERANKERS[rname], device=device, max_length=1024, trust_remote_code=True)
        def score(query, docs):
            return ce.predict([(query, d[:1500]) for d in docs], batch_size=32, show_progress_bar=False).tolist()
    name = f"rerank_{rname}_{base}" + (f"_top{depth}" if depth != 100 else "")
    rows = []; have = set()
    if resume and (RUNS / f"{name}.jsonl").exists():
        rows = list(load_run(name).values()); have = {r["qid"] for r in rows}
        print(f"[{name}] resuming: {len(have)} queries already scored, {len(base_run) - len(have)} to do", flush=True)
    for qid, r in base_run.items():
        if qid in have: continue
        cand = r["ranked"][:depth]
        if not cand:
            rows.append({"qid": qid, "ranked": [], "ms": r.get("ms")}); continue
        t0 = time.perf_counter()
        sc = score(qs[qid]["query"], [text_of[c] for c in cand])
        ranked = [c for c, _ in sorted(zip(cand, sc), key=lambda p: -p[1])]
        rows.append({"qid": qid, "ranked": ranked, "ms": (r.get("ms") or 0) + (time.perf_counter() - t0) * 1000})
    order = {q["qid"]: i for i, q in enumerate(queries())}
    rows.sort(key=lambda r: order.get(r["qid"], 1 << 30))
    write_run(name, rows, {"system": f"{RERANKERS[rname]} over top-{depth} of {base}", "components": [base]})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("system"); ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--resume", action="store_true", help="rerank only the queries missing from an existing run file"); a = ap.parse_args()
    s = a.system
    if s == "fts_and": run_fts("all")
    elif s == "fts_or": run_fts("any")
    elif s == "bm25s": run_bm25s()
    elif s.startswith("dense:"): run_dense(s.split(":", 1)[1], a.fp16)
    elif s.startswith("hybrid:"): x, y = s.split(":", 1)[1].split("+"); run_hybrid(x, y)
    elif s.startswith("guard:"): run_hybrid_guard(*s.split(":", 1)[1].split("+"))
    elif s.startswith("rerank:"):
        parts = s.split(":")
        run_rerank(parts[1], parts[2], depth=int(parts[3]) if len(parts) > 3 else 100, resume=a.resume)
    else: sys.exit(f"unknown system {s}")

if __name__ == "__main__":
    main()
