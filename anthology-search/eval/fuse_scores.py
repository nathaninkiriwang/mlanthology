"""Convex score fusion versus (guarded) reciprocal-rank fusion on the D-106 fixture.

Bruch, Gai, Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval" (TOIS 2023): a convex
combination of min–max-normalized lane scores is more robust than RRF, and RRF is sensitive to
its constant. This measures exactly that on our corpus and queries, on the CPU (EVAL_DEVICE=cpu
from the caller), with the same lanes the production engine uses:

  cfuse_<norm>_w<w>[_guard]   norm ∈ {mm: observed min–max over each lane's top-K list,
                                       tmm: theoretical min 0, observed max}  w = dense weight
  guardk<k>                   guarded RRF with a different constant (10, 200; production uses 60)

Usage: python eval/fuse_scores.py [dense-model]      (default qwen3-0.6b)
"""
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import PROMPTS, VEC, corpus, load_model  # noqa: E402
from retrieve import K, queries, write_run  # noqa: E402

GUARD = 5
WEIGHTS = (0.3, 0.5, 0.7)


def guarded_order(lists, scores_of, seats=GUARD):
    """Seat each lane's top `seats` first (round-robin), then the rest by the fused score."""
    seated, seen = [], set()
    for depth in range(seats):
        for ranked in lists:
            if depth < len(ranked) and ranked[depth] not in seen:
                seen.add(ranked[depth]); seated.append(ranked[depth])
    rest = sorted((d for d in scores_of if d not in seen), key=lambda d: -scores_of[d])
    return seated + rest


def rrf(lists, k):
    out = {}
    for ranked in lists:
        for rank, d in enumerate(ranked):
            out[d] = out.get(d, 0.0) + 1.0 / (k + rank + 1)
    return out


def main():
    import bm25s, Stemmer
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
    ids, texts = corpus()
    stemmer = Stemmer.Stemmer("english")
    t0 = time.time()
    bm = bm25s.BM25(); bm.index(bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False), show_progress=False)
    print(f"[fuse] bm25s indexed in {time.time() - t0:.0f}s", flush=True)
    vec = np.load(VEC / f"{name}.f16.npy").astype(np.float32)
    meta = json.loads((VEC / f"{name}.meta.json").read_text(encoding="utf-8"))
    model, device = load_model(name, False, meta.get("backend", "hf"), meta.get("bf16", False))
    _, qprefix = PROMPTS[name]
    qs = queries()
    t0 = time.time()
    qemb = model.encode([qprefix + q["query"] for q in qs], batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    print(f"[fuse] {len(qs)} queries encoded on {device} in {time.time() - t0:.0f}s", flush=True)

    runs = {f"cfuse_{norm}_w{w}{g}": [] for norm in ("mm", "tmm") for w in WEIGHTS for g in ("", "_guard")}
    runs.update({"guardk10": [], "guardk200": [], "guardk60_check": []})
    for q, e in zip(qs, qemb):
        t1 = time.perf_counter()
        qt = bm25s.tokenize([q["query"]], stopwords="en", stemmer=stemmer, show_progress=False, return_ids=False)[0]
        vocab = bm.vocab_dict
        known = [t for t in qt if t in vocab]
        sb = np.asarray(bm.get_scores(known), dtype=np.float32) if known else np.zeros(len(ids), dtype=np.float32)
        sd = vec @ e
        top_b = np.argpartition(-sb, K)[:K]; top_b = top_b[np.argsort(-sb[top_b])]; top_b = [i for i in top_b if sb[i] > 0]
        top_d = np.argpartition(-sd, K)[:K]; top_d = top_d[np.argsort(-sd[top_d])].tolist()
        list_b = [ids[i] for i in top_b]; list_d = [ids[i] for i in top_d]
        cand = list(dict.fromkeys(top_b + top_d))
        ms = (time.perf_counter() - t1) * 1000
        for norm in ("mm", "tmm"):
            if top_b:
                b_max = float(sb[top_b[0]]); b_min = float(sb[top_b[-1]]) if norm == "mm" else 0.0
            else:
                b_max, b_min = 1.0, 0.0
            d_max = float(sd[top_d[0]]); d_min = float(sd[top_d[-1]]) if norm == "mm" else 0.0
            in_b, in_d = set(top_b), set(top_d)
            def nb(i):
                if i not in in_b: return 0.0
                return (float(sb[i]) - b_min) / max(b_max - b_min, 1e-9)
            def nd(i):
                if i not in in_d: return 0.0
                return (float(sd[i]) - d_min) / max(d_max - d_min, 1e-9)
            for w in WEIGHTS:
                fused = {ids[i]: (1 - w) * nb(i) + w * nd(i) for i in cand}
                plain = sorted(fused, key=lambda d: -fused[d])[:K]
                runs[f"cfuse_{norm}_w{w}"].append({"qid": q["qid"], "ranked": plain, "ms": ms})
                runs[f"cfuse_{norm}_w{w}_guard"].append({"qid": q["qid"], "ranked": guarded_order([list_b, list_d], fused)[:K], "ms": ms})
        for k, label in ((10, "guardk10"), (200, "guardk200"), (60, "guardk60_check")):
            scores = rrf([list_b, list_d], k)
            runs[label].append({"qid": q["qid"], "ranked": guarded_order([list_b, list_d], scores)[:K], "ms": ms})
    for label, rows in runs.items():
        write_run(label, rows, {"system": f"{label} over bm25s + dense_{name} (top-{K} per lane)", "components": ["bm25s", f"dense_{name}"]})
    print("[fuse] done", flush=True)


if __name__ == "__main__":
    main()
