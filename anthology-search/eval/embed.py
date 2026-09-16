"""Embed the whole corpus (title + abstract) with one model → index/vec/<name>.{f16.npy,ids.json,meta.json}.

`--probe --limit N` encodes N docs and prints docs/s plus the full-corpus extrapolation (D-106:
larger models are measured before they are committed to hours of compute). Query-side prefixes
live here too (PROMPTS) so retrieve.py encodes queries the way each model expects.
"""
import argparse, json, sqlite3, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "index" / "mla.sqlite3"
VEC = HERE.parent / "index" / "vec"

# (document prefix, query prefix) — the model card's own convention, or none.
PROMPTS = {
    "nomic": ("search_document: ", "search_query: "),
    "mxbai": ("", "Represent this sentence for searching relevant passages: "),
    "bge-m3": ("", ""),
    "gemma": ("title: none | text: ", "task: search result | query: "),
    "qwen3-0.6b": ("", "Instruct: Given a research query, retrieve relevant papers\nQuery: "),
    "qwen3-4b": ("", "Instruct: Given a research query, retrieve relevant papers\nQuery: "),
    "qwen3-8b": ("", "Instruct: Given a research query, retrieve relevant papers\nQuery: "),
    "specter2": ("", ""),
    "yuan": ("", ""),  # IEITYuan/Yuan-embedding-2.0-en: plain sentence-transformers, no prompt (card)
    "nemotron-1b": ("passage: ", "query: "),  # card: `query: ` / `passage: ` prefixes, mean pooling
    "nemotron-8b": ("passage: ", "query: "),
    "octen-8b": ("", "Instruct: Given a research query, retrieve relevant papers\nQuery: "),  # base Qwen3-Embedding-8B
}
MODELS = {
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
    "bge-m3": "BAAI/bge-m3",
    "gemma": "google/embeddinggemma-300m",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
    "specter2": "allenai/specter2_base",
    "yuan": "IEITYuan/Yuan-embedding-2.0-en",
    "nemotron-1b": "nvidia/Nemotron-3-Embed-1B-BF16",
    "nemotron-8b": "nvidia/Nemotron-3-Embed-8B-BF16",
    "octen-8b": "Octen/Octen-Embedding-8B",
}

def corpus(limit=None):
    con = sqlite3.connect(DB)
    q = "select rowid, citekey, title, abstract from papers order by rowid"
    if limit: q += f" limit {int(limit)}"
    rows = con.execute(q).fetchall()
    ids = [r[1] for r in rows]
    texts = [(f"{r[2]}. {r[3]}" if r[3] else r[2]) for r in rows]
    return ids, texts

OLLAMA = {"nomic": "nomic-embed-text", "mxbai": "mxbai-embed-large", "gemma": "embeddinggemma"}

class OllamaModel:
    """The same model served by Ollama (GGUF weights — documented as such in the report). Used
    where the Hugging Face remote code no longer runs on current transformers (nomic)."""
    def __init__(self, tag: str):
        self.tag = tag; self.max_seq_length = 512
    def get_embedding_dimension(self):
        return len(self._embed(["dimension probe"])[0])
    def _embed(self, texts):
        import json as _json, urllib.request
        req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=_json.dumps({"model": self.tag, "input": texts, "truncate": True}).encode(), headers={"Content-Type": "application/json"})
        return _json.load(urllib.request.urlopen(req, timeout=600))["embeddings"]
    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        import numpy as np
        out = []
        for s in range(0, len(texts), batch_size):
            out.extend(self._embed(texts[s:s + batch_size]))
        arr = np.asarray(out, dtype=np.float32)
        if normalize_embeddings:
            arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
        return arr

def load_model(name: str, fp16: bool, backend: str = "hf", bf16: bool = False):
    if backend == "ollama":
        return OllamaModel(OLLAMA[name]), "ollama"
    return _load_hf(name, fp16, bf16)

def _load_hf(name: str, fp16, bf16: bool = False):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    kwargs = {"trust_remote_code": True, "device": device}
    if bf16:
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    elif fp16:
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    model = SentenceTransformer(MODELS[name], **kwargs)
    return model, device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--probe", action="store_true", help="time N docs, extrapolate, write nothing")
    ap.add_argument("--backend", default="hf", choices=("hf", "ollama"))
    a = ap.parse_args()
    import numpy as np
    ids, texts = corpus(a.limit)
    total = sqlite3.connect(DB).execute("select count(*) from papers").fetchone()[0]
    doc_prefix, _ = PROMPTS[a.name]
    t_load = time.time(); model, device = load_model(a.name, a.fp16, a.backend, a.bf16); model.max_seq_length = a.max_seq
    print(f"[{a.name}] loaded {MODELS[a.name]} on {device} in {time.time() - t_load:.0f}s; encoding {len(ids)} docs", flush=True)
    t0 = time.time()
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    out = np.zeros((len(ids), dim), dtype=np.float16)
    CH = 4096
    for s in range(0, len(texts), CH):
        chunk = [doc_prefix + t for t in texts[s:s + CH]]
        emb = model.encode(chunk, batch_size=a.batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        out[s:s + len(chunk)] = emb.astype(np.float16)
        done = s + len(chunk); el = time.time() - t0
        print(f"[{a.name}] {done}/{len(ids)} docs, {done / el:.1f} docs/s, {el / 60:.1f} min elapsed", flush=True)
    el = time.time() - t0; rate = len(ids) / el
    print(f"[{a.name}] {len(ids)} docs in {el:.0f}s = {rate:.1f} docs/s → full corpus ({total}) ≈ {total / rate / 3600:.2f} h; dim {dim}", flush=True)
    if a.probe:
        return
    VEC.mkdir(parents=True, exist_ok=True)
    np.save(VEC / f"{a.name}.f16.npy", out)
    (VEC / f"{a.name}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (VEC / f"{a.name}.meta.json").write_text(json.dumps({"model": MODELS[a.name], "dim": dim, "docs": len(ids), "seconds": el, "docs_per_s": rate, "device": device, "fp16": a.fp16, "bf16": a.bf16, "max_seq": a.max_seq, "prefix": doc_prefix, "backend": a.backend}, indent=2), encoding="utf-8")
    print(f"[{a.name}] saved {VEC / (a.name + '.f16.npy')}", flush=True)

if __name__ == "__main__":
    main()
