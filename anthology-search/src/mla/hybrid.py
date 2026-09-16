"""The hybrid lanes (mla 0.2.0, D-107): a stemmed BM25 lane, a dense lane per embedding model,
guarded fusion across them, and a reranker over a pool.

This is the stack the retrieval evaluation measured on THIS corpus (kernel-build/SEARCH-EVAL.md,
1,142 known-item queries over all 286k papers): the FTS5 lane answers nothing sentence-shaped;
BM25 with Porter stemming is the floor; a dense lane alone does not beat it but is the only lane
that finds paraphrases and title-only papers; plain reciprocal-rank fusion silences one lane's
top-5 in 12% of queries while seating each lane's top-5 first silences none; and an LLM reranker
over the fused top-50 is worth ~+0.09 R@10 at seconds per query — once per sweep, not per query.

Everything heavy is imported LAZILY. `mla` stays stdlib-only unless the `hybrid` extra is
installed, and the `fts` lane never touches this module. A missing lane or a missing extra is
`LaneUnavailable` — the operator's fault (exit 4 at the CLI, "build it with …"), never a thin
result.

Lane files live beside the SQLite index:

    index/bm25/            bm25s' own files + ids.json + meta.json
    index/vec/<name>.f16.npy  .ids.json  .meta.json      one vector per paper, corpus order
    index/vec/DEFAULT      the name of the dense lane a hybrid search uses when none is given
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from mla.index import meta as index_meta
from mla.query import _SELECT, Filters, _hit

GUARD_SEATS = 5            # each lane's top-5 is seated before fusion (measured: zero silenced lists)
RRF_K = 60
DEPTH_CAP = 1000
DEFAULT_MARKER = "DEFAULT"
DEFAULT_RERANKER = "qwen3-0.6b"
HASHED_DIM = 256
_MAX_SQL_PARAMS = 900

# Embedding models the lane knows how to build and query. `doc` / `query` are each card's own
# prefixes; `ollama` names the same weights served by Ollama where that exists. `hashed` is a
# deterministic feature-hashing embedder for tests and offline smoke — named, never default.
MODELS: dict[str, dict] = {
    "qwen3-0.6b": {"hf": "Qwen/Qwen3-Embedding-0.6B", "ollama": "qwen3-embedding:0.6b", "doc": "",
                   "query": "Instruct: Given a research query, retrieve relevant papers\nQuery: "},
    "qwen3-4b": {"hf": "Qwen/Qwen3-Embedding-4B", "ollama": "qwen3-embedding:4b", "doc": "",
                 "query": "Instruct: Given a research query, retrieve relevant papers\nQuery: "},
    "qwen3-8b": {"hf": "Qwen/Qwen3-Embedding-8B", "ollama": "qwen3-embedding:8b", "doc": "",
                 "query": "Instruct: Given a research query, retrieve relevant papers\nQuery: "},
    "octen-8b": {"hf": "Octen/Octen-Embedding-8B", "ollama": None, "doc": "",
                 "query": "Instruct: Given a research query, retrieve relevant papers\nQuery: "},
    "yuan": {"hf": "IEITYuan/Yuan-embedding-2.0-en", "ollama": None, "doc": "", "query": ""},
    "nemotron-1b": {"hf": "nvidia/Nemotron-3-Embed-1B-BF16", "ollama": None,
                    "doc": "passage: ", "query": "query: "},
    "nemotron-8b": {"hf": "nvidia/Nemotron-3-Embed-8B-BF16", "ollama": None,
                    "doc": "passage: ", "query": "query: "},
    "mxbai": {"hf": "mixedbread-ai/mxbai-embed-large-v1", "ollama": "mxbai-embed-large", "doc": "",
              "query": "Represent this sentence for searching relevant passages: "},
    "bge-m3": {"hf": "BAAI/bge-m3", "ollama": "bge-m3", "doc": "", "query": ""},
    "nomic": {"hf": "nomic-ai/nomic-embed-text-v1.5", "ollama": "nomic-embed-text",
              "doc": "search_document: ", "query": "search_query: "},
    "gemma": {"hf": "google/embeddinggemma-300m", "ollama": "embeddinggemma",
              "doc": "title: none | text: ", "query": "task: search result | query: "},
    "specter2": {"hf": "allenai/specter2_base", "ollama": None, "doc": "", "query": ""},
    "hashed": {"hf": None, "ollama": None, "doc": "", "query": ""},
}
RERANKERS: dict[str, str | None] = {
    "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B",
    "bge-m3": "BAAI/bge-reranker-v2-m3",
    "hashed": None,
}
BACKENDS = ("hf", "ollama", "hashed")
MODES = ("fts", "bm25", "dense", "hybrid")


class LaneUnavailable(RuntimeError):
    """A lane or the `hybrid` extra is missing — the operator's fault, with the fix in it."""


# --- lazy imports ------------------------------------------------------------------------------

def _np():
    try:
        import numpy
    except ImportError as exc:                       # pragma: no cover - environment-specific
        raise LaneUnavailable(
            "MLA-HYBRID: the `hybrid` extra is not installed (numpy missing). "
            "Fix: uv sync --extra hybrid") from exc
    return numpy


def _bm25s():
    try:
        import bm25s
        import Stemmer
    except ImportError as exc:                       # pragma: no cover - environment-specific
        raise LaneUnavailable(
            "MLA-HYBRID: the `hybrid` extra is not installed (bm25s / PyStemmer missing). "
            "Fix: uv sync --extra hybrid") from exc
    return bm25s, Stemmer.Stemmer("english")


def extra_installed() -> bool:
    try:
        import bm25s  # noqa: F401
        import numpy  # noqa: F401
        import Stemmer  # noqa: F401
    except ImportError:
        return False
    return True


def torch_installed() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def device_name() -> str:
    """`MLA_DEVICE` wins (tests and a busy GPU pin `cpu`); else MPS when present; else CPU."""
    if pinned := os.environ.get("MLA_DEVICE", "").strip():
        return pinned
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


# --- corpus access -----------------------------------------------------------------------------

def corpus_texts(connection: sqlite3.Connection, limit: int | None = None) -> tuple[list[str], list[str]]:
    """Every paper as `title. abstract` (title alone where there is none), in rowid order."""
    sql = "SELECT citekey, title, abstract FROM papers ORDER BY rowid"
    if limit:
        sql += f" LIMIT {int(limit)}"
    ids, texts = [], []
    for citekey, title, abstract in connection.execute(sql):
        ids.append(citekey)
        texts.append(f"{title}. {abstract}" if abstract else title)
    return ids, texts


def corpus_sha_of(connection: sqlite3.Connection) -> str:
    return index_meta(connection).get("corpus_sha", "")


def rows_by_citekey(connection: sqlite3.Connection, citekeys: list[str]) -> dict[str, sqlite3.Row]:
    out: dict[str, sqlite3.Row] = {}
    for start in range(0, len(citekeys), _MAX_SQL_PARAMS):
        chunk = citekeys[start:start + _MAX_SQL_PARAMS]
        marks = ", ".join("?" * len(chunk))
        for row in connection.execute(
                f"SELECT {_SELECT} FROM papers p WHERE p.citekey IN ({marks})", chunk):
            out[row["citekey"]] = row
    return out


def _write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


# --- the BM25 lane -----------------------------------------------------------------------------

def bm25_dir(index_path: Path) -> Path:
    return Path(index_path).parent / "bm25"


def build_bm25(connection: sqlite3.Connection, index_path: Path, progress=None) -> dict:
    """Index title+abstract with Porter stemming and English stopwords; atomic install."""
    bm25s, stemmer = _bm25s()
    ids, texts = corpus_texts(connection)
    started = time.time()
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    target = bm25_dir(index_path)
    staging = target.with_name("bm25.building")
    if staging.exists():
        import shutil
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    retriever.save(str(staging), show_progress=False)
    meta = {
        "lane": "bm25", "docs": len(ids), "corpus_sha": corpus_sha_of(connection),
        "stemmer": "porter-english", "stopwords": "en",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started, 1),
    }
    _write_json(staging / "ids.json", ids)
    _write_json(staging / "meta.json", meta)
    if target.exists():
        import shutil
        old = target.with_name("bm25.prev")
        if old.exists():
            shutil.rmtree(old)
        target.rename(old)
        staging.rename(target)
        shutil.rmtree(old)
    else:
        staging.rename(target)
    if progress:
        progress(len(ids))
    return meta


@dataclass
class Bm25Lane:
    retriever: object
    stemmer: object
    ids: list[str]
    meta: dict

    @classmethod
    def load(cls, index_path: Path) -> "Bm25Lane":
        bm25s, stemmer = _bm25s()
        directory = bm25_dir(index_path)
        if not (directory / "meta.json").is_file():
            raise LaneUnavailable(
                f"MLA-HYBRID: no BM25 lane at {directory}. Build it with `mla bm25 build` "
                "(about 20 seconds).")
        retriever = bm25s.BM25.load(str(directory), mmap=True, show_progress=False)
        ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(retriever, stemmer, ids, meta)

    def query_tokens(self, query: str) -> tuple[list[str], list[str]]:
        bm25s, _ = _bm25s()
        tokens = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer,
                                show_progress=False, return_ids=False)[0]
        vocab = self.retriever.vocab_dict
        unknown = [t for t in tokens if t not in vocab]
        return tokens, unknown

    def scores(self, query: str):
        """A score for every paper (0 where no term matches) plus the stems the lane never saw."""
        np = _np()
        tokens, unknown = self.query_tokens(query)
        known = [t for t in tokens if t not in unknown]
        if not known:
            return np.zeros(len(self.ids), dtype=np.float32), unknown
        return np.asarray(self.retriever.get_scores(known), dtype=np.float32), unknown


# --- encoders ----------------------------------------------------------------------------------

class HashedEncoder:
    """Feature hashing over word unigrams and bigrams, L2-normalized. Deterministic, dependency-
    free beyond numpy, and honest about what it is: a test double that behaves like an embedder
    (shared vocabulary ⇒ nearby vectors), not a model."""
    backend = "hashed"

    def __init__(self, dim: int = HASHED_DIM):
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 64):
        np = _np()
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, bucket] += sign
        norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out / norms


class OllamaEncoder:
    """The same weights served by Ollama (GGUF). Stdlib HTTP; the server holds the model."""
    backend = "ollama"

    def __init__(self, tag: str, url: str | None = None):
        self.tag = tag
        self.url = (url or os.environ.get("MLA_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/")

    def _embed(self, texts: list[str]):
        import urllib.request
        request = urllib.request.Request(
            f"{self.url}/api/embed",
            data=json.dumps({"model": self.tag, "input": texts, "truncate": True}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)["embeddings"]

    def encode(self, texts: list[str], batch_size: int = 32):
        np = _np()
        rows: list = []
        for start in range(0, len(texts), batch_size):
            rows.extend(self._embed(texts[start:start + batch_size]))
        matrix = np.asarray(rows, dtype=np.float32)
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


class HfEncoder:
    """sentence-transformers over the Hugging Face weights, on `device_name()`."""
    backend = "hf"

    def __init__(self, model_id: str, fp16: bool = False, bf16: bool = False, max_seq: int = 512):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:                   # pragma: no cover - environment-specific
            raise LaneUnavailable(
                "MLA-HYBRID: the `hybrid` extra is not installed (torch / sentence-transformers "
                "missing). Fix: uv sync --extra hybrid") from exc
        kwargs: dict = {"trust_remote_code": True, "device": device_name()}
        if bf16:
            kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
        elif fp16:
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = SentenceTransformer(model_id, **kwargs)
        self.model.max_seq_length = max_seq
        self.device = kwargs["device"]

    def encode(self, texts: list[str], batch_size: int = 32):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False).astype("float32")


def make_encoder(name: str, backend: str = "hf", fp16: bool = False, bf16: bool = False):
    if name not in MODELS:
        raise LaneUnavailable(
            f"MLA-HYBRID: unknown embedding model {name!r}. Known: {', '.join(sorted(MODELS))}.")
    if backend not in BACKENDS:
        raise LaneUnavailable(f"MLA-HYBRID: unknown backend {backend!r}. Known: {', '.join(BACKENDS)}.")
    spec = MODELS[name]
    if name == "hashed" or backend == "hashed":
        if name != "hashed":
            raise LaneUnavailable("MLA-HYBRID: the `hashed` backend only serves the `hashed` model.")
        return HashedEncoder()
    if backend == "ollama":
        if not spec["ollama"]:
            raise LaneUnavailable(f"MLA-HYBRID: {name} has no Ollama build; use --backend hf.")
        return OllamaEncoder(spec["ollama"])
    return HfEncoder(spec["hf"], fp16=fp16, bf16=bf16)


# --- the vector lane ---------------------------------------------------------------------------

def vec_dir(index_path: Path) -> Path:
    return Path(index_path).parent / "vec"


def vec_paths(index_path: Path, name: str) -> tuple[Path, Path, Path]:
    directory = vec_dir(index_path)
    return directory / f"{name}.f16.npy", directory / f"{name}.ids.json", directory / f"{name}.meta.json"


def list_vec(index_path: Path) -> dict[str, dict]:
    """Every dense lane present, by model name, with its meta."""
    out: dict[str, dict] = {}
    directory = vec_dir(index_path)
    if not directory.is_dir():
        return out
    for meta_path in sorted(directory.glob("*.meta.json")):
        name = meta_path.name[:-len(".meta.json")]
        npy, ids, _ = vec_paths(index_path, name)
        if npy.is_file() and ids.is_file():
            try:
                out[name] = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                out[name] = {"error": "unreadable meta"}
    return out


def default_dense(index_path: Path) -> str | None:
    """`MLA_DENSE` wins; else the DEFAULT marker; else the only lane present; else None."""
    if pinned := os.environ.get("MLA_DENSE", "").strip():
        return pinned
    marker = vec_dir(index_path) / DEFAULT_MARKER
    if marker.is_file():
        name = marker.read_text(encoding="utf-8").strip()
        if name:
            return name
    present = list_vec(index_path)
    return next(iter(present)) if len(present) == 1 else None


def set_default_dense(index_path: Path, name: str) -> Path:
    if name not in list_vec(index_path):
        raise LaneUnavailable(
            f"MLA-HYBRID: no dense lane named {name!r} under {vec_dir(index_path)}; "
            f"present: {', '.join(sorted(list_vec(index_path))) or 'none'}. "
            f"Build it with `mla vec build --model {name}`.")
    marker = vec_dir(index_path) / DEFAULT_MARKER
    marker.write_text(name + "\n", encoding="utf-8")
    return marker


def build_vec(connection: sqlite3.Connection, index_path: Path, name: str, *,
              backend: str = "hf", encoder=None, batch: int = 32, limit: int | None = None,
              fp16: bool = False, bf16: bool = False, max_seq: int = 512, progress=None) -> dict:
    """Embed the papers this lane lacks (and drop the ones the corpus lost); atomic install.

    Incremental by citekey: a weekly refresh that adds a few hundred papers embeds a few hundred
    papers. The stored order is corpus order, so a lane and the BM25 lane index the same rows.
    """
    np = _np()
    if name not in MODELS:
        raise LaneUnavailable(
            f"MLA-HYBRID: unknown embedding model {name!r}. Known: {', '.join(sorted(MODELS))}.")
    encoder = encoder or make_encoder(name, backend, fp16=fp16, bf16=bf16)
    if hasattr(encoder, "model"):
        encoder.model.max_seq_length = max_seq
    doc_prefix = MODELS[name]["doc"]
    ids, texts = corpus_texts(connection, limit=limit)
    npy_path, ids_path, meta_path = vec_paths(index_path, name)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: dict[str, int] = {}
    existing = None
    if npy_path.is_file() and ids_path.is_file():
        old_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        existing = np.load(npy_path, mmap_mode="r")
        if existing.shape[0] == len(old_ids):
            existing_rows = {citekey: row for row, citekey in enumerate(old_ids)}
        else:
            existing = None
    todo = [(row, citekey) for row, citekey in enumerate(ids) if citekey not in existing_rows]
    removed = len(existing_rows) - (len(ids) - len(todo))

    started = time.time()
    fresh: dict[int, object] = {}
    done = 0
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        vectors = encoder.encode([doc_prefix + texts[row] for row, _ in chunk], batch_size=batch)
        for (row, _), vector in zip(chunk, vectors):
            fresh[row] = vector
        done += len(chunk)
        if progress and (done % (batch * 32) == 0 or done == len(todo)):
            progress(done, len(todo), time.time() - started)
    if fresh:
        dim = len(next(iter(fresh.values())))
    elif existing is not None:
        dim = existing.shape[1]
    else:
        dim = HASHED_DIM if name == "hashed" else 0
    matrix = np.zeros((len(ids), dim), dtype=np.float16)
    for row, citekey in enumerate(ids):
        if row in fresh:
            matrix[row] = fresh[row]
        else:
            matrix[row] = existing[existing_rows[citekey]]
    elapsed = time.time() - started
    meta = {
        "lane": "dense", "model": name, "model_id": MODELS[name]["hf"], "backend": getattr(encoder, "backend", backend),
        "dim": int(dim), "docs": len(ids), "corpus_sha": corpus_sha_of(connection),
        "prefix": doc_prefix, "query_prefix": MODELS[name]["query"], "max_seq": max_seq,
        "fp16": fp16, "bf16": bf16, "limit": limit,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(elapsed, 1),
        "docs_per_s": round(len(todo) / elapsed, 2) if elapsed > 0 and todo else None,
        "incremental": {"embedded": len(todo), "kept": len(ids) - len(todo), "removed": max(removed, 0)},
    }
    tmp = npy_path.with_suffix(".tmp.npy")
    np.save(tmp, matrix)
    tmp.replace(npy_path)
    _write_json(ids_path, ids)
    _write_json(meta_path, meta)
    return meta


@dataclass
class VecLane:
    name: str
    matrix: object
    ids: list[str]
    meta: dict

    @classmethod
    def load(cls, index_path: Path, name: str) -> "VecLane":
        np = _np()
        npy_path, ids_path, meta_path = vec_paths(index_path, name)
        if not (npy_path.is_file() and ids_path.is_file()):
            present = ", ".join(sorted(list_vec(index_path))) or "none"
            raise LaneUnavailable(
                f"MLA-HYBRID: no dense lane {name!r} under {vec_dir(index_path)} (present: {present}). "
                f"Build it with `mla vec build --model {name}`.")
        matrix = np.load(npy_path).astype(np.float32)
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if matrix.shape[0] != len(ids):
            raise LaneUnavailable(
                f"MLA-HYBRID: dense lane {name!r} is inconsistent ({matrix.shape[0]} vectors, "
                f"{len(ids)} ids). Rebuild it with `mla vec build --model {name}`.")
        return cls(name, matrix, ids, meta)

    def scores(self, query_vector):
        return self.matrix @ query_vector


# --- masks, fusion -------------------------------------------------------------------------------

def filter_mask(connection: sqlite3.Connection, filters: Filters | None, position: dict[str, int], n: int):
    """A boolean mask over lane rows for the papers the filters admit — None when unfiltered.

    Applied INSIDE the lanes, so a `--since 2018 --family theory` search still returns its full
    limit; post-filtering a fixed top-k loses exactly the recall a facet was meant to buy."""
    if filters is None:
        return None
    where, params = filters.sql()
    if not where:
        return None
    np = _np()
    mask = np.zeros(n, dtype=bool)
    for (citekey,) in connection.execute(
            f"SELECT p.citekey FROM papers p WHERE {' AND '.join(where)}", params):
        row = position.get(citekey)
        if row is not None:
            mask[row] = True
    return mask


def top_positions(scores, mask, k: int, positive_only: bool) -> list[int]:
    np = _np()
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, scores.shape[0])
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    out = []
    for row in top.tolist():
        value = scores[row]
        if value == -np.inf or (positive_only and value <= 0):
            break
        out.append(row)
    return out


def guarded_rrf(lists: list[list[str]], seats: int = GUARD_SEATS, k: int = RRF_K) -> list[tuple[str, float]]:
    """Seat each list's top `seats` first (round-robin, in list order), then reciprocal-rank
    fusion over everything. No list can be silenced by the others' agreement."""
    rrf: dict[str, float] = {}
    for ranked in lists:
        for rank, key in enumerate(ranked):
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (k + rank + 1)
    seated: list[str] = []
    seen: set[str] = set()
    for depth in range(seats):
        for ranked in lists:
            if depth < len(ranked) and ranked[depth] not in seen:
                seen.add(ranked[depth])
                seated.append(ranked[depth])
    rest = sorted((key for key in rrf if key not in seen), key=lambda key: (-rrf[key], key))
    return [(key, rrf[key]) for key in seated] + [(key, rrf[key]) for key in rest]


# --- the engine --------------------------------------------------------------------------------

@dataclass
class LaneAnswer:
    ranked: list[str]
    total: int | None = None
    unknown_terms: list[str] = field(default_factory=list)


@dataclass
class Engine:
    """Lanes loaded once; many searches. The sweep's batch mode exists because of this: a model
    load per query would turn a 24-query ladder into minutes."""

    connection: sqlite3.Connection
    index_path: Path
    dense: str | None = None
    backend: str | None = None
    bm25: Bm25Lane | None = None
    vec: VecLane | None = None
    encoder: object = None
    position: dict[str, int] = field(default_factory=dict)

    def ensure(self, mode: str) -> None:
        if mode not in MODES or mode == "fts":
            raise LaneUnavailable(f"MLA-HYBRID: mode must be one of {', '.join(m for m in MODES if m != 'fts')}")
        if mode in ("bm25", "hybrid") and self.bm25 is None:
            self.bm25 = Bm25Lane.load(self.index_path)
            self.position = {citekey: row for row, citekey in enumerate(self.bm25.ids)}
        if mode in ("dense", "hybrid") and self.vec is None:
            name = self.dense or default_dense(self.index_path)
            if not name:
                raise LaneUnavailable(
                    "MLA-HYBRID: no dense lane is marked default and none was named. Build one "
                    "with `mla vec build --model <name>` and mark it with `mla vec default <name>`.")
            self.vec = VecLane.load(self.index_path, name)
            self.dense = name
            if self.bm25 is not None and self.bm25.ids != self.vec.ids:
                raise LaneUnavailable(
                    "MLA-HYBRID: the BM25 lane and the dense lane index different papers "
                    "(rebuild both: `mla bm25 build` and `mla vec build --model "
                    f"{name}`).")
            if not self.position:
                self.position = {citekey: row for row, citekey in enumerate(self.vec.ids)}
            if self.encoder is None:
                backend = self.backend or self.vec.meta.get("backend") or "hf"
                self.encoder = make_encoder(name, backend, fp16=bool(self.vec.meta.get("fp16")),
                                            bf16=bool(self.vec.meta.get("bf16")))

    @property
    def ids(self) -> list[str]:
        return (self.bm25 or self.vec).ids

    def lane_bm25(self, query: str, mask, depth: int) -> LaneAnswer:
        np = _np()
        scores, unknown = self.bm25.scores(query)
        positive = scores > 0
        if mask is not None:
            positive &= mask
        total = int(np.count_nonzero(positive))
        rows = top_positions(scores, mask, depth, positive_only=True)
        return LaneAnswer([self.ids[row] for row in rows], total, unknown)

    def lane_dense(self, query: str, mask, depth: int) -> LaneAnswer:
        prefix = MODELS[self.vec.name]["query"]
        vector = self.encoder.encode([prefix + query], batch_size=1)[0]
        rows = top_positions(self.vec.scores(vector), mask, depth, positive_only=False)
        return LaneAnswer([self.ids[row] for row in rows], None, [])

    def search(self, query: str, *, mode: str = "hybrid", limit: int = 20, offset: int = 0,
               filters: Filters | None = None, abstracts: bool = False,
               depth: int | None = None) -> dict:
        self.ensure(mode)
        if not query or not query.strip():
            raise LaneUnavailable("MLA-HYBRID: an empty query has nothing to embed or score")
        want = offset + limit
        depth = min(depth or max(want * 2, 50), DEPTH_CAP)
        mask = filter_mask(self.connection, filters, self.position, len(self.ids))
        lanes: dict[str, LaneAnswer] = {}
        if mode in ("bm25", "hybrid"):
            lanes["bm25"] = self.lane_bm25(query, mask, depth)
        if mode in ("dense", "hybrid"):
            lanes["dense"] = self.lane_dense(query, mask, depth)
        if mode == "hybrid":
            fused = guarded_rrf([lanes["bm25"].ranked, lanes["dense"].ranked])
        else:
            only = next(iter(lanes.values()))
            fused = [(key, 1.0 / (RRF_K + rank + 1)) for rank, key in enumerate(only.ranked)]
        page = fused[offset:want]
        rows = rows_by_citekey(self.connection, [key for key, _ in page])
        rank_in = {lane: {key: rank + 1 for rank, key in enumerate(answer.ranked)}
                   for lane, answer in lanes.items()}
        hits = []
        for key, score in page:
            row = rows.get(key)
            if row is None:
                continue
            hit = _hit(row, score=score, abstracts=abstracts)
            hit["lanes"] = {lane: rank_in[lane].get(key) for lane in lanes}
            hits.append(hit)
        unknown = sorted({term for answer in lanes.values() for term in answer.unknown_terms})
        totals = {lane: answer.total for lane, answer in lanes.items()}
        return {
            "mode": mode,
            "dense": self.vec.name if self.vec is not None else None,
            "total": totals.get("bm25"),
            "totals": totals,
            "unknown_terms": unknown,
            "depth": depth,
            "hits": hits,
        }


# --- the reranker ------------------------------------------------------------------------------

class Reranker:
    """`score(query, texts)` — bigger is better. Qwen3-Reranker (a causal LM read for its yes/no
    logit), a cross-encoder, or the hashed cosine for tests."""

    def __init__(self, name: str = DEFAULT_RERANKER):
        if name not in RERANKERS:
            raise LaneUnavailable(
                f"MLA-HYBRID: unknown reranker {name!r}. Known: {', '.join(sorted(RERANKERS))}.")
        self.name = name
        self.model_id = RERANKERS[name]
        self.device = "cpu"
        if name == "hashed":
            self._encoder = HashedEncoder()
            return
        try:
            import torch
        except ImportError as exc:                   # pragma: no cover - environment-specific
            raise LaneUnavailable(
                "MLA-HYBRID: the `hybrid` extra is not installed (torch missing). "
                "Fix: uv sync --extra hybrid") from exc
        self.device = device_name()
        if name == "qwen3-0.6b":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device).eval()
            self.yes_id = self.tok.convert_tokens_to_ids("yes")
            self.no_id = self.tok.convert_tokens_to_ids("no")
        else:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_id, device=self.device, max_length=1024, trust_remote_code=True)

    _PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
               "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
               "<|im_end|>\n<|im_start|>user\n")
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    _INSTRUCT = "Given a research query, judge whether the paper is relevant."

    def score(self, query: str, texts: list[str], batch: int = 16) -> list[float]:
        if not texts:
            return []
        if self.name == "hashed":
            np = _np()
            vectors = self._encoder.encode([query] + texts)
            return (vectors[1:] @ vectors[0]).astype(float).tolist()
        import torch
        if self.name == "qwen3-0.6b":
            out: list[float] = []
            for start in range(0, len(texts), batch):
                prompts = [f"{self._PREFIX}<Instruct>: {self._INSTRUCT}\n<Query>: {query}\n"
                           f"<Document>: {text[:1500]}{self._SUFFIX}" for text in texts[start:start + batch]]
                encoded = self.tok(prompts, padding=True, truncation=True, max_length=1024,
                                   return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = self.model(**encoded).logits[:, -1, :]
                    pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
                    out.extend(torch.log_softmax(pair, dim=1)[:, 1].cpu().tolist())
            return out
        return self.model.predict([(query, text[:1500]) for text in texts], batch_size=32,
                                  show_progress_bar=False).tolist()


def rerank_pool(connection: sqlite3.Connection, reranker: Reranker, query: str,
                pool: list[dict]) -> dict:
    """Score a pool once. Entries carry `id` and either a `citekey` (resolved here to
    title+abstract) or their own `text`; entries with neither are reported, not guessed."""
    if not query or not query.strip():
        raise LaneUnavailable("MLA-HYBRID: rerank needs a non-empty query")
    citekeys = [str(entry.get("citekey")) for entry in pool if entry.get("citekey")]
    rows = rows_by_citekey(connection, citekeys) if citekeys else {}
    texts: list[str] = []
    ids: list[str] = []
    skipped: list[str] = []
    for entry in pool:
        entry_id = str(entry.get("id") or entry.get("citekey") or "")
        row = rows.get(str(entry.get("citekey") or ""))
        text = (f"{row['title']}. {row['abstract']}" if row and row["abstract"] else
                (row["title"] if row else entry.get("text")))
        if not entry_id or not text:
            skipped.append(entry_id or "?")
            continue
        ids.append(entry_id)
        texts.append(str(text))
    started = time.time()
    scores = reranker.score(query, texts)
    ranked = sorted(zip(ids, scores), key=lambda pair: -pair[1])
    return {
        "model": reranker.name, "model_id": reranker.model_id, "device": reranker.device,
        "query": query, "scored": len(ids), "skipped": skipped,
        "seconds": round(time.time() - started, 2),
        "ranked": [{"id": entry_id, "score": round(float(score), 4)} for entry_id, score in ranked],
    }


# --- status ------------------------------------------------------------------------------------

def lanes_status(connection: sqlite3.Connection, index_path: Path) -> dict:
    """What `mla doctor` and `mla stats` say about the lanes. Never raises."""
    sha = corpus_sha_of(connection)
    papers = int(index_meta(connection).get("papers", 0) or 0)
    report: dict = {"hybrid_extra": extra_installed(), "torch": torch_installed(),
                    "device": device_name() if torch_installed() else None,
                    "bm25": {"present": False}, "vec": {"default": None, "models": {}}}
    directory = bm25_dir(index_path)
    if (directory / "meta.json").is_file():
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            report["bm25"] = {"present": True, "docs": meta.get("docs"),
                              "stale": meta.get("corpus_sha") != sha, "built_at": meta.get("built_at")}
        except ValueError:
            report["bm25"] = {"present": True, "error": "unreadable meta"}
    try:
        report["vec"]["default"] = default_dense(index_path)
    except OSError:
        pass
    for name, meta in list_vec(index_path).items():
        docs = meta.get("docs")
        report["vec"]["models"][name] = {
            "docs": docs, "dim": meta.get("dim"), "backend": meta.get("backend"),
            "built_at": meta.get("built_at"),
            "stale": (meta.get("corpus_sha") != sha) if meta.get("corpus_sha") else None,
            "missing": (papers - int(docs)) if isinstance(docs, int) else None,
        }
    report["reranker"] = {"default": DEFAULT_RERANKER, "available": torch_installed()}
    return report
