"""The hybrid lanes (0.2.0, D-107) on the synthetic corpus, with the `hashed` backend so no
model is downloaded and no GPU is touched. Skipped wholesale when the `hybrid` extra is absent."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("bm25s")
pytest.importorskip("Stemmer")

from mla import hybrid  # noqa: E402
from mla.cli import main  # noqa: E402
from mla.index import build, connect  # noqa: E402

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture(scope="module")
def lane_index(tmp_path_factory, data_dir) -> Path:
    """A private index (the lanes live beside it) so the FTS fixtures stay untouched."""
    target = tmp_path_factory.mktemp("lanes") / "mla.sqlite3"
    build(data_dir=data_dir, index_path=target)
    connection = connect(target)
    try:
        hybrid.build_bm25(connection, target)
        hybrid.build_vec(connection, target, "hashed", backend="hashed", batch=4)
        hybrid.set_default_dense(target, "hashed")
    finally:
        connection.close()
    return target


def run_cli(*args: str, stdin: str | None = None, index: Path | None = None,
            data: Path | None = None) -> tuple[int, dict | str, str]:
    command = [sys.executable, "-c", "import sys; sys.path.insert(0, %r); from mla.cli import main; sys.exit(main())" % SRC]
    if index is not None:
        command += ["--index", str(index)]
    if data is not None:
        command += ["--data", str(data)]
    done = subprocess.run(command + list(args), input=stdin, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "MLA_DEVICE": "cpu", "HOME": str(Path.home())})
    try:
        payload = json.loads(done.stdout)
    except ValueError:
        payload = done.stdout
    return done.returncode, payload, done.stderr


# --- lanes ---------------------------------------------------------------------------------------

def test_bm25_lane_stems_and_reports_unknown_terms(lane_index):
    connection = connect(lane_index)
    lane = hybrid.Bm25Lane.load(lane_index)
    scores, unknown = lane.scores("learnable concepts zzzunseen")
    assert unknown == ["zzzunseen"]
    # `learnable` stems to `learn`, so the BM25 lane finds "Learning With Hints" — the exact
    # thing the FTS lane (no stemming, pinned by test_query) deliberately does not.
    top = [lane.ids[row] for row in hybrid.top_positions(scores, None, 3, positive_only=True)]
    assert any("learning" in key or "valiant" in key for key in top), top
    assert lane.meta["docs"] == len(lane.ids) == int(dict(
        connection.execute("SELECT key, value FROM meta"))["papers"])
    connection.close()


def test_vec_lane_is_incremental_by_citekey(lane_index, tmp_path):
    connection = connect(lane_index)
    first = hybrid.build_vec(connection, lane_index, "hashed", backend="hashed", batch=4)
    assert first["incremental"]["embedded"] == 0 and first["incremental"]["kept"] == first["docs"]
    # Drop one vector on disk and rebuild: exactly one paper is embedded again.
    npy, ids_path, _ = hybrid.vec_paths(lane_index, "hashed")
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    import numpy as np
    matrix = np.load(npy)
    np.save(npy, matrix[1:])
    ids_path.write_text(json.dumps(ids[1:]), encoding="utf-8")
    second = hybrid.build_vec(connection, lane_index, "hashed", backend="hashed", batch=4)
    assert second["incremental"] == {"embedded": 1, "kept": len(ids) - 1, "removed": 0}
    assert json.loads(ids_path.read_text(encoding="utf-8")) == ids
    assert np.load(npy).shape[0] == len(ids)
    connection.close()


def test_guarded_rrf_seats_every_list_first():
    a = ["a1", "a2", "a3", "shared", "a5", "a6"]
    b = ["b1", "b2", "b3", "b4", "b5", "shared"]
    fused = [key for key, _ in hybrid.guarded_rrf([a, b], seats=2)]
    assert fused[:4] == ["a1", "b1", "a2", "b2"]           # round-robin seats, list order
    assert set(fused) == set(a) | set(b)
    # After the seats, the doc both lists agree on outranks the ones only one list saw.
    assert fused.index("shared") < fused.index("a5")
    assert fused.index("shared") < fused.index("b4")


def test_filter_mask_is_applied_inside_the_lanes(lane_index):
    connection = connect(lane_index)
    engine = hybrid.Engine(connection, lane_index)
    from mla.query import Filters
    answer = engine.search("learning", mode="bm25", limit=10, filters=Filters(venues=("colt",)))
    assert answer["hits"], "the stemmed lane finds COLT's learning papers"
    assert {hit["venue"] for hit in answer["hits"]} == {"colt"}
    assert answer["total"] == len(answer["hits"])
    everything = engine.search("learning", mode="bm25", limit=10)
    assert everything["total"] >= answer["total"]
    connection.close()


def test_hybrid_search_carries_lane_ranks_and_totals(lane_index):
    connection = connect(lane_index)
    engine = hybrid.Engine(connection, lane_index)
    answer = engine.search("kernel methods support vectors", mode="hybrid", limit=5)
    assert answer["mode"] == "hybrid" and answer["dense"] == "hashed"
    assert set(answer["totals"]) == {"bm25", "dense"} and answer["totals"]["dense"] is None
    assert answer["hits"][0]["citekey"].startswith("scholkopf")
    for hit in answer["hits"]:
        assert set(hit["lanes"]) == {"bm25", "dense"}
        assert "score" in hit
    connection.close()


def test_rerank_pool_resolves_citekeys_and_reports_textless_entries(lane_index):
    connection = connect(lane_index)
    reranker = hybrid.Reranker("hashed")
    verdict = hybrid.rerank_pool(connection, reranker, "kernel methods", [
        {"id": "scholkopf2001neurips-kernel", "citekey": "scholkopf2001neurips-kernel"},
        {"id": "foreign-1", "text": "Kernel methods for structured data"},
        {"id": "foreign-2"},
    ])
    assert verdict["scored"] == 2 and verdict["skipped"] == ["foreign-2"]
    assert [row["id"] for row in verdict["ranked"]][0] in {"scholkopf2001neurips-kernel", "foreign-1"}
    connection.close()


# --- the CLI contract ---------------------------------------------------------------------------

def test_search_default_mode_is_the_untouched_fts_contract(lane_index):
    code, payload, _ = run_cli("search", "kernels", index=lane_index)
    assert code == 0 and payload["mode"] == "all" and "lanes" not in payload["hits"][0]
    assert "dense" not in payload


def test_search_hybrid_mode_keeps_the_harness_keys(lane_index):
    code, payload, err = run_cli("search", "learnable concepts", "--mode", "hybrid", "-n", "3", index=lane_index)
    assert code == 0, err
    for key in ("provider", "query", "match", "total", "returned", "corpus", "hits", "mode", "dense", "totals"):
        assert key in payload, key
    assert payload["mode"] == "hybrid" and payload["dense"] == "hashed" and payload["returned"] <= 3
    assert payload["hits"][0]["lanes"]
    assert "learnabl" in payload.get("unknown_terms", []) or payload["total"] >= 1


def test_search_hybrid_refuses_a_typo_venue_before_loading_lanes(lane_index):
    code, _, err = run_cli("search", "learning", "--mode", "hybrid", "--venue", "nips", index=lane_index)
    assert code == 3 and "MLA-VENUE" in err


def test_missing_lane_is_exit_4_with_the_fix(tmp_path, data_dir):
    bare = tmp_path / "bare.sqlite3"
    build(data_dir=data_dir, index_path=bare)
    code, _, err = run_cli("search", "learning", "--mode", "bm25", index=bare)
    assert code == 4 and "mla bm25 build" in err
    code, _, err = run_cli("search", "learning", "--mode", "dense", index=bare)
    assert code == 4 and "mla vec build" in err


def test_batch_answers_many_queries_in_one_process(lane_index):
    spec = {"queries": [
        {"id": "brief", "query": "kernel methods and support vectors", "limit": 3},
        {"id": "colt", "query": "learning theory", "mode": "bm25", "limit": 5, "filters": {"venues": ["colt"]}},
    ]}
    code, payload, err = run_cli("batch", stdin=json.dumps(spec), index=lane_index)
    assert code == 0, err
    assert payload["batch"] is True and payload["queries"] == 2 and payload["dense"] == "hashed"
    by_id = {row["id"]: row for row in payload["results"]}
    assert by_id["brief"]["mode"] == "hybrid" and by_id["brief"]["hits"][0]["citekey"].startswith("scholkopf")
    assert by_id["colt"]["filters"] == {"venues": ["colt"]}
    assert all(hit["venue"] == "colt" for hit in by_id["colt"]["hits"])


def test_batch_refuses_the_whole_ladder_on_a_bad_filter(lane_index):
    spec = {"queries": [{"id": "ok", "query": "learning"}, {"id": "bad", "query": "learning", "filters": {"venues": ["nips"]}}]}
    code, _, err = run_cli("batch", stdin=json.dumps(spec), index=lane_index)
    assert code == 3 and "MLA-VENUE" in err
    spec = {"queries": [{"id": "bad", "query": "learning", "filters": {"venue": "colt"}}]}
    code, _, err = run_cli("batch", stdin=json.dumps(spec), index=lane_index)
    assert code == 3 and "MLA-FILTER" in err


def test_rerank_cli_scores_a_pool(lane_index):
    pool = {"pool": [{"id": "scholkopf2001neurips-kernel", "citekey": "scholkopf2001neurips-kernel"},
                     {"id": "x", "text": "Learning with hints"}]}
    code, payload, err = run_cli("rerank", "--query", "kernel methods", "--model", "hashed",
                                 stdin=json.dumps(pool), index=lane_index)
    assert code == 0, err
    assert payload["model"] == "hashed" and payload["scored"] == 2 and len(payload["ranked"]) == 2


def test_doctor_and_stats_report_the_lanes_without_moving_ok(lane_index, data_dir):
    code, payload, _ = run_cli("doctor", index=lane_index, data=data_dir)
    assert code == 0 and payload["ok"] is True
    lanes = payload["lanes"]
    assert lanes["bm25"]["present"] is True and lanes["bm25"]["stale"] is False
    assert lanes["vec"]["default"] == "hashed" and lanes["vec"]["models"]["hashed"]["missing"] == 0
    code, payload, _ = run_cli("stats", index=lane_index, data=data_dir)
    assert code == 0 and payload["lanes"]["vec"]["default"] == "hashed"


def test_vec_default_refuses_an_unbuilt_lane(lane_index):
    code, _, err = run_cli("vec", "default", "qwen3-8b", index=lane_index)
    assert code == 4 and "mla vec build --model qwen3-8b" in err
