import gzip
import json

import pytest

from mla.corpus import CorpusError, corpus_sha, normalize, read, source_files
from mla.index import IndexUnavailable, build, connect, meta, stale


def test_build_reads_both_container_shapes(db):
    assert db.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 6
    venues = {row[0] for row in db.execute("SELECT DISTINCT venue FROM papers")}
    assert {"neurips", "cvpr", "icmlw", "colt"} <= venues


def test_meta_records_provenance(db):
    info = meta(db)
    assert len(info["corpus_sha"]) == 64
    assert info["papers"] == "6"
    assert info["year_min"] == "1984" and info["year_max"] == "2024"
    assert info["schema_version"] == "1"


def test_corpus_sha_is_stable_and_content_addressed(data_dir, tmp_path):
    first = corpus_sha(source_files(data_dir))
    assert first == corpus_sha(source_files(data_dir))
    extra = data_dir / "papers" / "zz-2025.json.gz"
    with gzip.open(extra, "wt", encoding="utf-8") as fh:
        json.dump({"venue": "zz", "year": "2025", "papers": []}, fh)
    try:
        assert corpus_sha(source_files(data_dir)) != first
    finally:
        extra.unlink()
    assert corpus_sha(source_files(data_dir)) == first


def test_stale_tracks_the_corpus(db, data_dir):
    assert stale(db, data_dir) is False


def test_venues_table_is_summarised(db):
    row = db.execute("SELECT * FROM venues WHERE venue='colt'").fetchone()
    assert row["papers"] == 2
    assert (row["year_min"], row["year_max"]) == (1984, 1988)
    assert row["abstracts"] == 1        # angluin1988 has no abstract upstream


def test_missing_index_is_an_actionable_error(tmp_path):
    with pytest.raises(IndexUnavailable, match="mla index"):
        connect(tmp_path / "absent.sqlite3")


def test_missing_corpus_fails_closed(tmp_path):
    with pytest.raises(CorpusError):
        list(read(tmp_path / "nowhere"))


def test_empty_corpus_dir_fails_closed(tmp_path):
    (tmp_path / "papers").mkdir()
    with pytest.raises(CorpusError):
        source_files(tmp_path)


def test_rows_without_a_citekey_or_title_are_dropped():
    assert normalize({"bibtex_key": "", "title": "x"}) is None
    assert normalize({"bibtex_key": "k", "title": ""}) is None


def test_build_is_atomic(tmp_path, data_dir):
    target = tmp_path / "out.sqlite3"
    build(data_dir=data_dir, index_path=target)
    assert target.is_file()
    assert not target.with_suffix(".building").exists()
