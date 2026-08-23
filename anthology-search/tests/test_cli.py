"""The CLI is the contract a harness depends on: JSON on stdout, meaning in the exit code."""

import json

import pytest

from mla.cli import EXIT_NO_INDEX, EXIT_OK, EXIT_REFUSED, EXIT_USAGE, main


@pytest.fixture
def run(index_path, data_dir, capsys):
    def _run(*argv):
        code = main(["--index", str(index_path), "--data", str(data_dir), *argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err
    return _run


def payload(out):
    return json.loads(out)


def test_search_emits_one_json_object(run):
    code, out, _ = run("search", "residual learning", "-n", "2")
    assert code == EXIT_OK
    body = payload(out)
    assert body["provider"] == "mla"
    assert body["hits"][0]["citekey"] == "he2016cvpr-deep"
    assert body["total"] >= 1
    assert len(body["corpus"]["sha"]) == 64      # every result names its corpus


def test_hit_shape_is_stable(run):
    _, out, _ = run("search", "residual", "-n", "1")
    hit = payload(out)["hits"][0]
    for field in ("paper_id", "citekey", "title", "authors", "year", "venue",
                  "venue_type", "doi", "url", "anthology_url", "pdf_url", "source"):
        assert field in hit, field
    assert hit["source"] == "mla"
    assert isinstance(hit["authors"], list)


def test_abstracts_are_opt_in(run):
    _, out, _ = run("search", "residual", "-n", "1")
    assert "abstract" not in payload(out)["hits"][0]
    _, out, _ = run("search", "residual", "-n", "1", "--abstracts")
    assert payload(out)["hits"][0]["abstract"]


def test_unknown_venue_is_refused_not_answered_empty(run):
    code, _, err = run("search", "x", "--venue", "nips")
    assert code == EXIT_REFUSED
    assert "nips" in err and "neurips" in err


def test_unknown_family_is_refused(run):
    code, _, err = run("search", "x", "--family", "quantum")
    assert code == EXIT_REFUSED and "MLA-FAMILY" in err


def test_family_expands_to_venues(run):
    _, out, _ = run("search", "learning", "--family", "theory", "-n", "5")
    assert "colt" in payload(out)["filters"]["venues"]


def test_family_skips_members_this_index_lacks(run):
    """Losing one member must not cancel the sweep — it is reported, not fatal."""
    _, out, _ = run("search", "learning", "--family", "theory", "-n", "5")
    body = payload(out)
    assert body["filters"]["venues"] == ["colt"]
    assert "isipta" in body["family_venues_absent"]


def test_family_with_no_members_present_is_refused(run):
    code, _, err = run("search", "x", "--family", "robotics")
    assert code == EXIT_REFUSED and "carries none of" in err


def test_inverted_year_span_is_a_usage_error(run):
    code, _, err = run("search", "x", "--since", "2020", "--until", "2010")
    assert code == EXIT_USAGE and "MLA-YEAR" in err


def test_empty_query_without_filters_is_a_usage_error(run):
    code, _, err = run("search", "")
    assert code == EXIT_USAGE and "at least one filter" in err


def test_filter_shutout_is_diagnosed(run):
    _, out, _ = run("search", "attention", "--venue", "colt")
    body = payload(out)
    assert body["total"] == 0
    assert body["unfiltered_total"] == 2
    assert "filters cut this" in body["note"]


def test_no_workshops_drops_workshop_papers(run):
    _, out, _ = run("search", "attention", "--no-workshops")
    assert "tyukin2024icmlw-attention" not in [h["citekey"] for h in payload(out)["hits"]]


def test_get_reports_absence_without_failing(run):
    code, out, _ = run("get", "not-a-key")
    assert code == EXIT_OK and payload(out)["found"] is False


def test_doi_resolves(run):
    _, out, _ = run("doi", "10.1109/CVPR.2016.90")
    assert payload(out)["paper"]["citekey"] == "he2016cvpr-deep"


def test_bibtex_uses_the_canonical_venue_name(run):
    code, out, _ = run("bibtex", "vaswani2017neurips-attention")
    assert code == EXIT_OK
    assert out.startswith("@inproceedings{vaswani2017neurips-attention,")
    assert "Conference on Neural Information Processing Systems" in out
    assert "{{Attention is All you Need}}" in out       # braced against case-mangling


def test_bibtex_missing_key_is_refused(run):
    code, _, err = run("bibtex", "not-a-key")
    assert code == EXIT_REFUSED and "not-a-key" in err


def test_bibtex_partial_success_still_emits(run):
    code, out, err = run("bibtex", "he2016cvpr-deep", "not-a-key")
    assert code == EXIT_OK
    assert "he2016cvpr-deep" in out and "not-a-key" in err


def test_venues_lists_families_and_counts(run):
    _, out, _ = run("venues")
    body = payload(out)
    assert body["count"] >= 4
    assert {"core-ml", "theory", "vision"} <= set(body["families"])
    colt = next(v for v in body["venues"] if v["venue"] == "colt")
    assert colt["family"] == "theory"
    assert colt["venue_name"] == "Annual Conference on Learning Theory"


def test_stats_publishes_sparse_field_coverage(run):
    _, out, _ = run("stats")
    coverage = payload(out)["field_coverage"]
    assert coverage["code_url"]["papers"] == 1
    assert 0 <= coverage["code_url"]["pct"] <= 100


def test_doctor_is_green_on_a_fresh_index(run):
    code, out, _ = run("doctor")
    assert code == EXIT_OK and payload(out)["ok"] is True


def test_doctor_reports_a_missing_index(tmp_path, capsys):
    code = main(["--index", str(tmp_path / "gone.sqlite3"), "doctor"])
    body = json.loads(capsys.readouterr().out)
    assert code == EXIT_NO_INDEX
    assert body["reason"] == "index-missing" and body["fix"] == "mla index"


def test_search_against_a_missing_index_exits_no_index(tmp_path, capsys):
    code = main(["--index", str(tmp_path / "gone.sqlite3"), "search", "x"])
    assert code == EXIT_NO_INDEX
    assert "mla index" in capsys.readouterr().err
