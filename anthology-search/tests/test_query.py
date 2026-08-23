import pytest

from mla.query import Filters, QueryError, by_doi, corpus_stamp, get, search


def keys(result):
    return [hit["citekey"] for hit in result.hits]


def test_basic_search_finds_the_paper(db):
    assert "he2016cvpr-deep" in keys(search(db, "residual learning"))


def test_exact_title_outranks_the_near_miss(db):
    # Both titles contain every term; the one the caller actually named must win.
    assert keys(search(db, "attention is all you need", limit=2))[0] == \
        "vaswani2017neurips-attention"


def test_abstracts_are_searched_not_just_titles(db):
    assert "he2016cvpr-deep" in keys(search(db, "difficult to train"))


def test_field_restriction_excludes_abstract_matches(db):
    assert keys(search(db, "difficult to train", fields=("title",))) == []


def test_venue_filter(db):
    assert keys(search(db, "attention", filters=Filters(venues=("neurips",)))) == \
        ["vaswani2017neurips-attention"]


def test_year_span_filter(db):
    result = search(db, "learn*", filters=Filters(year_min=1984, year_max=1988))
    assert set(keys(result)) == {"angluin1988colt-learning", "valiant1984colt-theory"}


def test_the_index_does_not_stem(db):
    """A deliberate property, not an oversight.

    `learning` does not match `learnable`. Stemming would quietly widen a recorded query
    beyond what it says, and every query here is meant to be reproducible from the text
    that was logged. The prefix operator is the explicit remedy, and the caller chooses
    when to reach for it.
    """
    assert "valiant1984colt-theory" not in keys(search(db, "learning"))
    assert "valiant1984colt-theory" in keys(search(db, "learn*"))


def test_venue_type_filter_drops_workshops(db):
    result = search(db, "attention", filters=Filters(venue_types=("conference",)))
    assert "tyukin2024icmlw-attention" not in keys(result)


def test_author_slug_filter(db):
    assert keys(search(db, "", limit=5, filters=Filters(author="he-kaiming"))) == \
        ["he2016cvpr-deep"]


def test_author_filter_ignores_diacritics(db):
    assert keys(search(db, "", filters=Filters(author="scholkopf"))) == \
        ["scholkopf2001neurips-kernel"]


def test_has_code_filter(db):
    assert keys(search(db, "", limit=9, filters=Filters(has_code=True))) == \
        ["he2016cvpr-deep"]


def test_browse_is_newest_first(db):
    result = search(db, "", limit=9, filters=Filters(venues=("colt",)))
    assert keys(result) == ["angluin1988colt-learning", "valiant1984colt-theory"]
    assert "score" not in result.hits[0]     # nothing was ranked, so nothing claims a score


def test_empty_query_without_filters_is_refused(db):
    with pytest.raises(QueryError, match="at least one filter"):
        search(db, "")


def test_total_counts_beyond_the_page(db):
    result = search(db, "learning", limit=1)
    assert len(result.hits) == 1
    assert result.total > 1


def test_zero_with_filters_reports_the_unfiltered_count(db):
    result = search(db, "attention", filters=Filters(venues=("colt",)))
    assert result.total == 0
    assert result.unfiltered_total == 2      # the filter did this, not the query


def test_zero_without_filters_has_no_diagnosis(db):
    assert search(db, "zzzznotaword").unfiltered_total is None


def test_dropped_terms_are_surfaced(db):
    assert search(db, "α attention").dropped_terms == ["α"]


def test_any_mode_widens(db):
    narrow = search(db, "residual quantum", mode="all")
    wide = search(db, "residual quantum", mode="any")
    assert narrow.total == 0 and wide.total >= 1


def test_get_and_missing_get(db):
    assert get(db, "he2016cvpr-deep")["year"] == 2016
    assert get(db, "no-such-key") is None


def test_by_doi_accepts_bare_and_url_forms(db):
    for form in ("10.1109/CVPR.2016.90", "https://doi.org/10.1109/CVPR.2016.90",
                 "10.1109/cvpr.2016.90"):
        assert by_doi(db, form)["citekey"] == "he2016cvpr-deep"
    assert by_doi(db, "10.0000/nope") is None


def test_hit_urls_prefer_the_publisher_record(db):
    hit = get(db, "vaswani2017neurips-attention")
    assert hit["url"] == "https://example.invalid/a.pdf"        # venue_url empty -> pdf
    assert hit["anthology_url"].endswith("/neurips/2017/vaswani2017neurips-attention")


def test_limits_are_bounded(db):
    for bad in (0, -1, 10_000):
        with pytest.raises(QueryError):
            search(db, "learning", limit=bad)
    with pytest.raises(QueryError):
        search(db, "learning", offset=-1)


def test_unknown_venue_type_is_refused(db):
    with pytest.raises(QueryError, match="unknown venue type"):
        search(db, "learning", filters=Filters(venue_types=("keynote",)))


def test_unknown_field_is_refused(db):
    with pytest.raises(QueryError, match="unknown search field"):
        search(db, "learning", fields=("fulltext",))


def test_corpus_stamp_names_the_bytes(db):
    stamp = corpus_stamp(db)
    assert len(stamp["sha"]) == 64 and stamp["papers"] == 6
