import pytest

from mla.text import compile_match, fold


@pytest.mark.parametrize("raw,expected", [
    ("Schölkopf", "scholkopf"),
    ("ENTRY-WISE", "entry-wise"),
    ("Ł", "l"),
])
def test_fold_strips_case_and_diacritics(raw, expected):
    assert fold(raw) == expected


def test_bare_terms_are_quoted_and_anded():
    assert compile_match("entrywise eigenvector")[0] == '"entrywise" AND "eigenvector"'


def test_any_mode_ors():
    assert compile_match("a1 b2", "any")[0] == '"a1" OR "b2"'


def test_quoted_phrase_survives_as_one_term():
    expression, dropped = compile_match('"leave-one-out" bound')
    assert expression == '"leave one out" AND "bound"'
    assert dropped == []


def test_explicit_operator_is_not_double_joined():
    assert compile_match("neural OR network")[0] == '"neural" OR "network"'


def test_prefix_star_is_preserved():
    assert compile_match("eigen* value")[0] == '"eigen"* AND "value"'


def test_punctuation_can_never_reach_fts_syntax():
    # Any of these raw would be an `fts5: syntax error`.
    for hostile in ('a:b', 'NEAR(', 'x"y', "(paren)", "^caret", "a-b*c"):
        expression, _ = compile_match(hostile)
        assert '"' in expression or expression == ""
        assert not expression.strip().endswith(":")


def test_unindexable_terms_are_reported_not_swallowed():
    expression, dropped = compile_match("ℓ∞ eigenvector")
    assert expression == '"eigenvector"'
    assert dropped == ["ℓ∞"]


def test_all_terms_unindexable_yields_empty_match():
    expression, dropped = compile_match("α β")
    assert expression == ""
    assert dropped == ["α", "β"]


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        compile_match("x", "sometimes")
