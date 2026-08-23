"""Text folding and the FTS5 query compiler.

A scout's query is free text written by a model: it carries hyphens, colons, maths
symbols and stray quotes. Handing that to SQLite's ``MATCH`` raw is how you get an
``fts5: syntax error near ":"`` mid-sweep. Everything a caller types is therefore
compiled into a guaranteed-valid MATCH expression here, and any term the tokenizer
cannot index is reported back rather than silently swallowed — a query that quietly
lost half its terms looks like a thin field, which is the wrong diagnosis.
"""

from __future__ import annotations

import re
import unicodedata

# Bare words FTS5 reads as operators; anything else we quote.
OPERATORS = frozenset({"AND", "OR", "NOT", "NEAR"})
_TOKEN_RE = re.compile(r'"[^"]*"|\S+')
_ALNUM_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Latin letters NFKD will not take apart, because the mark is a stroke or a ligature
# rather than a combining accent. Without these, `--author lukasz` misses Łukasz Kaiser
# and `--author sorensen` misses Sørensen.
_TRANSLITERATE = str.maketrans({
    "ł": "l", "Ł": "l", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ð": "d", "Ð": "d",
    "ħ": "h", "Ħ": "h", "ı": "i", "İ": "i", "ŀ": "l", "Ŀ": "l", "ĸ": "k",
    "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ß": "ss", "þ": "th", "Þ": "th",
})


def fold(value: str) -> str:
    """Lowercase and strip diacritics, matching the index's ``unicode61`` tokenizer.

    ``Schölkopf`` and ``Scholkopf`` fold together, so an author filter typed either way
    finds the same papers. The index stores author names pre-folded with this same
    function, so the two sides can never drift apart.
    """
    decomposed = unicodedata.normalize("NFKD", value.translate(_TRANSLITERATE))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


def _indexable(term: str) -> str:
    """The part of a term the tokenizer will actually index, or "" if none of it is."""
    return _ALNUM_RE.sub(" ", fold(term)).strip()


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def compile_match(query: str, mode: str = "all") -> tuple[str, list[str]]:
    """Free text -> ``(match_expression, dropped_terms)``.

    Supported by design, because a search ladder needs them: ``"quoted phrases"``,
    the bare operators ``AND`` / ``OR`` / ``NOT``, parentheses, and a trailing ``*``
    for prefix matching (``eigen*``). Everything else is quoted literally and is
    therefore always safe.

    ``mode`` picks the operator joining adjacent bare terms — ``all`` (AND, precise)
    or ``any`` (OR, the widening rung).

    A term the tokenizer would reduce to less than two characters is dropped, not
    searched: ``ℓ∞`` NFKD-folds to a bare ``l``, and a one-letter term matches most of
    the corpus. Callers get it back in ``dropped`` so they can retry with the spelled
    form (``linf``, ``sup-norm``) instead of misreading the flood.
    """
    if mode not in ("all", "any"):
        raise ValueError(f"MLA-QUERY: mode must be 'all' or 'any', got {mode!r}")
    joiner = " AND " if mode == "all" else " OR "

    parts: list[str] = []
    dropped: list[str] = []
    explicit_operator = False

    for token in _TOKEN_RE.findall(query or ""):
        if token in OPERATORS:
            parts.append(token)
            explicit_operator = True
            continue
        prefix = token.endswith("*") and not token.startswith('"')
        bare = token[:-1] if prefix else token
        bare = bare[1:-1] if bare.startswith('"') and bare.endswith('"') else bare

        if len(indexable := _indexable(bare)) < 2:
            dropped.append(token)
            continue
        parts.append(_quote(indexable) + ("*" if prefix else ""))

    if not parts:
        return "", dropped

    # An explicit AND/OR/NOT means the caller punctuated the query themselves; joining
    # again would produce `"a" AND OR "b"`. Otherwise apply the mode's joiner.
    expression = " ".join(parts) if explicit_operator else joiner.join(parts)
    return expression, dropped
