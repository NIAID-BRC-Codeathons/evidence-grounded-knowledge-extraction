"""Cite or refuse: require a verbatim quote, and drop rows whose quote is not there.

The project's rule is that every emitted claim carries the exact passage that
supports it, or is not emitted at all. Retaining the passage (see extract.Citation)
makes that checkable; this module does the checking.

Deterministic on purpose -- no model judges another model's output here. A fuzzy
string match cannot be wrong in an interesting way, which is exactly what is wanted
for a gate: it answers "did the model copy this from the source, or invent it",
not "is this claim true". Those are different questions and only the first is
cheap and reliable. Entailment (does the passage actually *support* the claim) is
a separate, harder check.

The ladder is taken from experiment-01/verify.py, which measured these thresholds
against real extractions:

    exact substring  ->  normalized (case, whitespace, unicode punctuation)
                     ->  difflib ratio over a sliding window

A quote shorter than MIN_QUOTE_LEN is rejected without matching: a three-word
fragment matches almost any passage by accident, so a pass would mean nothing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .templates import Template

QUOTE_COLUMN = "Quote"

# Below this many characters a match carries no information.
MIN_QUOTE_LEN = 24
# Ratio at or above which a windowed difflib match counts as the same text.
FUZZY_THRESHOLD = 0.82

_WHITESPACE = re.compile(r"\s+")
# Smart quotes, dashes and ellipses differ between the model's output and the
# source text far more often than the words do.
_PUNCTUATION = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}

QUOTE_RULE = (
    'In the Quote column, copy one sentence VERBATIM from the source that states '
    'this finding. Copy it character for character, do not paraphrase, summarise, '
    'translate or shorten it. If no single sentence in the sources states the '
    'finding, leave the row out entirely rather than inventing a quote.'
)


@dataclass(frozen=True)
class QuoteCheck:
    """Why a quote passed or failed."""

    ok: bool
    method: str          # exact | normalized | fuzzy | too_short | missing | no_passage
    score: float = 0.0

    @property
    def reason(self) -> str:
        return "" if self.ok else f"quote_{self.method}"


# A full sentence per row is far more output than the other columns combined, and
# a reasoning model spends its budget thinking before it writes anything. Measured:
# with the declaration's 2500-token cap, gpt56sol returned a COMPLETELY EMPTY answer
# for a 6-source katG query with the quote column on -- not a truncated table, zero
# characters. The budget has to grow with the column or the gate silently produces
# nothing, which looks identical to "the model refused every row".
QUOTE_TOKEN_MULTIPLIER = 3
QUOTE_TOKEN_FLOOR = 8000


def with_quote_column(template: Template) -> Template:
    """A copy of the template that also asks for a verbatim quote.

    Derived rather than declared, so the gate works for any template -- including
    the local charter types -- without a second place to register columns.
    """
    columns = list(template.columns or [])
    if QUOTE_COLUMN in columns:
        return template
    base = template.max_output_tokens or 2500
    return replace(
        template,
        columns=columns + [QUOTE_COLUMN],
        guidance=list(template.guidance or []) + [QUOTE_RULE],
        max_output_tokens=max(base * QUOTE_TOKEN_MULTIPLIER, QUOTE_TOKEN_FLOOR),
    )


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    for source, target in _PUNCTUATION.items():
        text = text.replace(source, target)
    return _WHITESPACE.sub(" ", text).strip().lower()


def _fuzzy_best(needle: str, haystack: str) -> float:
    """Best difflib ratio for `needle` against any same-length window of `haystack`.

    Comparing against the whole passage would drown a one-sentence quote in a
    2000-character chunk, so the window tracks the quote's length. The step is a
    quarter of the window: fine enough not to straddle every candidate match,
    coarse enough to stay linear in practice.
    """
    if not needle or not haystack:
        return 0.0
    window = len(needle)
    if window >= len(haystack):
        return SequenceMatcher(None, needle, haystack).ratio()

    step = max(1, window // 4)
    best = 0.0
    for start in range(0, len(haystack) - window + 1, step):
        ratio = SequenceMatcher(None, needle, haystack[start:start + window]).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.995:
                break
    return best


def check(quote: str, passage: str) -> QuoteCheck:
    """Is `quote` actually present in `passage`?"""
    quote = (quote or "").strip().strip('"').strip()
    if not quote:
        return QuoteCheck(False, "missing")
    if not passage:
        return QuoteCheck(False, "no_passage")
    if len(quote) < MIN_QUOTE_LEN:
        return QuoteCheck(False, "too_short")

    if quote in passage:
        return QuoteCheck(True, "exact", 1.0)

    normal_quote, normal_passage = normalize(quote), normalize(passage)
    if normal_quote in normal_passage:
        return QuoteCheck(True, "normalized", 1.0)

    score = _fuzzy_best(normal_quote, normal_passage)
    if score >= FUZZY_THRESHOLD:
        return QuoteCheck(True, "fuzzy", score)
    return QuoteCheck(False, "mismatch", score)


def _row_passages(row) -> List[str]:
    return [c.passage for c in row.citations if c.passage]


def apply_gate(rows: Sequence[Any]) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Split rows into those whose quote checks out and those refused.

    A row is kept if its quote matches ANY of its cited passages -- a row can
    legitimately cite several chunks and quote from one of them. The check and
    the passage it matched are recorded on the row, so a reviewer can see why a
    row survived rather than taking the gate's word for it.
    """
    kept: List[Any] = []
    refused: List[Dict[str, Any]] = []

    for row in rows:
        quote = row.values.get(QUOTE_COLUMN, "")
        passages = _row_passages(row)

        best = QuoteCheck(False, "missing") if quote else QuoteCheck(False, "missing")
        if quote and not passages:
            best = QuoteCheck(False, "no_passage")
        for passage in passages:
            result = check(quote, passage)
            if result.ok:
                best = result
                break
            if result.score > best.score:
                best = result

        row.provenance["quote"] = quote
        row.provenance["quote_method"] = best.method
        row.provenance["quote_score"] = round(best.score, 3)

        if best.ok:
            if best.method != "exact":
                row.flags.append(f"quote_{best.method}")
            kept.append(row)
        else:
            refused.append({
                "outcome": "omit",
                "omit_reason": "quote_rejected",
                "detail": best.method,
                "score": round(best.score, 3),
                "quote": quote,
                "values": dict(row.values),
            })

    return kept, refused


def strip_quote_column(columns: Sequence[str], rows: Sequence[Any]) -> List[str]:
    """Remove the Quote column from the output shape after the gate has used it.

    The quote is provenance, not a finding, and leaving it in the columns would
    make it part of the dedup identity key -- two rows stating the same fact with
    different supporting sentences would stop merging.
    """
    for row in rows:
        row.values.pop(QUOTE_COLUMN, None)
        row.variants.pop(QUOTE_COLUMN, None)
    return [c for c in columns if c != QUOTE_COLUMN]
