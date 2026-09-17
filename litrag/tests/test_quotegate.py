"""The quote gate must be able to refuse.

A gate that passes everything is indistinguishable from no gate at all, and the
first live run passed 5 of 5 rows -- a good result, but not evidence that the
check works. These tests pin both directions: real quotes survive the normal
distortions of copying, and invented ones do not.
"""

import pytest

from litrag.extract import Citation, Row
from litrag.quotegate import (QUOTE_COLUMN, MIN_QUOTE_LEN, apply_gate, check,
                              strip_quote_column, with_quote_column)

PASSAGE = (
    "AbstractThe mutation S315T of the catalase-peroxidase (CP) protein KatG of "
    "Mycobacterium tuberculosis is the most common cause of isoniazid resistance "
    "in clinical isolates, and it retains most of the enzyme's catalase activity."
)


def row_with(quote, passage=PASSAGE):
    return Row(
        values={"Gene Name": "katG", "Mutation": "S315T", QUOTE_COLUMN: quote},
        citations=[Citation(marker=1, pmid="1", passage=passage)],
    )


# --- the gate accepts real quotes, including messily copied ones -------------

def test_exact_quote_passes():
    result = check("is the most common cause of isoniazid resistance", PASSAGE)
    assert result.ok and result.method == "exact"


def test_case_and_whitespace_differences_still_pass():
    """Models reflow whitespace and capitalisation when copying."""
    result = check("IS THE MOST   COMMON cause of\nisoniazid resistance", PASSAGE)
    assert result.ok and result.method == "normalized"


def test_smart_punctuation_still_passes():
    passage = "the enzyme’s catalase activity was retained in all isolates"
    result = check("the enzyme's catalase activity was retained", passage)
    assert result.ok


def test_small_wording_drift_passes_as_fuzzy():
    result = check("is the most common cause of isoniazid resistence", PASSAGE)
    assert result.ok and result.method == "fuzzy"
    assert result.score >= 0.82


# --- the gate refuses ---------------------------------------------------------

def test_invented_quote_is_refused():
    """The failure this whole feature exists to catch."""
    result = check(
        "S315T was shown to abolish catalase activity entirely in every isolate",
        PASSAGE,
    )
    assert not result.ok
    assert result.method == "mismatch"


def test_missing_quote_is_refused():
    assert check("", PASSAGE).method == "missing"


def test_quote_without_a_passage_is_refused():
    """No passage means nothing to check against -- refuse, never pass by default."""
    assert check("is the most common cause of isoniazid resistance", "").method == "no_passage"


def test_short_quote_is_refused_even_though_it_appears():
    """A fragment matches almost anything, so a pass would carry no information."""
    fragment = "KatG"
    assert fragment in PASSAGE
    assert len(fragment) < MIN_QUOTE_LEN
    assert check(fragment, PASSAGE).method == "too_short"


# --- row-level behaviour ------------------------------------------------------

def test_apply_gate_splits_kept_from_refused():
    kept, refused = apply_gate([
        row_with("is the most common cause of isoniazid resistance"),
        row_with("this sentence appears in no source whatsoever, anywhere at all"),
    ])
    assert len(kept) == 1
    assert len(refused) == 1
    assert refused[0]["omit_reason"] == "quote_rejected"
    # The refusal keeps the row's values so a reviewer can see what was dropped.
    assert refused[0]["values"]["Mutation"] == "S315T"


def test_a_row_passes_if_any_cited_passage_supports_it():
    """Rows legitimately cite several chunks and quote from one of them."""
    row = Row(
        values={"Mutation": "S315T", QUOTE_COLUMN: "retains most of the enzyme's catalase activity"},
        citations=[
            Citation(marker=1, pmid="1", passage="an unrelated passage about something else"),
            Citation(marker=2, pmid="2", passage=PASSAGE),
        ],
    )
    kept, refused = apply_gate([row])
    assert len(kept) == 1 and not refused


def test_gate_records_how_each_row_was_judged():
    row = row_with("is the most common cause of isoniazid resistance")
    apply_gate([row])
    assert row.provenance["quote_method"] == "exact"
    assert row.provenance["quote_score"] == 1.0


# --- template and column plumbing --------------------------------------------

def test_with_quote_column_adds_the_column_and_raises_the_token_budget(monkeypatch):
    from litrag.templates import Template

    base = Template(id="mutation", version=2, label="Mutation",
                    columns=["Gene Name", "Mutation", "Reference"],
                    max_output_tokens=2500)
    augmented = with_quote_column(base)
    assert augmented.columns[-1] == QUOTE_COLUMN
    # Measured: at 2500 tokens gpt56sol returned an empty answer with this column.
    assert augmented.max_output_tokens >= 8000
    assert base.columns[-1] != QUOTE_COLUMN, "the original must not be mutated"


def test_with_quote_column_is_idempotent():
    from litrag.templates import Template

    base = Template(id="mutation", version=2, label="Mutation",
                    columns=["Gene Name", QUOTE_COLUMN])
    assert with_quote_column(base) is base


def test_strip_quote_column_removes_it_from_rows_and_columns():
    """The quote is provenance; leaving it in columns would join the dedup key."""
    row = row_with("is the most common cause of isoniazid resistance")
    columns = strip_quote_column(["Gene Name", "Mutation", QUOTE_COLUMN], [row])
    assert QUOTE_COLUMN not in columns
    assert QUOTE_COLUMN not in row.values


# --- regression: provenance stamping must not erase the gate's record --------

def test_stamping_provenance_preserves_the_quote_gate_record():
    """prov.stamp() replaced row.provenance wholesale and silently dropped the
    quote, so the UI and the envelope saw no evidence the gate had ever run."""
    from litrag import provenance as prov

    row = row_with("is the most common cause of isoniazid resistance")
    apply_gate([row])
    assert row.provenance["quote_method"] == "exact"

    prov.stamp([row], {"model": "gpt56sol", "collection": "open-access"}, "q1")

    assert row.provenance["model"] == "gpt56sol", "the run record must land"
    assert row.provenance["quote_method"] == "exact", "the gate record must survive"
    assert row.provenance["quote"]


def test_citation_survives_a_round_trip_with_its_passage():
    """Batch resume restores rows from a sidecar via Row/Citation.from_dict.
    from_dict did not carry chunk_id, doc_id or passage, so a RESUMED run came
    back with citations but no evidence -- and the quote gate and the
    unsupported-claim axis silently had nothing to work with."""
    from litrag.extract import Citation

    original = Citation(marker=1, pmid="123", chunk_id="c-1", doc_id="d-1",
                        passage="the supporting sentence")
    restored = Citation.from_dict(original.to_dict())

    assert restored.pmid == "123"
    assert restored.chunk_id == "c-1"
    assert restored.doc_id == "d-1"
    assert restored.passage == "the supporting sentence"
