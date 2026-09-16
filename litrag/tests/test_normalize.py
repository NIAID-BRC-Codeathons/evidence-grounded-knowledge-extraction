"""Notation normalization -- the basis for collapsing duplicate facts."""

import pytest

from litrag.normalize import clean, is_null, normalize_gene, normalize_mutation


@pytest.mark.parametrize("value", [
    "S315T", "Ser315Thr", "katG-S315T", "p.Ser315Thr",
    "Ser315→Thr", "Ser315->Thr", "katG S315T", "ser315thr",
])
def test_amino_acid_notations_agree(value):
    """One katG fact appeared under five spellings in a single live response."""
    assert normalize_mutation(value, "katG") == "S315T"


def test_promoter_sign_is_preserved():
    """-15 and 15 are different sites; the minus must not be eaten."""
    assert normalize_mutation("c-15t", "inhA") != normalize_mutation("c15t", "inhA")
    assert normalize_mutation("c-15t", "inhA") == "n-15T"


@pytest.mark.parametrize("value", ["c-15t", "c.-15C>T", "c.-15T", "C-15T"])
def test_nucleotide_notations_agree(value):
    """The reference base is redundant -- position fixes it within a gene -- so
    spellings that omit it still have to match the ones that do not."""
    assert normalize_mutation(value, "inhA") == "n-15T"


def test_case_and_sign_disambiguate_nucleotide_from_amino_acid():
    """C15T is Cys15Thr; c15t is a base change. A negative position is always
    nucleotide, since no protein has residue -15."""
    assert normalize_mutation("C15T", "katG") == "C15T"
    assert normalize_mutation("c15t", "katG") == "n15T"
    assert normalize_mutation("C-15T", "inhA") == "n-15T"


@pytest.mark.parametrize("value", [
    "Ser315Thr (S315T)", "S315T (Ser315Thr)", "katG Ser315Thr [S315T]",
])
def test_parenthetical_gloss_is_stripped(value):
    """Qwen routinely restates the notation in parentheses; without stripping,
    the gloss defeats every pattern and the row never merges with its twin."""
    assert normalize_mutation(value, "katG") == "S315T"


def test_gloss_on_unparseable_text_is_kept():
    """Only strip a gloss when what remains actually parses."""
    assert "partial" in normalize_mutation("Deletion (partial)", "katG")


def test_distinct_mutations_stay_distinct():
    assert normalize_mutation("S315T", "katG") != normalize_mutation("S315N", "katG")
    assert normalize_mutation("Ser94Ala", "inhA") == "S94A"


def test_unparseable_mutation_falls_back_to_text():
    assert normalize_mutation("katG deletion", "katG") == "deletion"
    assert normalize_mutation("Deletion", "katG") == normalize_mutation("deletion", "katG")


def test_gene_decorations_are_stripped():
    assert normalize_gene("katG") == normalize_gene("katG gene") == normalize_gene("KatG protein")


@pytest.mark.parametrize("value", ["N/A", "n/a", "NA", "-", "", "  ", "None", "not reported"])
def test_null_spellings(value):
    assert is_null(value)
    assert clean(value) == ""


def test_real_value_is_not_null():
    assert not is_null("isoniazid resistance")
