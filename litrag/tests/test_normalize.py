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


# -- prose written out in words ------------------------------------------

@pytest.mark.parametrize("value,expected", [
    # The three from the dengue vaccine paper that prompted this.
    ("NS1-53 glycine to aspartate", "G53D"),
    ("NS3-250 glutamate to valine", "E250V"),
    ("5' UTR-57, C to T", "n57T"),
    # Other phrasings papers use.
    ("position 315 serine to threonine", "S315T"),
    ("codon 315, serine to threonine", "S315T"),
    ("glycine 53 to aspartate", "G53D"),
    ("glycine-53 to aspartate", "G53D"),
    ("serine to threonine at position 315", "S315T"),
    ("57 cytosine to thymine", "n57T"),
    ("aspartic acid 180 to alanine", "D180A"),
])
def test_prose_converts_to_standard_notation(value, expected):
    from litrag.normalize import standard_notation
    assert standard_notation(value) == expected


def test_already_standard_values_return_themselves():
    """Lets a caller compare written against standard instead of special-casing."""
    from litrag.normalize import standard_notation
    assert standard_notation("S315T", "katG") == "S315T"
    assert standard_notation("Ser315Thr", "katG") == "S315T"
    assert standard_notation("c-15t", "inhA") == "n-15T"


@pytest.mark.parametrize("value", [
    "katG deletion", "Multiple mutations (codons 315, 316)",
    "high-level resistance", "N/A", "", "the N-terminal region",
])
def test_non_mutations_are_not_converted(value):
    """An empty result lets a caller tell a real conversion from a guess."""
    from litrag.normalize import standard_notation
    assert standard_notation(value) == ""


def test_prose_and_notation_share_an_identity():
    """The point of the conversion: these must merge rather than sit apart."""
    forms = ["NS1-53 glycine to aspartate", "G53D", "Gly53Asp", "glycine 53 to aspartate"]
    assert len({normalize_mutation(f) for f in forms}) == 1


def test_ambiguous_single_letters_need_a_non_coding_qualifier():
    """"C to T" is Cys->Thr as readily as cytosine->thymine, so a bare one is
    only read as a base change where the qualifier names a non-coding region."""
    from litrag.normalize import standard_notation
    assert standard_notation("5'UTR-57 C to T") == "n57T"
    # "<word>-<digits>" reads the hyphen as a separator, matching 5'UTR-57.
    # A genuinely negative position is written without a word prefix.
    assert standard_notation("promoter-15 C to T") == "n15T"
    assert standard_notation("c.-15C>T") == "n-15T"
    # Spelled out, there is no ambiguity to resolve.
    assert standard_notation("57 cytosine to thymine") == "n57T"


@pytest.mark.parametrize("value,expected", [
    # Butrapet et al. write the same substitution with three-letter codes.
    ("NS1-53 Gly-to-Asp", "G53D"),
    ("NS1-53 Gly to Asp", "G53D"),
    ("NS3-250 Glu-to-Val", "E250V"),
    ("Ser315 to Thr", "S315T"),
    ("position 315 Ser to Thr", "S315T"),
])
def test_three_letter_codes_convert(value, expected):
    from litrag.normalize import standard_notation
    assert standard_notation(value) == expected


def test_full_names_and_codes_agree():
    """Two papers writing one substitution differently must merge."""
    forms = ["NS1-53 glycine to aspartate", "NS1-53 Gly-to-Asp", "G53D", "Gly53Asp"]
    assert len({normalize_mutation(f) for f in forms}) == 1
