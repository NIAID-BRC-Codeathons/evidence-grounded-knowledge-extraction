"""Matchers for the three charter relation types.

Two of the three slots in each relation are entities and compare cleanly. The
third is free text, and matching it is a judgement call with failure modes in
both directions. These tests pin the behaviour AND record the known failures,
so nobody later reads a recall number as a property of the extractor when it is
partly a property of this file.
"""

import pytest

from litrag.matchers import (CHARTER_MATCHERS, SCHEMA_ADDITIONS, _assoc_bucket,
                             _fuzzy, biomarker_match, mechanism_match,
                             phenotype_match)


# --- pathogen-host phenotype --------------------------------------------------

def pheno(pathogen="SARS-CoV-2", host="mouse", phenotype="weight loss"):
    return {"pathogen": pathogen, "host": host, "phenotype": phenotype}


def test_same_relation_matches_despite_wording():
    assert phenotype_match(
        pheno(phenotype="reduced weight gain in infected animals"),
        pheno(phenotype="reduced weight gain"),
    )


def test_different_host_does_not_match():
    """The same pathogen causes different effects in different hosts."""
    assert not phenotype_match(pheno(host="ferret"), pheno(host="mouse"))


def test_different_pathogen_does_not_match():
    assert not phenotype_match(pheno(pathogen="influenza A"), pheno())


def test_host_punctuation_and_case_are_ignored():
    assert phenotype_match(pheno(host="BALB/c Mice"), pheno(host="balb c mice"))


def test_empty_entities_never_match():
    """Two blank rows are not the same fact; they are two absences."""
    assert not phenotype_match(pheno(pathogen="", host=""), pheno(pathogen="", host=""))


def test_negation_blocks_a_phenotype_match():
    """The failure this guard exists for: a finding and its opposite share
    almost every word."""
    assert not phenotype_match(
        pheno(phenotype="no weight loss observed"),
        pheno(phenotype="weight loss observed"),
    )


# --- pathogen-mechanism-disease -----------------------------------------------

def mech(pathogen="pathogen Z", disease="diarrheal disease",
         mechanism="disruption of epithelial tight junctions via protein Y"):
    return {"pathogen": pathogen, "disease": disease, "mechanism": mechanism}


def test_mechanism_matches_on_shared_substance():
    assert mechanism_match(mech(), mech(mechanism="disrupts epithelial tight junctions"))


def test_unrelated_mechanisms_do_not_match():
    assert not mechanism_match(mech(), mech(mechanism="inhibits interferon signalling"))


def test_same_mechanism_different_disease_does_not_match():
    assert not mechanism_match(mech(disease="sepsis"), mech())


# --- biomarker-disease --------------------------------------------------------

def bio(biomarker="IL-6", disease="severe COVID-19", association="increased"):
    return {"biomarker": biomarker, "disease": disease, "association": association}


def test_direction_synonyms_agree():
    assert biomarker_match(bio(association="elevated in severe cases"),
                           bio(association="increased"))


def test_opposite_directions_do_not_match():
    """Same biomarker, same disease, opposite finding -- emphatically not a match."""
    assert not biomarker_match(bio(association="increased"), bio(association="decreased"))


def test_no_association_is_a_real_finding_and_matches_itself():
    assert biomarker_match(bio(association="no association"),
                           bio(association="unchanged"))


def test_null_result_does_not_match_a_positive_one():
    assert not biomarker_match(bio(association="unchanged"), bio(association="increased"))


@pytest.mark.parametrize("wording,bucket", [
    ("elevated", "up"), ("upregulated", "up"), ("higher in cases", "up"),
    ("decreased", "down"), ("downregulated", "down"),
    ("unchanged", "none"), ("no association", "none"),
    ("detected", None), ("", None),
])
def test_association_buckets(wording, bucket):
    assert _assoc_bucket(wording) == bucket


# --- the free-text matcher itself ---------------------------------------------

def test_subset_counts_as_a_match():
    assert _fuzzy("reduced polymerase activity", "reduced activity")


def test_unrelated_text_does_not_match():
    assert not _fuzzy("elevated serum glycan", "tight junction disruption")


def test_empty_text_never_matches():
    assert not _fuzzy("", "anything at all")


# --- known failure modes, recorded rather than hidden -------------------------

def test_KNOWN_FAILURE_synonyms_across_vocabularies_are_missed():
    """Depresses recall. Two ways of naming one concept share no tokens, so a
    correct extraction scores as a miss. Mitigation is extending
    experiment-01/synonyms.json, which is hand-curated; an ontology lookup was
    judged out of codeathon budget."""
    assert not _fuzzy("increased virulence", "increased pathogenicity")


def test_KNOWN_FAILURE_entity_synonyms_miss_at_the_hard_key():
    """Worse than the free-text case, because no threshold can rescue it: the
    biomarker slot is compared exactly, so one standard symbol and its expanded
    name are simply different entities. This is the strongest argument for the
    entity-linking axis (E6) feeding normalisation back into matching."""
    assert not biomarker_match(bio(biomarker="IL-6"), bio(biomarker="interleukin-6"))


def test_KNOWN_FAILURE_shared_scaffolding_can_over_match():
    """Inflates precision. Two mechanisms that are opposites in substance --
    synthesis versus degradation -- share every structural word and clear the
    threshold. Neither negation nor direction catches this, because neither
    word is a negator or a direction term."""
    assert _fuzzy("inhibition of host protein synthesis",
                  "inhibition of host protein degradation")


def test_KNOWN_WEAKNESS_short_slots_can_match_on_coincidence():
    """A number and a shared modifier are enough to clear Jaccard 0.5 when the
    slot is short. The match here happens to be semantically right, but it is
    right by accident, not because anything understood the synonym."""
    assert _fuzzy("IL-6 elevated", "interleukin-6 elevated")


# --- registry plumbing --------------------------------------------------------

def test_all_three_types_are_registered():
    assert set(CHARTER_MATCHERS) == {"phenotype", "mechanism", "biomarker"}


def test_schema_fields_are_the_snake_case_of_the_columns():
    """envelope.py derives field names from columns, so the two must agree or
    gold rows and run rows are read through different keys."""
    for entry in SCHEMA_ADDITIONS.values():
        derived = [c.lower().replace(" ", "_") for c in entry["columns"]]
        assert derived == entry["fields"]


def test_schema_ids_match_the_local_template_ids():
    """One id across template, matcher, dedup key and gold key -- the thing the
    existing ppi-extraction / ppi split got wrong."""
    from litrag.local_templates import BUILTIN
    from litrag.dedup import IDENTITY_COLUMNS

    builtin_ids = {t["id"] for t in BUILTIN}
    for type_id in SCHEMA_ADDITIONS:
        assert type_id in builtin_ids
        assert type_id in IDENTITY_COLUMNS
        assert type_id in CHARTER_MATCHERS


# --- the copies must not drift ------------------------------------------------

def _sibling_evaluate():
    """experiment-01/evaluate.py, if this checkout sits in the codeathon tree."""
    import importlib.util
    from pathlib import Path

    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        candidate = parent / "experiment-01" / "evaluate.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("sibling_evaluate", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def test_the_merged_copy_in_evaluate_py_still_agrees_with_this_one():
    """These matchers were merged into experiment-01/evaluate.py, so the logic
    now exists twice. Duplication that nothing checks is duplication that
    silently diverges -- this fails the moment one copy is edited alone.

    Skipped rather than failed when the sibling project is absent: this package
    must remain installable on its own.
    """
    evaluate = _sibling_evaluate()
    if evaluate is None:
        pytest.skip("experiment-01/evaluate.py not present in this checkout")

    cases = [
        ("phenotype", pheno(), pheno(phenotype="reduced weight gain")),
        ("phenotype", pheno(phenotype="no weight loss"), pheno(phenotype="weight loss")),
        ("phenotype", pheno(host="ferret"), pheno()),
        ("mechanism", mech(), mech(mechanism="disrupts epithelial tight junctions")),
        ("mechanism", mech(), mech(mechanism="inhibits interferon signalling")),
        ("biomarker", bio(association="elevated"), bio(association="increased")),
        ("biomarker", bio(association="increased"), bio(association="decreased")),
        ("biomarker", bio(biomarker="IL-6"), bio(biomarker="interleukin-6")),
    ]

    for data_type, left, right in cases:
        mine = CHARTER_MATCHERS[data_type](left, right)
        theirs = evaluate.MATCHERS[data_type](left, right)
        assert mine == theirs, (
            f"{data_type} matchers disagree on {left} vs {right}: "
            f"litrag says {mine}, evaluate.py says {theirs}"
        )


def test_the_registries_cover_the_same_charter_types():
    evaluate = _sibling_evaluate()
    if evaluate is None:
        pytest.skip("experiment-01/evaluate.py not present in this checkout")
    assert set(CHARTER_MATCHERS) <= set(evaluate.MATCHERS)
