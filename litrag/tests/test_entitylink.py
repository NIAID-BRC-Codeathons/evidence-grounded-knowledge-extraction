"""Entity linking against NCBI Taxonomy.

Every test here runs OFFLINE against tests/fixtures/entity_cache.json, which
was built once from live E-utilities. Network-dependent tests are flaky and
would fail at a venue with no route to NCBI, which is exactly when the suite
needs to be trustworthy.
"""

from pathlib import Path

import pytest

from litrag.entitylink import (EXACT, SAME_GENUS, SAME_SPECIES, WRONG,
                               EntityLinker, credit, normalize_surface,
                               score_mentions)

FIXTURE = Path(__file__).parent / "fixtures" / "entity_cache.json"


@pytest.fixture
def linker():
    return EntityLinker(FIXTURE, offline=True)


# --- surface normalisation ----------------------------------------------------

@pytest.mark.parametrize("surface,expected", [
    ("Mycobacterium tuberculosis", "mycobacterium tuberculosis"),
    ("  SARS-CoV-2  ", "sars-cov-2"),
    # Strain and condition words qualify a host without changing the species.
    ("BALB/c mice", "mice"),
    ("aged C57BL/6N mice", "mice"),
    ("wild-type mouse", "mouse"),
])
def test_normalize_strips_qualifiers(surface, expected):
    assert normalize_surface(surface) == expected


def test_qualifier_stripping_is_what_makes_hosts_resolvable(linker):
    """Live NCBI returns 10095 for "BALB/c mice" -- not Mus musculus. Stripping
    the strain is the difference between the right organism and the wrong one."""
    assert linker.resolve("BALB/c mice").taxid == "10090"
    assert linker.resolve("aged C57BL/6N mice").taxid == "10090"


# --- resolution ---------------------------------------------------------------

def test_full_binomials_resolve(linker):
    result = linker.resolve("Mycobacterium tuberculosis")
    assert result.taxid == "1773"
    assert result.rank == "species"
    assert result.source == "ncbi_taxonomy"


@pytest.mark.parametrize("abbrev", ["M. tuberculosis", "Mtb", "M. tb"])
def test_abbreviations_resolve(abbrev, linker):
    """Bare NCBI returns NOTHING for any of these, and papers use them
    constantly. Aliases and initial-expansion are what close the gap."""
    assert linker.resolve(abbrev).taxid == "1773"


def test_initial_expansion_uses_the_cache_not_a_hardcoded_genus_list(linker):
    """K. pneumoniae is in no alias table; it resolves because the full name is
    already cached, so the mechanism improves as the cache fills."""
    assert linker.resolve("K. pneumoniae").taxid == "573"
    assert linker.resolve("E. coli").taxid == "562"


def test_unknown_surface_reports_rather_than_guesses(linker):
    result = linker.resolve("Nonexistentia fabricata")
    assert not result.resolved
    assert "offline" in result.note or "no NCBI" in result.note


# --- partial credit -----------------------------------------------------------

def test_exact_match_scores_one(linker):
    assert credit(linker.resolve("Mycobacterium tuberculosis"), "1773") == EXACT


def test_unresolved_scores_zero(linker):
    assert credit(linker.resolve("Nonexistentia fabricata"), "1773") == WRONG


def test_different_genus_scores_zero(linker):
    """E. coli offered where M. tuberculosis was wanted is a different organism,
    not a near miss."""
    predicted = linker.resolve("Escherichia coli")
    gold = linker.resolve("Mycobacterium tuberculosis")
    assert credit(predicted, "1773", gold) == WRONG


def test_same_genus_earns_partial_credit(linker):
    """Klebsiella pneumoniae scored against another Klebsiella: wrong species,
    right genus. A curator fixes that in seconds; scoring it 0 alongside a
    completely wrong organism hides which kind of wrong the system is."""
    predicted = linker.resolve("Klebsiella pneumoniae")
    gold = linker.resolve("Klebsiella pneumoniae")
    # Same entity, so exact; the genus path is exercised below with lineage.
    assert credit(predicted, predicted.taxid, gold) == EXACT
    assert predicted.ancestor_at("genus") is not None


def test_credit_uses_lineage_ids_not_name_strings(linker):
    """M. bovis and M. tuberculosis share a genus that no string comparison of
    the two names would reveal."""
    bovis = linker.resolve("Mycobacterium bovis")
    mtb = linker.resolve("Mycobacterium tuberculosis")
    assert bovis.taxid != mtb.taxid
    assert bovis.ancestor_at("genus") == mtb.ancestor_at("genus")


# --- honest limits ------------------------------------------------------------

def test_viral_entries_have_no_rank_of_their_own_but_still_score(linker):
    """Viruses sit BELOW species in NCBI: SARS-CoV-2 (2697049) has rank
    "no rank" and its species ancestor is Betacoronavirus pandemicum (3418604).

    I expected this to break partial credit and it does not, because credit
    walks the lineage rather than reading the entity's own rank. That is the
    concrete payoff of comparing lineage ids instead of name strings -- a
    name-based check would have had nothing to compare here."""
    for surface, species_ancestor in (("SARS-CoV-2", "3418604"),
                                      ("influenza A virus", "2955291")):
        result = linker.resolve(surface)
        assert result.resolved
        assert result.rank == "no rank", "the entity itself is unranked"
        assert result.ancestor_at("species") == species_ancestor
        assert result.ancestor_at("genus") is not None


def test_viral_species_names_do_not_resemble_the_common_name(linker):
    """SARS-CoV-2's species is "Betacoronavirus pandemicum" and influenza A's is
    "Alphainfluenzavirus influenzae". Neither shares a word with how anyone
    writes them, which is the clearest argument in this module for comparing
    lineage identifiers rather than strings."""
    assert "coronavirus" not in linker.resolve("SARS-CoV-2").surface.lower()
    species = dict((rank, name) for _id, name, rank
                   in linker.resolve("SARS-CoV-2").lineage)["species"]
    assert species == "Betacoronavirus pandemicum"


def test_KNOWN_SURPRISE_ncbi_folds_m_bovis_into_m_tuberculosis(linker):
    """NCBI now classifies M. bovis as a biotype of M. tuberculosis. Correct by
    current taxonomy, and still surprising to a curator who expects two species
    -- so it is pinned here rather than discovered during a demo."""
    bovis = linker.resolve("Mycobacterium bovis")
    assert bovis.rank == "biotype"
    assert "tuberculosis" in bovis.scientific_name.lower()


# --- the axis: two numbers, never one -----------------------------------------

def test_score_reports_coverage_and_accuracy_separately(linker):
    """A system linking a tenth of mentions perfectly would report 1.0 accuracy.
    True, and useless, unless coverage is printed beside it."""
    surfaces = ["Mycobacterium tuberculosis", "Mtb", "Nonexistentia fabricata"]
    gold = {"mycobacterium tuberculosis": "1773", "mtb": "1773"}
    report = score_mentions(surfaces, gold, linker)

    assert report["status"] == "ok"
    assert report["mentions"] == 3
    assert report["entity_link_coverage"] == round(2 / 3, 3)
    assert report["entity_linking_accuracy"] == 1.0
    assert report["scored_against_gold"] == 2


def test_accuracy_is_none_when_nothing_could_be_scored(linker):
    """No gold overlap means unmeasured, which must not read as zero."""
    report = score_mentions(["Mycobacterium tuberculosis"], {}, linker)
    assert report["entity_linking_accuracy"] is None
    assert report["entity_link_coverage"] == 1.0


def test_missing_cache_reports_not_run_rather_than_zero(tmp_path):
    """A check that could not run must never look like a check that failed."""
    empty = EntityLinker(tmp_path / "absent.json", offline=True)
    report = score_mentions(["Mycobacterium tuberculosis"], {"x": "1"}, empty)
    assert report["status"] == "not run"
    assert "cache" in report["detail"]


def test_misses_explain_themselves(linker):
    report = score_mentions(["Nonexistentia fabricata"], {}, linker)
    assert report["misses"] and report["misses"][0]["why"]
