"""The glycosylation-sites data type.

Defined by LitRAG rather than the server, like `ast`, so it runs on a local
generator. The hazards specific to this data type are site notation and site
numbering.
"""

import pytest

from litrag.dedup import dedupe, identity_key
from litrag.extract import Row, extract
from litrag.local_templates import declarations
from litrag.normalize import normalize_glyco_type, normalize_site
from litrag.prompts import build_prompt
from litrag.templates import TemplateRegistry

COLUMNS = [
    "Organism", "Protein", "Strain", "Site", "Glycosylation Type",
    "Glycan", "Method", "Effect", "Assertion", "Reference",
]


@pytest.fixture
def glyco():
    return TemplateRegistry.from_declarations(declarations()).resolve("glycosylation")


def _answer(*rows):
    return "\n".join(["\t".join(COLUMNS)] + ["\t".join(r) for r in rows])


def test_requested_columns(glyco):
    assert glyco.columns == COLUMNS
    assert glyco.is_table and glyco.is_local


# -- site notation -------------------------------------------------------

@pytest.mark.parametrize("value", ["N234", "Asn234", "Asn-234", "N-234", "p.Asn234", "asn234"])
def test_site_notations_agree(value):
    assert normalize_site(value) == "N234"


def test_site_numbering_is_never_rewritten():
    """Glycosite numbering differs between isoforms, strains and constructs, so
    two positions are two sites even for the same residue."""
    assert normalize_site("N234") != normalize_site("N235")
    assert normalize_site("Asn331") == "N331"


def test_o_linked_sites():
    assert normalize_site("Thr678") == "T678"
    assert normalize_site("Ser325") == "S325"


def test_unparseable_site_falls_back_to_text():
    assert normalize_site("the N-terminal region") == "the n terminal region"
    assert normalize_site("") == ""


@pytest.mark.parametrize("value,expected", [
    ("N-linked", "N-linked"), ("N-glycosylation", "N-linked"),
    ("N linked glycosylation", "N-linked"), ("Asn-linked", "N-linked"),
    ("O-linked", "O-linked"), ("O-glycan", "O-linked"),
    ("C-mannosylation", "C-mannosylation"),
])
def test_glycosylation_type_vocabulary(value, expected):
    assert normalize_glyco_type(value) == expected


def test_n_and_o_linked_are_distinct():
    assert normalize_glyco_type("N-linked") != normalize_glyco_type("O-linked")


# -- identity and merging ------------------------------------------------

def test_identity_is_protein_times_site(glyco):
    a = Row(values={"Organism": "SARS-CoV-2", "Protein": "Spike", "Site": "N234"})
    b = Row(values={"Organism": "SARS-CoV-2", "Protein": "Spike", "Site": "Asn234"})
    assert identity_key(a, "glycosylation", glyco.columns) == \
        identity_key(b, "glycosylation", glyco.columns)
    assert len(dedupe([a, b], "glycosylation", glyco.columns)) == 1


def test_different_sites_stay_separate(glyco):
    rows = [Row(values={"Organism": "SARS-CoV-2", "Protein": "Spike", "Site": s})
            for s in ("N234", "N343", "N165")]
    assert len(dedupe(rows, "glycosylation", glyco.columns)) == 3


def test_same_site_on_different_proteins_stays_separate(glyco):
    a = Row(values={"Organism": "SARS-CoV-2", "Protein": "Spike", "Site": "N234"})
    b = Row(values={"Organism": "SARS-CoV-2", "Protein": "ORF8", "Site": "N234"})
    assert len(dedupe([a, b], "glycosylation", glyco.columns)) == 2


def test_glycan_and_method_are_not_identity(glyco):
    """Two papers characterising one site differently describe the same site."""
    a = Row(values={"Organism": "SARS-CoV-2", "Protein": "Spike", "Site": "N234",
                    "Glycan": "oligomannose", "Method": "mass spectrometry"})
    b = Row(values={"Organism": "SARS-CoV-2", "Protein": "Spike", "Site": "N234",
                    "Glycan": "Man5GlcNAc2", "Method": "cryo-EM"})
    merged = dedupe([a, b], "glycosylation", glyco.columns)
    assert len(merged) == 1 and merged[0].n_support == 2
    assert set(merged[0].variants["Glycan"]) == {"oligomannose", "Man5GlcNAc2"}


def test_conflicting_linkage_is_flagged(glyco):
    a = Row(values={"Organism": "X", "Protein": "P", "Site": "N100",
                    "Glycosylation Type": "N-linked"})
    b = Row(values={"Organism": "X", "Protein": "P", "Site": "N100",
                    "Glycosylation Type": "O-linked"})
    merged = dedupe([a, b], "glycosylation", glyco.columns)
    assert "merged_variants:Glycosylation Type" in merged[0].flags


def test_wording_of_linkage_is_not_a_conflict(glyco):
    a = Row(values={"Organism": "X", "Protein": "P", "Site": "N100",
                    "Glycosylation Type": "N-linked"})
    b = Row(values={"Organism": "X", "Protein": "P", "Site": "N100",
                    "Glycosylation Type": "N-glycosylation"})
    merged = dedupe([a, b], "glycosylation", glyco.columns)
    assert not any(f.startswith("merged_variants:Glycosylation Type")
                   for f in merged[0].flags)


# -- extraction ----------------------------------------------------------

def test_row_without_a_site_is_flagged(glyco, mutation_response):
    """A glycosylation row with no site identifies nothing."""
    answer = _answer(
        ["SARS-CoV-2", "Spike", "A/test/1/2020", "N/A", "N-linked", "N/A", "LC-MS", "N/A", "reported", "[1]"],
    )
    result = extract(answer, glyco, mutation_response["sources"])
    assert "missing:Site" in result.rows[0].flags


def test_row_with_only_a_site_survives(glyco, mutation_response):
    """That a site is glycosylated at all is a finding, even uncharacterised."""
    answer = _answer(
        ["SARS-CoV-2", "Spike", "A/test/1/2020", "N234", "N-linked", "N/A", "N/A", "N/A", "reported", "[1]"],
    )
    result = extract(answer, glyco, mutation_response["sources"])
    assert result.n_rows == 1 and result.dropped_empty == 0


def test_empty_row_is_still_dropped(glyco, mutation_response):
    answer = _answer(
        ["SARS-CoV-2", "Spike", "A/test/1/2020", "N/A", "N/A", "N/A", "N/A", "N/A", "reported", "N/A"],
    )
    result = extract(answer, glyco, mutation_response["sources"])
    assert result.n_rows == 0 and result.dropped_empty == 1


def test_multiple_sites_in_one_row_are_split(glyco, mutation_response):
    answer = _answer(
        ["SARS-CoV-2", "Spike", "A/test/1/2020", "N234, N343", "N-linked", "oligomannose, complex", "LC-MS", "shielding", "reported", "[1]"],
    )
    result = extract(answer, glyco, mutation_response["sources"])
    assert result.split_compound == 1
    assert [r.get("Site") for r in result.rows] == ["N234", "N343"]


# -- prompt --------------------------------------------------------------

def test_prompt_carries_the_numbering_warning(glyco, mutation_response):
    prompt, _, _ = build_prompt(glyco, mutation_response["sources"], "SARS-CoV-2")
    assert "never renumber" in prompt
    assert "One row per glycosylation site" in prompt
    assert "Do not infer the type from the residue alone" in prompt


def test_prompt_includes_the_assertion_vocabulary(glyco, mutation_response):
    """The column rule applies here too, since the template has an Assertion."""
    from litrag.prompts import ASSERTION_VALUES
    prompt, _, _ = build_prompt(glyco, mutation_response["sources"], "SARS-CoV-2")
    for value in ASSERTION_VALUES:
        assert value in prompt


@pytest.mark.parametrize("alias", [
    "glycosylation", "glyco", "glycan", "glycosite", "glycosites",
    "glycosylation_sites", "GLYCO",
])
def test_aliases_resolve(alias):
    """Every shorthand the README advertises has to actually work."""
    registry = TemplateRegistry.from_declarations(declarations())
    assert registry.resolve(alias).id == "glycosylation"


# -- strain and protein --------------------------------------------------

def test_strain_is_a_column(glyco):
    assert "Strain" in glyco.columns


def test_prompt_asks_for_protein_and_strain(glyco, mutation_response):
    prompt, _, _ = build_prompt(glyco, mutation_response["sources"], "Influenza A virus")
    assert "Always name the protein" in prompt
    assert "A/California/07/2009" in prompt
    assert "Do not infer a strain from the organism" in prompt


def test_strain_is_part_of_identity(glyco):
    """Glycosite numbering is strain-dependent, so position 146 on H3N2 HA is
    not the same site as position 146 on H1N1 HA."""
    h3 = Row(values={"Organism": "IAV", "Protein": "HA", "Strain": "H3N2", "Site": "N146"})
    h1 = Row(values={"Organism": "IAV", "Protein": "HA", "Strain": "H1N1", "Site": "N146"})
    assert len(dedupe([h3, h1], "glycosylation", glyco.columns)) == 2


def test_same_strain_and_site_merges(glyco):
    a = Row(values={"Organism": "IAV", "Protein": "HA", "Strain": "H3N2", "Site": "N146"})
    b = Row(values={"Organism": "IAV", "Protein": "HA", "Strain": "H3N2", "Site": "Asn146"})
    assert len(dedupe([a, b], "glycosylation", glyco.columns)) == 1


# -- bare positions ------------------------------------------------------

def test_bare_position_gains_its_residue(glyco, mutation_response):
    """Papers list influenza HA glycosites as "42, 44, 50". Under an N-linked
    heading the residue is asparagine by definition, so it can be supplied.
    """
    answer = _answer(
        ["Influenza A virus", "HA", "H3N2", "42", "N-linked", "N/A",
         "sequence analysis", "N/A", "reported", "[1]"],
    )
    row = extract(answer, glyco, mutation_response["sources"]).rows[0]
    assert row.get("Site") == "42", "source wording kept"
    assert row.display("Site") == "N42"


def test_o_linked_bare_position_is_left_alone(glyco, mutation_response):
    """O-linked sits on serine or threonine, so the residue is not derivable."""
    answer = _answer(
        ["Influenza A virus", "HA", "H3N2", "325", "O-linked", "N/A",
         "mass spectrometry", "N/A", "reported", "[1]"],
    )
    row = extract(answer, glyco, mutation_response["sources"]).rows[0]
    assert row.display("Site") == "325"
