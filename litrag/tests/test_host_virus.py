"""The host-virus interaction data type.

The curated chain is: viral protein -> interaction type -> host protein ->
biological consequence. The interaction verb is a closed vocabulary, which is
what makes the table queryable.
"""

import pytest

from litrag.dedup import dedupe, identity_key
from litrag.extract import Row, extract
from litrag.local_templates import declarations
from litrag.normalize import INTERACTION_TYPES, normalize_interaction
from litrag.prompts import build_prompt
from litrag.templates import TemplateRegistry

COLUMNS = [
    "Organism", "Viral Protein", "Interaction Type", "Host Protein",
    "Host Organism", "Biological Consequence", "Method", "Assertion", "Reference",
]


@pytest.fixture
def hv():
    return TemplateRegistry.from_declarations(declarations()).resolve("host-virus")


def _answer(*rows):
    return "\n".join(["\t".join(COLUMNS)] + ["\t".join(r) for r in rows])


def test_columns_capture_the_chain(hv):
    assert hv.columns == COLUMNS
    assert hv.is_table and hv.is_local


@pytest.mark.parametrize("alias", [
    "host-virus", "virus-host", "hvi", "host", "HOST_VIRUS",
])
def test_aliases_resolve(alias):
    registry = TemplateRegistry.from_declarations(declarations())
    assert registry.resolve(alias).id == "host-virus"


# -- the interaction vocabulary -------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("binds", "binds"), ("binding", "binds"), ("interacts with", "binds"),
    ("associates with", "binds"), ("physical interaction", "binds"),
    ("cleaves", "cleaves"), ("cleavage", "cleaves"), ("proteolysis", "cleaves"),
    ("inhibits", "inhibits"), ("blocks", "inhibits"), ("antagonizes", "inhibits"),
    ("suppresses", "inhibits"), ("inhibits nuclear import", "inhibits"),
    ("activates", "activates"), ("induces", "activates"), ("upregulates", "activates"),
    ("degrades", "degrades"), ("targets for degradation", "degrades"),
    ("ubiquitinates", "degrades"),
    ("relocalizes", "relocalizes"), ("sequesters", "relocalizes"),
    ("retains", "relocalizes"), ("mislocalises", "relocalizes"),
])
def test_interaction_synonyms(value, expected):
    assert normalize_interaction(value) == expected
    assert expected in INTERACTION_TYPES


def test_unusual_verb_stays_visible():
    """Forcing every verb into the six would file a finding under the wrong one."""
    assert normalize_interaction("phosphorylates") == "phosphorylates"


def test_distinct_verbs_stay_distinct():
    assert normalize_interaction("binds") != normalize_interaction("degrades")
    assert normalize_interaction("inhibits") != normalize_interaction("activates")


# -- identity --------------------------------------------------------------

def test_the_verb_is_part_of_the_claim(hv):
    """Binding STAT1 and degrading it are two findings, not one."""
    binds = Row(values={"Organism": "SARS-CoV-2", "Viral Protein": "ORF6",
                        "Interaction Type": "binds", "Host Protein": "STAT1"})
    degrades = Row(values={"Organism": "SARS-CoV-2", "Viral Protein": "ORF6",
                           "Interaction Type": "degrades", "Host Protein": "STAT1"})
    assert len(dedupe([binds, degrades], "host-virus", hv.columns)) == 2


def test_synonymous_verbs_merge(hv):
    a = Row(values={"Organism": "SARS-CoV-2", "Viral Protein": "ORF6",
                    "Interaction Type": "inhibits", "Host Protein": "STAT1"})
    b = Row(values={"Organism": "SARS-CoV-2", "Viral Protein": "ORF6",
                    "Interaction Type": "blocks", "Host Protein": "STAT1"})
    assert len(dedupe([a, b], "host-virus", hv.columns)) == 1


def test_direction_is_not_symmetric(hv):
    """Unlike a protein-protein pair, the viral protein acts on the host."""
    forward = Row(values={"Organism": "X", "Viral Protein": "ORF6",
                          "Interaction Type": "binds", "Host Protein": "STAT1"})
    reversed_ = Row(values={"Organism": "X", "Viral Protein": "STAT1",
                            "Interaction Type": "binds", "Host Protein": "ORF6"})
    assert identity_key(forward, "host-virus", hv.columns) != \
        identity_key(reversed_, "host-virus", hv.columns)


def test_different_host_targets_stay_separate(hv):
    rows = [Row(values={"Organism": "SARS-CoV-2", "Viral Protein": "ORF6",
                        "Interaction Type": "binds", "Host Protein": host})
            for host in ("NUP98", "RAE1", "KPNA2")]
    assert len(dedupe(rows, "host-virus", hv.columns)) == 3


def test_consequence_is_not_identity(hv):
    """Two papers reporting different downstream effects describe one interaction."""
    a = Row(values={"Organism": "X", "Viral Protein": "ORF6", "Interaction Type": "binds",
                    "Host Protein": "STAT1", "Biological Consequence": "reduced IFN"})
    b = Row(values={"Organism": "X", "Viral Protein": "ORF6", "Interaction Type": "binds",
                    "Host Protein": "STAT1", "Biological Consequence": "blocked ISG induction"})
    merged = dedupe([a, b], "host-virus", hv.columns)
    assert len(merged) == 1 and merged[0].n_support == 2
    assert len(merged[0].variants["Biological Consequence"]) == 2


# -- extraction ------------------------------------------------------------

def test_the_worked_example(hv, mutation_response):
    """ORF6 inhibits STAT nuclear import -> reduced interferon signalling."""
    answer = _answer(
        ["SARS-CoV-2", "ORF6", "inhibits", "STAT1 nuclear import", "human",
         "reduced interferon signalling", "co-immunoprecipitation", "measured", "[1]"],
    )
    row = extract(answer, hv, mutation_response["sources"]).rows[0]
    assert row.get("Viral Protein") == "ORF6"
    assert row.get("Interaction Type") == "inhibits"
    assert row.get("Host Protein") == "STAT1 nuclear import"
    assert row.get("Biological Consequence") == "reduced interferon signalling"
    assert row.citations[0].pmid


def test_a_row_with_no_consequence_still_counts(hv, mutation_response):
    """An interaction is a finding even when no downstream effect is reported."""
    answer = _answer(
        ["SARS-CoV-2", "ORF6", "binds", "NUP98", "human", "N/A",
         "co-immunoprecipitation", "measured", "[1]"],
    )
    result = extract(answer, hv, mutation_response["sources"])
    assert result.n_rows == 1 and result.dropped_empty == 0


def test_several_host_targets_are_split(hv, mutation_response):
    answer = _answer(
        ["SARS-CoV-2", "ORF6", "binds", "NUP98, RAE1", "human", "N/A",
         "co-IP", "measured", "[1]"],
    )
    result = extract(answer, hv, mutation_response["sources"])
    assert result.split_compound == 1
    assert [r.get("Host Protein") for r in result.rows] == ["NUP98", "RAE1"]


def test_host_protein_is_never_off_target(hv, mutation_response):
    """The gene asked for is the viral one; the host protein is the target."""
    answer = _answer(
        ["SARS-CoV-2", "ORF6", "inhibits", "STAT1", "human", "reduced IFN",
         "co-IP", "measured", "[1]"],
    )
    row = extract(answer, hv, mutation_response["sources"],
                  requested_genes=["ORF6"]).rows[0]
    assert "off_target_gene" not in row.flags


# -- Method validation -----------------------------------------------------

def test_an_evidence_category_is_not_a_method(hv, mutation_response):
    """Told plainly that Method is a technique, the model still filled all 40
    rows of one live query with "reported". The value is checkable, so it is."""
    answer = _answer(
        ["SARS-CoV-2", "ORF6", "inhibits", "STAT1", "human", "reduced IFN",
         "reported", "measured", "[1]"],
    )
    row = extract(answer, hv, mutation_response["sources"]).rows[0]
    assert row.get("Method") == ""
    assert "method_not_a_technique" in row.flags
    assert row.get("Assertion") == "measured", "the category itself is untouched"


def test_prediction_is_a_real_method(hv, mutation_response):
    """Excluded from the check on purpose: prediction genuinely is a technique."""
    answer = _answer(
        ["SARS-CoV-2", "NSP1", "inhibits", "40S ribosome", "human", "blocked translation",
         "prediction", "predicted", "[1]"],
    )
    row = extract(answer, hv, mutation_response["sources"]).rows[0]
    assert row.get("Method") == "prediction"
    assert "method_not_a_technique" not in row.flags


def test_a_genuine_method_is_untouched(hv, mutation_response):
    answer = _answer(
        ["SARS-CoV-2", "ORF6", "binds", "NUP98", "human", "N/A",
         "affinity purification mass spectrometry", "measured", "[1]"],
    )
    row = extract(answer, hv, mutation_response["sources"]).rows[0]
    assert row.get("Method") == "affinity purification mass spectrometry"
    assert row.flags == []


# -- prompt ----------------------------------------------------------------

def test_prompt_states_the_vocabulary_and_direction(hv, mutation_response):
    prompt, _, _ = build_prompt(hv, mutation_response["sources"], "SARS-CoV-2")
    for verb in INTERACTION_TYPES:
        assert verb in prompt
    assert "always viral protein acting on host target" in prompt
    assert "never reverse the pair" in prompt.lower()
    assert "never an evidence category" in prompt
