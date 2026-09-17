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


# -- both ends of the pair -------------------------------------------------
#
# Live at top_k=40 the model returned 263 rows of which 255 named only one end
# of the pair. They survived because Method held a real technique, so the
# evidence-free filter saw a row that reported something -- it just had nothing
# to report it about. Most read "nsp13  N/A  N/A  ...  affinity purification
# mass spectrometry  reported": an interactome paper the model could not resolve
# into named partners.

def test_the_pair_and_the_verb_are_required(hv):
    assert hv.required == ["Viral Protein", "Interaction Type", "Host Protein"]


@pytest.mark.parametrize("gap,cells", [
    ("Host Protein",
     ["SARS-CoV-2", "nsp13", "binds", "N/A", "human", "N/A",
      "affinity purification mass spectrometry", "reported", "[1]"]),
    ("Viral Protein",
     ["SARS-CoV-2", "N/A", "binds", "PUS7", "human", "N/A",
      "computational prediction", "measured", "[1]"]),
    ("Interaction Type",
     ["SARS-CoV-2", "nsp15", "N/A", "N/A", "human", "N/A",
      "affinity purification mass spectrometry", "reported", "[1]"]),
])
def test_half_a_pair_is_dropped(hv, mutation_response, gap, cells):
    """One end of the pair is not a partial finding -- it is no finding."""
    result = extract(_answer(cells), hv, mutation_response["sources"])
    assert result.rows == []
    assert result.dropped_incomplete == 1
    # Not the evidence-free counter: these rows DID report a method.
    assert result.dropped_empty == 0


def test_a_complete_pair_survives_beside_incomplete_ones(hv, mutation_response):
    result = extract(
        _answer(
            ["SARS-CoV-2", "nsp13", "binds", "N/A", "human", "N/A",
             "affinity purification mass spectrometry", "reported", "[1]"],
            ["SARS-CoV-2", "ORF6", "inhibits", "STAT1", "human",
             "reduced interferon signalling", "co-immunoprecipitation",
             "measured", "[1]"],
            ["SARS-CoV-2", "N/A", "binds", "ACE2", "human", "N/A",
             "affinity purification mass spectrometry", "reported", "[1]"],
        ),
        hv, mutation_response["sources"],
    )
    assert [r.get("Viral Protein") for r in result.rows] == ["ORF6"]
    assert result.dropped_incomplete == 2


def test_keep_empty_shows_which_end_is_missing(hv, mutation_response):
    """Dropping is the default, not the only option -- the rows stay auditable."""
    result = extract(
        _answer(
            ["SARS-CoV-2", "nsp13", "binds", "N/A", "human", "N/A",
             "affinity purification mass spectrometry", "reported", "[1]"],
        ),
        hv, mutation_response["sources"], keep_empty=True,
    )
    assert result.dropped_incomplete == 0
    assert "missing:Host Protein" in result.rows[0].flags


def test_a_column_is_flagged_once(hv, mutation_response):
    """Interaction Type is both a key-value column and a required one."""
    result = extract(
        _answer(
            ["SARS-CoV-2", "nsp15", "N/A", "N/A", "human", "N/A",
             "affinity purification mass spectrometry", "reported", "[1]"],
        ),
        hv, mutation_response["sources"], keep_empty=True,
    )
    flags = result.rows[0].flags
    assert flags.count("missing:Interaction Type") == 1


def test_a_template_without_required_columns_is_unaffected(mutation_response):
    """The mutation table has no required list, so nothing new is dropped."""
    registry = TemplateRegistry.from_declarations(declarations())
    assert registry.resolve("host-virus").required
    result = extract(
        mutation_response["answer"],
        _mutation_template(),
        mutation_response["sources"],
    )
    assert result.dropped_incomplete == 0
    assert result.rows


def _mutation_template():
    from litrag.templates import Template
    return Template(
        id="mutation", label="Mutation", output="table",
        columns=["Organism", "Gene Name", "Mutation", "Phenotype",
                 "Assertion", "Reference"],
    )


def test_the_prompt_asks_for_both_ends(hv):
    prompt, _, _ = build_prompt(hv, [{"content": "x", "metadata": {}}], "SARS-CoV-2")
    assert "Viral Protein, Interaction Type, Host Protein" in prompt
    assert "write no row at all" in prompt
    # The interactome trap that produced most of the junk rows.
    assert "without naming" in prompt
    assert "viral complex" in prompt


# -- direction ---------------------------------------------------------------
#
# "PUS7 binds 3' terminal regions of SARS-CoV-2 RNA" came back live. PUS7 is
# human and the RNA is viral, so the row is a real interaction written
# backwards. It cannot be repaired here -- reversing the pair would have to
# reverse the verb with it -- but it can be caught.

def test_a_host_target_named_after_the_virus_is_flagged(hv, mutation_response):
    result = extract(
        _answer(
            ["SARS-CoV-2", "PUS7", "binds", "3' terminal regions of SARS-CoV-2 RNA",
             "human", "N/A", "computational prediction", "measured", "[1]"],
        ),
        hv, mutation_response["sources"],
    )
    assert "reversed_pair" in result.rows[0].flags


def test_a_real_host_target_is_not_flagged(hv, mutation_response):
    result = extract(
        _answer(
            ["SARS-CoV-2", "ORF6", "inhibits", "STAT1", "human",
             "reduced interferon signalling", "co-immunoprecipitation",
             "measured", "[1]"],
        ),
        hv, mutation_response["sources"],
    )
    assert "reversed_pair" not in result.rows[0].flags


def test_spelling_of_the_organism_does_not_matter(hv, mutation_response):
    """"SARS CoV 2 RNA" is the same claim as "SARS-CoV-2 RNA"."""
    result = extract(
        _answer(
            ["SARS-CoV-2", "PUS7", "binds", "SARS CoV 2 genomic RNA", "human",
             "N/A", "computational prediction", "measured", "[1]"],
        ),
        hv, mutation_response["sources"],
    )
    assert "reversed_pair" in result.rows[0].flags


def test_a_phrase_is_not_a_list(hv, mutation_response):
    """"5' and 3' terminal regions" split into a Host Protein reading "5'"."""
    result = extract(
        _answer(
            ["SARS-CoV-2", "Nsp1", "binds",
             "5' and 3' terminal regions of host mRNA", "human",
             "impedes translation", "CLIP-seq", "measured", "[1]"],
        ),
        hv, mutation_response["sources"],
    )
    assert len(result.rows) == 1
    assert result.rows[0].get("Host Protein").startswith("5'")


def test_a_real_list_still_splits(hv, mutation_response):
    """The guard must not defeat ordinary compound splitting."""
    result = extract(
        _answer(
            ["SARS-CoV-2", "ORF6", "binds", "NUP98 and RAE1", "human", "N/A",
             "co-immunoprecipitation", "measured", "[1]"],
        ),
        hv, mutation_response["sources"],
    )
    assert sorted(r.get("Host Protein") for r in result.rows) == ["NUP98", "RAE1"]
