"""The AST data type.

Templates come from /v1/prompt-templates, which is read-only (POST -> 405), so
this type is defined by LitRAG and runs only on a local generator. These tests
cover that constraint and the extraction rules specific to susceptibility data.
"""

import httpx
import pytest

from litrag.client import RagStackClient
from litrag.config import Config
from litrag.dedup import dedupe, identity_key
from litrag.extract import Row, extract
from litrag.llm import PRESETS
from litrag.local_templates import BUILTIN, declarations
from litrag.normalize import normalize_antibiotic, normalize_sir
from litrag.pipeline import QuerySpec, run_query
from litrag.prompts import build_prompt
from litrag.templates import TemplateError, TemplateRegistry

AST_COLUMNS = [
    "Organism", "Strain", "GenBank Accession", "BioSample",
    "Antibiotic", "MIC", "SIR", "Reference",
]


@pytest.fixture
def ast_registry(registry):
    """Server templates plus the locally-defined ones."""
    decls = [
        {"id": t.id, "label": t.label, "output": t.output, "version": t.version,
         "hash": t.hash, "columns": t.columns, "slots": []}
        for t in registry
    ] + declarations()
    return TemplateRegistry.from_declarations(decls)


@pytest.fixture
def ast(ast_registry):
    return ast_registry.resolve("ast")


def _answer(*rows):
    head = "\t".join(AST_COLUMNS)
    return "\n".join([head] + ["\t".join(r) for r in rows])


# -- declaration ---------------------------------------------------------

def test_requested_columns_are_present(ast):
    assert ast.columns == AST_COLUMNS
    assert ast.is_table and ast.is_local


def test_aliases_resolve(ast_registry):
    for alias in ("ast", "amr", "susceptibility", "mic"):
        assert ast_registry.resolve(alias).id == "ast"


def test_local_template_still_has_a_hash(ast):
    """Provenance needs one even though the server did not supply it."""
    assert ast.hash and len(ast.hash) == 16


def test_hash_changes_when_the_declaration_does():
    a = declarations()[0]["hash"]
    modified = dict(BUILTIN[0], columns=BUILTIN[0]["columns"] + ["Extra"])
    from litrag.local_templates import _finalize
    assert _finalize(modified)["hash"] != a


def test_a_server_template_wins_a_name_clash(registry, mutation_response):
    """If the operator later adds this id, the hosted version should take over."""
    def handler(request):
        return httpx.Response(200, json={"templates": [
            {"id": "ast", "label": "Server AST", "output": "table",
             "version": 9, "columns": ["A", "B"], "slots": []},
        ]})

    client = RagStackClient(
        Config(api_key="k", base_url="https://example.invalid"),
        client=httpx.Client(transport=httpx.MockTransport(handler),
                            base_url="https://example.invalid"))
    resolved = TemplateRegistry.fetch(client).resolve("ast")
    assert resolved.version == 9 and not resolved.is_local


# -- the local-generator constraint --------------------------------------

def test_hosted_path_is_refused_with_an_explanation(ast_registry, mutation_response):
    """Better than letting the server 404 on an id it has never heard of."""
    client = RagStackClient(
        Config(api_key="k", base_url="https://example.invalid"),
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=mutation_response)),
            base_url="https://example.invalid"))
    with pytest.raises(TemplateError, match="needs a local generator"):
        run_query(client, ast_registry, QuerySpec(organism="E. coli", data_type="ast"))


def test_local_path_runs(ast_registry, mutation_response):
    from litrag.llm import LlmClient

    sources = mutation_response["sources"]
    answer = _answer(
        ["Escherichia coli", "1539851", "N/A", "SAMEA116133969",
         "Ampicillin", ">32 mg/L", "R", "[1]"],
    )

    def rag(request):
        return httpx.Response(200, json={"sources": sources})

    def gen(request):
        return httpx.Response(200, json={
            "model": "Qwen/Qwen3.6-35B-A3B",
            "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
            "usage": {}})

    client = RagStackClient(
        Config(api_key="k", base_url="https://example.invalid"),
        client=httpx.Client(transport=httpx.MockTransport(rag),
                            base_url="https://example.invalid"))
    llm = LlmClient(PRESETS["qwen"], client=httpx.Client(transport=httpx.MockTransport(gen)))

    run = run_query(client, ast_registry, QuerySpec(organism="Escherichia coli", data_type="ast"),
                    endpoint=PRESETS["qwen"], llm_client=llm)
    row = run.rows[0]
    assert row.get("BioSample") == "SAMEA116133969"
    assert row.get("MIC") == ">32 mg/L" and row.get("SIR") == "R"
    assert row.citations[0].pmid == "40580943"
    assert row.provenance["_template"] == "ast"
    assert row.provenance["_template_hash"]


# -- prompt --------------------------------------------------------------

def test_prompt_carries_the_type_specific_rules(ast, mutation_response):
    prompt, _, _ = build_prompt(ast, mutation_response["sources"], "Escherichia coli")
    assert "strain-antibiotic pair" in prompt
    assert "never apply a breakpoint yourself" in prompt
    assert "Do not guess them" in prompt
    assert "\t".join(AST_COLUMNS) in prompt


def test_prompt_forbids_inferring_one_measure_from_the_other(ast, mutation_response):
    """An MIC derived from an S/I/R letter would be fabricated data."""
    prompt, _, _ = build_prompt(ast, mutation_response["sources"], "E. coli")
    assert "Never infer one of MIC and SIR from the other" in prompt


# -- extraction ----------------------------------------------------------

def test_row_without_mic_or_sir_is_dropped(ast, mutation_response):
    """Naming a strain and a drug with no result reports nothing."""
    answer = _answer(
        ["E. coli", "K-12", "N/A", "N/A", "Ampicillin", "N/A", "N/A", "[1]"],
        ["E. coli", "K-12", "N/A", "N/A", "Meropenem", "0.5 mg/L", "S", "[1]"],
    )
    result = extract(answer, ast, mutation_response["sources"])
    assert result.dropped_empty == 1
    assert result.n_rows == 1 and result.rows[0].get("Antibiotic") == "Meropenem"


def test_mic_alone_is_enough(ast, mutation_response):
    """Papers often publish an MIC without the categorical call."""
    answer = _answer(["E. coli", "K-12", "N/A", "N/A", "Ampicillin", "8 mg/L", "N/A", "[1]"])
    result = extract(answer, ast, mutation_response["sources"])
    assert result.n_rows == 1 and result.dropped_empty == 0


def test_sir_alone_is_enough(ast, mutation_response):
    answer = _answer(["E. coli", "K-12", "N/A", "N/A", "Ampicillin", "N/A", "R", "[1]"])
    result = extract(answer, ast, mutation_response["sources"])
    assert result.n_rows == 1 and result.dropped_empty == 0


def test_mic_text_is_preserved_verbatim(ast, mutation_response):
    """Units and inequalities carry the meaning; they must survive untouched."""
    answer = _answer(
        ["E. coli", "A", "N/A", "N/A", "Ciprofloxacin", "≤0.125 mg/L", "S", "[1]"],
        ["E. coli", "B", "N/A", "N/A", "Ampicillin", ">32 mg/L", "R", "[1]"],
        ["E. coli", "C", "N/A", "N/A", "Gentamicin", "2 µg/mL", "S", "[1]"],
    )
    result = extract(answer, ast, mutation_response["sources"])
    assert [r.get("MIC") for r in result.rows] == \
        ["≤0.125 mg/L", ">32 mg/L", "2 µg/mL"]


def test_multiple_antibiotics_in_one_row_are_split(ast, mutation_response):
    answer = _answer(
        ["E. coli", "K-12", "N/A", "N/A", "Ampicillin, Meropenem",
         ">32 mg/L, 0.5 mg/L", "R, S", "[1]"],
    )
    result = extract(answer, ast, mutation_response["sources"])
    assert result.split_compound == 1
    assert [r.get("Antibiotic") for r in result.rows] == ["Ampicillin", "Meropenem"]
    assert [r.get("SIR") for r in result.rows] == ["R", "S"]


# -- identity and merging ------------------------------------------------

def test_identity_is_strain_times_antibiotic(ast):
    """Accession and BioSample identify the strain, not the measurement, so a
    paper that omits them must still merge with one that reports them."""
    with_ids = Row(values={"Organism": "E. coli", "Strain": "K-12",
                           "GenBank Accession": "CP012345",
                           "BioSample": "SAMN01", "Antibiotic": "Ampicillin"})
    without = Row(values={"Organism": "E. coli", "Strain": "K-12",
                          "Antibiotic": "ampicillin"})
    assert identity_key(with_ids, "ast", ast.columns) == identity_key(without, "ast", ast.columns)


def test_different_antibiotics_stay_separate(ast):
    a = Row(values={"Organism": "E. coli", "Strain": "K-12", "Antibiotic": "Ampicillin"})
    b = Row(values={"Organism": "E. coli", "Strain": "K-12", "Antibiotic": "Meropenem"})
    assert len(dedupe([a, b], "ast", ast.columns)) == 2


def test_different_strains_stay_separate(ast):
    a = Row(values={"Organism": "E. coli", "Strain": "K-12", "Antibiotic": "Ampicillin"})
    b = Row(values={"Organism": "E. coli", "Strain": "O157:H7", "Antibiotic": "Ampicillin"})
    assert len(dedupe([a, b], "ast", ast.columns)) == 2


def test_salt_forms_are_the_same_drug(ast):
    a = Row(values={"Organism": "E. coli", "Strain": "K-12", "Antibiotic": "Ampicillin sodium"})
    b = Row(values={"Organism": "E. coli", "Strain": "K-12", "Antibiotic": "Ampicillin"})
    assert len(dedupe([a, b], "ast", ast.columns)) == 1


def test_wording_of_sir_is_not_a_disagreement(ast):
    """"R" and "Resistant" agree; flagging that would be noise."""
    a = Row(values={"Organism": "E. coli", "Strain": "K", "Antibiotic": "Amp", "SIR": "R"})
    b = Row(values={"Organism": "E. coli", "Strain": "K", "Antibiotic": "Amp", "SIR": "Resistant"})
    merged = dedupe([a, b], "ast", ast.columns)
    assert len(merged) == 1
    assert not any(f.startswith("merged_variants:SIR") for f in merged[0].flags)


def test_conflicting_sir_is_flagged(ast):
    """Two papers calling the same strain-drug pair S and R is a real conflict."""
    a = Row(values={"Organism": "E. coli", "Strain": "K", "Antibiotic": "Amp", "SIR": "S"})
    b = Row(values={"Organism": "E. coli", "Strain": "K", "Antibiotic": "Amp", "SIR": "R"})
    merged = dedupe([a, b], "ast", ast.columns)
    assert len(merged) == 1
    assert "merged_variants:SIR" in merged[0].flags
    assert set(merged[0].variants["SIR"]) == {"S", "R"}


def test_differing_mics_are_preserved_as_variants(ast):
    a = Row(values={"Organism": "E. coli", "Strain": "K", "Antibiotic": "Amp", "MIC": "8 mg/L"})
    b = Row(values={"Organism": "E. coli", "Strain": "K", "Antibiotic": "Amp", "MIC": "16 mg/L"})
    merged = dedupe([a, b], "ast", ast.columns)
    assert merged[0].n_support == 2
    assert set(merged[0].variants["MIC"]) == {"8 mg/L", "16 mg/L"}


@pytest.mark.parametrize("word,expected", [
    ("R", "R"), ("Resistant", "R"), ("resistant (R)", "R"), ("non-susceptible", "R"),
    ("S", "S"), ("Susceptible", "S"), ("sensitive", "S"),
    ("I", "I"), ("Intermediate", "I"), ("SDD", "I"),
])
def test_sir_vocabulary(word, expected):
    assert normalize_sir(word) == expected


@pytest.mark.parametrize("name", [
    "Ciprofloxacin", "ciprofloxacin", "CIPROFLOXACIN", "Ciprofloxacin (CIP)",
])
def test_antibiotic_name_variants(name):
    assert normalize_antibiotic(name) == "ciprofloxacin"


def test_distinct_drugs_are_not_conflated():
    assert normalize_antibiotic("Ampicillin") != normalize_antibiotic("Amoxicillin")
    assert normalize_antibiotic("Cefotaxime") != normalize_antibiotic("Ceftazidime")
