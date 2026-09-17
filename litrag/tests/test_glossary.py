"""Column and flag definitions."""

import pytest

from litrag import glossary
from litrag.local_templates import declarations
from litrag.templates import TemplateRegistry


def test_every_column_of_every_template_is_defined(registry):
    """A column with no definition gets no tooltip, which reads as an oversight.

    Covers the server's templates and the locally-defined ones together.
    """
    decls = [
        {"id": t.id, "label": t.label, "output": t.output,
         "columns": t.columns, "slots": []}
        for t in registry
    ] + declarations()

    undefined = []
    for template in TemplateRegistry.from_declarations(decls):
        for column in template.columns or []:
            if not glossary.describe_column(column):
                undefined.append(f"{template.id}.{column}")
    assert not undefined, f"columns without a definition: {undefined}"


def test_derived_columns_are_defined():
    """Support, Citations and Flags are added by the UI, not by a template."""
    for column in ("Support", "Citations", "Flags"):
        assert glossary.describe_column(column)


def test_lookup_is_case_and_space_insensitive():
    assert glossary.describe_column("MIC") == glossary.describe_column("mic")
    assert glossary.describe_column("Gene  Name") == glossary.describe_column("gene name")
    assert glossary.describe_column(" Protein A ") == glossary.describe_column("protein a")


def test_unknown_column_has_no_definition():
    """A template added server-side should get no tooltip, not a wrong one."""
    assert glossary.describe_column("Some Future Column") == ""
    assert glossary.describe_column(None) == ""
    assert glossary.describe_column("") == ""


@pytest.mark.parametrize("flag", [
    "off_target_gene", "citation_not_in_sources", "no_citation",
    "unresolved_citation", "merged_variants", "compound_row",
    "missing", "evidence_free",
])
def test_every_flag_the_extractor_emits_is_defined(flag):
    assert glossary.describe_flag(flag)


def test_qualified_flags_resolve_on_their_prefix():
    """Flags carry a column suffix: merged_variants:Phenotype, missing:MIC/SIR."""
    assert glossary.describe_flag("merged_variants:Phenotype") == \
        glossary.describe_flag("merged_variants")
    assert glossary.describe_flag("missing:MIC/SIR") == glossary.describe_flag("missing")


def test_unknown_flag_has_no_definition():
    assert glossary.describe_flag("something_new") == ""
    assert glossary.describe_flag(None) == ""


def test_for_columns_skips_the_undefined():
    described = glossary.for_columns(["MIC", "Some Future Column", "SIR"])
    assert set(described) == {"MIC", "SIR"}


def test_definitions_read_as_sentences():
    """These are shown to curators, not to developers."""
    for entry in glossary.entries():
        text = entry["text"]
        assert text[0].isupper(), f"{entry['term']} does not start with a capital"
        assert text.rstrip().endswith("."), f"{entry['term']} is not a sentence"
        assert len(text) > 20, f"{entry['term']} is too terse to help"


def test_caveats_are_stated_where_they_matter():
    """The definitions carry the rules a curator would otherwise have to infer."""
    assert "never converted" in glossary.describe_column("MIC").lower()
    assert "never derived" in glossary.describe_column("SIR").lower()
    assert "never inferred" in glossary.describe_column("BioSample").lower()
    assert "order carries no meaning" in glossary.describe_column("Protein A").lower()
