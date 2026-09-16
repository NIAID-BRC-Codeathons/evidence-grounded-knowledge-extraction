"""Templates come from the server, not from hardcoded constants."""

import pytest

from litrag.templates import TemplateError, TemplateRegistry


def test_registry_matches_server(registry):
    assert registry.ids == [
        "literature-summary", "mutation", "ppi-extraction", "protein-function",
    ]


def test_columns_come_from_declaration(registry):
    ppi = registry.resolve("ppi-extraction")
    assert ppi.columns == [
        "Pathogen", "Protein A", "Protein B",
        "Interaction Type", "Method", "Assertion", "Reference",
    ]
    assert ppi.is_table and ppi.version == 2


def test_aliases_resolve(registry):
    assert registry.resolve("ppi").id == "ppi-extraction"
    assert registry.resolve("summary").id == "literature-summary"
    assert registry.resolve("MUTATION").id == "mutation"


def test_unknown_type_lists_alternatives(registry):
    with pytest.raises(TemplateError, match="unknown data type"):
        registry.resolve("nonsense")


def test_missing_required_slot_is_caught_locally(registry):
    """A 422 from the server is a worse error than one raised before the call."""
    with pytest.raises(TemplateError, match="requires a value for 'organism'"):
        registry.resolve("mutation").validate_vars({"genes": "katG"})


def test_overlong_slot_is_rejected(registry):
    with pytest.raises(TemplateError, match="at most"):
        registry.resolve("mutation").validate_vars({"organism": "x" * 500})


def test_unknown_slot_is_rejected(registry):
    with pytest.raises(TemplateError, match="no slot"):
        registry.resolve("mutation").validate_vars({"organism": "M. tb", "bogus": "1"})


def test_optional_slots_are_dropped_when_blank(registry):
    got = registry.resolve("mutation").validate_vars({"organism": "M. tb", "genes": "  "})
    assert got == {"organism": "M. tb"}


def test_new_server_template_needs_no_code_change():
    """The registry is data-driven, so an added template just works."""
    registry = TemplateRegistry.from_declarations([{
        "id": "operon", "label": "Operon", "output": "table",
        "version": 1, "columns": ["Organism", "Operon", "Reference"],
        "slots": [{"name": "organism", "required": True}],
    }])
    assert registry.resolve("operon").columns == ["Organism", "Operon", "Reference"]
