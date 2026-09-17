"""Output rendering."""

import csv
import io
import json

from litrag import formats
from litrag.extract import extract
from litrag.provenance import PROVENANCE_COLUMNS


def _extraction(registry, response, name="mutation"):
    template = registry.resolve(name)
    return template, extract(response["answer"], template, response["sources"])


def test_tsv_roundtrips(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    text = formats.render(result, result.rows, "tsv", provenance=False)
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    assert len(rows) == result.n_rows
    assert rows[0]["Gene Name"] == "katG"


def test_provenance_columns_present_and_omittable(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    with_prov = formats.render(result, result.rows, "tsv", provenance=True)
    without = formats.render(result, result.rows, "tsv", provenance=False)
    header = with_prov.splitlines()[0]
    for column in PROVENANCE_COLUMNS:
        assert column in header
    assert "_template_hash" not in without.splitlines()[0]


def test_csv_quotes_embedded_commas(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    result.rows[0].values["Phenotype"] = "resistance, high-level"
    text = formats.render(result, result.rows, "csv", provenance=False)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert parsed[0]["Phenotype"] == "resistance, high-level"


def test_json_includes_citations_and_flags(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    payload = json.loads(formats.render(result, result.rows, "json"))
    assert payload["n_rows"] == result.n_rows
    assert payload["columns"] == template.columns
    assert payload["rows"][0]["citations"][0]["pmid"] == "40580943"
    assert "flags" in payload["rows"][0]


def test_jsonl_is_one_object_per_line(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    text = formats.render(result, result.rows, "jsonl")
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == result.n_rows
    assert all(json.loads(line) for line in lines)


def test_markdown_escapes_pipes(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    result.rows[0].values["Phenotype"] = "a | b"
    text = formats.render(result, result.rows, "md")
    assert r"a \| b" in text


def test_pretty_table_is_aligned(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    text = formats.render(result, result.rows, "table")
    lines = text.splitlines()
    assert set(lines[1]) <= {"-", " "}
    assert "Gene Name" in lines[0]


def test_prose_answer_renders_as_text(registry, mutation_response):
    template = registry.resolve("summary")
    result = extract("Prose about katG [1].", template, mutation_response["sources"])
    assert formats.render(result, result.rows, "tsv").strip() == "Prose about katG [1]."
    assert json.loads(formats.render(result, result.rows, "json"))["answer"]


def test_unknown_format_is_rejected(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    try:
        formats.render(result, result.rows, "xlsx")
    except ValueError as exc:
        assert "unknown format" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_chunk_ids_are_exported(registry, mutation_response):
    """A curated row must be traceable to the passages behind it; these ids go
    straight to /v1/chunks?ids= to fetch the text back."""
    template, result = _extraction(registry, mutation_response)
    text = formats.render(result, result.rows, "tsv", provenance=False)
    header = text.splitlines()[0]
    assert "_chunk_ids" in header and "_markers" in header

    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    assert rows[0]["_chunk_ids"]
    assert rows[0]["_markers"]


def test_json_carries_chunk_references(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    payload = json.loads(formats.render(result, result.rows, "json"))
    chunks = payload["rows"][0]["citations"][0]["chunks"]
    assert chunks and chunks[0]["chunk_id"]
    assert "start_char" in chunks[0]
    assert payload["rows"][0]["chunk_ids"]


def test_source_summary_exposes_the_span(registry, mutation_response):
    template, result = _extraction(registry, mutation_response)
    payload = json.loads(formats.render(
        result, result.rows, "json", sources=mutation_response["sources"]))
    source = payload["sources"][0]
    assert source["chunk_id"] and "start_char" in source


def test_merged_alternatives_are_joined_in_flat_formats(registry, mutation_response):
    """A flat table has nowhere to put alternatives except the cell.

    Exporting only the representative value silently dropped the fact that the
    sources disagreed.
    """
    template, result = _extraction(registry, mutation_response)
    row = result.rows[0]
    row.values["Phenotype"] = "isoniazid resistance"
    row.variants["Phenotype"] = [
        "isoniazid resistance", "moderate-level isoniazid resistance",
    ]

    text = formats.render(result, [row], "tsv", provenance=False)
    cell = list(csv.DictReader(io.StringIO(text), delimiter="\t"))[0]["Phenotype"]
    assert cell == "isoniazid resistance; moderate-level isoniazid resistance"


def test_unmerged_cell_is_unchanged(registry, mutation_response):
    """No variants means no separator -- a plain row must look plain."""
    template, result = _extraction(registry, mutation_response)
    row = result.rows[0]
    row.variants.clear()
    text = formats.render(result, [row], "tsv", provenance=False)
    cell = list(csv.DictReader(io.StringIO(text), delimiter="\t"))[0]["Gene Name"]
    assert cell == row.get("Gene Name") and ";" not in cell


def test_json_keeps_alternatives_structured(registry, mutation_response):
    """Flat formats join; JSON should not, since it can represent both."""
    template, result = _extraction(registry, mutation_response)
    row = result.rows[0]
    row.values["Phenotype"] = "a"
    row.variants["Phenotype"] = ["a", "b"]
    payload = json.loads(formats.render(result, [row], "json"))
    assert payload["rows"][0]["Phenotype"] == "a"
    assert payload["rows"][0]["variants"]["Phenotype"] == ["a", "b"]
