"""Row and record shapes for the extraction pipeline.

Single source of truth: extract/CONTRACT.md. This module mirrors that
contract exactly, including the display column sets in Literature.js
lines 17-34. Python standard library only. Importable on its own,
prints nothing on import.
"""

import copy
import sys

ASSERTION_PLACEHOLDER = "provisional, pending the leads"

DATA_TYPES = {
    "mutation": {
        "label": "Mutation",
        "columns": ["Organism", "Gene Name", "Mutation", "Phenotype", "Assertion", "Reference"],
        "fields": ["organism", "gene_name", "mutation", "phenotype", "assertion", "reference"],
    },
    "ppi": {
        "label": "Protein-Protein Interaction (PPI)",
        "columns": ["Pathogen", "Protein A", "Protein B", "Interaction Type", "Method", "Assertion", "Reference"],
        "fields": ["pathogen", "protein_a", "protein_b", "interaction_type", "method", "assertion", "reference"],
    },
}

OUTCOMES = ("cite", "qualify", "omit", "refuse")

OMIT_REASONS = (
    "unknown_chunk",
    "quote_mismatch",
    "schema_fail",
    "wrong_organism",
    "wrong_gene",
    "duplicate",
)

VERDICTS = (
    "correct",
    "wrong entity",
    "wrong relation or value",
    "not supported by the quote",
    "duplicate",
)

# Evidence identifiers that must be null rather than "" or "N/A" when absent.
_EVIDENCE_IDENTIFIER_FIELDS = ("pubmed_id", "pmc_id", "doi")


def row_id(gene, data_type, counter):
    """Build a row id: <gene lowercased>__<data type>__<4-digit counter>."""
    return "{}__{}__{:04d}".format(gene.lower(), data_type, counter)


def _empty_evidence():
    return {
        "chunk_id": None,
        "doc_id": None,
        "pubmed_id": None,
        "pmc_id": None,
        "doi": None,
        "title": None,
        "year": None,
        "passage": None,
        "passage_char_offset": None,
    }


def new_row(data_type):
    """Return a row skeleton for data_type with every field present."""
    if data_type not in DATA_TYPES:
        raise ValueError("unknown data type: {!r}".format(data_type))

    row = {
        "row_id": None,
        "outcome": None,
    }
    for field in DATA_TYPES[data_type]["fields"]:
        row[field] = ASSERTION_PLACEHOLDER if field == "assertion" else None
    row["evidence"] = _empty_evidence()
    row["citation_partial"] = False
    row["failure_layer_tags"] = []
    return row


def _is_blank(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def validate_row(data_type, row):
    """Return a list of problem strings, empty when row is valid."""
    problems = []

    if data_type not in DATA_TYPES:
        problems.append("unknown data type: {!r}".format(data_type))
        return problems

    for field in DATA_TYPES[data_type]["fields"]:
        if field not in row or _is_blank(row.get(field)):
            problems.append("missing or empty display field: {}".format(field))

    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        problems.append("missing evidence block")
    else:
        for field in ("chunk_id", "doc_id", "passage"):
            if _is_blank(evidence.get(field)):
                problems.append("missing or empty evidence.{}".format(field))
        for field in _EVIDENCE_IDENTIFIER_FIELDS:
            value = evidence.get(field)
            if isinstance(value, str) and (value == "" or value == "N/A"):
                problems.append(
                    "evidence.{} must be null when absent, not {!r}".format(field, value)
                )

    outcome = row.get("outcome")
    if outcome not in OUTCOMES:
        problems.append("outcome not in allowed set: {!r}".format(outcome))

    return problems


def refusal_record(data_type, gene, reason, chunks_searched, papers_searched):
    return {
        "outcome": "refuse",
        "data_type": data_type,
        "gene": gene,
        "reason": reason,
        "chunks_searched": chunks_searched,
        "papers_searched": papers_searched,
    }


def omitted_record(row, omit_reason, detail):
    if omit_reason not in OMIT_REASONS:
        raise ValueError("unknown omit reason: {!r}".format(omit_reason))
    record = copy.deepcopy(row)
    record["outcome"] = "omit"
    record["omit_reason"] = omit_reason
    record["detail"] = detail
    return record


def run_record(run_id, generated_at, model, prompt_sha256, prompt_edited, query):
    return {
        "run_id": run_id,
        "generated_at": generated_at,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "prompt_edited": prompt_edited,
        "query": query,
    }


def verification_record(row_id_value):
    return {
        "row_id": row_id_value,
        "verdict": None,
        "opened_paper": None,
        "seconds": None,
        "note": "",
    }


def output_envelope(run, retrieval, rows, omitted, refusals):
    emitted = len(rows)
    omitted_count = len(omitted)
    tally = {
        "proposed": emitted + omitted_count,
        "emitted": emitted,
        "omitted": omitted_count,
    }
    return {
        "schema_version": 1,
        "run": run,
        "retrieval": retrieval,
        "rows": rows,
        "omitted": omitted,
        "refusals": refusals,
        "tally": tally,
    }


def check_tally(envelope):
    """Return a list of problem strings, empty when the envelope's tally holds."""
    problems = []
    tally = envelope.get("tally", {})
    proposed = tally.get("proposed")
    emitted = tally.get("emitted")
    omitted = tally.get("omitted")

    if emitted is None or omitted is None or proposed is None or emitted + omitted != proposed:
        problems.append(
            "tally.emitted ({}) + tally.omitted ({}) != tally.proposed ({})".format(
                emitted, omitted, proposed
            )
        )

    if not envelope.get("rows") and not envelope.get("refusals"):
        problems.append("rows is empty but refusals holds no record")

    return problems


def _valid_mutation_row():
    row = new_row("mutation")
    row["row_id"] = row_id("PB2", "mutation", 7)
    row["outcome"] = "cite"
    row["organism"] = "Influenza A virus"
    row["gene_name"] = "PB2"
    row["mutation"] = "E627K"
    row["phenotype"] = "increased polymerase activity in mammalian cells"
    row["reference"] = "Subbarao 1993, J Virol"
    row["evidence"] = {
        "chunk_id": "chunk-123",
        "doc_id": "doc-456",
        "pubmed_id": "24899203",
        "pmc_id": "PMC4136279",
        "doi": "10.1128/jvi.00422-14",
        "title": "PB2 Mutations D701N and S714R Promote Adaptation ...",
        "year": 2014,
        "passage": "the exact sentence copied from the chunk",
        "passage_char_offset": 812,
    }
    return row


def _valid_ppi_row():
    row = new_row("ppi")
    row["row_id"] = row_id("PB2", "ppi", 1)
    row["outcome"] = "cite"
    row["pathogen"] = "Influenza A virus"
    row["protein_a"] = "PB2"
    row["protein_b"] = "importin-alpha"
    row["interaction_type"] = "binding"
    row["method"] = "co-immunoprecipitation"
    row["reference"] = "Tarendeau 2007, Nat Struct Mol Biol"
    row["evidence"] = {
        "chunk_id": "chunk-abc",
        "doc_id": "doc-def",
        "pubmed_id": "17334375",
        "pmc_id": None,
        "doi": "10.1038/nsmb1201",
        "title": "Structure and nuclear import function of the C-terminal domain of influenza virus polymerase PB2 subunit",
        "year": 2007,
        "passage": "PB2 interacts with importin-alpha via its C-terminal domain",
        "passage_char_offset": 44,
    }
    return row


def _self_check():
    results = []

    valid_mutation = _valid_mutation_row()
    problems = validate_row("mutation", valid_mutation)
    ok = problems == []
    results.append(("valid mutation row passes validate_row", ok))

    valid_ppi = _valid_ppi_row()
    problems = validate_row("ppi", valid_ppi)
    ok = problems == []
    results.append(("valid ppi row passes validate_row", ok))

    def mutate(base, apply_fn):
        row = copy.deepcopy(base)
        apply_fn(row)
        return row

    cases = [
        ("missing chunk_id is rejected", lambda r: r["evidence"].__setitem__("chunk_id", None)),
        ("empty passage is rejected", lambda r: r["evidence"].__setitem__("passage", "")),
        ("blank assertion is rejected", lambda r: r.__setitem__("assertion", "")),
        ("outcome 'maybe' is rejected", lambda r: r.__setitem__("outcome", "maybe")),
        ("pmc_id 'N/A' is rejected", lambda r: r["evidence"].__setitem__("pmc_id", "N/A")),
        ("missing display field is rejected", lambda r: r.pop("phenotype")),
    ]

    for label, apply_fn in cases:
        row = mutate(valid_mutation, apply_fn)
        problems = validate_row("mutation", row)
        ok = len(problems) > 0
        results.append((label, ok))

    all_ok = True
    for label, ok in results:
        status = "PASS" if ok else "FAIL"
        print("{}: {}".format(status, label))
        if not ok:
            all_ok = False

    return all_ok


if __name__ == "__main__":
    success = _self_check()
    sys.exit(0 if success else 1)
