"""Collapse rows that state the same fact.

Identity is per-template, because what makes two rows "the same" differs by data
type: a protein interaction is symmetric (A-B is B-A), a mutation is identified
by its position and substitution regardless of notation. Merged rows keep the
union of their supporting citations and a count, so a fact found by three
queries is visibly better supported than one found once.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .extract import Citation, Row
from .normalize import (normalize_antibiotic, normalize_gene,
                        normalize_mutation, normalize_sir, normalize_text)

# Which columns establish identity, per template. Columns absent from a row are
# skipped, so a partially-populated table still dedups on what it has.
IDENTITY_COLUMNS: Dict[str, Sequence[str]] = {
    "mutation": ("Organism", "Gene Name", "Mutation"),
    "protein-function": ("Organism", "Gene Name", "Function"),
    "ppi-extraction": ("Pathogen", "Protein A", "Protein B"),
    # One measurement is one strain tested against one drug. Accession and
    # BioSample identify the strain rather than the observation, and MIC/SIR
    # are the result, so none of them belong in the key.
    "ast": ("Organism", "Strain", "Antibiotic"),
}

# Pairs treated as unordered, because the relation they describe is symmetric.
SYMMETRIC_PAIRS: Dict[str, Tuple[str, str]] = {
    "ppi-extraction": ("Protein A", "Protein B"),
}

_GENE_COLUMNS = {"gene name", "gene", "protein a", "protein b", "protein"}
_MUTATION_COLUMNS = {"mutation", "variant", "allele"}
_ANTIBIOTIC_COLUMNS = {"antibiotic", "drug", "antimicrobial", "agent"}
_SIR_COLUMNS = {"sir", "interpretation", "category", "phenotype (sir)"}
_REFERENCE_COLUMNS = {"reference", "references", "citation", "citations", "source"}


def _normalize_cell(column: str, value: str, gene_hint: str = "") -> str:
    key = column.strip().lower()
    if key in _MUTATION_COLUMNS:
        return normalize_mutation(value, gene_hint)
    if key in _GENE_COLUMNS:
        return normalize_gene(value)
    if key in _ANTIBIOTIC_COLUMNS:
        return normalize_antibiotic(value)
    if key in _SIR_COLUMNS:
        return normalize_sir(value)
    return normalize_text(value)


def identity_key(row: Row, template_id: str, columns: Sequence[str]) -> Tuple:
    """A hashable identity for a row under its template's rules."""
    identity = [c for c in IDENTITY_COLUMNS.get(template_id, tuple()) if c in columns]
    if not identity:
        # Unknown template: fall back to every column except the reference,
        # which varies between papers reporting the same fact.
        identity = [c for c in columns if c.strip().lower() not in {
            "reference", "references", "citation", "citations", "source",
        }]

    gene_hint = ""
    for column in identity:
        if column.strip().lower() in _GENE_COLUMNS:
            gene_hint = row.get(column)
            break

    symmetric = SYMMETRIC_PAIRS.get(template_id)
    if symmetric and all(c in identity for c in symmetric):
        left, right = symmetric
        pair = tuple(sorted((
            _normalize_cell(left, row.get(left)),
            _normalize_cell(right, row.get(right)),
        )))
        others = tuple(
            _normalize_cell(c, row.get(c), gene_hint)
            for c in identity if c not in symmetric
        )
        return (template_id,) + others + pair

    return (template_id,) + tuple(
        _normalize_cell(c, row.get(c), gene_hint) for c in identity
    )


def _merge_citations(existing: List[Citation], incoming: Iterable[Citation]) -> List[Citation]:
    # Citation.identity, not the marker: once passages are split across parallel
    # calls each call numbers its sources from 1, so "[1]" in two batches is two
    # different papers. Keying on the marker silently dropped the second.
    seen = {c.identity for c in existing}
    merged = list(existing)
    for citation in incoming:
        key = citation.identity
        if key in seen:
            continue
        seen.add(key)
        merged.append(citation)
    return merged


def _richer(candidate: str, current: str) -> bool:
    """Prefer the more informative of two values for the same field."""
    return len(candidate.strip()) > len(current.strip())


def dedupe(
    rows: Sequence[Row],
    template_id: str,
    columns: Sequence[str],
) -> List[Row]:
    """Merge rows sharing an identity, preserving citations and support counts."""
    merged: Dict[Tuple, Row] = {}
    order: List[Tuple] = []

    for row in rows:
        key = identity_key(row, template_id, columns)
        existing = merged.get(key)

        if existing is None:
            clone = Row(
                values=dict(row.values),
                citations=list(row.citations),
                flags=list(row.flags),
                n_support=row.n_support,
                query_ids=list(row.query_ids),
                provenance=dict(row.provenance),
                variants={k: list(v) for k, v in row.variants.items()},
            )
            merged[key] = clone
            order.append(key)
            continue

        existing.n_support += row.n_support
        existing.citations = _merge_citations(existing.citations, row.citations)
        for query_id in row.query_ids:
            if query_id not in existing.query_ids:
                existing.query_ids.append(query_id)

        # Keep the fuller description in non-identity columns, but record that
        # the sources disagreed. Merging "isoniazid resistance" into
        # "moderate-level isoniazid resistance" would otherwise present one
        # paper's qualifier as if every source had reported it.
        identity = set(IDENTITY_COLUMNS.get(template_id, tuple()))
        for column in columns:
            # Reference columns are expected to differ -- merging rows from
            # different papers is the point -- and the real citations are
            # tracked separately on the row.
            if column in identity or column.strip().lower() in _REFERENCE_COLUMNS:
                continue
            incoming = row.values.get(column, "")
            if not incoming:
                continue
            current = existing.values.get(column, "")
            if current and _normalize_cell(column, incoming) != _normalize_cell(column, current):
                existing.variants.setdefault(column, [])
                for value in (current, incoming):
                    if value and value not in existing.variants[column]:
                        existing.variants[column].append(value)
                flag = f"merged_variants:{column}"
                if flag not in existing.flags:
                    existing.flags.append(flag)
            if _richer(incoming, current):
                existing.values[column] = incoming

        for flag in row.flags:
            if flag not in existing.flags:
                existing.flags.append(flag)
        # A fact confirmed by another row is no longer uncited.
        if existing.citations and "no_citation" in existing.flags:
            existing.flags.remove("no_citation")

    return [merged[k] for k in order]
