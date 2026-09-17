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
                        normalize_glyco_type, normalize_mutation,
                        normalize_site, normalize_sir, normalize_text)

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
    # A site on a protein is the unit. Glycan, method and effect are things
    # observed about that site, not part of what identifies it.
    "glycosylation": ("Organism", "Protein", "Site"),
}

# Pairs treated as unordered, because the relation they describe is symmetric.
SYMMETRIC_PAIRS: Dict[str, Tuple[str, str]] = {
    "ppi-extraction": ("Protein A", "Protein B"),
}

# Columns that are one measurement rather than several, and so must be carried
# across a merge together. Resolving them independently picks a winner per
# column and can emit a combination no source reported: from (0.5, Susceptible)
# and (>128, Resistant) the longest-value rule takes ">128" and "Susceptible",
# which is clinically impossible. The AST template already forbids the model
# from inferring one of MIC and SIR from the other -- the merge must not either.
CORRELATED_COLUMNS: Dict[str, Sequence[Tuple[str, ...]]] = {
    "ast": (("MIC", "SIR"),),
}

_GENE_COLUMNS = {"gene name", "gene", "protein a", "protein b", "protein"}
_MUTATION_COLUMNS = {"mutation", "variant", "allele"}
_ANTIBIOTIC_COLUMNS = {"antibiotic", "drug", "antimicrobial", "agent"}
_SIR_COLUMNS = {"sir", "interpretation", "category", "phenotype (sir)"}
_SITE_COLUMNS = {"site", "position", "residue"}
_GLYCO_TYPE_COLUMNS = {"glycosylation type", "glycan type", "linkage"}
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
    if key in _SITE_COLUMNS:
        return normalize_site(value)
    if key in _GLYCO_TYPE_COLUMNS:
        return normalize_glyco_type(value)
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


def _citation_key(citation: Citation) -> str:
    """Stable "is this the same paper?" key.

    The marker is the LAST resort, not the fallback it used to be. Passages are
    now split across several concurrent generation calls, and each call numbers
    its own sources from 1 -- so "[1]" in two batches is two different papers,
    and keying on the marker silently merged them, dropping one paper's evidence
    as a duplicate of an unrelated one. The Dengue and Influenza collections
    publish no pmid and no pmcid at all, so their chunks land in exactly that
    fallback.

    Publication identifiers stay ahead of doc_id deliberately. Checked against
    the live index: 16 of 111 PMIDs from one query came back under more than one
    doc_id, one of them under five, because PMC gives figures and tables their
    own ids. Keying on doc_id first would report one paper as several
    independent sources and inflate n_support.
    """
    doc = next((c.doc_id for c in citation.chunks if c.doc_id), "")
    chunk = next((c.chunk_id for c in citation.chunks if c.chunk_id), "")
    return (citation.pmid or citation.doi or citation.pmcid or doc or chunk
            or f"marker:{citation.marker}")


def _merge_citations(existing: List[Citation], incoming: Iterable[Citation]) -> List[Citation]:
    """Union two citation lists, keeping every supporting passage.

    When two merged rows cite the same paper through different chunks, both
    chunks are evidence for the combined row -- discarding one would hide where
    half the support came from.
    """
    merged = list(existing)
    by_key = {_citation_key(c): c for c in merged}

    for citation in incoming:
        key = _citation_key(citation)
        current = by_key.get(key)
        if current is None:
            by_key[key] = citation
            merged.append(citation)
            continue
        seen = {c.chunk_id or c.marker for c in current.chunks}
        for chunk in citation.chunks:
            if (chunk.chunk_id or chunk.marker) not in seen:
                seen.add(chunk.chunk_id or chunk.marker)
                current.chunks.append(chunk)
        current.chunks.sort(key=lambda c: c.marker)
    return merged


def _richer(candidate: str, current: str) -> bool:
    """Prefer the more informative of two values for the same field."""
    return len(candidate.strip()) > len(current.strip())


def _stated(value: str) -> bool:
    """Whether a cell carries a finding at all.

    The AST template tells the model to write "N/A" into whichever of MIC and
    SIR a paper did not report, so an N/A half is a gap another row may fill,
    not a contradiction to flag.
    """
    text = (value or "").strip()
    return bool(text) and text.lower() not in {"n/a", "na", "none", "-"}


def _merge_correlated(existing: Row, incoming_row: Row, group: Sequence[str]) -> None:
    """Carry a group of columns across a merge as a single measurement."""
    current = [existing.values.get(c, "") for c in group]
    incoming = [incoming_row.values.get(c, "") for c in group]

    # Alternatives are still recorded per column: the table and the UI look
    # them up by column name, so a composite key would hide them entirely.
    disagreed = False
    for column, was, now in zip(group, current, incoming):
        if not (_stated(was) and _stated(now)):
            continue
        if _normalize_cell(column, was) == _normalize_cell(column, now):
            continue
        disagreed = True
        existing.variants.setdefault(column, [])
        for value in (was, now):
            if value not in existing.variants[column]:
                existing.variants[column].append(value)
        flag = f"merged_variants:{column}"
        if flag not in existing.flags:
            existing.flags.append(flag)

    if disagreed:
        # Keep the pair already displayed rather than taking the richer value in
        # each column independently. It came from the higher-ranked row and is
        # internally coherent; splicing half of this one together with half of
        # that one is exactly what invented ">128 / Susceptible".
        flag = "conflicting_result:" + "/".join(group)
        if flag not in existing.flags:
            existing.flags.append(flag)
        return

    # Nothing contradicts, so let this row complete any half that was missing.
    for column, was, now in zip(group, current, incoming):
        if _stated(now) and (not _stated(was) or _richer(now, was)):
            existing.values[column] = now


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

        # Correlated groups are resolved first, as a unit, and then held out of
        # the per-column pass below so it cannot recombine their halves.
        correlated: set = set()
        for group in CORRELATED_COLUMNS.get(template_id, ()):
            if all(c in columns for c in group):
                _merge_correlated(existing, row, group)
                correlated.update(group)

        for column in columns:
            if column in correlated:
                continue
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
