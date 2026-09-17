"""Turn a model's table answer into validated, citation-resolved rows.

The TSV cleaning follows the approach in Literature.js `_parseTSV` -- strip code
fences, tolerate pipe tables, drop separator rules -- and then adds the parts a
curation system needs: citations resolved to real identifiers, evidence-free
rows removed, and out-of-scope genes flagged rather than silently kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .normalize import clean, is_null, normalize_gene
from .templates import Template

_FENCE = re.compile(r"```[a-zA-Z]*\n?")
_SEPARATOR_ROW = re.compile(r"^[\s\-|=:+]+$")
# Any number of markers in one bracket: [1], [1,3], [1-3], [1, 2, 3], [1-3; 7].
# The previous pattern allowed at most two numbers, so "[1, 2, 3]" matched
# nothing at all and the row fell through to surname guessing. That was rare
# with ten passages in context and is routine with several hundred.
_CITATION_REF = re.compile(r"\[(\d+(?:\s*[-,;–]\s*\d+)*)\]")
_CITATION_SPAN = re.compile(r"(\d+)\s*[-–]\s*(\d+)")
# A span wider than this is a parsing artefact, not a citation. Without a bound,
# a hallucinated "[1-99999]" would allocate that many Citation objects.
MAX_CITATION_SPAN = 50

# Columns that identify what a row is about rather than what was found about it.
# A row with no content outside these is a restatement of the query.
_SUBJECT_COLUMNS = {
    "organism", "pathogen", "gene name", "gene", "protein a", "protein b",
    "protein", "species",
    # For susceptibility testing the subject is the strain-drug pair, and the
    # identifiers that pin the strain down. The evidence is the MIC or SIR.
    "strain", "isolate", "antibiotic", "drug", "antimicrobial", "agent",
    "genbank accession", "accession", "biosample",
}
_REFERENCE_COLUMNS = {"reference", "references", "citation", "citations", "source"}
# The column that carries the actual finding for each table type.
_KEY_VALUE_COLUMNS = {"mutation", "function", "interaction type"}
# Columns where at least one of a group must be present for the row to say
# anything. An AST row needs an MIC or an SIR; neither alone is required.
_EITHER_OR_COLUMNS = [{"mic", "sir"}]


@dataclass
class Citation:
    """A resolved literature reference."""

    marker: int
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    title: Optional[str] = None
    first_author: Optional[str] = None
    resolved: bool = True
    # "marker" for a [n] reference, "author" for a prose name match.
    matched_by: str = "marker"
    # Which retrieved document and chunk this marker pointed at. The marker
    # itself is only meaningful within the prompt that produced it -- once
    # passages are split across parallel calls, every call numbers its sources
    # from 1, so `doc_id` is the only stable identity a citation carries. Some
    # corpora (Dengue, Influenza_2024_2025) publish no pmid or pmcid at all,
    # which is exactly when the identifiers below are all None.
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None

    @property
    def identity(self) -> str:
        """Stable key for "is this the same paper?" across batches."""
        return (self.pmid or self.doi or self.pmcid or self.doc_id
                or f"marker:{self.marker}")

    @property
    def url(self) -> Optional[str]:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        if self.pmcid:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{self.pmcid}/"
        return None

    def short(self) -> str:
        """A compact human-readable citation."""
        if not self.resolved:
            return f"[{self.marker}] unresolved"
        bits = []
        if self.first_author:
            bits.append(f"{self.first_author} et al.")
        if self.journal:
            bits.append(self.journal)
        if self.year:
            bits.append(f"({self.year})")
        if self.pmid:
            bits.append(f"PMID:{self.pmid}")
        elif self.doi:
            bits.append(f"DOI:{self.doi}")
        return " ".join(bits) or f"[{self.marker}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker": self.marker,
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "doi": self.doi,
            "journal": self.journal,
            "year": self.year,
            "title": self.title,
            "first_author": self.first_author,
            "url": self.url,
            "resolved": self.resolved,
            "matched_by": self.matched_by,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
        }


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        return cls(
            marker=int(data.get("marker", 0)),
            pmid=data.get("pmid"), pmcid=data.get("pmcid"), doi=data.get("doi"),
            journal=data.get("journal"), year=data.get("year"),
            title=data.get("title"), first_author=data.get("first_author"),
            resolved=bool(data.get("resolved", True)),
            matched_by=data.get("matched_by", "marker"),
            doc_id=data.get("doc_id"), chunk_id=data.get("chunk_id"),
        )


@dataclass
class Row:
    """One extracted fact."""

    values: Dict[str, str]
    citations: List[Citation] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    n_support: int = 1
    query_ids: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    # Differing values seen for a non-identity column across merged rows,
    # kept so a merge never hides disagreement between sources.
    variants: Dict[str, List[str]] = field(default_factory=dict)

    def get(self, column: str) -> str:
        return self.values.get(column, "")

    @property
    def citation_text(self) -> str:
        return "; ".join(c.short() for c in self.citations)

    @property
    def pmids(self) -> List[str]:
        return [c.pmid for c in self.citations if c.pmid]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "citations": [c.to_dict() for c in self.citations],
            "flags": self.flags,
            "n_support": self.n_support,
            "query_ids": self.query_ids,
            "provenance": self.provenance,
            "variants": self.variants,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Row":
        return cls(
            values=dict(data.get("values", {})),
            citations=[Citation.from_dict(c) for c in data.get("citations", [])],
            flags=list(data.get("flags", [])),
            n_support=int(data.get("n_support", 1)),
            query_ids=list(data.get("query_ids", [])),
            provenance=dict(data.get("provenance", {})),
            variants={k: list(v) for k, v in (data.get("variants") or {}).items()},
        )


@dataclass
class Extraction:
    """The result of parsing one answer, with counters for the run summary."""

    rows: List[Row] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    dropped_empty: int = 0
    dropped_malformed: int = 0
    split_compound: int = 0
    # True when the model emitted the table rotated and it was repaired.
    transposed: bool = False
    unresolved_citations: int = 0
    raw_answer: str = ""
    is_table: bool = True

    @property
    def n_rows(self) -> int:
        return len(self.rows)


def _split_cells(line: str, delimiter: str) -> List[str]:
    cells = [c.strip() for c in line.split(delimiter)]
    if delimiter == "|":
        # Pipe tables have empty leading/trailing cells from the border pipes.
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
    return cells


def _maybe_transpose(
    rows: List[List[str]], columns: Sequence[str]
) -> "tuple[List[List[str]], bool]":
    """Rotate a table a model emitted sideways.

    Llama-4-Scout sometimes answers with one line per FIELD rather than one per
    record -- "Organism<TAB>value1<TAB>value2", "Gene Name<TAB>..." -- which a
    positional parser turns into rows of pure nonsense. Detect it by seeing the
    expected column names filling the first cell of most lines, and rotate back.
    """
    if len(rows) < 2:
        return rows, False

    expected = {c.strip().lower(): c for c in columns}
    leading = [r[0].strip().lower() for r in rows if r and r[0].strip()]
    if not leading:
        return rows, False

    matches = sum(1 for name in leading if name in expected)
    # Most lines must start with a column name, and the table must be wider
    # than a single record, or this is just an ordinary table whose first
    # column happens to repeat a header word.
    if matches < max(2, len(expected) // 2) or matches < len(leading) - 1:
        return rows, False

    width = max(len(r) for r in rows)
    if width < 2:
        return rows, False

    by_field: Dict[str, List[str]] = {}
    for row in rows:
        if not row or row[0].strip().lower() not in expected:
            continue
        canonical = expected[row[0].strip().lower()]
        by_field[canonical] = [c for c in row[1:]]

    n_records = max((len(v) for v in by_field.values()), default=0)
    rotated: List[List[str]] = []
    for index in range(n_records):
        record = []
        for column in columns:
            values = by_field.get(column, [])
            record.append(values[index] if index < len(values) else "")
        if any(cell.strip() for cell in record):
            rotated.append(record)
    return (rotated, True) if rotated else (rows, False)


def _looks_like_header(cells: Sequence[str], columns: Sequence[str]) -> bool:
    normalized = {c.strip().lower() for c in cells if c.strip()}
    expected = {c.strip().lower() for c in columns}
    return len(normalized & expected) >= max(2, len(expected) // 2)


def parse_citations(text: str, sources: Sequence[Dict[str, Any]]) -> List[Citation]:
    """Map [n] markers onto the retrieved sources.

    Markers are 1-based positions into the sources list -- verified against live
    responses. Several chunks of one paper can be retrieved separately, so the
    result is deduplicated by document.
    """
    citations: List[Citation] = []
    seen: set = set()

    for match in _CITATION_REF.finditer(text or ""):
        # Split on list separators first, then decide span-or-single per item.
        # "[1-3]" is a span of three papers, "[1,3]" is two, and "[1-3, 7]" is
        # both at once -- which the old single-shot test for a dash could not
        # express.
        for part in re.split(r"\s*[,;]\s*", match.group(1)):
            part = part.strip()
            if not part:
                continue
            span = _CITATION_SPAN.fullmatch(part)
            if span:
                low, high = sorted((int(span.group(1)), int(span.group(2))))
                if high - low >= MAX_CITATION_SPAN:
                    continue
                markers = range(low, high + 1)
            elif part.isdigit():
                markers = [int(part)]
            else:
                continue

            for marker in markers:
                if marker in seen:
                    continue
                seen.add(marker)
                citations.append(_citation_for(marker, sources))

    if not citations:
        citations = _match_by_text(text, sources)

    # Collapse chunks of the same document into one citation.
    deduped: List[Citation] = []
    doc_seen: set = set()
    for citation in citations:
        key = citation.identity
        if key in doc_seen:
            continue
        doc_seen.add(key)
        deduped.append(citation)
    return deduped


def _surnames(authors: Any) -> List[str]:
    """Surnames from an author list, lowercased."""
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    if not isinstance(authors, list):
        return []
    names = []
    for author in authors:
        parts = str(author).replace(",", " ").split()
        if parts:
            surname = parts[-1].strip(".").lower()
            if len(surname) >= 3:
                names.append(surname)
    return names


def _match_by_text(text: str, sources: Sequence[Dict[str, Any]]) -> List[Citation]:
    """Resolve a reference written as prose rather than a [n] marker.

    The model does not always cite by number -- it may write "Kei Haga et al."
    Matching is deliberately strict: only the FIRST author counts, because that
    is what "X et al." denotes. A looser rule mis-attributes claims, and a wrong
    citation in a curated table is worse than an absent one -- an unmatched
    reference is reported as unresolved so a curator can check it by hand.
    """
    lowered = (text or "").lower()
    if not lowered.strip():
        return []

    year_match = re.search(r"\b(19|20)\d{2}\b", lowered)
    year = year_match.group(0) if year_match else None

    matches: List[Citation] = []
    for index, source in enumerate(sources):
        meta = source.get("metadata", {}) or {}
        surnames = _surnames(meta.get("authors"))
        if not surnames:
            continue
        first_surname = surnames[0]
        if not re.search(rf"\b{re.escape(first_surname)}\b", lowered):
            continue
        source_year = str(meta.get("year") or "")
        if year and source_year and year != source_year:
            continue
        citation = _citation_for(index + 1, sources)
        citation.matched_by = "author"
        matches.append(citation)
    return matches


def _citation_for(marker: int, sources: Sequence[Dict[str, Any]]) -> Citation:
    index = marker - 1
    if index < 0 or index >= len(sources):
        return Citation(marker=marker, resolved=False)

    source = sources[index]
    meta = source.get("metadata", {}) or {}
    authors = meta.get("authors") or []
    first_author = None
    if isinstance(authors, list) and authors:
        first_author = str(authors[0])
    elif isinstance(authors, str) and authors.strip():
        first_author = authors.split(",")[0].strip()

    year = meta.get("year")
    return Citation(
        marker=marker,
        pmid=_as_str(meta.get("pmid")),
        pmcid=_as_str(meta.get("pmcid")),
        doi=_as_str(meta.get("doi")),
        journal=_as_str(meta.get("journal")),
        year=_as_str(year),
        title=_as_str(meta.get("title")),
        first_author=first_author,
        resolved=True,
        # Top level on a retrieved source, not inside metadata.
        doc_id=_as_str(source.get("doc_id")),
        chunk_id=_as_str(source.get("chunk_id")),
    )


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_reference_text(text: str) -> bool:
    """True when a reference cell names something rather than being blank/N-A."""
    stripped = clean(text)
    if not stripped:
        return False
    # Needs at least one word of three or more letters to count as a name.
    return bool(re.search(r"[A-Za-z]{3,}", stripped))


def _is_evidence_free(values: Dict[str, str], columns: Sequence[str]) -> bool:
    """True when a row names a subject but reports nothing about it.

    Observed live: `SARS-CoV-2  NSP13  Spike  N/A  N/A  N/A  N/A` -- the query
    restated as a finding.
    """
    evidence_columns = [
        c for c in columns
        if c.strip().lower() not in _SUBJECT_COLUMNS
        and c.strip().lower() not in _REFERENCE_COLUMNS
    ]
    if not evidence_columns:
        evidence_columns = [c for c in columns if c.strip().lower() not in _SUBJECT_COLUMNS]
    if not evidence_columns:
        return False
    return all(is_null(values.get(c)) for c in evidence_columns)


# Columns whose value identifies the row's subject and must be singular. A row
# reading "katG, inhA" with mutations "Ser315Thr, c-15t" is two facts packed
# into one, and dedup cannot match either against its single-valued twin.
_SPLITTABLE_COLUMNS = {
    "gene name", "gene", "mutation", "variant", "allele", "protein",
    "antibiotic", "drug", "antimicrobial", "mic", "sir",
}
_LIST_SEPARATOR = re.compile(r"\s*[,;]\s*|\s+and\s+")


def _split_compound(
    values: Dict[str, str], columns: Sequence[str]
) -> "tuple[List[Dict[str, str]], bool]":
    """Split a row that packed several facts into list-valued cells.

    Returns the expanded rows and whether the split was ambiguous. Splitting
    only happens when every multi-valued column yields the same number of
    parts, so "katG, inhA" + "Ser315Thr, c-15t" becomes two aligned rows while
    a mismatched pairing is left intact and flagged for a curator to read.
    """
    splittable = [c for c in columns if c.strip().lower() in _SPLITTABLE_COLUMNS]
    parts_by_column: Dict[str, List[str]] = {}
    for column in splittable:
        value = values.get(column, "")
        if not value:
            continue
        parts = [p.strip() for p in _LIST_SEPARATOR.split(value) if p.strip()]
        if len(parts) > 1:
            parts_by_column[column] = parts

    if not parts_by_column:
        return [values], False

    counts = {len(v) for v in parts_by_column.values()}
    if len(counts) != 1:
        return [values], True

    count = counts.pop()
    expanded: List[Dict[str, str]] = []
    for index in range(count):
        clone = dict(values)
        for column, parts in parts_by_column.items():
            clone[column] = parts[index]
        expanded.append(clone)
    return expanded, False


def _gene_columns(columns: Sequence[str]) -> List[str]:
    return [
        c for c in columns
        if c.strip().lower() in {"gene name", "gene", "protein a", "protein b", "protein"}
    ]


def extract_table(
    answer: str,
    template: Template,
    sources: Sequence[Dict[str, Any]],
    requested_genes: Optional[Sequence[str]] = None,
    keep_empty: bool = False,
) -> Extraction:
    """Parse a table answer into rows with resolved citations."""
    columns = list(template.columns or [])
    result = Extraction(columns=columns, raw_answer=answer, is_table=True)

    cleaned = _FENCE.sub("", answer or "").replace("```", "").strip()
    lines = [
        line for line in cleaned.splitlines()
        if line.strip() and not _SEPARATOR_ROW.match(line.strip())
    ]
    if not lines:
        return result

    delimiter = "\t"
    if "\t" not in lines[0] and "|" in lines[0]:
        delimiter = "|"

    start = 0
    header = _split_cells(lines[0], delimiter)
    if _looks_like_header(header, columns):
        start = 1

    gene_keys = {normalize_gene(g) for g in (requested_genes or []) if normalize_gene(g)}
    gene_cols = _gene_columns(columns)

    parsed_lines = [_split_cells(line, delimiter) for line in lines[start:]]
    parsed_lines, was_transposed = _maybe_transpose(parsed_lines, columns)
    result.transposed = was_transposed

    for cells in parsed_lines:
        if not any(c.strip() for c in cells):
            continue
        if len(cells) < 2:
            result.dropped_malformed += 1
            continue
        line = delimiter.join(cells)

        parsed = {col: clean(cells[i]) if i < len(cells) else "" for i, col in enumerate(columns)}

        if not keep_empty and _is_evidence_free(parsed, columns):
            result.dropped_empty += 1
            continue

        expanded, ambiguous = _split_compound(parsed, columns)
        if len(expanded) > 1:
            result.split_compound += len(expanded) - 1

        for values in expanded:
            flags: List[str] = []
            if ambiguous:
                flags.append("compound_row")
            if _is_evidence_free(values, columns):
                flags.append("evidence_free")

            reference_text = " ".join(
                cells[i] for i, col in enumerate(columns)
                if i < len(cells) and col.strip().lower() in _REFERENCE_COLUMNS
            ) or line
            citations = parse_citations(reference_text, sources)
            unresolved = [c for c in citations if not c.resolved]
            result.unresolved_citations += len(unresolved)
            if unresolved:
                flags.append("unresolved_citation")
            if not citations:
                # Distinguish "said nothing" from "cited a paper that is not in
                # the retrieved set" -- the latter usually means the model
                # picked up a reference from inside a chunk's bibliography.
                if _has_reference_text(reference_text):
                    flags.append("citation_not_in_sources")
                else:
                    flags.append("no_citation")

            # A mutation table row with no mutation names a subject without
            # saying anything specific about it.
            for column in columns:
                if column.strip().lower() in _KEY_VALUE_COLUMNS and not values.get(column):
                    flags.append(f"missing:{column}")

            for group in _EITHER_OR_COLUMNS:
                present = [c for c in columns if c.strip().lower() in group]
                if present and all(is_null(values.get(c)) for c in present):
                    flags.append("missing:" + "/".join(sorted(group)).upper())

            # A katG query legitimately returns inhA rows. Flag, do not drop --
            # the finding is real, it just was not what was asked for.
            if gene_keys and gene_cols:
                row_genes = {normalize_gene(values.get(c)) for c in gene_cols}
                row_genes.discard("")
                if row_genes and not (row_genes & gene_keys):
                    flags.append("off_target_gene")

            result.rows.append(Row(values=values, citations=citations, flags=flags))

    return result


def extract_text(answer: str, template: Template, sources: Sequence[Dict[str, Any]]) -> Extraction:
    """Wrap a prose answer, still resolving its citation markers."""
    extraction = Extraction(columns=[], raw_answer=answer, is_table=False)
    extraction.rows = []
    citations = parse_citations(answer or "", sources)
    extraction.unresolved_citations = sum(1 for c in citations if not c.resolved)
    extraction.rows.append(Row(values={"Answer": answer or ""}, citations=citations))
    return extraction


def extract(
    answer: str,
    template: Template,
    sources: Sequence[Dict[str, Any]],
    requested_genes: Optional[Sequence[str]] = None,
    keep_empty: bool = False,
) -> Extraction:
    """Parse an answer according to the template's declared output shape."""
    if template.is_table:
        return extract_table(
            answer, template, sources,
            requested_genes=requested_genes, keep_empty=keep_empty,
        )
    return extract_text(answer, template, sources)
