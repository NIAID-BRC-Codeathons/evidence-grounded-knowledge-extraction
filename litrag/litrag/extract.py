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
_CITATION_REF = re.compile(r"\[(\d+(?:\s*[-,–]\s*\d+)?)\]")

# Columns that identify what a row is about rather than what was found about it.
# A row with no content outside these is a restatement of the query.
_SUBJECT_COLUMNS = {
    "organism", "pathogen", "gene name", "gene", "protein a", "protein b",
    "protein", "species",
    # For susceptibility testing the subject is the strain-drug pair, and the
    # identifiers that pin the strain down. The evidence is the MIC or SIR.
    "strain", "isolate", "antibiotic", "drug", "antimicrobial", "agent",
    "genbank accession", "accession", "biosample",
    # A glycosylation row is about a site on a protein; the glycan, method and
    # effect are what is reported about it.
    "site", "position", "residue",
}
_REFERENCE_COLUMNS = {"reference", "references", "citation", "citations", "source"}
# Assertion classifies the status of a claim, not its content. Since it is now
# filled from a fixed vocabulary on every row, counting it as evidence would
# mean no row is ever evidence-free and the filter would never fire.
_STATUS_COLUMNS = {"assertion", "confidence", "evidence"}
# The column that carries the actual finding for each table type.
_KEY_VALUE_COLUMNS = {"mutation", "function", "interaction type", "site"}
# Columns where at least one of a group must be present for the row to say
# anything. An AST row needs an MIC or an SIR; neither alone is required.
_EITHER_OR_COLUMNS = [{"mic", "sir"}]


@dataclass
class ChunkRef:
    """One retrieved passage supporting a claim.

    The chunk, not the paper, is the real unit of evidence: it is the text the
    model actually read. Two chunks of one paper are two pieces of support and
    must both survive, which is why a Citation carries a list of these rather
    than a single marker.
    """

    marker: int
    chunk_id: str = ""
    doc_id: str = ""
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    score: Optional[float] = None

    @property
    def span(self) -> str:
        if self.start_char is None or self.end_char is None:
            return ""
        return f"{self.start_char}-{self.end_char}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker": self.marker, "chunk_id": self.chunk_id,
            "doc_id": self.doc_id, "start_char": self.start_char,
            "end_char": self.end_char, "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRef":
        return cls(
            marker=int(data.get("marker", 0)),
            chunk_id=data.get("chunk_id", "") or "",
            doc_id=data.get("doc_id", "") or "",
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            score=data.get("score"),
        )


@dataclass
class Citation:
    """A resolved literature reference, with the passages that support it."""

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
    # Every retrieved passage backing this citation, in marker order.
    chunks: List[ChunkRef] = field(default_factory=list)

    @property
    def url(self) -> Optional[str]:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        if self.pmcid:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{self.pmcid}/"
        return None

    @property
    def markers(self) -> List[int]:
        return [c.marker for c in self.chunks] or [self.marker]

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks if c.chunk_id]

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
            "chunks": [c.to_dict() for c in self.chunks],
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
            chunks=[ChunkRef.from_dict(c) for c in data.get("chunks", [])],
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

    def display(self, column: str) -> str:
        """The cell value, with every merged alternative joined by "; ".

        When rows merge, the representative value alone hides that the sources
        said different things. Listing them all inline keeps a flat table
        honest without needing a second line per cell.
        """
        primary = self.values.get(column, "")
        alternatives = self.variants.get(column)
        if not alternatives:
            return primary
        ordered = [primary] if primary else []
        for value in alternatives:
            if value and value not in ordered:
                ordered.append(value)
        return "; ".join(ordered)

    @property
    def citation_text(self) -> str:
        return "; ".join(c.short() for c in self.citations)

    @property
    def pmids(self) -> List[str]:
        return [c.pmid for c in self.citations if c.pmid]

    @property
    def chunk_ids(self) -> List[str]:
        """Every passage backing this row, across all its citations."""
        ids: List[str] = []
        for citation in self.citations:
            for chunk_id in citation.chunk_ids:
                if chunk_id not in ids:
                    ids.append(chunk_id)
        return ids

    @property
    def markers(self) -> List[int]:
        seen: List[int] = []
        for citation in self.citations:
            for marker in citation.markers:
                if marker not in seen:
                    seen.append(marker)
        return sorted(seen)

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


_BARE_NUMBERS = re.compile(r"\d+")
# A cell that is nothing but source numbers: "3", "3, 5", "1-3". Anything else
# in it -- a letter, a dot, a slash -- means the digits belong to something
# that is not a marker, and a DOI is the common case: every DOI starts "10.",
# so harvesting its digits attributes the row to source 10.
_BARE_ONLY = re.compile(r"[\d\s,;–-]+")


def parse_citations(
    text: str,
    sources: Sequence[Dict[str, Any]],
    bare_numbers: bool = False,
) -> List[Citation]:
    """Map [n] markers onto the retrieved sources.

    Markers are 1-based positions into the sources list -- verified against live
    responses. Several chunks of one paper can be retrieved separately, so the
    result is deduplicated by document.

    ``bare_numbers`` allows "3" to mean "[3]". Only pass it for text taken from
    a reference column, where a lone integer can only be a source number --
    elsewhere it would turn a position or a dose into a citation. Even there it
    applies only when the cell holds nothing but numbers and separators: a
    reference written as a DOI or an "Author Year, Journal 5:231" string must
    fall through to ``_match_by_text`` and be flagged, not be resolved from
    whichever of its digits happen to be in range.
    """
    citations: List[Citation] = []
    seen: set = set()

    for match in _CITATION_REF.finditer(text or ""):
        body = match.group(1)
        parts = re.split(r"\s*[-,–]\s*", body)
        try:
            numbers = [int(p) for p in parts if p.strip()]
        except ValueError:
            continue
        # "[1-3]" is a span of three papers; "[1, 2]" is two. The separator is
        # what distinguishes them, so test it explicitly rather than relying on
        # and/or precedence.
        is_span = len(numbers) == 2 and re.search(r"[-–]", body) is not None
        span = range(min(numbers), max(numbers) + 1) if is_span else numbers

        for marker in span:
            if marker in seen:
                continue
            seen.add(marker)
            citations.append(_citation_for(marker, sources))

    if not citations and bare_numbers and _BARE_ONLY.fullmatch((text or "").strip()):
        for token in _BARE_NUMBERS.findall(text or ""):
            marker = int(token)
            if marker in seen or not 1 <= marker <= len(sources):
                continue
            seen.add(marker)
            citations.append(_citation_for(marker, sources))

    if not citations:
        citations = _match_by_text(text, sources)

    # One citation per document, but every supporting passage is kept: three
    # chunks of one paper are three pieces of evidence, and dropping two of
    # them would throw away exactly the provenance a curator needs.
    deduped: List[Citation] = []
    by_document: Dict[str, Citation] = {}
    for citation in citations:
        key = (citation.pmid or citation.doi or citation.pmcid
               or f"marker:{citation.marker}")
        existing = by_document.get(key)
        if existing is None:
            by_document[key] = citation
            deduped.append(citation)
            continue
        seen = {c.chunk_id or c.marker for c in existing.chunks}
        for chunk in citation.chunks:
            if (chunk.chunk_id or chunk.marker) not in seen:
                existing.chunks.append(chunk)
    for citation in deduped:
        citation.chunks.sort(key=lambda c: c.marker)
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
    chunk = ChunkRef(
        marker=marker,
        chunk_id=source.get("chunk_id", "") or "",
        doc_id=source.get("doc_id", "") or "",
        start_char=meta.get("start_char"),
        end_char=meta.get("end_char"),
        score=source.get("score"),
    )
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
        chunks=[chunk],
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
        and c.strip().lower() not in _STATUS_COLUMNS
    ]
    if not evidence_columns:
        evidence_columns = [
            c for c in columns
            if c.strip().lower() not in _SUBJECT_COLUMNS
            and c.strip().lower() not in _STATUS_COLUMNS
        ]
    if not evidence_columns:
        return False
    return all(is_null(values.get(c)) for c in evidence_columns)


# Columns whose value identifies the row's subject and must be singular. A row
# reading "katG, inhA" with mutations "Ser315Thr, c-15t" is two facts packed
# into one, and dedup cannot match either against its single-valued twin.
_SPLITTABLE_COLUMNS = {
    "gene name", "gene", "mutation", "variant", "allele", "protein",
    "antibiotic", "drug", "antimicrobial", "mic", "sir",
    "site", "position", "residue",
}
_LIST_SEPARATOR = re.compile(r"\s*[,;]\s*|\s+and\s+")


def _split_list(value: str) -> List[str]:
    """Split "a, b and c" into parts, ignoring separators inside brackets.

    "Multiple mutations (codons 315, 316, 309)" is one value, not three: the
    commas belong to the parenthetical. Splitting blindly produced the rows
    "Multiple mutations (codons 315", "316" and "309)".
    """
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char in "([{":
            depth += 1
            current.append(char)
            index += 1
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            current.append(char)
            index += 1
            continue
        if depth == 0:
            match = _LIST_SEPARATOR.match(value, index)
            if match and match.end() > index:
                parts.append("".join(current))
                current = []
                index = match.end()
                continue
        current.append(char)
        index += 1
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


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
        parts = _split_list(value)
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

            reference_cells = [
                cells[i] for i, col in enumerate(columns)
                if i < len(cells) and col.strip().lower() in _REFERENCE_COLUMNS
            ]
            from_reference_column = bool(" ".join(reference_cells).strip())
            reference_text = " ".join(reference_cells) or line
            citations = parse_citations(
                reference_text, sources, bare_numbers=from_reference_column
            )
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
