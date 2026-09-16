"""Output writers.

The same extracted rows render as TSV, CSV, JSON, JSONL, or Markdown so the
result can go straight into a spreadsheet, a pipeline, or a report.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .extract import Extraction, Row
from .provenance import ANNOTATION_COLUMNS, PROVENANCE_COLUMNS

FORMATS = ("tsv", "csv", "json", "jsonl", "md", "table")


def _annotations(row: Row) -> Dict[str, Any]:
    return {
        "_n_support": row.n_support,
        "_flags": ";".join(row.flags),
        "_citations": row.citation_text,
        "_pmids": ";".join(row.pmids),
    }


def header(columns: Sequence[str], provenance: bool = True) -> List[str]:
    cols = list(columns) + list(ANNOTATION_COLUMNS)
    if provenance:
        cols += PROVENANCE_COLUMNS
    return cols


def row_to_dict(
    row: Row,
    columns: Sequence[str],
    provenance: bool = True,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {c: row.get(c) for c in columns}
    record.update(_annotations(row))
    if provenance:
        for column in PROVENANCE_COLUMNS:
            record[column] = row.provenance.get(column, "")
    return record


def to_delimited(
    rows: Sequence[Row],
    columns: Sequence[str],
    delimiter: str = "\t",
    provenance: bool = True,
) -> str:
    buffer = io.StringIO()
    cols = header(columns, provenance)
    writer = csv.DictWriter(
        buffer, fieldnames=cols, delimiter=delimiter,
        lineterminator="\n", extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row_to_dict(row, columns, provenance))
    return buffer.getvalue()


def to_json(
    rows: Sequence[Row],
    columns: Sequence[str],
    provenance: bool = True,
    meta: Optional[Dict[str, Any]] = None,
    sources: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "columns": list(columns),
        "n_rows": len(rows),
        "rows": [
            {
                **{c: row.get(c) for c in columns},
                "citations": [c.to_dict() for c in row.citations],
                "flags": row.flags,
                "variants": row.variants,
                "n_support": row.n_support,
                "query_ids": row.query_ids,
                **({"provenance": row.provenance} if provenance else {}),
            }
            for row in rows
        ],
    }
    if meta:
        payload["meta"] = meta
    if sources is not None:
        payload["sources"] = [_source_summary(s) for s in sources]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def to_jsonl(rows: Sequence[Row], columns: Sequence[str], provenance: bool = True) -> str:
    lines = []
    for row in rows:
        record = {c: row.get(c) for c in columns}
        record["citations"] = [c.to_dict() for c in row.citations]
        record["flags"] = row.flags
        record["n_support"] = row.n_support
        if provenance:
            record["provenance"] = row.provenance
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def to_markdown(rows: Sequence[Row], columns: Sequence[str], provenance: bool = False) -> str:
    cols = list(columns) + ["_n_support", "_citations"]
    if provenance:
        cols += PROVENANCE_COLUMNS

    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(escape(c) for c in cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        record = row_to_dict(row, columns, provenance)
        out.append("| " + " | ".join(escape(record.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def to_pretty_table(rows: Sequence[Row], columns: Sequence[str], max_width: int = 34) -> str:
    """Aligned plain-text table for terminal output."""
    cols = list(columns) + ["n", "flags"]

    def cell(row: Row, column: str) -> str:
        if column == "n":
            return str(row.n_support)
        if column == "flags":
            return ",".join(f.split(":")[0] for f in row.flags)
        return row.get(column)

    def truncate(text: str) -> str:
        text = str(text).replace("\n", " ")
        return text if len(text) <= max_width else text[: max_width - 1] + "…"

    widths = {c: len(truncate(c)) for c in cols}
    for row in rows:
        for column in cols:
            widths[column] = max(widths[column], len(truncate(cell(row, column))))

    lines = ["  ".join(truncate(c).ljust(widths[c]) for c in cols).rstrip(),
             "  ".join("-" * widths[c] for c in cols)]
    for row in rows:
        lines.append("  ".join(truncate(cell(row, c)).ljust(widths[c]) for c in cols).rstrip())
    return "\n".join(lines) + "\n"


def _source_summary(source: Dict[str, Any]) -> Dict[str, Any]:
    meta = source.get("metadata", {}) or {}
    authors = meta.get("authors") or []
    return {
        "doc_id": source.get("doc_id"),
        "chunk_id": source.get("chunk_id"),
        "score": source.get("score"),
        "title": meta.get("title"),
        "journal": meta.get("journal"),
        "year": meta.get("year"),
        "pmid": meta.get("pmid"),
        "pmcid": meta.get("pmcid"),
        "doi": meta.get("doi"),
        "authors": authors if isinstance(authors, list) else [authors],
        "content": source.get("content", ""),
    }


def render(
    extraction: Extraction,
    rows: Sequence[Row],
    fmt: str,
    provenance: bool = True,
    meta: Optional[Dict[str, Any]] = None,
    sources: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Render rows in the requested format."""
    columns = extraction.columns
    if not extraction.is_table:
        if fmt in ("json", "jsonl"):
            payload = {"answer": extraction.raw_answer,
                       "citations": [c.to_dict() for c in extraction.rows[0].citations]
                       if extraction.rows else []}
            if meta:
                payload["meta"] = meta
            if sources is not None:
                payload["sources"] = [_source_summary(s) for s in sources]
            return json.dumps(payload, indent=2, ensure_ascii=False)
        return extraction.raw_answer.rstrip() + "\n"

    if fmt == "tsv":
        return to_delimited(rows, columns, "\t", provenance)
    if fmt == "csv":
        return to_delimited(rows, columns, ",", provenance)
    if fmt == "json":
        return to_json(rows, columns, provenance, meta, sources)
    if fmt == "jsonl":
        return to_jsonl(rows, columns, provenance)
    if fmt == "md":
        return to_markdown(rows, columns, provenance)
    if fmt == "table":
        return to_pretty_table(rows, columns)
    raise ValueError(f"unknown format '{fmt}'. Choose from: {', '.join(FORMATS)}")
