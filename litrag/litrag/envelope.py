"""Render a run as the scoring envelope experiment-01's evaluate.py consumes.

LitRAG and the scorer were built independently, so neither knows the other's
shapes. Rather than fork the scorer -- its matchers are format-agnostic and its
notation handling is worth keeping -- this module translates outward, leaving
both sides untouched.

Two deliberate choices:

- Column names are display strings ("Gene Name"), the scorer reads snake_case
  ("gene_name"). The mapping is derived, not hardcoded, so a new template works
  without editing this file.
- `data_type` is remapped where the two vocabularies disagree: LitRAG's server
  template id is `ppi-extraction`, the scorer's matcher key is `ppi`. Anything
  not in the map passes through unchanged.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from .extract import Citation, Row

SCHEMA_VERSION = 1

# LitRAG template id -> evaluate.py MATCHERS key. Absent means "same name".
DATA_TYPE_ALIASES = {"ppi-extraction": "ppi"}

_NON_WORD = re.compile(r"[^0-9a-z]+")


def field_name(column: str) -> str:
    """"Gene Name" -> "gene_name". Stable for any template's columns."""
    return _NON_WORD.sub("_", (column or "").strip().lower()).strip("_")


def scorer_data_type(template_id: str) -> str:
    return DATA_TYPE_ALIASES.get(template_id, template_id)


def _primary_citation(row: Row) -> Optional[Citation]:
    """The citation a row is scored against.

    Rows can carry several. The scorer compares one pubmed_id per row, so pick
    the first that resolved to an identifier -- markers arrive in the order the
    model cited them, so the first is the one it leaned on.
    """
    for citation in row.citations:
        if citation.pmid or citation.pmcid or citation.doi:
            return citation
    return row.citations[0] if row.citations else None


def _evidence(citation: Optional[Citation]) -> Dict[str, Any]:
    if citation is None:
        return {}
    return {
        "chunk_id": citation.chunk_id,
        "doc_id": citation.doc_id,
        "pubmed_id": citation.pmid,
        "pmc_id": citation.pmcid,
        "doi": citation.doi,
        "title": citation.title,
        "year": citation.year,
        # Empty until E11 lands everywhere; the quote gate needs it non-empty.
        "passage": citation.passage,
        "passage_char_offset": None,
    }


def _paper_manifest(sources: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One entry per document, with how many of its chunks were retrieved.

    The scorer restricts gold to papers that were actually retrieved, so this
    list decides what recall is measured against. Getting it wrong silently
    inflates or deflates the score, which is why chunks are counted per doc
    rather than assumed one-to-one.
    """
    by_doc: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        meta = source.get("metadata", {}) or {}
        doc_id = source.get("doc_id") or meta.get("pmid") or meta.get("doi")
        if not doc_id:
            continue
        entry = by_doc.setdefault(str(doc_id), {
            "doc_id": str(doc_id),
            "pubmed_id": meta.get("pmid"),
            "pmc_id": meta.get("pmcid"),
            "doi": meta.get("doi"),
            "year": meta.get("year"),
            "title": meta.get("title"),
            "chunks": 0,
        })
        entry["chunks"] += 1
    return list(by_doc.values())


def _row_record(index: int, row: Row, columns: Sequence[str]) -> Dict[str, Any]:
    citation = _primary_citation(row)
    record: Dict[str, Any] = {
        "row_id": f"r{index:04d}",
        "outcome": "emit",
        "evidence": _evidence(citation),
        "citation_partial": bool(citation and not citation.resolved),
        "failure_layer_tags": list(row.flags),
        "n_support": row.n_support,
    }
    for column in columns:
        record[field_name(column)] = row.get(column)
    if row.variants:
        record["variants"] = {field_name(k): v for k, v in row.variants.items()}
    return record


def build_envelope(
    run,
    run_id: Optional[str] = None,
    prompt_edited: bool = False,
) -> Dict[str, Any]:
    """Convert a pipeline.RunResult into the scorer's run envelope."""
    spec = run.spec
    template = run.template
    result = run.result
    columns = list(run.columns)
    sources = list(result.sources or [])

    manifest = _paper_manifest(sources)
    rows = [_row_record(i, row, columns) for i, row in enumerate(run.rows, start=1)]

    # The scorer reads tally.proposed as the denominator of unsupported-claim
    # rate, so it must count every row the model offered, not the survivors.
    proposed = run.extraction.n_rows or len(rows)
    dropped = run.extraction.dropped_empty + run.extraction.dropped_malformed

    omitted: List[Dict[str, Any]] = []
    for _ in range(run.extraction.dropped_empty):
        omitted.append({"outcome": "omit", "omit_reason": "evidence_free"})
    for _ in range(run.extraction.dropped_malformed):
        omitted.append({"outcome": "omit", "omit_reason": "malformed_row"})

    identity = run_id or hashlib.sha256(
        f"{spec.identity()}|{result.prompt_hash}|{result.model}".encode("utf-8")
    ).hexdigest()[:12]

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": identity,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": result.model,
            "endpoint": result.endpoint,
            "generator": result.generator,
            "prompt_sha256": result.prompt_hash or result.template_hash or "",
            "prompt_edited": prompt_edited,
            "query": {
                "organism": spec.organism,
                # The scorer keys gold files by gene, and a multi-gene query has
                # no single one; join so the caller can see why gold missed.
                "gene": spec.gene_list[0] if spec.gene_list else "",
                "genes": spec.gene_list,
                "aliases": [],
                "additional_terms": spec.other_terms,
                "data_type": scorer_data_type(template.id),
                "collection": spec.collection_label,
                "top_k": spec.top_k,
                "filters": {},
            },
        },
        "retrieval": {
            "chunks_returned": len(sources),
            "chunks_after_dedupe": len(sources),
            "duplicates_removed": 0,
            "papers": len(manifest),
            "paper_manifest": manifest,
        },
        "rows": rows,
        "omitted": omitted,
        "refusals": [],
        # proposed counts what the model offered, before dedup. Merged rows are
        # neither emitted separately nor omitted, so they are named explicitly --
        # otherwise the tally silently fails to balance and the difference looks
        # like lost rows.
        "tally": {
            "proposed": proposed,
            "emitted": len(rows),
            "omitted": dropped,
            "merged": max(0, proposed - len(rows) - dropped),
        },
    }
