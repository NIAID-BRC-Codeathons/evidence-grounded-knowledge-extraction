"""Stamp rows with what produced them.

Every field here comes back from the API today. Recording them means a table
can be traced to the exact template version, model, and retrieval settings that
generated it -- the difference between a result and a reproducible result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .client import QueryResult

# Underscore-prefixed so provenance sorts away from the curated data columns.
PROVENANCE_COLUMNS: List[str] = [
    "_query_id",
    "_template",
    "_template_version",
    "_template_hash",
    "_model",
    "_collection",
    "_top_k",
    # How much literature this row was drawn from. _top_k alone stopped
    # describing a run once passages could be expanded beyond the search hits
    # and split across several calls: the same _top_k=50 can mean 50 passages
    # or 285. An exported table has to say which, or it cannot be reproduced.
    "_depth",
    "_n_passages",
    "_n_papers",
    "_n_batches",
    "_retrieval_mode",
    "_retrieved_at",
    "_request_id",
    "_generator",
    "_llm_endpoint",
    "_prompt_hash",
]

# Columns describing the extraction itself rather than its origin.
ANNOTATION_COLUMNS: List[str] = [
    "_n_support", "_flags", "_citations", "_pmids",
    # The passages a claim came from -- the unit of evidence a curator checks,
    # and what /v1/chunks?ids= takes to fetch the text back.
    "_chunk_ids", "_markers",
    # Standard notation for a value the source wrote in prose, so a row can be
    # matched against one that used notation.
    "_standard_notation",
    # What the source actually wrote, where a value was canonicalised.
    "_as_written",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(
    result: QueryResult,
    query_id: str,
    collection: Optional[str],
    top_k: int,
    retrieval_mode: str = "hybrid",
    retrieved_at: Optional[str] = None,
    depth: str = "standard",
    n_batches: int = 1,
) -> Dict[str, Any]:
    """The provenance record for one query."""
    papers = len({s.get("doc_id") for s in (result.sources or [])
                  if s.get("doc_id")})
    return {
        "_query_id": query_id,
        "_template": result.template or "",
        "_template_version": result.template_version if result.template_version is not None else "",
        "_template_hash": result.template_hash or "",
        "_model": result.model or "",
        "_collection": collection or "default",
        "_top_k": top_k,
        "_depth": depth,
        "_n_passages": len(result.sources or []),
        "_n_papers": papers,
        "_n_batches": n_batches,
        "_retrieval_mode": retrieval_mode,
        "_retrieved_at": retrieved_at or utc_now(),
        "_request_id": result.request_id or "",
        "_generator": result.generator,
        "_llm_endpoint": result.endpoint,
        "_prompt_hash": result.prompt_hash,
    }


def stamp(rows: Iterable[Any], record: Dict[str, Any], query_id: str) -> None:
    """Attach a provenance record to each row, in place."""
    for row in rows:
        row.provenance = dict(record)
        if query_id and query_id not in row.query_ids:
            row.query_ids.append(query_id)
