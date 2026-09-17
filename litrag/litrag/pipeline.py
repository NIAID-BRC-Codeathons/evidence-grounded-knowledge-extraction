"""One curation query, end to end.

This is the single code path behind both the CLI and the web UI: build the
query, call /v1/query, extract rows, resolve citations, dedupe, stamp
provenance. Neither surface reimplements any of it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import provenance as prov
from .client import QueryResult, RagStackClient
from .llm import LlmClient, LlmEndpoint
from .prompts import build_prompt
from .dedup import dedupe
from .extract import Extraction, Row, extract
from .templates import Template, TemplateError, TemplateRegistry

# Room left for the chat template's own wrapping around our prompt.
CONTEXT_MARGIN = 1024
# Never squeeze the retrieved context below this, whatever the output needs.
MIN_CONTEXT_TOKENS = 4000
# A table row costs roughly this much to write, so more sources means more
# output is needed before the answer gets cut off.
# Measured against Qwen at top_k=100: rows carry verbose assertions and
# phenotypes, so 110 tokens a source still truncated the table.
TOKENS_PER_SOURCE = 170
MAX_OUTPUT_TOKENS = 20000


def plan_output_tokens(template: Template, n_sources: int) -> int:
    """How much room the answer needs.

    The template declares a cap sized for the hosted path's own retrieval. On
    the local path we choose top_k, so at 100 sources that cap cuts the table
    off mid-row -- observed as truncated=True with rows silently missing.
    """
    declared = template.max_output_tokens or 2500
    if not template.is_table:
        return declared
    return max(declared, min(MAX_OUTPUT_TOKENS, TOKENS_PER_SOURCE * max(1, n_sources)))


@dataclass
class QuerySpec:
    """What the user asked for, independent of how they asked."""

    organism: str
    genes: str = ""
    other_terms: str = ""
    data_type: str = "literature-summary"
    top_k: int = 10
    collection: Optional[str] = None
    # Resolved corpus ids. Several means one multi-collection request, which
    # the API supports up to five.
    collections: List[str] = field(default_factory=list)
    retrieval_mode: str = "hybrid"
    max_context_tokens: Optional[int] = None
    keep_empty: bool = False
    no_dedupe: bool = False
    query_id: str = ""
    # Backend name recorded in the identity hash so switching models
    # invalidates a resume rather than silently reusing the old answer.
    backend: str = "server"

    @property
    def gene_list(self) -> List[str]:
        return [g.strip() for g in self.genes.replace(";", ",").split(",") if g.strip()]

    def template_vars(self) -> Dict[str, str]:
        values = {"organism": self.organism}
        if self.genes:
            values["genes"] = self.genes
        if self.other_terms:
            values["other_terms"] = self.other_terms
        return values

    def search_text(self, template: Optional[Template] = None) -> str:
        """The retrieval query.

        Mirrors how the original widget composed its search string: the field
        values joined together, with the data-type label included so retrieval
        is biased toward documents discussing that kind of finding.
        """
        parts = [self.organism, self.genes]
        if template is not None and template.is_table:
            parts.append(template.label)
        parts.append(self.other_terms)
        return " ".join(p.strip() for p in parts if p and p.strip())

    @property
    def collection_label(self) -> str:
        """What provenance records for the corpora searched."""
        if self.collections:
            return "+".join(self.collections)
        return self.collection or "default"

    def identity(self) -> str:
        """Stable id for this spec, used for resume and row attribution."""
        if self.query_id:
            return self.query_id
        raw = "|".join([
            self.organism, self.genes, self.other_terms,
            self.data_type, str(self.top_k),
            ",".join(self.collections) or (self.collection or ""),
            self.backend,
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class RunResult:
    """Everything one query produced."""

    spec: QuerySpec
    template: Template
    extraction: Extraction
    rows: List[Row]
    result: QueryResult
    request_body: Dict[str, Any] = field(default_factory=dict)

    @property
    def columns(self) -> List[str]:
        return self.extraction.columns

    def summary(self) -> Dict[str, Any]:
        return {
            "query_id": self.spec.identity(),
            "organism": self.spec.organism,
            "genes": self.spec.genes,
            "data_type": self.template.id,
            "n_sources": len(self.result.sources),
            # Asked for vs actually shown to the model: a smaller context
            # window silently drops the tail of a large retrieval.
            "top_k": self.spec.top_k,
            "sources_dropped": max(0, self.spec.top_k - len(self.result.sources)),
            "collections": self.spec.collection_label,
            "n_rows_raw": self.extraction.n_rows,
            "n_rows": len(self.rows),
            "dropped_empty": self.extraction.dropped_empty,
            "dropped_malformed": self.extraction.dropped_malformed,
            "unresolved_citations": self.extraction.unresolved_citations,
            "model": self.result.model,
            "truncated": bool(self.result.truncated),
            "generator": self.result.generator,
            "endpoint": self.result.endpoint,
            "prompt_hash": self.result.prompt_hash,
            "template_version": self.result.template_version,
            "template_hash": self.result.template_hash,
            "request_id": self.result.request_id,
            "elapsed_s": round(self.result.elapsed_s, 2) if self.result.elapsed_s else None,
        }


def build_request(spec: QuerySpec, template: Template) -> Dict[str, Any]:
    """The exact payload /v1/query will receive, without sending it."""
    template_vars = template.validate_vars(spec.template_vars())
    return RagStackClient.build_query_body(
        query=spec.search_text(template),
        top_k=spec.top_k,
        template=template.id,
        template_vars=template_vars,
        collection=spec.collection,
        collections=spec.collections,
        # The graph backend reports itself disabled on this deployment, so
        # asking for it would only cost latency.
        use_graph=False,
        retrieval_mode=spec.retrieval_mode,
    )


def _generate_locally(
    client: RagStackClient,
    spec: QuerySpec,
    template: Template,
    endpoint: LlmEndpoint,
    llm_client: Optional[LlmClient] = None,
) -> QueryResult:
    """Retrieve from RAGStack, then generate against a directly-addressed model.

    The hosted /v1/query cannot select a model -- its `llm` field returns an
    empty answer with no sources -- so choosing a generator means splitting
    retrieval from generation. The template declaration still defines the
    columns; only the prompt wrapping them is ours, and its hash is recorded.
    """
    sources = client.retrieve(
        query=spec.search_text(template),
        top_k=spec.top_k,
        collection=spec.collection,
        collections=spec.collections,
        use_graph=False,
        retrieval_mode=spec.retrieval_mode,
    )

    # A large top_k produces both a bigger prompt and more rows to write, and
    # the two compete for one context window. Size the output first, then give
    # the prompt what is left -- so raising top_k truncates retrieved context
    # rather than cutting the answer off mid-table.
    output_tokens = plan_output_tokens(template, len(sources))
    context_budget = spec.max_context_tokens or max(
        MIN_CONTEXT_TOKENS, endpoint.context_tokens - output_tokens - CONTEXT_MARGIN
    )

    prompt, prompt_hash, included = build_prompt(
        template,
        sources,
        organism=spec.organism,
        genes=spec.genes,
        other_terms=spec.other_terms,
        max_context_tokens=context_budget,
    )

    owns = llm_client is None
    llm = llm_client or LlmClient(endpoint)
    try:
        completion = llm.complete(prompt, max_tokens=output_tokens)
    finally:
        if owns:
            llm.close()

    if included < len(sources):
        # Keep only what the model was actually shown: a citation marker past
        # this point could not have come from a passage it read.
        sources = list(sources[:included])

    return QueryResult(
        answer=completion.text,
        sources=sources,
        rewritten_queries=[],
        # The declaration that defined these columns, even though the prompt
        # body was ours -- _generator and _prompt_hash keep that unambiguous.
        template=template.id,
        template_version=template.version,
        template_hash=template.hash,
        model=completion.model,
        truncated=completion.truncated,
        elapsed_s=completion.elapsed_s,
        generator="local",
        endpoint=completion.endpoint,
        prompt_hash=prompt_hash,
    )


def run_query(
    client: RagStackClient,
    registry: TemplateRegistry,
    spec: QuerySpec,
    endpoint: Optional[LlmEndpoint] = None,
    llm_client: Optional[LlmClient] = None,
) -> RunResult:
    """Execute one curation query and return processed rows.

    With `endpoint` set, retrieval and generation are split and the named model
    does the extraction. Without it, the hosted /v1/query does both.
    """
    template = registry.resolve(spec.data_type)
    template_vars = template.validate_vars(spec.template_vars())

    if template.is_local and endpoint is None:
        raise TemplateError(
            f"data type '{template.id}' is defined by LitRAG, not by the server, "
            f"so it needs a local generator. The hosted path does not know it. "
            f"Re-run without --llm server (the default is qwen)."
        )

    body = build_request(spec, template)

    if endpoint is not None:
        result = _generate_locally(client, spec, template, endpoint, llm_client)
    else:
        result = client.query(
            query=spec.search_text(template),
            top_k=spec.top_k,
            template=template.id,
            template_vars=template_vars,
            collection=spec.collection,
            use_graph=False,
            retrieval_mode=spec.retrieval_mode,
        )

    extraction = extract(
        result.answer,
        template,
        result.sources,
        requested_genes=spec.gene_list,
        keep_empty=spec.keep_empty,
    )

    query_id = spec.identity()
    rows = list(extraction.rows)
    if extraction.is_table and not spec.no_dedupe:
        rows = dedupe(rows, template.id, extraction.columns)

    record = prov.build(
        result,
        query_id=query_id,
        collection=spec.collection_label,
        top_k=spec.top_k,
        retrieval_mode=spec.retrieval_mode,
    )
    prov.stamp(rows, record, query_id)

    return RunResult(
        spec=spec,
        template=template,
        extraction=extraction,
        rows=rows,
        result=result,
        request_body=body,
    )


def merge_runs(
    runs: Sequence[RunResult],
    no_dedupe: bool = False,
) -> "tuple[List[Row], List[str]]":
    """Combine several runs into one table.

    Rows are deduplicated across queries per template, so a fact found by three
    different searches becomes one row with n_support=3 rather than three rows.
    Templates with different column sets stay in separate groups.
    """
    by_template: Dict[str, List[Row]] = {}
    columns_by_template: Dict[str, List[str]] = {}

    for run in runs:
        if not run.extraction.is_table:
            continue
        by_template.setdefault(run.template.id, []).extend(run.rows)
        columns_by_template.setdefault(run.template.id, run.columns)

    merged: List[Row] = []
    columns: List[str] = []
    for template_id, rows in by_template.items():
        cols = columns_by_template[template_id]
        merged.extend(rows if no_dedupe else dedupe(rows, template_id, cols))
        for column in cols:
            if column not in columns:
                columns.append(column)
    return merged, columns
