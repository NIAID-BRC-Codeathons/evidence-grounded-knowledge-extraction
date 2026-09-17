"""One curation query, end to end.

This is the single code path behind both the CLI and the web UI: build the
query, call /v1/query, extract rows, resolve citations, dedupe, stamp
provenance. Neither surface reimplements any of it.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import provenance as prov
from . import retrieval
from .client import ApiError, QueryResult, RagStackClient
from .llm import LlmClient, LlmEndpoint, LlmError
from .prompts import build_prompt
from .dedup import dedupe
from .extract import Extraction, Row, extract
from .templates import Template, TemplateError, TemplateRegistry


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
    # How much to read. "standard" is the original behaviour: top_k chunks, one
    # call, no expansion. See retrieval.plan.
    depth: str = retrieval.STANDARD
    # Chunk count of the corpus being searched, which is what depth adapts to.
    # The registry knows it before the query runs; 0 means "unknown", which
    # plan() treats as a large corpus -- the conservative direction.
    collection_count: int = 0
    concurrency: int = retrieval.MAX_CONCURRENCY

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
            # Depth changes how much was read, so a resumed batch must not mix
            # shallow and deep rows into one table under the same id.
            self.depth,
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
            # How much was actually read, which "n_sources" alone no longer
            # conveys: a passage is a fragment, and several belong to one paper.
            "n_papers": len({s.get("doc_id") for s in self.result.sources
                             if s.get("doc_id")}),
            "n_expanded": sum(1 for s in self.result.sources if s.get("expanded")),
            "depth": self.spec.depth,
            "n_batches": self.extraction.n_batches,
            "n_batches_failed": self.extraction.n_batches_failed,
            "collections": self.spec.collection_label,
            "n_rows_raw": self.extraction.n_rows,
            "n_rows": len(self.rows),
            "dropped_empty": self.extraction.dropped_empty,
            "dropped_malformed": self.extraction.dropped_malformed,
            "unresolved_citations": self.extraction.unresolved_citations,
            "model": self.result.model,
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


@dataclass
class _BatchOutcome:
    """What one batch produced, or why it produced nothing."""

    index: int
    completion: Any = None
    extraction: Optional[Extraction] = None
    prompt_hash: str = ""
    error: Optional[str] = None


def _run_batches(
    llm: LlmClient,
    spec: QuerySpec,
    template: Template,
    batches: Sequence[Sequence[Dict[str, Any]]],
    on_batch: Optional[Callable[[int, int], None]] = None,
) -> List[_BatchOutcome]:
    """Send every batch, concurrently, and extract each against its own slice.

    The concurrency shape is lifted from `batch.py`: a ThreadPoolExecutor, a
    future-to-key dict, `as_completed`, and a per-future try/except so one
    failure cannot take the query down. One `LlmClient` is shared across workers
    -- httpx.Client is thread-safe and pooling the connections is the point.

    A single batch runs inline. Threads for one call would add nothing but a
    stack frame, and keeping that path identical to the old one is what lets
    `depth=standard` stay a true control arm.
    """
    total = len(batches)
    max_tokens = template.max_output_tokens or 2500

    def work(index: int) -> _BatchOutcome:
        slice_ = batches[index]
        prompt, prompt_hash, _ = build_prompt(
            template, slice_,
            organism=spec.organism, genes=spec.genes,
            other_terms=spec.other_terms,
            max_context_tokens=spec.max_context_tokens,
        )
        completion = llm.complete(prompt, max_tokens=max_tokens)
        # Extract HERE, against `slice_` -- the exact passages this prompt
        # numbered. Deferring it to the caller would mean resolving a marker
        # against the wrong list.
        extraction = extract(
            completion.text, template, slice_,
            requested_genes=spec.gene_list, keep_empty=spec.keep_empty,
        )
        return _BatchOutcome(index, completion, extraction, prompt_hash)

    if total <= 1:
        try:
            outcomes = [work(0)]
        except (LlmError, ValueError) as exc:
            outcomes = [_BatchOutcome(0, error=str(exc))]
        if on_batch:
            on_batch(1, total)
        return outcomes

    outcomes: List[_BatchOutcome] = []
    # Never more workers than batches (idle threads help nobody), and never
    # more than the server rewards -- see retrieval.MAX_CONCURRENCY.
    workers = max(1, min(spec.concurrency, total, retrieval.MAX_CONCURRENCY))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, i): i for i in range(total)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcomes.append(future.result())
            except (ApiError, TemplateError, LlmError, ValueError) as exc:
                # One lost batch costs a fraction of the answer; aborting costs
                # all of it. The loss is counted and surfaced, never hidden.
                outcomes.append(_BatchOutcome(index, error=str(exc)))
            if on_batch:
                on_batch(len(outcomes), total)

    # as_completed yields in finish order; restore submission order so the
    # merged answer and the row order do not depend on which batch was fastest.
    return sorted(outcomes, key=lambda o: o.index)


def _merge_extractions(done: Sequence[_BatchOutcome], total: int) -> Extraction:
    """Fold per-batch extractions into one, summing the counters.

    Rows keep the citations their own batch resolved, so the later dedupe merges
    them on real document identity rather than on a marker number that means
    something different in every batch.
    """
    first = done[0].extraction
    merged = Extraction(
        columns=list(first.columns),
        is_table=first.is_table,
        transposed=any(o.extraction.transposed for o in done),
        raw_answer="\n\n".join(o.extraction.raw_answer for o in done
                               if o.extraction.raw_answer),
    )
    for outcome in done:
        part = outcome.extraction
        merged.rows.extend(part.rows)
        merged.dropped_empty += part.dropped_empty
        merged.dropped_malformed += part.dropped_malformed
        merged.split_compound += part.split_compound
        merged.unresolved_citations += part.unresolved_citations
        # A later batch may declare a column the first one did not, e.g. when a
        # batch returned no rows at all.
        for column in part.columns:
            if column not in merged.columns:
                merged.columns.append(column)

    merged.n_batches = total
    merged.n_batches_failed = total - len(done)
    return merged


def rewrite_reference_cells(rows: Sequence[Row], columns: Sequence[str]) -> None:
    """Replace per-batch `[n]` markers with the papers they resolved to.

    Markers are assigned per prompt, so once a query runs as several batches the
    same `[17]` means a different paper in each one -- measured on a live run,
    one marker text covered three distinct PMIDs. The resolved citations on the
    row are correct, but the Reference CELL still held the model's raw text, so
    an exported table showed three different papers under one identical label
    and gave a reader no way to tell them apart.

    Only applied to batched runs: with a single batch the markers still index
    the one source list, and leaving them untouched keeps that path byte-exact.
    """
    from .extract import _REFERENCE_COLUMNS

    targets = [c for c in columns if c.strip().lower() in _REFERENCE_COLUMNS]
    if not targets:
        return
    for row in rows:
        if not row.citations:
            continue
        labels = []
        for citation in row.citations:
            if citation.pmid:
                labels.append(f"PMID:{citation.pmid}")
            elif citation.doi:
                labels.append(f"DOI:{citation.doi}")
            elif citation.pmcid:
                labels.append(citation.pmcid)
        if labels:
            for column in targets:
                row.values[column] = "; ".join(dict.fromkeys(labels))


def _combined_hash(hashes: Sequence[str]) -> str:
    """One stable key for a run made of several prompts.

    The ledger joins a run to the prompt that produced it. With N prompts there
    is no single hash, so record a hash of the sorted batch hashes: identical
    inputs give an identical key, and any prompt change moves it.
    """
    if len(hashes) == 1:
        return hashes[0]
    digest = hashlib.sha256("|".join(sorted(hashes)).encode("utf-8"))
    return digest.hexdigest()[:16]


def gather(
    client: RagStackClient,
    spec: QuerySpec,
    template: Template,
    plan: Optional[retrieval.RetrievalPlan] = None,
) -> Tuple[List[Dict[str, Any]], retrieval.RetrievalPlan]:
    """Retrieve, then read around each hit. No LLM call, so this is free.

    Split out from generation so a caller can show the user what is about to be
    read -- the UI's "reviewing N passages from M papers" and the CLI's banner
    both come from here -- without paying for a completion first.
    """
    plan = plan or retrieval.plan(spec.collection_count, spec.depth, spec.top_k)
    sources = client.retrieve(
        query=spec.search_text(template),
        top_k=plan.top_k,
        collection=spec.collection,
        collections=spec.collections,
        use_graph=False,
        retrieval_mode=spec.retrieval_mode,
    )
    sources = retrieval.dedupe_chunks(sources)
    # Multi-collection requests have no single collection to scope a chunk
    # lookup to, and the endpoint requires one -- it returns an empty list
    # rather than an error without it. Reading around hits is skipped in that
    # case rather than silently returning nothing.
    scope = spec.collection or (spec.collections[0] if len(spec.collections) == 1 else None)
    if scope:
        if plan.depth == retrieval.FULL:
            sources = retrieval.dedupe_chunks(
                retrieval.complete_documents(client, sources, scope)
            )
        elif plan.expands:
            sources = retrieval.dedupe_chunks(
                retrieval.expand(client, sources, scope, hops=plan.hops)
            )
    return retrieval.order(sources), plan


def _generate_locally(
    client: RagStackClient,
    spec: QuerySpec,
    template: Template,
    endpoint: LlmEndpoint,
    llm_client: Optional[LlmClient] = None,
    on_batch: Optional[Callable[[int, int], None]] = None,
    on_gather: Optional[Callable[[Dict[str, int], int], None]] = None,
) -> Tuple[QueryResult, Extraction]:
    """Retrieve from RAGStack, then generate against a directly-addressed model.

    The hosted /v1/query cannot select a model -- its `llm` field returns an
    empty answer with no sources -- so choosing a generator means splitting
    retrieval from generation. The template declaration still defines the
    columns; only the prompt wrapping them is ours, and its hash is recorded.

    Passages are packed into batches and sent concurrently. **Each batch is
    extracted against its own slice, here, rather than by the caller against one
    flat list.** That is not an optimisation -- it is the only way the citation
    markers stay meaningful. Sources are numbered from 1 within each prompt, so
    `[2]` in one batch and `[2]` in another are different papers; resolving each
    batch's markers against the passages that batch actually saw turns them into
    real PMIDs and doc ids before anything merges.
    """
    started = time.monotonic()
    owns = llm_client is None
    llm = llm_client or LlmClient(endpoint)
    try:
        sources, plan = gather(client, spec, template)
        # Ask the model how much it can take rather than assuming. Context
        # windows differ by more than 2x between the two models on one host, so
        # a fixed budget is a guess that happens to be safe. Cached on the
        # client: one lookup before the fan-out, not one per batch.
        budget = plan.batch_chars
        if budget:
            budget = retrieval.batch_chars_for(llm.context_limit(), budget)
        batches = retrieval.pack(sources, budget) or [[]]

        if len(batches) > retrieval.MAX_BATCHES:
            # Refuse loudly. Silently trimming would report a smaller corpus as
            # though it were everything, which is the failure mode this whole
            # change exists to remove.
            raise LlmError(
                f"{len(sources)} passages would need {len(batches)} generation "
                f"calls, over the limit of {retrieval.MAX_BATCHES}. Narrow the "
                f"query, pick a smaller collection, or lower --top-k."
            )
        if on_gather:
            # Before any generation, so a caller can say how much is about to be
            # read while the user waits for it.
            on_gather(retrieval.summarise(sources), len(batches))

        outcomes = _run_batches(llm, spec, template, batches, on_batch)
    finally:
        if owns:
            llm.close()

    done = [o for o in outcomes if o.extraction is not None]
    if not done:
        errors = [o.error for o in outcomes if o.error]
        raise LlmError(
            f"all {len(batches)} generation batches failed. First error: "
            f"{errors[0] if errors else 'unknown'}"
        )

    extraction = _merge_extractions(done, len(batches))
    first = done[0].completion

    return QueryResult(
        answer=extraction.raw_answer,
        sources=sources,
        rewritten_queries=[],
        # The declaration that defined these columns, even though the prompt
        # body was ours -- _generator and _prompt_hash keep that unambiguous.
        template=template.id,
        template_version=template.version,
        template_hash=template.hash,
        model=first.model,
        truncated=any(o.completion.truncated for o in done),
        # Wall clock, not the sum of the batches: they ran concurrently, so
        # summing would report a duration that never elapsed.
        elapsed_s=time.monotonic() - started,
        generator="local",
        endpoint=first.endpoint,
        prompt_hash=_combined_hash([o.prompt_hash for o in done]),
    ), extraction


def run_query(
    client: RagStackClient,
    registry: TemplateRegistry,
    spec: QuerySpec,
    endpoint: Optional[LlmEndpoint] = None,
    llm_client: Optional[LlmClient] = None,
    on_batch: Optional[Callable[[int, int], None]] = None,
    on_gather: Optional[Callable[[Dict[str, int], int], None]] = None,
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
        # The local path extracts per batch, against the passages each prompt
        # actually numbered, so it hands back the extraction rather than letting
        # us redo it here against a flat list the markers do not index into.
        result, extraction = _generate_locally(
            client, spec, template, endpoint, llm_client,
            on_batch=on_batch, on_gather=on_gather,
        )
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
    if extraction.n_batches > 1:
        rewrite_reference_cells(rows, extraction.columns)

    record = prov.build(
        result,
        query_id=query_id,
        collection=spec.collection_label,
        top_k=spec.top_k,
        retrieval_mode=spec.retrieval_mode,
        depth=spec.depth,
        n_batches=extraction.n_batches,
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
