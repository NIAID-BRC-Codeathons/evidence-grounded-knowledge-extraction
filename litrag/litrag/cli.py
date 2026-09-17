"""Command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__, formats, retrieval
from .batch import BatchDefaults, load_specs, run_batch
from .client import ApiError, RagStackClient
from .collections import ALL, CollectionRegistry, clean_title
from .config import ConfigError, load_config
from .llm import (DEFAULT_BACKEND, PRESETS, SERVER, LlmClient, LlmError,
                  resolve_endpoint)
from .pipeline import QuerySpec, build_request, merge_runs, run_query
from .templates import TemplateError, TemplateRegistry

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Automated literature curation over the BV-BRC RAGStack API.",
)


def _err(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)


def _info(message: str) -> None:
    typer.secho(message, fg=typer.colors.CYAN, err=True)


def _connect(api_key: Optional[str], base_url: Optional[str], timeout: float = 300.0) -> RagStackClient:
    try:
        config = load_config(api_key=api_key, base_url=base_url, timeout=timeout)
    except ConfigError as exc:
        _err(str(exc))
        raise typer.Exit(2)
    return RagStackClient(config)


def _collections(client: RagStackClient) -> CollectionRegistry:
    try:
        return CollectionRegistry.fetch(client)
    except ApiError as exc:
        _err(f"Could not load collections: {exc}")
        raise typer.Exit(1)


def _resolve_collections(registry: CollectionRegistry, value):
    try:
        return registry.resolve(value)
    except ValueError as exc:
        _err(str(exc))
        raise typer.Exit(2)


def _print_sources(sources) -> None:
    """One entry per PAPER, not per passage.

    This used to print a numbered block per source, which was fine at ten and is
    a wall of near-identical text at several hundred. The number was also
    actively misleading: it looked like the [n] citation marker, but markers are
    now assigned per batch, so the two have nothing to do with each other.
    """
    papers: dict = {}
    for source in sources:
        meta = source.get("metadata") or {}
        key = meta.get("pmid") or meta.get("doi") or source.get("doc_id")
        entry = papers.setdefault(key, {
            "title": clean_title(meta.get("title")) or "Untitled",
            "journal": meta.get("journal") or "?", "year": meta.get("year") or "",
            "pmid": meta.get("pmid") or "-", "doi": meta.get("doi") or "-",
            "passages": 0, "expanded": 0, "best": None,
        })
        entry["passages"] += 1
        entry["expanded"] += 1 if source.get("expanded") else 0
        score = source.get("score")
        if isinstance(score, (int, float)):
            entry["best"] = score if entry["best"] is None else max(entry["best"], score)

    typer.echo(f"\nSources: {len(papers)} papers, {len(sources)} passages", err=True)
    for entry in sorted(papers.values(), key=lambda e: -(e["best"] or 0)):
        best = f"{entry['best']:.3f}" if entry["best"] is not None else "-"
        note = f"{entry['passages']} passage" + ("s" if entry["passages"] != 1 else "")
        if entry["expanded"]:
            note += f", {entry['expanded']} expanded"
        typer.echo(
            f"  {entry['title'][:78]}\n"
            f"      {entry['journal']} {entry['year']} "
            f"PMID:{entry['pmid']} DOI:{entry['doi']} "
            f"| {note}, best score {best}",
            err=True,
        )


def _report_gather(counts: dict, n_batches: int) -> None:
    """Say how much literature is about to be read, before the waiting starts.

    Worth printing even for one batch: "10 passages from 8 papers" is the single
    most surprising fact about how this tool works, and it was invisible.
    """
    bits = [f"reviewing {counts['n_passages']} passages "
            f"from {counts['n_papers']} papers"]
    if counts.get("n_expanded"):
        bits.append(f"({counts['n_expanded']} read around the search hits)")
    if n_batches > 1:
        bits.append(f"in {n_batches} parallel batches")
    _info(" ".join(bits))

    # The honest coverage line. A passage count sounds thorough and says nothing
    # about whether any paper was finished.
    complete, papers = counts.get("n_papers_complete", 0), counts["n_papers"]
    corpus = counts.get("collection_chunks") or 0
    line = f"  {complete} of {papers} papers read end to end"
    if complete == 0:
        line += " -- every paper is a partial window, so a finding buried " \
                "mid-paper can still be missed (try --depth full)"
    if corpus:
        share = 100.0 * counts["n_passages"] / corpus
        line += (f" | {counts['n_passages']:,} of {corpus:,} chunks in this "
                 f"collection ({share:.4g}%)")
    _info(line)


def _report_batch(done: int, total: int) -> None:
    if total > 1:
        _info(f"  [{done}/{total}] batch complete")


def _collection_count(registry: CollectionRegistry, ids) -> int:
    """Chunks in the corpus being searched -- what depth adapts to.

    Several collections at once: take the largest, since the depth chosen for
    the biggest corpus is the one that keeps the run affordable.
    """
    counts = [c.count for c in registry if c.id in set(ids or [])]
    return max(counts) if counts else 0


def _registry(client: RagStackClient) -> TemplateRegistry:
    try:
        return TemplateRegistry.fetch(client)
    except ApiError as exc:
        _err(f"Could not load prompt templates: {exc}")
        raise typer.Exit(1)


def _write(text: str, output: Optional[Path]) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        _info(f"Wrote {output}")
    else:
        sys.stdout.write(text)


API_KEY = typer.Option(None, "--api-key", help="API key (else $LITRAG_API_KEY or ~/.config/litrag/config.toml).")
BASE_URL = typer.Option(None, "--base-url", help="API base URL.")
LLM = typer.Option(
    DEFAULT_BACKEND, "--llm",
    help=f"Generator: {SERVER} (hosted), {', '.join(sorted(PRESETS))}, or a URL.",
)
LLM_MODEL = typer.Option(None, "--llm-model", help="Override the model id at that endpoint.")
THINKING = typer.Option(
    None, "--thinking/--no-thinking",
    help="Force reasoning mode on a model that supports it (Qwen defaults off).",
)


def _endpoint(llm, llm_model, thinking):
    try:
        return resolve_endpoint(llm, model=llm_model, thinking=thinking)
    except LlmError as exc:
        _err(str(exc))
        raise typer.Exit(2)


def _dry_run(spec, template, endpoint) -> None:
    """Show exactly what will be sent, for either generation path."""
    if endpoint is None:
        typer.echo(json.dumps(build_request(spec, template), indent=2))
        return

    from .prompts import build_prompt
    typer.echo(json.dumps({
        "retrieve": {
            "endpoint": "/v1/retrieve", "query": spec.search_text(template),
            "top_k": spec.top_k, "collection": spec.collection,
        },
        "generate": {
            "endpoint": f"{endpoint.base_url}/chat/completions",
            "model": endpoint.model, "thinking": endpoint.thinking,
            "max_tokens": template.max_output_tokens or 2500,
        },
    }, indent=2))
    prompt, digest, _ = build_prompt(
        template, [], organism=spec.organism, genes=spec.genes,
        other_terms=spec.other_terms,
    )
    typer.echo(f"\n--- prompt (hash {digest}, context omitted) ---\n{prompt}")


@app.command()
def version(
    api_key: Optional[str] = API_KEY,
    base_url: Optional[str] = BASE_URL,
) -> None:
    """Show litrag and server versions."""
    typer.echo(f"litrag {__version__}")
    try:
        config = load_config(api_key=api_key, base_url=base_url)
    except ConfigError:
        typer.echo("server  (no API key configured)")
        return
    with RagStackClient(config) as client:
        try:
            info = client.version()
            typer.echo(f"server  {info.get('version')} ({info.get('git_tag')})")
        except ApiError as exc:
            _err(f"server unreachable: {exc}")


@app.command()
def templates(
    api_key: Optional[str] = API_KEY,
    base_url: Optional[str] = BASE_URL,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List the data types this tenant offers, with their output columns."""
    with _connect(api_key, base_url) as client:
        registry = _registry(client)

    if as_json:
        typer.echo(json.dumps([
            {
                "id": t.id, "label": t.label, "output": t.output,
                "version": t.version, "hash": t.hash, "columns": t.columns,
                "slots": [
                    {"name": s.name, "required": s.required, "max_len": s.max_len}
                    for s in t.slots
                ],
            }
            for t in registry
        ], indent=2))
        return

    for template in registry:
        version_text = f"v{template.version}" if template.version else ""
        typer.secho(f"{template.id}  ", fg=typer.colors.GREEN, nl=False)
        typer.echo(f"{version_text}  [{template.output}]  {template.label}")
        if template.columns:
            typer.echo(f"    columns: {' | '.join(template.columns)}")
        required = [s.name for s in template.slots if s.required]
        optional = [s.name for s in template.slots if not s.required]
        typer.echo(f"    slots:   required={required or '-'} optional={optional or '-'}")


@app.command()
def collections(
    api_key: Optional[str] = API_KEY,
    base_url: Optional[str] = BASE_URL,
) -> None:
    """List searchable literature collections."""
    with _connect(api_key, base_url) as client:
        registry = _collections(client)

    for item in registry:
        marker = " (default)" if item.id == registry.default else ""
        typer.secho(f"{item.id}{marker}", fg=typer.colors.GREEN, nl=False)
        typer.echo(f"  {item.display_name}")
        typer.echo(f"    {item.size_note} | {item.chunk_method}")
        if item.label:
            typer.echo(f"    {item.label}")

    if len(registry) > 1:
        typer.secho(ALL, fg=typer.colors.GREEN, nl=False)
        typer.echo(f"  Search every collection at once "
                   f"({registry.total_count / 1e6:.1f}M chunks)")


@app.command()
def query(
    organism: str = typer.Option(..., "--organism", "-O", help="Organism of interest."),
    genes: str = typer.Option("", "--genes", "-g", help="Comma-separated genes/proteins."),
    other_terms: str = typer.Option("", "--other-terms", "-t", help="Additional search terms."),
    data_type: str = typer.Option(
        "summary", "--type", "-T",
        help="Data type: ppi, protein-function, mutation, summary (see `litrag templates`).",
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=100, help="Chunks to retrieve."),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c",
        help=f"Collection id, comma-separated ids, or '{ALL}'. Defaults to PubMed Central.",
    ),
    fmt: str = typer.Option("table", "--format", "-f", help="table|tsv|csv|json|jsonl|md."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write to file."),
    no_provenance: bool = typer.Option(False, "--no-provenance", help="Omit provenance columns."),
    keep_empty: bool = typer.Option(False, "--keep-empty", help="Keep evidence-free rows."),
    no_dedupe: bool = typer.Option(False, "--no-dedupe", help="Do not merge duplicate facts."),
    show_sources: bool = typer.Option(False, "--show-sources", help="Print retrieved sources."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the request body and exit."),
    depth: str = typer.Option(
        retrieval.STANDARD, "--depth", "-d",
        help="standard (top_k chunks, one call) | adaptive (read around each hit, "
             "batched in parallel; depth set by corpus size) | full (read the "
             "retrieved papers as completely as possible -- small corpora only).",
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", "-j", min=1, max=16,
        help="Batches generated at once when --depth is not standard.",
    ),
    llm: str = LLM,
    llm_model: Optional[str] = LLM_MODEL,
    thinking: Optional[bool] = THINKING,
    api_key: Optional[str] = API_KEY,
    base_url: Optional[str] = BASE_URL,
) -> None:
    """Run one curation query."""
    if fmt not in formats.FORMATS:
        _err(f"unknown format '{fmt}'. Choose from: {', '.join(formats.FORMATS)}")
        raise typer.Exit(2)

    if depth not in retrieval.DEPTHS:
        _err(f"unknown depth '{depth}'. Choose from: {', '.join(retrieval.DEPTHS)}")
        raise typer.Exit(2)

    endpoint = _endpoint(llm, llm_model, thinking)
    if depth != retrieval.STANDARD and endpoint is None:
        # The hosted endpoint owns its prompt, so it cannot be handed a batch of
        # passages. Saying so beats silently ignoring the flag.
        _err(f"--depth {depth} needs a directly-addressed model; the hosted "
             f"backend builds its own prompt and cannot be given a batch of "
             f"passages. Pass --llm qwen (or another model) alongside it.")
        raise typer.Exit(2)
    spec = QuerySpec(
        organism=organism, genes=genes, other_terms=other_terms,
        data_type=data_type, top_k=top_k,
        keep_empty=keep_empty, no_dedupe=no_dedupe,
        backend=endpoint.name if endpoint else SERVER,
        depth=depth, concurrency=concurrency,
    )

    with _connect(api_key, base_url) as client:
        collection_registry = _collections(client)
        spec.collections = _resolve_collections(collection_registry, collection)
        spec.collection_count = _collection_count(collection_registry, spec.collections)
        registry = _registry(client)
        try:
            template = registry.resolve(spec.data_type)
            if dry_run:
                _dry_run(spec, template, endpoint)
                return
            run = run_query(client, registry, spec, endpoint=endpoint,
                            on_gather=_report_gather, on_batch=_report_batch)
        except LlmError as exc:
            _err(str(exc))
            raise typer.Exit(1)
        except TemplateError as exc:
            _err(str(exc))
            raise typer.Exit(2)
        except ApiError as exc:
            _err(str(exc))
            raise typer.Exit(1)

    summary = run.summary()
    _info(
        f"{summary['n_sources']} passages from {summary['n_papers']} papers "
        f"in {summary['collections']} "
        f"-> {summary['n_rows']} rows "
        f"({summary['dropped_empty']} evidence-free dropped) "
        f"| {summary['model']} ({summary['generator']}) "
        f"| {summary['data_type']} v{summary['template_version']} "
        f"| {summary['elapsed_s']}s"
    )
    if summary["n_batches_failed"]:
        # A partial answer that looks complete is worse than a failure.
        _err(f"WARNING: {summary['n_batches_failed']} of {summary['n_batches']} "
             f"batches failed. These rows are drawn from part of the retrieved "
             f"literature, not all of it.")

    text = formats.render(
        run.extraction, run.rows, fmt,
        provenance=not no_provenance,
        meta=summary,
        sources=run.result.sources if (show_sources or fmt in ("json",)) else None,
    )
    _write(text, output)

    if show_sources and fmt not in ("json", "jsonl"):
        _print_sources(run.result.sources)


@app.command()
def batch(
    spec_file: Path = typer.Argument(..., exists=True, readable=True, help="TSV/CSV/JSON/YAML of queries."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write merged table here."),
    fmt: str = typer.Option("tsv", "--format", "-f", help="table|tsv|csv|json|jsonl|md."),
    concurrency: int = typer.Option(4, "--concurrency", "-j", min=1, max=16),
    data_type: str = typer.Option("summary", "--type", "-T", help="Default data type for rows that omit one."),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=100),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c",
        help=f"Collection id, comma-separated ids, or '{ALL}'.",
    ),
    resume: bool = typer.Option(False, "--resume", help="Skip queries already in the progress file."),
    progress: Optional[Path] = typer.Option(None, "--progress", help="Progress sidecar (default: <output>.progress.jsonl)."),
    no_provenance: bool = typer.Option(False, "--no-provenance"),
    keep_empty: bool = typer.Option(False, "--keep-empty"),
    no_dedupe: bool = typer.Option(False, "--no-dedupe", help="Do not merge duplicate facts across queries."),
    llm: str = LLM,
    llm_model: Optional[str] = LLM_MODEL,
    thinking: Optional[bool] = THINKING,
    api_key: Optional[str] = API_KEY,
    base_url: Optional[str] = BASE_URL,
) -> None:
    """Run many queries from a file and merge the results into one table."""
    if fmt not in formats.FORMATS:
        _err(f"unknown format '{fmt}'. Choose from: {', '.join(formats.FORMATS)}")
        raise typer.Exit(2)

    endpoint = _endpoint(llm, llm_model, thinking)
    defaults = BatchDefaults(
        data_type=data_type, top_k=top_k, collection=collection,
        keep_empty=keep_empty, no_dedupe=no_dedupe,
        backend=endpoint.name if endpoint else SERVER,
    )

    progress_path = progress
    if progress_path is None and output is not None:
        progress_path = output.with_suffix(output.suffix + ".progress.jsonl")

    generator = endpoint.described() if endpoint else "hosted /v1/query"

    completed = {"n": 0}

    def report(spec: QuerySpec, run, error: Optional[str]) -> None:
        completed["n"] += 1
        prefix = f"[{completed['n']}/{len(specs)}]"
        if error:
            typer.secho(f"{prefix} FAIL {spec.organism} ({spec.genes}): {error}",
                        fg=typer.colors.RED, err=True)
        else:
            summary = run.summary()
            typer.secho(
                f"{prefix} ok   {spec.organism} ({spec.genes or '-'}) "
                f"-> {summary['n_rows']} rows, {summary['n_sources']} sources, "
                f"{summary['elapsed_s']}s",
                fg=typer.colors.GREEN, err=True,
            )

    with _connect(api_key, base_url) as client:
        defaults.collections = _resolve_collections(_collections(client), collection)
        try:
            specs = load_specs(spec_file, defaults)
        except (ValueError, RuntimeError) as exc:
            _err(str(exc))
            raise typer.Exit(2)

        _info(f"{len(specs)} queries from {spec_file} (concurrency {concurrency}, "
              f"generator: {generator}, collections: {'+'.join(defaults.collections)})")

        registry = _registry(client)
        # Validate every spec before spending any calls: a typo in row 40
        # should not surface after 39 successful queries.
        try:
            for spec in specs:
                template = registry.resolve(spec.data_type)
                template.validate_vars(spec.template_vars())
        except TemplateError as exc:
            _err(str(exc))
            raise typer.Exit(2)

        outcome = run_batch(
            client, registry, specs,
            concurrency=concurrency,
            progress_path=progress_path,
            resume=resume,
            on_done=report,
            endpoint=endpoint,
        )

    if outcome.skipped:
        _info(f"{outcome.skipped} already done, restored from progress file")

    rows, columns = merge_runs(outcome.runs, no_dedupe=no_dedupe)

    # Rows recovered from the sidecar belong in the output too; without them a
    # fully-resumed run would write an empty table over a complete one.
    if outcome.restored_rows:
        for column in outcome.restored_columns:
            if column not in columns:
                columns.append(column)
        rows = list(outcome.restored_rows) + rows
        template_id = outcome.restored_template_id
        if template_id and not no_dedupe:
            from .dedup import dedupe as _dedupe
            rows = _dedupe(rows, template_id, columns)

    prose = [r for r in outcome.runs if not r.extraction.is_table]
    if rows or not prose:
        from .extract import Extraction
        merged = Extraction(columns=columns, is_table=True)
        text = formats.render(merged, rows, fmt, provenance=not no_provenance)
    else:
        text = "\n\n".join(
            f"### {r.spec.organism} {r.spec.genes}\n{r.extraction.raw_answer}" for r in prose
        ) + "\n"

    if not rows and output and output.exists() and output.stat().st_size > 0:
        _err(
            f"Refusing to overwrite {output} with an empty table "
            f"({len(outcome.failures)} queries failed). Existing file left untouched."
        )
        raise typer.Exit(1)

    _write(text, output)

    _info(
        f"Done: {len(outcome.runs)} ok, {len(outcome.failures)} failed, "
        f"{outcome.skipped} restored -> {len(rows)} merged rows"
    )
    if outcome.failures:
        for failure in outcome.failures:
            _err(f"  failed: {failure.organism}: {failure.error}")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port", "-p"),
    api_key: Optional[str] = API_KEY,
    base_url: Optional[str] = BASE_URL,
) -> None:
    """Start the web UI."""
    try:
        load_config(api_key=api_key, base_url=base_url)
    except ConfigError as exc:
        _err(str(exc))
        raise typer.Exit(2)

    import os

    import uvicorn

    if api_key:
        os.environ["LITRAG_API_KEY"] = api_key
    if base_url:
        os.environ["LITRAG_BASE_URL"] = base_url

    _info(f"LitRAG UI on http://{host}:{port}")
    uvicorn.run("litrag.server:app", host=host, port=port, log_level="info")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
