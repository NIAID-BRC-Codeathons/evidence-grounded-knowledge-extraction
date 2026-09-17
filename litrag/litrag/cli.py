"""Command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__, formats
from .batch import BatchDefaults, load_specs, run_batch
from .client import ApiError, RagStackClient
from .collections import ALL, CollectionRegistry
from .config import ConfigError, load_config
from . import glossary as _glossary
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
def glossary(
    term: Optional[str] = typer.Argument(None, help="Show one column or flag."),
) -> None:
    """Explain the output columns and the row flags."""
    import textwrap

    def show(name: str, text: str) -> None:
        typer.secho(f"{name}", fg=typer.colors.GREEN)
        for line in textwrap.wrap(text, width=76):
            typer.echo(f"    {line}")

    if term:
        text = _glossary.describe_column(term) or _glossary.describe_flag(term)
        if not text:
            _err(f"no glossary entry for '{term}'")
            raise typer.Exit(1)
        show(term, text)
        return

    typer.secho("Columns", bold=True)
    for name, text in _glossary.COLUMNS.items():
        show(name, text)
    typer.secho("\nAdded to every table", bold=True)
    for name, text in _glossary.DERIVED.items():
        show(name, text)
    typer.secho("\nRow flags", bold=True)
    for name, text in _glossary.FLAGS.items():
        show(name, text)


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
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=100, help="Chunks to retrieve (1-100)."),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c",
        help=f"Collection id, comma-separated ids, or '{ALL}'. Defaults to PubMed Central.",
    ),
    fmt: str = typer.Option("table", "--format", "-f", help="table|tsv|csv|json|jsonl|md."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write to file."),
    no_provenance: bool = typer.Option(False, "--no-provenance", help="Omit provenance columns."),
    keep_empty: bool = typer.Option(False, "--keep-empty", help="Keep evidence-free rows."),
    no_dedupe: bool = typer.Option(False, "--no-dedupe", help="Do not merge duplicate facts."),
    retrieval_mode: str = typer.Option(
        "fused", "--retrieval-mode",
        help="fused (hybrid+bm25, default), hybrid, vector, or bm25.",
    ),
    show_sources: bool = typer.Option(False, "--show-sources", help="Print retrieved sources."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the request body and exit."),
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

    endpoint = _endpoint(llm, llm_model, thinking)
    spec = QuerySpec(
        organism=organism, genes=genes, other_terms=other_terms,
        data_type=data_type, top_k=top_k,
        keep_empty=keep_empty, no_dedupe=no_dedupe,
        retrieval_mode=retrieval_mode,
        backend=endpoint.name if endpoint else SERVER,
    )

    with _connect(api_key, base_url) as client:
        spec.collections = _resolve_collections(_collections(client), collection)
        registry = _registry(client)
        try:
            template = registry.resolve(spec.data_type)
            if dry_run:
                _dry_run(spec, template, endpoint)
                return
            run = run_query(client, registry, spec, endpoint=endpoint)
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
        f"{summary['n_sources']} sources from {summary['collections']} "
        f"-> {summary['n_rows']} rows "
        f"({summary['dropped_empty']} evidence-free dropped) "
        f"| {summary['model']} ({summary['generator']}) "
        f"| {summary['data_type']} v{summary['template_version']} "
        f"| {summary['elapsed_s']}s"
    )
    if summary.get("sources_dropped"):
        _err(
            f"Warning: {summary['sources_dropped']} of {summary['top_k']} retrieved "
            f"sources did not fit this model's context and were not used. "
            f"Use --llm qwen for a larger window, or lower --top-k."
        )
    if summary.get("truncated"):
        # A cut-off table is missing rows; it must not look like a full result.
        _err(
            "Warning: the model hit its output limit, so this table is "
            "incomplete. Lower --top-k, or narrow the query."
        )

    text = formats.render(
        run.extraction, run.rows, fmt,
        provenance=not no_provenance,
        meta=summary,
        sources=run.result.sources if (show_sources or fmt in ("json",)) else None,
    )
    _write(text, output)

    if show_sources and fmt not in ("json", "jsonl"):
        typer.echo("\nSources:", err=True)
        for index, source in enumerate(run.result.sources, start=1):
            meta = source.get("metadata", {}) or {}
            typer.echo(
                f"  [{index}] {meta.get('title', 'Untitled')[:80]}\n"
                f"      {meta.get('journal', '?')} {meta.get('year', '')} "
                f"PMID:{meta.get('pmid', '-')} DOI:{meta.get('doi', '-')} "
                f"score={source.get('score', 0):.3f}",
                err=True,
            )


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
