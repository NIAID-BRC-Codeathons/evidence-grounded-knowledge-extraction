"""FastAPI backend for the web UI.

The API key stays in this process and is never sent to the browser -- the
original widget relied on the caller's own session, so a standalone page had no
safe way to hold one. Every endpoint here is a thin shell over the same
pipeline the CLI uses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pathlib import Path

from . import __version__, formats
from .client import ApiError, RagStackClient
from .collections import ALL, CollectionRegistry, clean_title
from .config import ConfigError, load_config
from .llm import (DEFAULT_BACKEND, PRESETS, SERVER, LlmError, resolve_endpoint)
from .pipeline import QuerySpec, build_request, run_query
from .templates import TemplateError, TemplateRegistry

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="LitRAG", version=__version__)

_state: Dict[str, Any] = {"registry": None}


def _client() -> RagStackClient:
    try:
        return RagStackClient(load_config())
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _registry(client: RagStackClient) -> TemplateRegistry:
    """Templates change only when the operator edits them; cache per process."""
    if _state["registry"] is None:
        try:
            _state["registry"] = TemplateRegistry.fetch(client)
        except ApiError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _state["registry"]


class QueryBody(BaseModel):
    organism: str = Field(..., min_length=1)
    genes: str = ""
    other_terms: str = ""
    data_type: str = "literature-summary"
    top_k: int = Field(10, ge=1, le=50)
    collection: Optional[str] = None
    keep_empty: bool = False
    no_dedupe: bool = False
    llm: str = DEFAULT_BACKEND


def _resolve(client: RagStackClient, value):
    """Turn the UI's collection choice into ids, rejecting unknown ones."""
    try:
        return _collection_registry(client).resolve(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _endpoint(body: QueryBody):
    try:
        return resolve_endpoint(body.llm)
    except LlmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _collection_registry(client: RagStackClient) -> CollectionRegistry:
    try:
        return CollectionRegistry.fetch(client)
    except ApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _spec(body: QueryBody, endpoint=None, collections=None) -> QuerySpec:
    return QuerySpec(
        organism=body.organism.strip(),
        genes=body.genes.strip(),
        other_terms=body.other_terms.strip(),
        data_type=body.data_type,
        top_k=body.top_k,
        collection=body.collection or None,
        keep_empty=body.keep_empty,
        no_dedupe=body.no_dedupe,
        backend=endpoint.name if endpoint is not None else SERVER,
        collections=list(collections or []),
    )


@app.get("/api/health")
def health() -> Dict[str, Any]:
    with _client() as client:
        try:
            return {"litrag": __version__, "server": client.version()}
        except ApiError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/templates")
def templates() -> Dict[str, Any]:
    with _client() as client:
        registry = _registry(client)
    return {
        "templates": [
            {
                "id": t.id, "label": t.label, "output": t.output,
                "version": t.version, "hash": t.hash, "columns": t.columns,
                # Local types need a local generator; the UI warns rather than
                # letting the user discover it as a failed query.
                "local": t.is_local,
                "slots": [
                    {"name": s.name, "required": s.required,
                     "max_len": s.max_len, "label": s.label}
                    for s in t.slots
                ],
            }
            for t in registry
        ]
    }


@app.get("/api/backends")
def backends() -> Dict[str, Any]:
    """Generators the UI can offer."""
    return {
        "default": DEFAULT_BACKEND,
        "backends": [
            {"id": SERVER, "label": "Hosted RAGStack (Llama-4-Scout)",
             "note": "Server-side versioned prompt template"},
        ] + [
            {"id": name, "label": preset.described(), "note": preset.model}
            for name, preset in sorted(PRESETS.items())
        ],
    }


@app.get("/api/collections")
def collections() -> Dict[str, Any]:
    """Searchable corpora, plus the option to search them all at once."""
    with _client() as client:
        registry = _collection_registry(client)

    items = [
        {
            "id": c.id,
            "name": c.display_name,
            "label": c.label,
            "count": c.count,
            "size_note": c.size_note,
            "chunk_method": c.chunk_method,
        }
        for c in registry
    ]
    if len(registry) > 1:
        items.append({
            "id": ALL,
            "name": "All collections",
            "label": "Search every corpus in one request",
            "count": registry.total_count,
            "size_note": f"{registry.total_count / 1e6:.1f}M chunks",
            "chunk_method": "",
        })

    return {"collections": items, "default": registry.default}


@app.post("/api/request")
def preview_request(body: QueryBody) -> Dict[str, Any]:
    """The exact payload /v1/query would receive.

    Replaces the original widget's "View / Edit Prompt" dialog. The prompt text
    itself is owned by the server and deliberately not exposed, so what a client
    can honestly show is the request it is about to make.
    """
    with _client() as client:
        registry = _registry(client)
        try:
            template = registry.resolve(body.data_type)
            endpoint = _endpoint(body)
            spec = _spec(body, endpoint, _resolve(client, body.collection))
            declaration = {
                "id": template.id, "version": template.version,
                "hash": template.hash, "output": template.output,
                "columns": template.columns,
            }
            if endpoint is None:
                return {
                    "endpoint": "/v1/query",
                    "body": build_request(spec, template),
                    "template": declaration,
                }

            from .prompts import build_prompt
            prompt, digest, _ = build_prompt(
                template, [], organism=spec.organism, genes=spec.genes,
                other_terms=spec.other_terms,
            )
            return {
                "endpoint": f"{endpoint.base_url}/chat/completions",
                "body": {
                    "retrieve": {"endpoint": "/v1/retrieve",
                                 "query": spec.search_text(template),
                                 "top_k": spec.top_k,
                                 "collection": spec.collection},
                    "generate": {"model": endpoint.model,
                                 "thinking": endpoint.thinking,
                                 "max_tokens": template.max_output_tokens or 2500},
                },
                "template": declaration,
                "prompt": prompt,
                "prompt_hash": digest,
            }
        except TemplateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/query")
def run(body: QueryBody) -> Dict[str, Any]:
    endpoint = _endpoint(body)
    with _client() as client:
        chosen = _resolve(client, body.collection)
        registry = _registry(client)
        try:
            result = run_query(client, registry, _spec(body, endpoint, chosen),
                               endpoint=endpoint)
        except TemplateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LlmError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ApiError as exc:
            status = 502 if exc.status is None else exc.status
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    return {
        "summary": result.summary(),
        "is_table": result.extraction.is_table,
        "columns": result.columns,
        "answer": result.extraction.raw_answer,
        "request_body": result.request_body,
        "rows": [
            {
                "values": row.values,
                "citations": [c.to_dict() for c in row.citations],
                "flags": row.flags,
                "variants": row.variants,
                "n_support": row.n_support,
                "provenance": row.provenance,
            }
            for row in result.rows
        ],
        "sources": [_source_view(s) for s in result.result.sources],
    }


@app.post("/api/export", response_class=PlainTextResponse)
def export(body: QueryBody, fmt: str = "tsv") -> PlainTextResponse:
    if fmt not in formats.FORMATS:
        raise HTTPException(status_code=400, detail=f"unknown format '{fmt}'")

    endpoint = _endpoint(body)
    with _client() as client:
        chosen = _resolve(client, body.collection)
        registry = _registry(client)
        try:
            result = run_query(client, registry, _spec(body, endpoint, chosen),
                               endpoint=endpoint)
        except TemplateError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ApiError, LlmError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    text = formats.render(
        result.extraction, result.rows, fmt,
        provenance=True, meta=result.summary(),
        sources=result.result.sources if fmt == "json" else None,
    )
    media = {
        "tsv": "text/tab-separated-values", "csv": "text/csv",
        "json": "application/json", "jsonl": "application/x-ndjson",
        "md": "text/markdown", "table": "text/plain",
    }[fmt]
    suffix = {"table": "txt"}.get(fmt, fmt)
    return PlainTextResponse(
        text, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="litrag.{suffix}"'},
    )


def _source_view(source: Dict[str, Any]) -> Dict[str, Any]:
    summary = formats._source_summary(source)
    summary["title"] = clean_title(summary.get("title"))
    # Stamped only on multi-collection requests; lets the UI show provenance
    # per source when several corpora were searched.
    summary["collection"] = source.get("collection", "")
    return summary


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
