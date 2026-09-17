"""Typed client for the RAGStack API.

Only the read-only query surface is wrapped. Ingest, collection management,
grading, and admin endpoints exist on the server but are deliberately out of
scope -- a curation tool should not be able to mutate the corpus it cites.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

from .config import Config

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
BACKOFF_BASE = 1.5


class ApiError(RuntimeError):
    """An API call failed. Carries the server's request id where available."""

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        request_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        self.status = status
        self.request_id = request_id
        self.detail = detail
        parts = [message]
        if detail and detail not in message:
            parts.append(detail)
        if request_id:
            parts.append(f"(x-request-id: {request_id})")
        super().__init__(" ".join(parts))


@dataclass
class QueryResult:
    """A /v1/query response plus the transport metadata we need for provenance."""

    answer: str
    sources: List[Dict[str, Any]]
    rewritten_queries: List[str] = field(default_factory=list)
    template: Optional[str] = None
    template_version: Optional[int] = None
    template_hash: Optional[str] = None
    model: Optional[str] = None
    truncated: Optional[bool] = None
    request_id: Optional[str] = None
    elapsed_s: Optional[float] = None
    # "server" for hosted /v1/query, "local" for a directly-addressed vLLM.
    generator: str = "server"
    endpoint: str = ""
    # Set only on the local path, where LitRAG owns the prompt and the server's
    # template hash no longer describes what was sent.
    prompt_hash: str = ""

    @classmethod
    def from_response(
        cls,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
        elapsed_s: Optional[float] = None,
    ) -> "QueryResult":
        return cls(
            answer=payload.get("answer", ""),
            sources=payload.get("sources", []) or [],
            rewritten_queries=payload.get("rewritten_queries", []) or [],
            template=payload.get("template"),
            template_version=payload.get("template_version"),
            template_hash=payload.get("template_hash"),
            model=payload.get("model"),
            truncated=payload.get("truncated"),
            request_id=request_id,
            elapsed_s=elapsed_s,
        )


# The API caps a multi-collection request at five (QueryRequest.collections).
MAX_COLLECTIONS = 5


def _apply_collections(
    body: Dict[str, Any],
    collection: Optional[str],
    collections: Optional[Sequence[str]],
) -> None:
    """Set whichever collection field the request needs.

    `collections` takes precedence; a single-entry list is sent as the scalar
    `collection` so the response stays byte-identical to a plain single-corpus
    request, which is what the server's compatibility guarantee is built on.
    """
    if collections:
        unique = list(dict.fromkeys(c for c in collections if c))
        if len(unique) > MAX_COLLECTIONS:
            raise ApiError(
                f"at most {MAX_COLLECTIONS} collections per request, got {len(unique)}"
            )
        if len(unique) == 1:
            body["collection"] = unique[0]
        elif unique:
            body["collections"] = unique
        return
    if collection:
        body["collection"] = collection


# Retrieval modes combined by the default "fused" strategy.
FUSED_MODES = ("hybrid", "bm25")
# Reciprocal rank fusion constant. 60 is the value from the original paper and
# damps the influence of any single list's top ranks.
RRF_K = 60


def fuse_rankings(
    rankings: Sequence[Sequence[Dict[str, Any]]], k: int = RRF_K
) -> List[Dict[str, Any]]:
    """Merge ranked source lists by reciprocal rank fusion.

    A chunk scores 1/(k + rank) in each list it appears in. Chunks found by
    both modes rise; a chunk found by only one still places, which is the point
    -- that is how a BM25-only hit survives into the final set.
    """
    scores: Dict[str, float] = {}
    first_seen: Dict[str, Dict[str, Any]] = {}
    for ranking in rankings:
        for rank, source in enumerate(ranking, start=1):
            key = source.get("chunk_id") or source.get("doc_id") or repr(source)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, source)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [first_seen[key] for key in ordered]


class RagStackClient:
    """Thin wrapper over the RAGStack HTTP API with retry and error shaping."""

    def __init__(self, config: Config, client: Optional[httpx.Client] = None) -> None:
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "X-API-Key": config.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "RagStackClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- transport ---------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "tuple[Dict[str, Any], Optional[str], float]":
        last_error: Optional[ApiError] = None
        started = time.monotonic()

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.request(
                    method, path, json=json_body, params=params
                )
            except httpx.TimeoutException as exc:
                last_error = ApiError(f"request to {path} timed out: {exc}")
            except httpx.HTTPError as exc:
                last_error = ApiError(f"request to {path} failed: {exc}")
            else:
                request_id = response.headers.get("x-request-id")
                if response.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                    last_error = ApiError(
                        f"{method} {path} returned {response.status_code}",
                        status=response.status_code,
                        request_id=request_id,
                    )
                elif response.is_success:
                    elapsed = time.monotonic() - started
                    return response.json(), request_id, elapsed
                else:
                    raise ApiError(
                        f"{method} {path} failed with HTTP {response.status_code}",
                        status=response.status_code,
                        request_id=request_id,
                        detail=_extract_detail(response),
                    )

            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE ** attempt)

        raise last_error or ApiError(f"{method} {path} failed after {MAX_ATTEMPTS} attempts")

    # -- endpoints ---------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        payload, _, _ = self._request("GET", "/health")
        return payload

    def version(self) -> Dict[str, Any]:
        payload, _, _ = self._request("GET", "/v1/version")
        return payload

    def prompt_templates(self) -> List[Dict[str, Any]]:
        payload, _, _ = self._request("GET", "/v1/prompt-templates")
        return payload.get("templates", [])

    def collections(self) -> Dict[str, Any]:
        payload, _, _ = self._request("GET", "/v1/collections")
        return payload

    def chunks(
        self,
        ids: Sequence[str],
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch specific chunks by id -- used to read around a retrieved hit.

        `collection` is not optional in practice. The endpoint scopes the lookup,
        and for any collection that is not the server's default it returns an
        empty list rather than an error when the parameter is absent. A caller
        that forgets it sees "no such chunks", which is indistinguishable from a
        corpus whose chunks record no neighbours.
        """
        wanted = [str(i) for i in ids if i]
        if not wanted:
            return []
        params: Dict[str, Any] = {"ids": ",".join(wanted)}
        if collection:
            params["collection"] = collection
        payload, _, _ = self._request("GET", "/v1/chunks", params=params)
        return payload.get("chunks", [])

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        collections: Optional[Sequence[str]] = None,
        use_graph: bool = False,
        retrieval_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "use_graph": use_graph,
            "retrieval_mode": retrieval_mode,
        }
        _apply_collections(body, collection, collections)
        if filters:
            body["filters"] = filters
        payload, _, _ = self._request("POST", "/v1/retrieve", json_body=body)
        return payload.get("sources", [])

    def retrieve_fused(
        self,
        query: str,
        top_k: int = 5,
        modes: Sequence[str] = FUSED_MODES,
        collection: Optional[str] = None,
        collections: Optional[Sequence[str]] = None,
        retrieval_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve under several modes and fuse the rankings.

        Dense retrieval ranks a chunk by what it is *about*, so a passage that
        mentions NS1-53 while being about vaccine stability sinks below the top
        100 -- measured on the Dengue corpus, where BM25 ranked that same chunk
        12th and hybrid missed it entirely. Gene names, mutation codes and
        accessions are exactly the literal tokens BM25 is good at and dense
        similarity is not, so a curation tool should not rely on either alone.

        Fused with reciprocal rank fusion, which needs only the rankings and so
        does not care that the two modes score on different scales.
        """
        rankings = [
            self.retrieve(
                query=query, top_k=top_k, collection=collection,
                collections=collections, use_graph=False,
                retrieval_mode=mode, filters=filters,
            )
            for mode in modes
        ]
        return fuse_rankings(rankings)[:top_k]

    def query(
        self,
        query: str,
        top_k: int = 5,
        template: Optional[str] = None,
        template_vars: Optional[Dict[str, str]] = None,
        collection: Optional[str] = None,
        collections: Optional[Sequence[str]] = None,
        use_graph: bool = False,
        retrieval_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        body = self.build_query_body(
            query=query,
            top_k=top_k,
            template=template,
            template_vars=template_vars,
            collection=collection,
            collections=collections,
            use_graph=use_graph,
            retrieval_mode=retrieval_mode,
            filters=filters,
        )
        payload, request_id, elapsed = self._request("POST", "/v1/query", json_body=body)
        return QueryResult.from_response(payload, request_id=request_id, elapsed_s=elapsed)

    @staticmethod
    def build_query_body(
        query: str,
        top_k: int = 5,
        template: Optional[str] = None,
        template_vars: Optional[Dict[str, str]] = None,
        collection: Optional[str] = None,
        collections: Optional[Sequence[str]] = None,
        use_graph: bool = False,
        retrieval_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The exact JSON payload /v1/query will receive.

        Separated from the call itself so `--dry-run` and the UI's "View request"
        panel can show the real body without issuing a request.
        """
        body: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "use_graph": use_graph,
            "retrieval_mode": retrieval_mode,
        }
        if template:
            body["template"] = template
        if template_vars:
            body["template_vars"] = template_vars
        _apply_collections(body, collection, collections)
        if filters:
            body["filters"] = filters
        return body


def _extract_detail(response: httpx.Response) -> Optional[str]:
    """Pull the server's `detail` message out of an error body."""
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:300] if text else None

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return str(detail)[:300]
    return None
