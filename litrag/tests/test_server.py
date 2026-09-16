"""HTTP surface, with the upstream API mocked."""

import httpx
import pytest
from fastapi.testclient import TestClient

from litrag import server
from litrag.client import RagStackClient
from litrag.config import Config

# A key the server must never echo back. Using a sentinel rather than a real
# tenant prefix keeps the assertion meaningful in a public repository.
SENTINEL_KEY = "rk-sentinel-must-not-leak-0123456789"


@pytest.fixture
def client(monkeypatch, registry, mutation_response):
    collections = {
        "collections": [
            {"id": "asm-semantic", "label": "ASM papers", "count": 6718269,
             "chunk_method": "semantic", "state": "active"},
            {"id": "open-access", "label": "PMC open access", "count": 47625155,
             "chunk_method": "fixed_token", "state": "active"},
        ],
        "default": "open-access",
    }

    def handler(request):
        path = request.url.path
        if path.endswith("/v1/collections"):
            return httpx.Response(200, json=collections)
        if path.endswith("/v1/version"):
            return httpx.Response(200, json={"version": "0.1.0", "git_tag": "v1.6.2"})
        if path.endswith("/v1/query"):
            return httpx.Response(200, json=mutation_response,
                                  headers={"x-request-id": "req-1"})
        return httpx.Response(404, json={"detail": "not found"})

    def fake_client():
        return RagStackClient(
            Config(api_key=SENTINEL_KEY, base_url="https://example.invalid"),
            client=httpx.Client(transport=httpx.MockTransport(handler),
                                base_url="https://example.invalid"),
        )

    monkeypatch.setattr(server, "_client", fake_client)
    server._state["registry"] = registry
    return TestClient(server.app)


def test_index_and_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_templates_endpoint_mirrors_server(client):
    payload = client.get("/api/templates").json()
    ids = [t["id"] for t in payload["templates"]]
    assert "ppi-extraction" in ids
    ppi = next(t for t in payload["templates"] if t["id"] == "ppi-extraction")
    assert ppi["columns"][0] == "Pathogen" and ppi["version"] == 2


def test_collections_endpoint(client):
    payload = client.get("/api/collections").json()
    assert payload["default"] == "open-access"


def test_query_returns_rows_with_provenance(client):
    response = client.post("/api/query", json={
        "organism": "Mycobacterium tuberculosis", "genes": "katG",
        "data_type": "mutation", "top_k": 6, "llm": "server",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_table"] and payload["rows"]
    assert payload["columns"][0] == "Organism"
    assert payload["rows"][0]["provenance"]["_template_hash"] == "84cd8501d39dfaa4"
    assert payload["sources"][0]["pmid"]


def test_missing_organism_is_rejected(client):
    assert client.post("/api/query", json={"organism": "", "data_type": "mutation"}).status_code == 422


def test_backends_endpoint_lists_generators(client):
    payload = client.get("/api/backends").json()
    ids = [b["id"] for b in payload["backends"]]
    assert payload["default"] == "qwen"
    assert {"server", "qwen", "llama"} <= set(ids)


def test_unknown_backend_is_a_400(client):
    response = client.post("/api/query", json={
        "organism": "M. tb", "data_type": "mutation", "llm": "gpt9",
    })
    assert response.status_code == 400
    assert "unknown backend" in response.json()["detail"]


def test_local_request_preview_shows_the_prompt(client):
    """On the local path LitRAG owns the prompt, so it can honestly show it --
    the one thing the hosted path cannot do."""
    payload = client.post("/api/request", json={
        "organism": "M. tb", "genes": "katG", "data_type": "mutation", "llm": "qwen",
    }).json()
    assert payload["endpoint"].endswith("/chat/completions")
    assert "TSV" in payload["prompt"]
    assert len(payload["prompt_hash"]) == 16


def test_unknown_data_type_is_a_400(client):
    response = client.post("/api/query", json={"organism": "M. tb", "data_type": "nope"})
    assert response.status_code == 400
    assert "unknown data type" in response.json()["detail"]


def test_request_preview_exposes_body_not_prompt(client):
    """The prompt text is the server's; what we can honestly show is the request."""
    payload = client.post("/api/request", json={
        "organism": "M. tb", "genes": "katG", "data_type": "mutation", "llm": "server",
    }).json()
    assert payload["endpoint"] == "/v1/query"
    assert "prompt" not in payload
    assert payload["body"]["template"] == "mutation"
    assert payload["body"]["use_graph"] is False
    assert payload["template"]["hash"]
    assert "system" not in payload["template"] and "user" not in payload["template"]


def test_export_formats(client):
    body = {"organism": "M. tb", "genes": "katG", "data_type": "mutation", "llm": "server"}
    tsv = client.post("/api/export?fmt=tsv", json=body)
    assert tsv.status_code == 200
    assert "_template_hash" in tsv.text.splitlines()[0]
    assert "attachment" in tsv.headers["content-disposition"]
    assert client.post("/api/export?fmt=json", json=body).status_code == 200
    assert client.post("/api/export?fmt=xlsx", json=body).status_code == 400


def test_api_key_never_reaches_the_browser(client):
    """The standalone UI holds no credential; the server does."""
    body = {"organism": "M. tb", "genes": "katG", "data_type": "mutation", "llm": "server"}
    for response in (
        client.get("/api/templates"),
        client.get("/api/collections"),
        client.post("/api/query", json=body),
        client.post("/api/request", json=body),
        client.get("/"),
        client.get("/static/app.js"),
    ):
        assert SENTINEL_KEY not in response.text
        assert "X-API-Key" not in response.text
        assert "api_key" not in response.text.lower()


def test_collections_endpoint_offers_an_all_option(client):
    payload = client.get("/api/collections").json()
    ids = [c["id"] for c in payload["collections"]]
    assert payload["default"] == "open-access"
    assert "all" in ids
    everything = next(c for c in payload["collections"] if c["id"] == "all")
    assert everything["count"] >= max(
        c["count"] for c in payload["collections"] if c["id"] != "all")


def test_collections_have_readable_names(client):
    payload = client.get("/api/collections").json()
    pmc = next(c for c in payload["collections"] if c["id"] == "open-access")
    assert pmc["name"] == "PubMed Central (open access)"
    assert pmc["size_note"].endswith("chunks")


def test_unknown_collection_is_a_400(client):
    response = client.post("/api/query", json={
        "organism": "M. tb", "data_type": "mutation", "collection": "pubmed",
    })
    assert response.status_code == 400
    assert "unknown collection" in response.json()["detail"]
