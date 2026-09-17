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


# --- prompt lab plumbing ------------------------------------------------------

def test_query_body_accepts_the_prompt_lab_fields():
    """The UI cannot reach the gateway without llm_model, and cannot run an A/B
    without system_prompt. Both were missing from QueryBody until E7b."""
    body = server.QueryBody(
        organism="M. tuberculosis", llm="argo", llm_model="gpt56sol",
        system_prompt="You are a careful curator.", quote_gate=True,
    )
    assert body.llm_model == "gpt56sol"
    assert body.quote_gate is True
    assert body.system_prompt.startswith("You are")


def test_query_body_defaults_keep_the_original_behaviour():
    body = server.QueryBody(organism="M. tuberculosis")
    assert body.llm_model is None
    assert body.system_prompt is None
    assert body.quote_gate is False


def test_spec_carries_the_system_prompt_through():
    body = server.QueryBody(organism="M. tb", system_prompt="be careful")
    assert server._spec(body).system_prompt == "be careful"


def test_two_system_prompts_give_two_identities():
    """Resume must not hand one variant's rows back for the other."""
    a = server._spec(server.QueryBody(organism="M. tb", system_prompt="prompt one"))
    b = server._spec(server.QueryBody(organism="M. tb", system_prompt="prompt two"))
    plain = server._spec(server.QueryBody(organism="M. tb"))
    assert len({a.identity(), b.identity(), plain.identity()}) == 3


def test_argo_without_a_model_is_rejected_before_the_query_runs():
    """Failing here beats failing inside a worker pool half way through a batch."""
    with pytest.raises(Exception) as excinfo:
        server._endpoint(server.QueryBody(organism="M. tb", llm="argo"))
    assert "model" in str(excinfo.value).lower()


# --- CLI: eval subcommand and batch pre-flight --------------------------------

def test_eval_rejects_a_missing_envelope(tmp_path):
    from typer.testing import CliRunner
    from litrag.cli import app

    result = CliRunner().invoke(app, ["eval", str(tmp_path / "nope.json")])
    assert result.exit_code == 2
    assert "no envelope" in result.output.lower() or result.exception is not None


def test_eval_runs_the_scorer_it_is_given(tmp_path):
    """The scorer is a separate, stdlib-only codebase. Shelling out keeps the
    two projects uncoupled; importing it would make one depend on the other."""
    from typer.testing import CliRunner
    from litrag.cli import app

    envelope = tmp_path / "run.json"
    envelope.write_text("{}")
    scorer = tmp_path / "fake_evaluate.py"
    scorer.write_text("import sys; print('scored', sys.argv[1])")

    result = CliRunner().invoke(
        app, ["eval", str(envelope), "--scorer", str(scorer)])
    assert result.exit_code == 0
    assert "scored" in result.output


def test_eval_says_so_when_no_scorer_can_be_found(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from litrag.cli import app

    envelope = tmp_path / "run.json"
    envelope.write_text("{}")
    # An isolated cwd with no experiment-01 anywhere above it.
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["eval", str(envelope)])
    assert result.exit_code == 2
    assert "--scorer" in result.output or "evaluate.py" in result.output
