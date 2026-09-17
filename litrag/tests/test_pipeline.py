"""End-to-end pipeline against a mocked transport (no network)."""

import json

import httpx
import pytest

from litrag.client import ApiError, RagStackClient
from litrag.config import Config
from litrag.pipeline import QuerySpec, build_request, merge_runs, run_query


def make_client(handler):
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="https://example.invalid")
    return RagStackClient(Config(api_key="k", base_url="https://example.invalid"), client=http)


def responder(payload, status=200, captured=None):
    def handler(request):
        if captured is not None:
            captured.append(json.loads(request.content) if request.content else {})
        return httpx.Response(status, json=payload, headers={"x-request-id": "req-abc"})
    return handler


def test_run_query_produces_deduped_rows(registry, mutation_response):
    client = make_client(responder(mutation_response))
    spec = QuerySpec(organism="Mycobacterium tuberculosis", genes="katG",
                     data_type="mutation", top_k=6)
    run = run_query(client, registry, spec)

    assert run.template.id == "mutation"
    assert len(run.rows) < run.extraction.n_rows
    assert run.summary()["request_id"] == "req-abc"


def test_provenance_is_stamped_on_every_row(registry, mutation_response):
    client = make_client(responder(mutation_response))
    run = run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"))
    for row in run.rows:
        prov = row.provenance
        assert prov["_template"] == "mutation"
        assert prov["_template_version"] == 2
        assert prov["_template_hash"] == "84cd8501d39dfaa4"
        assert prov["_model"].startswith("RedHatAI/")
        assert prov["_retrieved_at"].endswith("Z")
        assert prov["_request_id"] == "req-abc"


def test_request_body_shape(registry):
    """The graph backend reports itself disabled, so we never request it."""
    spec = QuerySpec(organism="SARS-CoV-2", genes="Spike", other_terms="entry",
                     data_type="ppi", top_k=7, collection="open-access")
    body = build_request(spec, registry.resolve("ppi"))
    assert body["template"] == "ppi-extraction"
    assert body["template_vars"] == {
        "organism": "SARS-CoV-2", "genes": "Spike", "other_terms": "entry",
    }
    assert body["top_k"] == 7
    assert body["collection"] == "open-access"
    assert body["use_graph"] is False
    assert "Protein-Protein Interaction" in body["query"]


def test_validation_happens_before_any_call(registry):
    """A missing organism must not cost a request."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"answer": "", "sources": [], "rewritten_queries": []})

    client = make_client(handler)
    with pytest.raises(Exception):
        run_query(client, registry, QuerySpec(organism="", data_type="mutation"))
    assert calls == []


def test_server_error_is_surfaced_with_request_id(registry):
    def handler(request):
        return httpx.Response(
            404, json={"detail": "unknown prompt template: 'nope'"},
            headers={"x-request-id": "req-xyz"},
        )

    client = make_client(handler)
    with pytest.raises(ApiError) as excinfo:
        run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"))
    assert "unknown prompt template" in str(excinfo.value)
    assert "req-xyz" in str(excinfo.value)


def test_query_id_is_stable_and_distinguishing():
    a = QuerySpec(organism="M. tb", genes="katG", data_type="mutation")
    b = QuerySpec(organism="M. tb", genes="katG", data_type="mutation")
    c = QuerySpec(organism="M. tb", genes="inhA", data_type="mutation")
    assert a.identity() == b.identity()
    assert a.identity() != c.identity()


def test_merge_across_runs_accumulates_support(registry, mutation_response):
    client = make_client(responder(mutation_response))
    runs = [
        run_query(client, registry, QuerySpec(organism="M. tb", genes=g, data_type="mutation"))
        for g in ("katG", "inhA")
    ]
    merged, columns = merge_runs(runs)
    assert columns == registry.resolve("mutation").columns
    assert any(row.n_support > 1 for row in merged)
    assert all(row.query_ids for row in merged)


def test_no_dedupe_preserves_every_row(registry, mutation_response):
    client = make_client(responder(mutation_response))
    spec = QuerySpec(organism="M. tb", data_type="mutation", no_dedupe=True)
    run = run_query(client, registry, spec)
    assert len(run.rows) == run.extraction.n_rows


def test_retry_then_success(registry, mutation_response, monkeypatch):
    monkeypatch.setattr("litrag.client.time.sleep", lambda _s: None)
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(200, json=mutation_response)

    client = make_client(handler)
    run = run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"))
    assert attempts["n"] == 2 and run.rows


# -- local generation ----------------------------------------------------

def local_client(answer, sources, captured=None, model="Qwen/Qwen3.6-35B-A3B"):
    """An LlmClient whose transport returns a fixed completion."""
    from litrag.llm import PRESETS, LlmClient

    def handler(request):
        if captured is not None:
            captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10},
        })

    return LlmClient(PRESETS["qwen"],
                     client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_local_path_retrieves_then_generates(registry, mutation_response):
    """Model choice requires splitting retrieval from generation, because the
    hosted endpoint's `llm` field returns an empty answer with no sources."""
    from litrag.llm import PRESETS

    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"sources": mutation_response["sources"]})

    client = make_client(handler)
    llm = local_client(mutation_response["answer"], mutation_response["sources"])
    run = run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"),
                    endpoint=PRESETS["qwen"], llm_client=llm)

    assert paths == ["/v1/retrieve"], "must not call /v1/query on the local path"
    assert run.rows
    assert run.result.generator == "local"
    assert run.result.model == "Qwen/Qwen3.6-35B-A3B"


def test_local_provenance_records_prompt_hash(registry, mutation_response):
    """LitRAG owns the local prompt, so its hash replaces the server's as the
    reproducibility anchor."""
    from litrag.llm import PRESETS

    client = make_client(responder({"sources": mutation_response["sources"]}))
    llm = local_client(mutation_response["answer"], mutation_response["sources"])
    run = run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"),
                    endpoint=PRESETS["qwen"], llm_client=llm)

    prov = run.rows[0].provenance
    assert prov["_generator"] == "local"
    assert prov["_llm_endpoint"] == "http://mango.cels.anl.gov:8004/v1"
    assert len(prov["_prompt_hash"]) == 16
    # The declaration still defined the columns, so it stays recorded.
    assert prov["_template"] == "mutation" and prov["_template_version"] == 2


def test_server_path_has_no_prompt_hash(registry, mutation_response):
    client = make_client(responder(mutation_response))
    run = run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"))
    assert run.rows[0].provenance["_generator"] == "server"
    assert run.rows[0].provenance["_prompt_hash"] == ""


def test_backend_is_part_of_query_identity():
    """Resuming after switching models must re-run, not reuse the old answer."""
    server = QuerySpec(organism="M. tb", data_type="mutation", backend="server")
    qwen = QuerySpec(organism="M. tb", data_type="mutation", backend="qwen")
    llama = QuerySpec(organism="M. tb", data_type="mutation", backend="llama")
    assert len({server.identity(), qwen.identity(), llama.identity()}) == 3


def test_local_generation_failure_propagates(registry, mutation_response):
    from litrag.llm import PRESETS, LlmClient, LlmError

    client = make_client(responder({"sources": mutation_response["sources"]}))
    llm = LlmClient(PRESETS["qwen"], client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down"))))

    with pytest.raises(LlmError):
        run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"),
                  endpoint=PRESETS["qwen"], llm_client=llm)


def test_local_max_tokens_follows_the_template(registry, mutation_response):
    """The declaration caps table output at 2500 tokens; honour it."""
    from litrag.llm import PRESETS

    captured = []
    client = make_client(responder({"sources": mutation_response["sources"]}))
    llm = local_client(mutation_response["answer"], mutation_response["sources"], captured)
    run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"),
              endpoint=PRESETS["qwen"], llm_client=llm)
    assert captured[0]["max_tokens"] == registry.resolve("mutation").max_output_tokens


# -- multi-collection ----------------------------------------------------

def test_single_collection_sends_the_scalar_field(registry):
    """One corpus must use `collection`, keeping the response byte-identical to
    a plain single-corpus request."""
    body = build_request(
        QuerySpec(organism="M. tb", data_type="mutation", collections=["open-access"]),
        registry.resolve("mutation"))
    assert body["collection"] == "open-access"
    assert "collections" not in body


def test_several_collections_send_the_array(registry):
    body = build_request(
        QuerySpec(organism="M. tb", data_type="mutation",
                  collections=["open-access", "asm-semantic"]),
        registry.resolve("mutation"))
    assert body["collections"] == ["open-access", "asm-semantic"]
    assert "collection" not in body


def test_collection_choice_is_part_of_query_identity():
    """Searching a different corpus is a different query, so a resume must
    re-run rather than hand back the previous corpus's answer."""
    pmc = QuerySpec(organism="M. tb", data_type="mutation", collections=["open-access"])
    both = QuerySpec(organism="M. tb", data_type="mutation",
                     collections=["open-access", "asm-semantic"])
    assert pmc.identity() != both.identity()


def test_provenance_records_every_corpus_searched(registry, mutation_response):
    client = make_client(responder(mutation_response))
    spec = QuerySpec(organism="M. tb", data_type="mutation",
                     collections=["open-access", "asm-semantic"])
    run = run_query(client, registry, spec)
    assert run.rows[0].provenance["_collection"] == "open-access+asm-semantic"
    assert run.summary()["collections"] == "open-access+asm-semantic"


# -- context and output budgeting ----------------------------------------

def test_output_scales_with_the_number_of_sources(registry):
    """The template's cap is sized for the hosted path's own retrieval.

    On the local path we choose top_k, so at 100 sources that cap cut the table
    off mid-row -- observed live as truncated=True with rows silently missing.
    """
    from litrag.pipeline import MAX_OUTPUT_TOKENS, plan_output_tokens
    template = registry.resolve("mutation")
    declared = template.max_output_tokens

    assert plan_output_tokens(template, 10) == declared
    assert plan_output_tokens(template, 100) > declared
    assert plan_output_tokens(template, 100) <= MAX_OUTPUT_TOKENS
    # Monotonic: more sources never asks for less room.
    sizes = [plan_output_tokens(template, n) for n in (1, 10, 25, 50, 100)]
    assert sizes == sorted(sizes)


def test_output_budget_is_capped(registry):
    from litrag.pipeline import MAX_OUTPUT_TOKENS, plan_output_tokens
    assert plan_output_tokens(registry.resolve("mutation"), 10_000) == MAX_OUTPUT_TOKENS


def test_prose_templates_keep_their_declared_cap(registry):
    from litrag.pipeline import plan_output_tokens
    template = registry.resolve("literature-summary")
    assert plan_output_tokens(template, 100) == (template.max_output_tokens or 2500)


def test_prompt_is_bounded_by_the_model_context(registry, mutation_response):
    """Llama's window is half Qwen's, so the same top_k must truncate context
    for one and not the other rather than overflowing."""
    from litrag.llm import PRESETS

    sources = mutation_response["sources"] * 40  # a deliberately huge retrieval
    captured = {}

    def gen(request):
        captured["prompt"] = json.loads(request.content)["messages"][0]["content"]
        captured["max_tokens"] = json.loads(request.content)["max_tokens"]
        return httpx.Response(200, json={
            "model": "m", "choices": [{"message": {"content": ""},
                                       "finish_reason": "stop"}], "usage": {}})

    from litrag.llm import LlmClient
    client = make_client(responder({"sources": sources}))
    llm = LlmClient(PRESETS["llama"], client=httpx.Client(transport=httpx.MockTransport(gen)))
    run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation", top_k=100),
              endpoint=PRESETS["llama"], llm_client=llm)

    estimated = len(captured["prompt"]) / 3.5
    assert estimated + captured["max_tokens"] < PRESETS["llama"].context_tokens


def test_truncation_is_surfaced_in_the_summary(registry, mutation_response):
    """A cut-off table is missing rows and must not look like a full result."""
    from litrag.llm import PRESETS, LlmClient

    def gen(request):
        return httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {"content": mutation_response["answer"]},
                         "finish_reason": "length"}],
            "usage": {}})

    client = make_client(responder({"sources": mutation_response["sources"]}))
    llm = LlmClient(PRESETS["qwen"], client=httpx.Client(transport=httpx.MockTransport(gen)))
    run = run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"),
                    endpoint=PRESETS["qwen"], llm_client=llm)
    assert run.summary()["truncated"] is True


def test_complete_answer_is_not_marked_truncated(registry, mutation_response):
    client = make_client(responder(mutation_response))
    run = run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"))
    assert run.summary()["truncated"] is False


def test_dropped_sources_are_reported(registry, mutation_response):
    """A smaller context window silently drops the tail of a large retrieval.

    The caller asked for N sources; if the model only saw fewer, say so.
    """
    from litrag.llm import PRESETS, LlmClient

    sources = mutation_response["sources"] * 40
    def gen(request):
        return httpx.Response(200, json={
            "model": "m", "choices": [{"message": {"content": mutation_response["answer"]},
                                       "finish_reason": "stop"}], "usage": {}})

    client = make_client(responder({"sources": sources}))
    llm = LlmClient(PRESETS["llama"], client=httpx.Client(transport=httpx.MockTransport(gen)))
    run = run_query(client, registry,
                    QuerySpec(organism="M. tb", data_type="mutation", top_k=100),
                    endpoint=PRESETS["llama"], llm_client=llm)
    summary = run.summary()
    assert summary["sources_dropped"] > 0
    assert summary["n_sources"] < len(sources)
    # Sources kept must be exactly those the model saw, so a citation marker
    # can never point at a passage that was cut.
    assert len(run.result.sources) == summary["n_sources"]


def test_nothing_dropped_when_everything_fits(registry, mutation_response):
    client = make_client(responder(mutation_response))
    run = run_query(client, registry,
                    QuerySpec(organism="M. tb", data_type="mutation", top_k=6))
    assert run.summary()["sources_dropped"] == 0
