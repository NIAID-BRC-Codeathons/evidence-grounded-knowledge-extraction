"""End-to-end pipeline against a mocked transport (no network)."""

import json

import httpx
import pytest

from litrag.client import ApiError, RagStackClient
from litrag.config import Config
from litrag.pipeline import (QuerySpec, build_request, hosted_template_vars,
                            merge_runs, run_query)
from litrag.templates import TemplateError


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

    assert set(paths) == {"/v1/retrieve"}, "must not call /v1/query on the local path"
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


# -- fused retrieval -----------------------------------------------------

def test_fused_retrieval_queries_both_modes(registry, mutation_response):
    """Dense retrieval ranks a chunk by what it is about, so a passage that
    merely mentions an identifier sinks. Measured on the Dengue corpus: BM25
    ranked the target chunk 12th and hybrid missed it in the top 100 entirely.
    """
    from litrag.llm import PRESETS

    modes = []

    def handler(request):
        modes.append(json.loads(request.content)["retrieval_mode"])
        return httpx.Response(200, json={"sources": mutation_response["sources"]})

    client = make_client(handler)
    llm = local_client(mutation_response["answer"], mutation_response["sources"])
    run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"),
              endpoint=PRESETS["qwen"], llm_client=llm)
    assert modes == ["hybrid", "bm25"]


def test_explicit_mode_makes_one_call(registry, mutation_response):
    from litrag.llm import PRESETS

    modes = []

    def handler(request):
        modes.append(json.loads(request.content)["retrieval_mode"])
        return httpx.Response(200, json={"sources": mutation_response["sources"]})

    client = make_client(handler)
    llm = local_client(mutation_response["answer"], mutation_response["sources"])
    run_query(client, registry,
              QuerySpec(organism="M. tb", data_type="mutation", retrieval_mode="bm25"),
              endpoint=PRESETS["qwen"], llm_client=llm)
    assert modes == ["bm25"]


def test_hosted_path_never_receives_the_fused_mode(registry, mutation_response):
    """"fused" is ours; /v1/query retrieves server-side and would reject it."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=mutation_response)

    client = make_client(handler)
    run_query(client, registry, QuerySpec(organism="M. tb", data_type="mutation"))
    assert bodies[0]["retrieval_mode"] == "hybrid"


def test_fusion_keeps_a_hit_found_by_only_one_mode():
    """The whole point: a BM25-only hit must survive into the fused set."""
    from litrag.client import fuse_rankings
    dense = [{"chunk_id": f"d{i}"} for i in range(10)]
    keyword = [{"chunk_id": "d0"}, {"chunk_id": "target"}]
    fused = [s["chunk_id"] for s in fuse_rankings([dense, keyword])]
    assert "target" in fused
    assert fused.index("target") < fused.index("d9")


def test_fusion_rewards_agreement():
    from litrag.client import fuse_rankings
    a = [{"chunk_id": "x"}, {"chunk_id": "both"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "both"}]
    assert fuse_rankings([a, b])[0]["chunk_id"] == "both"


def test_fusion_is_deterministic():
    from litrag.client import fuse_rankings
    a = [{"chunk_id": "p"}, {"chunk_id": "q"}]
    b = [{"chunk_id": "q"}, {"chunk_id": "p"}]
    assert fuse_rankings([a, b]) == fuse_rankings([a, b])


# --- advanced instructions ------------------------------------------------
#
# The hosted path has no slot of its own for free-text guidance, so it rides
# inside other_terms; the local path writes its own prompt section instead.


def test_instructions_ride_in_other_terms_on_the_hosted_path(registry):
    spec = QuerySpec(organism="M. tb", genes="katG", other_terms="isoniazid",
                     data_type="mutation", instructions="Report only confirmed mutations.")
    values = hosted_template_vars(spec, registry.resolve("mutation"))
    assert values["other_terms"] == "isoniazid; Report only confirmed mutations."


def test_instructions_stand_alone_when_there_are_no_other_terms(registry):
    spec = QuerySpec(organism="M. tb", data_type="mutation",
                     instructions="Report only confirmed mutations.")
    values = hosted_template_vars(spec, registry.resolve("mutation"))
    assert values["other_terms"] == "Report only confirmed mutations."


def test_instructions_do_not_steer_retrieval(registry):
    """Guidance shapes the answer; it must not change which papers are found."""
    template = registry.resolve("mutation")
    plain = QuerySpec(organism="M. tb", genes="katG", data_type="mutation")
    guided = QuerySpec(organism="M. tb", genes="katG", data_type="mutation",
                       instructions="Report positions as A226, K128.")
    assert guided.search_text(template) == plain.search_text(template)


def test_overlong_instructions_are_rejected_before_the_call(registry):
    spec = QuerySpec(organism="M. tb", data_type="mutation", instructions="x" * 250)
    with pytest.raises(TemplateError) as excinfo:
        hosted_template_vars(spec, registry.resolve("mutation"))
    message = str(excinfo.value)
    assert "200" in message, "the message should name the slot's limit"
    assert "--llm qwen" in message, "and point at the path with no limit"


def test_the_budget_accounts_for_other_terms_already_in_the_slot(registry):
    """Other terms spend the same 200 characters, so they shrink the budget."""
    instructions = "y" * 150
    template = registry.resolve("mutation")
    assert hosted_template_vars(
        QuerySpec(organism="M. tb", data_type="mutation", instructions=instructions), template
    )["other_terms"] == instructions

    with pytest.raises(TemplateError):
        hosted_template_vars(
            QuerySpec(organism="M. tb", data_type="mutation",
                      other_terms="z" * 60, instructions=instructions), template
        )


def test_build_request_stays_safe_for_a_local_run(registry):
    """Provenance is built on both paths, so the hosted cap must not bind here."""
    spec = QuerySpec(organism="M. tb", data_type="mutation", instructions="x" * 5000)
    body = build_request(spec, registry.resolve("mutation"))
    assert "other_terms" not in body["template_vars"]


def test_run_query_sends_the_folded_instructions(registry, mutation_response):
    bodies = []
    client = make_client(responder(mutation_response, captured=bodies))
    run_query(client, registry, QuerySpec(
        organism="M. tb", data_type="mutation", instructions="Confirmed only."))
    assert bodies[0]["template_vars"]["other_terms"] == "Confirmed only."


def test_instructions_change_the_identity_hash():
    """A resume must not hand back answers written under different guidance."""
    plain = QuerySpec(organism="M. tb", genes="katG", data_type="mutation")
    guided = QuerySpec(organism="M. tb", genes="katG", data_type="mutation",
                       instructions="Confirmed only.")
    assert plain.identity() != guided.identity()
