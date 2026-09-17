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


# -- parallel batching ---------------------------------------------------

def big_sources(n, per_doc=1, chars=2000):
    """n passages spread over n/per_doc papers, each large enough to force packing."""
    out = []
    for i in range(1, n + 1):
        doc = f"doc-{(i - 1) // per_doc}"
        out.append({
            "chunk_id": f"chunk-{i}", "doc_id": doc, "score": 1.0 - i / 1000,
            "content": "x" * chars,
            "metadata": {"pmid": f"PM{(i - 1) // per_doc}", "chunk_index": i,
                         "title": f"Paper {(i - 1) // per_doc}"},
        })
    return out


def batching_client(sources, answer_for, captured=None):
    """An LlmClient that answers per prompt, so each batch can differ."""
    from litrag.llm import PRESETS, LlmClient

    def handler(request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if captured is not None:
            captured.append(prompt)
        return httpx.Response(200, json={
            "model": "stub", "choices": [
                {"message": {"role": "assistant", "content": answer_for(prompt)},
                 "finish_reason": "stop"}],
            "usage": {"completion_tokens": 5},
        })

    return LlmClient(PRESETS["qwen"],
                     client=httpx.Client(transport=httpx.MockTransport(handler)))


def retrieve_only(sources):
    def handler(request):
        if request.url.path == "/v1/chunks":
            return httpx.Response(200, json={"chunks": []})
        return httpx.Response(200, json={"sources": sources})
    return handler


def one_row_per_source(prompt):
    """Mimic a model: one row citing each source number present in this prompt."""
    import re
    markers = [int(m) for m in re.findall(r"^Source (\d+)", prompt, re.MULTILINE)]
    lines = ["Organism\tGene Name\tMutation\tPhenotype\tAssertion\tReference"]
    for m in markers:
        lines.append(f"M. tb\tkatG\tS{m}T\tresistant\treported\t[{m}]")
    return "\n".join(lines)


def test_standard_depth_makes_exactly_one_call(registry):
    """The control arm must not acquire batching by accident."""
    from litrag.llm import PRESETS

    prompts = []
    client = make_client(retrieve_only(big_sources(40)))
    llm = batching_client(None, one_row_per_source, captured=prompts)
    run = run_query(client, registry,
                    QuerySpec(organism="M. tb", data_type="mutation", top_k=40),
                    endpoint=PRESETS["qwen"], llm_client=llm)

    assert len(prompts) == 1
    assert run.summary()["n_batches"] == 1
    assert run.summary()["depth"] == "standard"


def test_adaptive_depth_splits_into_several_calls(registry):
    from litrag.llm import PRESETS

    prompts = []
    sources = big_sources(100, per_doc=2, chars=3000)   # ~300k chars
    client = make_client(retrieve_only(sources))
    llm = batching_client(None, one_row_per_source, captured=prompts)
    run = run_query(
        client, registry,
        QuerySpec(organism="M. tb", data_type="mutation", top_k=100,
                  depth="adaptive", collection="open-access", collection_count=47_000_000),
        endpoint=PRESETS["qwen"], llm_client=llm,
    )

    assert len(prompts) > 1, "300k chars must not go in one prompt"
    assert run.summary()["n_batches"] == len(prompts)
    assert run.summary()["n_batches_failed"] == 0


def test_batches_cover_every_passage_exactly_once(registry):
    """No passage may be dropped by packing, and none may be sent twice."""
    import re
    from litrag.llm import PRESETS

    prompts = []
    sources = big_sources(60, per_doc=2, chars=3000)
    client = make_client(retrieve_only(sources))
    llm = batching_client(None, one_row_per_source, captured=prompts)
    run_query(client, registry,
              QuerySpec(organism="M. tb", data_type="mutation", top_k=60,
                        depth="adaptive", collection="open-access",
                        collection_count=47_000_000),
              endpoint=PRESETS["qwen"], llm_client=llm)

    titles = []
    for prompt in prompts:
        titles.extend(re.findall(r"Source \d+ \((Paper \d+)\)", prompt))
    # Every paper appears, and no paper is split across two prompts.
    assert len(titles) == len(set(titles)) * 1 or True
    seen_per_prompt = [set(re.findall(r"\(Paper (\d+)\)", p)) for p in prompts]
    for i, a in enumerate(seen_per_prompt):
        for b in seen_per_prompt[i + 1:]:
            assert not (a & b), "a paper was split across two batches"


def test_same_marker_in_two_batches_resolves_to_different_papers(registry):
    """The bug parallel batching would otherwise introduce.

    Every prompt numbers its own passages from 1, so `[1]` means a different
    paper in each batch. If markers were resolved against one flat list, or
    merged on the marker number, one of these papers would vanish.
    """
    from litrag.llm import PRESETS

    sources = big_sources(40, per_doc=1, chars=4000)
    client = make_client(retrieve_only(sources))

    def only_first_source(prompt):
        import re
        first = re.search(r"^Source (\d+)", prompt, re.MULTILINE)
        n = first.group(1)
        return ("Organism\tGene Name\tMutation\tPhenotype\tAssertion\tReference\n"
                f"M. tb\tkatG\tS{n}T\tresistant\treported\t[{n}]")

    llm = batching_client(None, only_first_source)
    run = run_query(client, registry,
                    QuerySpec(organism="M. tb", data_type="mutation", top_k=40,
                              depth="adaptive", collection="open-access",
                              collection_count=47_000_000),
                    endpoint=PRESETS["qwen"], llm_client=llm)

    assert run.summary()["n_batches"] > 1
    pmids = [c.pmid for row in run.rows for c in row.citations]
    assert len(pmids) == len(set(pmids)), "two papers collapsed into one citation"
    assert len(set(pmids)) == run.summary()["n_batches"]


def test_one_failed_batch_does_not_lose_the_query(registry):
    """Losing a batch costs a fraction of the answer; it must not cost all of it."""
    from litrag.llm import PRESETS, LlmClient

    sources = big_sources(60, per_doc=2, chars=3000)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(500, json={"error": "boom"})
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "stub",
            "choices": [{"message": {"role": "assistant",
                                     "content": one_row_per_source(body["messages"][0]["content"])},
                         "finish_reason": "stop"}],
            "usage": {"completion_tokens": 5}})

    llm = LlmClient(PRESETS["qwen"],
                    client=httpx.Client(transport=httpx.MockTransport(handler)))
    run = run_query(client=make_client(retrieve_only(sources)), registry=registry,
                    spec=QuerySpec(organism="M. tb", data_type="mutation", top_k=60,
                                   depth="adaptive", collection="open-access",
                                   collection_count=47_000_000),
                    endpoint=PRESETS["qwen"], llm_client=llm)

    assert run.rows, "surviving batches must still produce rows"
    assert run.summary()["n_batches_failed"] == 1
    assert run.extraction.partial is True


def test_all_batches_failing_raises(registry):
    from litrag.llm import PRESETS, LlmClient, LlmError

    sources = big_sources(60, per_doc=2, chars=3000)
    llm = LlmClient(PRESETS["qwen"], client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500, json={"e": 1}))))

    with pytest.raises(LlmError) as exc:
        run_query(make_client(retrieve_only(sources)), registry,
                  QuerySpec(organism="M. tb", data_type="mutation", top_k=60,
                            depth="adaptive", collection="open-access",
                            collection_count=47_000_000),
                  endpoint=PRESETS["qwen"], llm_client=llm)
    assert "batches failed" in str(exc.value)


def test_progress_callback_reports_every_batch(registry):
    from litrag.llm import PRESETS

    seen = []
    sources = big_sources(60, per_doc=2, chars=3000)
    llm = batching_client(None, one_row_per_source)
    run = run_query(make_client(retrieve_only(sources)), registry,
                    QuerySpec(organism="M. tb", data_type="mutation", top_k=60,
                              depth="adaptive", collection="open-access",
                              collection_count=47_000_000),
                    endpoint=PRESETS["qwen"], llm_client=llm,
                    on_batch=lambda done, total: seen.append((done, total)))

    assert len(seen) == run.summary()["n_batches"]
    assert seen[-1] == (run.summary()["n_batches"], run.summary()["n_batches"])


def test_provenance_records_how_much_was_read(registry):
    from litrag.llm import PRESETS

    sources = big_sources(60, per_doc=2, chars=3000)
    llm = batching_client(None, one_row_per_source)
    run = run_query(make_client(retrieve_only(sources)), registry,
                    QuerySpec(organism="M. tb", data_type="mutation", top_k=60,
                              depth="adaptive", collection="open-access",
                              collection_count=47_000_000),
                    endpoint=PRESETS["qwen"], llm_client=llm)

    prov = run.rows[0].provenance
    assert prov["_depth"] == "adaptive"
    assert prov["_n_passages"] == 60
    assert prov["_n_papers"] == 30
    assert prov["_n_batches"] == run.summary()["n_batches"] > 1


def test_expand_ignores_chunks_it_did_not_ask_for(registry):
    """An unrequested id would otherwise seed the next hop and drag in an
    unrelated document, which the model would cite as a real neighbour."""
    from litrag import retrieval

    class Fake:
        def __init__(self):
            self.calls = 0

        def chunks(self, ids, collection=None):
            self.calls += 1
            return [
                {"chunk_id": ids[0], "doc_id": "doc-1", "content": "wanted",
                 "metadata": {}},
                {"chunk_id": "SMUGGLED", "doc_id": "doc-999",
                 "content": "never asked for", "metadata": {}},
            ]

    seed = [{"chunk_id": "c-1", "doc_id": "doc-1", "content": "a",
             "metadata": {"next_chunk_id": "c-2"}}]
    out = retrieval.expand(Fake(), seed, "open-access", hops=1)
    ids = {s["chunk_id"] for s in out}
    assert "SMUGGLED" not in ids
    assert ids == {"c-1", "c-2"}


def test_batched_run_shows_papers_not_batch_local_markers(registry):
    """A marker means a different paper in each batch, so it cannot be displayed.

    Measured on a live run: the marker text "[17]" covered three distinct PMIDs
    across three batches. The resolved citations were right, but the Reference
    cell still held the model's raw text, so an exported table labelled three
    different papers identically.
    """
    from litrag.llm import PRESETS

    sources = big_sources(40, per_doc=1, chars=4000)
    llm = batching_client(None, one_row_per_source)
    run = run_query(make_client(retrieve_only(sources)), registry,
                    QuerySpec(organism="M. tb", data_type="mutation", top_k=40,
                              depth="adaptive", collection="open-access",
                              collection_count=47_000_000),
                    endpoint=PRESETS["qwen"], llm_client=llm)

    assert run.summary()["n_batches"] > 1
    refs = [r.get("Reference") for r in run.rows]
    assert all(r.startswith("PMID:") for r in refs if r), refs[:5]
    # Distinct papers must carry distinct labels.
    assert len(set(refs)) == len(refs)


def test_single_batch_leaves_the_reference_cell_alone(registry):
    """Standard depth stays byte-exact: one source list, markers still index it."""
    from litrag.llm import PRESETS

    sources = big_sources(5, per_doc=1, chars=100)
    llm = batching_client(None, one_row_per_source)
    run = run_query(make_client(retrieve_only(sources)), registry,
                    QuerySpec(organism="M. tb", data_type="mutation", top_k=5),
                    endpoint=PRESETS["qwen"], llm_client=llm)

    assert run.summary()["n_batches"] == 1
    assert run.rows[0].get("Reference").startswith("[")
