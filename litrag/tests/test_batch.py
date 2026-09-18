"""Batch input parsing, concurrency, and resume."""

import json

import httpx
import pytest

from litrag.batch import BatchDefaults, load_progress, load_specs, run_batch
from litrag.client import RagStackClient
from litrag.config import Config


def make_client(payload, counter=None):
    def handler(request):
        if counter is not None:
            counter.append(request)
        return httpx.Response(200, json=payload, headers={"x-request-id": "r1"})
    return RagStackClient(
        Config(api_key="k", base_url="https://example.invalid"),
        client=httpx.Client(transport=httpx.MockTransport(handler),
                            base_url="https://example.invalid"),
    )


def test_load_tsv(tmp_path):
    path = tmp_path / "q.tsv"
    path.write_text(
        "organism\tgenes\tdata_type\n"
        "SARS-CoV-2\tSpike,NSP13\tppi\n"
        "Mycobacterium tuberculosis\tkatG\tmutation\n",
        encoding="utf-8",
    )
    specs = load_specs(path)
    assert [s.organism for s in specs] == ["SARS-CoV-2", "Mycobacterium tuberculosis"]
    assert specs[0].gene_list == ["Spike", "NSP13"]


def test_load_csv_with_aliased_headers(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text("Species,Gene,Type\nE. coli,lacZ,protein-function\n", encoding="utf-8")
    specs = load_specs(path)
    assert specs[0].organism == "E. coli"
    assert specs[0].genes == "lacZ"
    assert specs[0].data_type == "protein-function"


def test_load_yaml(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "queries:\n  - organism: SARS-CoV-2\n    genes: Spike\n    data_type: ppi\n",
        encoding="utf-8",
    )
    assert load_specs(path)[0].organism == "SARS-CoV-2"


def test_load_json(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"organism": "E. coli", "genes": "lacZ"}]), encoding="utf-8")
    assert load_specs(path)[0].genes == "lacZ"


def test_defaults_fill_missing_columns(tmp_path):
    path = tmp_path / "q.tsv"
    path.write_text("organism\nE. coli\n", encoding="utf-8")
    specs = load_specs(path, BatchDefaults(data_type="mutation", top_k=15,
                                           collection="open-access"))
    assert specs[0].data_type == "mutation"
    assert specs[0].top_k == 15
    assert specs[0].collection == "open-access"


def test_missing_organism_is_an_error(tmp_path):
    path = tmp_path / "q.tsv"
    path.write_text("genes\nkatG\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no organism"):
        load_specs(path)


def test_batch_runs_every_spec(registry, mutation_response, tmp_path):
    calls = []
    client = make_client(mutation_response, calls)
    specs = load_specs(_write_specs(tmp_path, 3))
    outcome = run_batch(client, registry, specs, concurrency=2)
    assert len(outcome.runs) == 3 and not outcome.failures
    assert len(calls) == 3


def test_progress_file_enables_resume(registry, mutation_response, tmp_path):
    progress = tmp_path / "p.jsonl"
    specs = load_specs(_write_specs(tmp_path, 3))

    first = run_batch(make_client(mutation_response), registry, specs[:2],
                      concurrency=1, progress_path=progress)
    assert len(first.runs) == 2
    assert len(load_progress(progress)) == 2

    calls = []
    second = run_batch(make_client(mutation_response, calls), registry, specs,
                       concurrency=1, progress_path=progress, resume=True)
    assert second.skipped == 2
    assert len(second.runs) == 1
    assert len(calls) == 1, "completed queries must not be re-run"


def test_failure_is_recorded_without_stopping_the_run(registry, mutation_response, tmp_path):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] == 2:
            return httpx.Response(400, json={"detail": "bad request"})
        return httpx.Response(200, json=mutation_response)

    client = RagStackClient(
        Config(api_key="k", base_url="https://example.invalid"),
        client=httpx.Client(transport=httpx.MockTransport(handler),
                            base_url="https://example.invalid"),
    )
    specs = load_specs(_write_specs(tmp_path, 3))
    outcome = run_batch(client, registry, specs, concurrency=1)
    assert len(outcome.runs) == 2 and len(outcome.failures) == 1
    assert outcome.n_total == 3


def _write_specs(tmp_path, count):
    path = tmp_path / "specs.tsv"
    lines = ["organism\tgenes\tdata_type"]
    for index in range(count):
        lines.append(f"Organism{index}\tgene{index}\tmutation")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_resume_restores_rows_not_just_counts(registry, mutation_response, tmp_path):
    """A fully-resumed run must rebuild the whole table.

    Regression: the sidecar once held only summaries, so resuming a finished
    batch produced zero rows and overwrote the completed output with an empty
    file -- silent loss of the entire run.
    """
    progress = tmp_path / "p.jsonl"
    specs = load_specs(_write_specs(tmp_path, 3))

    first = run_batch(make_client(mutation_response), registry, specs,
                      concurrency=1, progress_path=progress)
    expected = sum(len(run.rows) for run in first.runs)
    assert expected > 0

    second = run_batch(make_client(mutation_response), registry, specs,
                       concurrency=1, progress_path=progress, resume=True)
    assert second.skipped == 3 and not second.runs
    assert len(second.restored_rows) == expected
    assert second.restored_columns == registry.resolve("mutation").columns
    assert second.restored_template_id == "mutation"


def test_restored_rows_keep_citations_and_provenance(registry, mutation_response, tmp_path):
    progress = tmp_path / "p.jsonl"
    specs = load_specs(_write_specs(tmp_path, 1))
    run_batch(make_client(mutation_response), registry, specs,
              concurrency=1, progress_path=progress)

    resumed = run_batch(make_client(mutation_response), registry, specs,
                        concurrency=1, progress_path=progress, resume=True)
    row = resumed.restored_rows[0]
    assert row.provenance["_template_hash"] == "84cd8501d39dfaa4"
    assert row.citations and row.citations[0].pmid
    assert row.provenance["_retrieved_at"].endswith("Z")


def test_partial_resume_runs_only_missing_queries(registry, mutation_response, tmp_path):
    progress = tmp_path / "p.jsonl"
    specs = load_specs(_write_specs(tmp_path, 4))

    run_batch(make_client(mutation_response), registry, specs[:2],
              concurrency=1, progress_path=progress)

    calls = []
    resumed = run_batch(make_client(mutation_response, calls), registry, specs,
                        concurrency=1, progress_path=progress, resume=True)
    assert len(calls) == 2, "only the unfinished queries should be requested"
    assert resumed.skipped == 2 and len(resumed.runs) == 2
    assert resumed.restored_rows


def test_instructions_apply_to_every_query_in_the_batch(tmp_path):
    """One curation job asks one question, so the guidance is run-wide."""
    path = tmp_path / "q.tsv"
    path.write_text("organism\tgenes\nH5N1\tHA\nH5N1\tNA\n", encoding="utf-8")
    defaults = BatchDefaults(instructions="Confirmed findings only.")
    specs = load_specs(path, defaults)
    assert len(specs) == 2
    assert all(s.instructions == "Confirmed findings only." for s in specs)


def test_batch_instructions_invalidate_a_resume(tmp_path):
    path = tmp_path / "q.tsv"
    path.write_text("organism\tgenes\nH5N1\tHA\n", encoding="utf-8")
    plain = load_specs(path)[0]
    guided = load_specs(path, BatchDefaults(instructions="Confirmed findings only."))[0]
    assert plain.identity() != guided.identity()
