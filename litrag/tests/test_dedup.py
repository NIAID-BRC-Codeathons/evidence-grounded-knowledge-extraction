"""Cross-query deduplication."""

from litrag.dedup import dedupe, identity_key
from litrag.extract import Row, extract


def test_katg_notations_collapse_to_one_row(registry, mutation_response):
    """The headline case: five spellings of one mutation in one live response."""
    template = registry.resolve("mutation")
    result = extract(
        mutation_response["answer"], template, mutation_response["sources"],
        requested_genes=["katG"],
    )
    merged = dedupe(result.rows, template.id, result.columns)

    s315t = [r for r in merged if r.get("Gene Name") == "katG"
             and r.get("Mutation") in {"S315T", "Ser315Thr", "katG-S315T"}]
    assert len(s315t) == 1, "S315T / Ser315Thr / katG-S315T must merge"
    assert s315t[0].n_support >= 3
    # Support from several distinct papers, not one paper counted repeatedly.
    assert len(set(s315t[0].pmids)) >= 3


def test_dedup_reduces_row_count(registry, mutation_response):
    template = registry.resolve("mutation")
    result = extract(mutation_response["answer"], template, mutation_response["sources"])
    merged = dedupe(result.rows, template.id, result.columns)
    assert len(merged) < result.n_rows


def test_distinct_facts_are_not_merged(registry, mutation_response):
    template = registry.resolve("mutation")
    result = extract(mutation_response["answer"], template, mutation_response["sources"])
    merged = dedupe(result.rows, template.id, result.columns)
    inha = {r.get("Mutation") for r in merged if r.get("Gene Name") == "inhA"}
    assert len(inha) >= 2, "c-15t and Ser94Ala are different mutations"


def test_protein_interaction_is_symmetric(registry):
    """A-B and B-A are the same interaction."""
    template = registry.resolve("ppi")
    columns = template.columns
    forward = Row(values={"Pathogen": "SARS-CoV-2", "Protein A": "Spike", "Protein B": "ACE2"})
    reverse = Row(values={"Pathogen": "SARS-CoV-2", "Protein A": "ACE2", "Protein B": "Spike"})
    assert identity_key(forward, template.id, columns) == identity_key(reverse, template.id, columns)
    assert len(dedupe([forward, reverse], template.id, columns)) == 1


def test_mutation_identity_is_not_symmetric(registry):
    """Symmetry is a property of interactions, not of every table."""
    template = registry.resolve("mutation")
    columns = template.columns
    a = Row(values={"Organism": "M. tb", "Gene Name": "katG", "Mutation": "S315T"})
    b = Row(values={"Organism": "M. tb", "Gene Name": "S315T", "Mutation": "katG"})
    assert identity_key(a, template.id, columns) != identity_key(b, template.id, columns)


def test_merging_records_disagreement(registry):
    """Merging must not present one paper's qualifier as everyone's finding."""
    template = registry.resolve("mutation")
    columns = template.columns
    plain = Row(values={"Organism": "M. tb", "Gene Name": "katG",
                        "Mutation": "S315T", "Phenotype": "isoniazid resistance"})
    qualified = Row(values={"Organism": "M. tb", "Gene Name": "katG",
                            "Mutation": "Ser315Thr",
                            "Phenotype": "moderate-level isoniazid resistance"})
    merged = dedupe([plain, qualified], template.id, columns)
    assert len(merged) == 1
    assert "merged_variants:Phenotype" in merged[0].flags
    assert len(merged[0].variants["Phenotype"]) == 2


def test_reference_differences_are_not_treated_as_disagreement(registry):
    """Different papers reporting one fact is the point, not a conflict."""
    template = registry.resolve("mutation")
    columns = template.columns
    rows = [
        Row(values={"Organism": "M. tb", "Gene Name": "katG",
                    "Mutation": "S315T", "Reference": "[1]"}),
        Row(values={"Organism": "M. tb", "Gene Name": "katG",
                    "Mutation": "S315T", "Reference": "[2]"}),
    ]
    merged = dedupe(rows, template.id, columns)
    assert not any(f.startswith("merged_variants") for f in merged[0].flags)


def test_support_count_accumulates(registry):
    template = registry.resolve("mutation")
    columns = template.columns
    rows = [
        Row(values={"Organism": "M. tb", "Gene Name": "katG", "Mutation": m})
        for m in ("S315T", "Ser315Thr", "p.Ser315Thr", "katG-S315T")
    ]
    merged = dedupe(rows, template.id, columns)
    assert len(merged) == 1 and merged[0].n_support == 4


def test_citations_from_different_papers_survive_the_same_marker():
    """Two papers both cited as [1] in different batches must both survive.

    Once passages are split across parallel LLM calls, each call numbers its own
    sources from 1, so `[1]` in batch A and `[1]` in batch B are different
    papers. The merge key fell back to the marker number when a source had no
    PMID/DOI/PMCID, so the second paper was silently dropped as a duplicate of
    the first -- no error, no flag, no count.

    This is not hypothetical: the Dengue and Influenza_2024_2025 collections
    carry no pmid and no pmcid at all, so their chunks land in exactly that
    fallback.
    """
    from litrag.dedup import _merge_citations
    from litrag.extract import Citation

    first = Citation(marker=1, doc_id="doc-aaa", title="Paper A")
    second = Citation(marker=1, doc_id="doc-bbb", title="Paper B")

    merged = _merge_citations([first], [second])
    assert len(merged) == 2, "two distinct papers collapsed into one"
    assert {c.doc_id for c in merged} == {"doc-aaa", "doc-bbb"}


def test_same_paper_from_two_batches_still_merges():
    """The other half: one paper reached twice must NOT become two citations."""
    from litrag.dedup import _merge_citations
    from litrag.extract import Citation

    merged = _merge_citations(
        [Citation(marker=1, doc_id="doc-aaa")],
        [Citation(marker=7, doc_id="doc-aaa")],
    )
    assert len(merged) == 1
