"""Extraction behaviour, pinned against real recorded API responses.

Every case here reproduces something a live query actually returned.
"""

from litrag.extract import extract, parse_citations
from litrag.dedup import dedupe


def test_evidence_free_row_is_dropped(registry, ppi_response):
    """Live PPI run returned `SARS-CoV-2 NSP13 Spike N/A N/A N/A N/A`.

    That row restates the question and reports nothing, so it must not reach a
    curated table.
    """
    template = registry.resolve("ppi")
    result = extract(ppi_response["answer"], template, ppi_response["sources"])
    assert result.dropped_empty == 1
    for row in result.rows:
        assert not (row.get("Protein B") == "Spike" and row.get("Interaction Type") == "")


def test_keep_empty_retains_and_flags(registry, ppi_response):
    template = registry.resolve("ppi")
    result = extract(ppi_response["answer"], template, ppi_response["sources"], keep_empty=True)
    assert result.dropped_empty == 0
    assert any("evidence_free" in row.flags for row in result.rows)


def test_citation_markers_resolve_to_real_papers(registry, mutation_response):
    """[1] is the first retrieved source -- verified against the live response."""
    template = registry.resolve("mutation")
    result = extract(mutation_response["answer"], template, mutation_response["sources"])
    first = result.rows[0]
    assert first.citations[0].pmid == "40580943"
    assert first.citations[0].journal == "Genome Biology and Evolution"
    assert first.citations[0].url.startswith("https://doi.org/")


def test_second_marker_maps_to_second_source(registry, mutation_response):
    citations = parse_citations("[2]", mutation_response["sources"])
    assert citations[0].pmid == "29459669"


def test_repeated_document_yields_one_citation(registry, mutation_response):
    """Sources 3, 5 and 6 were three chunks of PMID 19578178."""
    citations = parse_citations("[3][5][6]", mutation_response["sources"])
    assert [c.pmid for c in citations] == ["19578178"]


def test_out_of_range_marker_is_unresolved(mutation_response):
    citations = parse_citations("[99]", mutation_response["sources"])
    assert citations and not citations[0].resolved


def test_off_target_gene_is_flagged_not_dropped(registry, mutation_response):
    """A katG query returned inhA rows -- real findings, just not what was asked."""
    template = registry.resolve("mutation")
    result = extract(
        mutation_response["answer"], template, mutation_response["sources"],
        requested_genes=["katG"],
    )
    inha = [r for r in result.rows if r.get("Gene Name") == "inhA"]
    assert inha, "inhA rows should be kept"
    assert all("off_target_gene" in r.flags for r in inha)


def test_compound_row_is_split(registry, mutation_response):
    """A live row packed two facts: 'katG, inhA' with 'Ser315Thr, c-15t'."""
    template = registry.resolve("mutation")
    answer = (
        "Organism\tGene Name\tMutation\tPhenotype\tAssertion\tReference\n"
        "M. tuberculosis\tkatG, inhA\tSer315Thr, c-15t\tINH resistance\tTrue\t[1]\n"
    )
    result = extract(answer, template, mutation_response["sources"])
    assert result.split_compound == 1
    assert {r.get("Gene Name") for r in result.rows} == {"katG", "inhA"}


def test_misaligned_compound_row_is_flagged_not_split(registry, mutation_response):
    """Two genes but three mutations cannot be paired; leave it for a human."""
    template = registry.resolve("mutation")
    answer = (
        "Organism\tGene Name\tMutation\tPhenotype\tAssertion\tReference\n"
        "M. tuberculosis\tkatG, inhA\tA1B, C2D, E3F\tINH resistance\tTrue\t[1]\n"
    )
    result = extract(answer, template, mutation_response["sources"])
    assert result.split_compound == 0
    assert "compound_row" in result.rows[0].flags


def test_prose_reference_matches_first_author_only(ppi_response):
    """"X et al." names the first author; matching deeper mis-attributes claims."""
    sources = ppi_response["sources"]
    first_author = sources[0]["metadata"]["authors"][0]
    surname = first_author.split()[-1]
    assert parse_citations(f"{surname} et al.", sources)[0].matched_by == "author"
    assert parse_citations("Nonexistentperson et al.", sources) == []


def test_reference_outside_retrieved_set_is_flagged(registry, ppi_response):
    """A live run cited 'Kei Haga et al.', who is in no retrieved source --
    the model had picked the name out of a chunk's bibliography."""
    template = registry.resolve("ppi")
    answer = (
        "Pathogen\tProtein A\tProtein B\tInteraction Type\tMethod\tAssertion\tReference\n"
        "SARS-CoV-2\tSpike\tACE2\tPhysical\tAssay\tValidated\tKei Haga et al., N/A\n"
    )
    result = extract(answer, template, ppi_response["sources"])
    assert "citation_not_in_sources" in result.rows[0].flags


def test_pipe_table_is_parsed(registry, mutation_response):
    """Models sometimes answer with a markdown table despite a TSV instruction."""
    template = registry.resolve("mutation")
    answer = (
        "| Organism | Gene Name | Mutation | Phenotype | Assertion | Reference |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| M. tuberculosis | katG | S315T | INH resistance | True | [1] |\n"
    )
    result = extract(answer, template, mutation_response["sources"])
    assert len(result.rows) == 1
    assert result.rows[0].get("Mutation") == "S315T"


def test_code_fences_are_stripped(registry, mutation_response):
    template = registry.resolve("mutation")
    answer = (
        "```tsv\nOrganism\tGene Name\tMutation\tPhenotype\tAssertion\tReference\n"
        "M. tuberculosis\tkatG\tS315T\tINH resistance\tTrue\t[1]\n```"
    )
    result = extract(answer, template, mutation_response["sources"])
    assert len(result.rows) == 1


def test_prose_template_returns_text(registry, mutation_response):
    template = registry.resolve("summary")
    result = extract("Some prose about katG [1].", template, mutation_response["sources"])
    assert not result.is_table
    assert result.rows[0].citations[0].pmid == "40580943"


def test_citation_ranges_and_lists_are_distinguished(mutation_response):
    """[1-3] spans three papers; [1, 2] names two. The separator decides."""
    sources = mutation_response["sources"]
    assert [c.marker for c in parse_citations("[1-3]", sources)] == [1, 2, 3]
    assert [c.marker for c in parse_citations("[1–3]", sources)] == [1, 2, 3]
    assert [c.marker for c in parse_citations("[1, 2]", sources)] == [1, 2]
    assert [c.marker for c in parse_citations("[1,2]", sources)] == [1, 2]
    assert [c.marker for c in parse_citations("[2]", sources)] == [2]


def test_separate_markers_are_collected(mutation_response):
    markers = [c.marker for c in parse_citations("[1][4]", mutation_response["sources"])]
    assert markers == [1, 4]


def test_transposed_table_is_repaired(registry, mutation_response):
    """Llama-4-Scout sometimes answers sideways -- one line per FIELD instead of
    per record. Parsed positionally that yields rows of pure nonsense, so the
    rotation has to be detected and undone."""
    template = registry.resolve("mutation")
    answer = "\n".join([
        "Organism\tMycobacterium tuberculosis\tMycobacterium tuberculosis",
        "Gene Name\tkatG\tinhA",
        "Mutation\tS315T\tc-15t",
        "Phenotype\tisoniazid resistance\tlow-level resistance",
        "Assertion\tconfirmed\tconfirmed",
        "Reference\t[1]\t[2]",
    ])
    result = extract(answer, template, mutation_response["sources"])
    assert result.transposed is True
    assert result.n_rows == 2
    assert result.rows[0].get("Gene Name") == "katG"
    assert result.rows[1].get("Mutation") == "c-15t"
    assert result.rows[0].citations[0].pmid == "40580943"


def test_normal_table_is_not_flagged_as_transposed(registry, mutation_response):
    template = registry.resolve("mutation")
    result = extract(mutation_response["answer"], template, mutation_response["sources"])
    assert result.transposed is False
    assert result.n_rows == 9


def test_transposed_table_with_ragged_fields(registry, mutation_response):
    """A rotated table where one field has fewer values than another must still
    line each value up with the right record rather than shifting them."""
    template = registry.resolve("mutation")
    answer = "\n".join([
        "Organism\tM. tuberculosis\tM. tuberculosis",
        "Gene Name\tkatG\tinhA",
        "Mutation\tS315T",                       # only the first record has one
        "Phenotype\tINH resistance\tINH resistance",
        "Reference\t[1]\t[2]",
    ])
    result = extract(answer, template, mutation_response["sources"])
    assert result.transposed is True
    assert result.n_rows == 2
    assert result.rows[0].get("Gene Name") == "katG"
    assert result.rows[0].get("Mutation") == "S315T"
    # The missing value must leave a gap, not pull inhA's phenotype forward.
    assert result.rows[1].get("Gene Name") == "inhA"
    assert result.rows[1].get("Mutation") == ""
    assert result.rows[1].get("Phenotype") == "INH resistance"
    assert "missing:Mutation" in result.rows[1].flags


def test_transposed_record_with_no_evidence_is_still_dropped(registry, mutation_response):
    """Rotation happens before filtering, so an empty record is caught as usual."""
    template = registry.resolve("mutation")
    answer = "\n".join([
        "Organism\tM. tuberculosis\tM. tuberculosis",
        "Gene Name\tkatG\tinhA",
        "Mutation\tS315T",
        "Reference\t[1]\t[2]",
    ])
    result = extract(answer, template, mutation_response["sources"])
    assert result.n_rows == 1 and result.dropped_empty == 1


def test_single_record_table_is_not_rotated(registry, mutation_response):
    """One data row is not evidence of transposition."""
    template = registry.resolve("mutation")
    answer = (
        "Organism\tGene Name\tMutation\tPhenotype\tAssertion\tReference\n"
        "M. tuberculosis\tkatG\tS315T\tINH resistance\tTrue\t[1]\n"
    )
    result = extract(answer, template, mutation_response["sources"])
    assert result.transposed is False
    assert result.n_rows == 1


def _fake_sources(n):
    return [
        {"chunk_id": f"chunk-{i}", "doc_id": f"doc-{i}", "content": "text",
         "metadata": {"pmid": f"1000{i}", "title": f"Paper {i}"}}
        for i in range(1, n + 1)
    ]


def test_citation_list_of_three_is_parsed():
    """`[1, 2, 3]` resolved to NOTHING, silently.

    The marker regex captured at most two numbers, so a three-item list matched
    nothing at all and the row fell through to surname guessing. With ten
    passages in context a model rarely cites three sources at once; with several
    hundred it does so constantly, which is what makes this worth fixing now.
    """
    citations = parse_citations("[1, 2, 3]", _fake_sources(5))
    assert [c.pmid for c in citations] == ["10001", "10002", "10003"]


def test_long_and_mixed_citation_lists():
    sources = _fake_sources(9)
    assert len(parse_citations("[1, 2, 3, 4, 5]", sources)) == 5
    # A span and a singleton in one bracket.
    assert {c.marker for c in parse_citations("[1-3, 7]", sources)} == {1, 2, 3, 7}
    # Semicolons separate too.
    assert len(parse_citations("[2; 4]", sources)) == 2


def test_span_and_list_still_mean_different_things():
    """Regression guard: [1-3] is three papers, [1,3] is two. Do not conflate."""
    sources = _fake_sources(5)
    assert {c.marker for c in parse_citations("[1-3]", sources)} == {1, 2, 3}
    assert {c.marker for c in parse_citations("[1,3]", sources)} == {1, 3}


def test_citation_round_trips_through_dict():
    """Guard: a field added to Citation must survive to_dict -> from_dict.

    Passages have been lost this way before -- a field was added to the
    dataclass and to to_dict, and from_dict was left behind, so anything written
    to disk and read back came home empty with no error.
    """
    from litrag.extract import Citation

    original = Citation(marker=3, pmid="123", doc_id="doc-x", chunk_id="chunk-y")
    restored = Citation.from_dict(original.to_dict())
    assert restored.doc_id == "doc-x"
    assert restored.chunk_id == "chunk-y"
    assert restored.identity == original.identity


def test_citation_identity_prefers_pmid_then_falls_back_to_doc_id():
    from litrag.extract import Citation

    assert Citation(marker=1, pmid="9", doc_id="d").identity == "9"
    assert Citation(marker=1, doc_id="d").identity == "d"
    # Nothing at all to key on: the marker is the last resort, not the default.
    assert Citation(marker=4).identity == "marker:4"


def test_citation_carries_doc_id_from_the_source():
    sources = _fake_sources(2)
    sources[0]["doc_id"] = "doc-alpha"
    citations = parse_citations("[1]", sources)
    assert citations[0].doc_id == "doc-alpha"


def test_one_paper_split_into_sub_documents_counts_once():
    """PMC gives figures and tables their own doc_id.

    Measured on the live index: 16 of 111 PMIDs from a single query came back
    under more than one doc_id, one of them under five, because PMC splits an
    article into sub-documents (`PMC4643029#figure-3`). Keying citation identity
    on doc_id ahead of the PMID would therefore report one paper as five
    independent supporting sources and inflate n_support -- the number used to
    rank what a curator checks first.
    """
    from litrag.dedup import _merge_citations
    from litrag.extract import Citation

    article = Citation(marker=1, pmid="34147065", doc_id="086c5b46-article")
    figure = Citation(marker=2, pmid="34147065", doc_id="329ee653-figure-3")

    assert article.identity == figure.identity
    assert len(_merge_citations([article], [figure])) == 1


def test_identity_never_falls_back_to_a_bare_marker_when_a_chunk_is_known():
    """A marker means a different paper in every batch, so it must be last."""
    from litrag.extract import Citation

    assert Citation(marker=1, chunk_id="c-1").identity == "c-1"
    assert Citation(marker=1, doc_id="d-1", chunk_id="c-1").identity == "d-1"
