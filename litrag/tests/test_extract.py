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


# -- passage-level provenance --------------------------------------------

def test_citation_records_the_chunk_it_came_from(mutation_response):
    """The chunk is the text the model actually read, so it is the evidence."""
    sources = mutation_response["sources"]
    citation = parse_citations("[1]", sources)[0]
    assert len(citation.chunks) == 1
    chunk = citation.chunks[0]
    assert chunk.chunk_id == sources[0]["chunk_id"]
    assert chunk.doc_id == sources[0]["doc_id"]
    assert chunk.marker == 1
    assert chunk.score == sources[0]["score"]


def test_every_passage_of_one_paper_survives(mutation_response):
    """Sources 3, 5 and 6 are three chunks of PMID 19578178.

    They collapse to one citation, but all three passages are kept: dropping
    two would discard exactly the provenance a curator needs to verify.
    """
    citations = parse_citations("[3][5][6]", mutation_response["sources"])
    assert len(citations) == 1
    assert [c.marker for c in citations[0].chunks] == [3, 5, 6]
    assert len(set(citations[0].chunk_ids)) == 3


def test_chunk_spans_are_captured(mutation_response):
    citation = parse_citations("[1]", mutation_response["sources"])[0]
    chunk = citation.chunks[0]
    meta = mutation_response["sources"][0]["metadata"]
    assert chunk.start_char == meta.get("start_char")
    assert chunk.end_char == meta.get("end_char")
    assert chunk.span == f"{chunk.start_char}-{chunk.end_char}"


def test_row_exposes_all_supporting_chunks(registry, mutation_response):
    template = registry.resolve("mutation")
    result = extract(mutation_response["answer"], template, mutation_response["sources"])
    row = result.rows[0]
    assert row.chunk_ids and all(row.chunk_ids)
    assert row.markers == sorted(row.markers)


def test_unresolved_citation_has_no_chunk(mutation_response):
    citation = parse_citations("[99]", mutation_response["sources"])[0]
    assert not citation.resolved and citation.chunks == []


def test_citation_round_trips_with_its_chunks(mutation_response):
    """Resume rebuilds rows from the sidecar; passages must come back too."""
    from litrag.extract import Citation
    original = parse_citations("[3][5][6]", mutation_response["sources"])[0]
    restored = Citation.from_dict(original.to_dict())
    assert [c.marker for c in restored.chunks] == [3, 5, 6]
    assert restored.chunk_ids == original.chunk_ids


# -- bracket-aware list splitting ----------------------------------------

def test_parenthetical_list_is_not_split(registry, mutation_response):
    """Live regression: "Multiple mutations (codons 315, 316, 309)" was split
    into the three rows "Multiple mutations (codons 315", "316" and "309)"."""
    template = registry.resolve("mutation")
    answer = (
        "Organism\tGene Name\tMutation\tPhenotype\tAssertion\tReference\n"
        "M. tuberculosis\tkatG\tMultiple mutations (codons 315, 316, 309)\t"
        "High-level resistance\tTrue\t[1]\n"
    )
    result = extract(answer, template, mutation_response["sources"])
    assert result.split_compound == 0
    assert result.n_rows == 1
    assert result.rows[0].get("Mutation") == "Multiple mutations (codons 315, 316, 309)"


def test_genuine_list_still_splits_alongside_a_parenthetical(registry, mutation_response):
    template = registry.resolve("mutation")
    answer = (
        "Organism\tGene Name\tMutation\tPhenotype\tAssertion\tReference\n"
        "M. tuberculosis\tkatG, inhA\tSer315Thr (S315T), c-15t\t"
        "INH resistance\tTrue\t[1]\n"
    )
    result = extract(answer, template, mutation_response["sources"])
    assert result.split_compound == 1
    assert [r.get("Mutation") for r in result.rows] == ["Ser315Thr (S315T)", "c-15t"]


def test_display_joins_merged_alternatives():
    from litrag.extract import Row
    row = Row(values={"Phenotype": "antigenic change"},
              variants={"Phenotype": ["antigenic change", "enhanced replication"]})
    assert row.display("Phenotype") == "antigenic change; enhanced replication"
    # get() still returns the representative value on its own.
    assert row.get("Phenotype") == "antigenic change"


def test_display_puts_the_representative_value_first():
    from litrag.extract import Row
    row = Row(values={"A": "chosen"}, variants={"A": ["other", "chosen"]})
    assert row.display("A") == "chosen; other"


def test_display_does_not_repeat_a_value():
    from litrag.extract import Row
    row = Row(values={"A": "x"}, variants={"A": ["x", "x", "y"]})
    assert row.display("A") == "x; y"


def test_display_without_variants_is_the_plain_value():
    from litrag.extract import Row
    assert Row(values={"A": "x"}).display("A") == "x"
    assert Row(values={}).display("A") == ""


def test_assertion_does_not_count_as_evidence(registry, ppi_response):
    """Regression: once Assertion was filled from a fixed vocabulary on every
    row, counting it as evidence meant no row was ever evidence-free and the
    filter stopped firing."""
    template = registry.resolve("ppi")
    answer = (
        "Pathogen\tProtein A\tProtein B\tInteraction Type\tMethod\tAssertion\tReference\n"
        "SARS-CoV-2\tNSP13\tSpike\tN/A\tN/A\treported\tN/A\n"
    )
    result = extract(answer, template, ppi_response["sources"])
    assert result.n_rows == 0 and result.dropped_empty == 1


def test_bare_number_in_a_reference_column_resolves(registry, mutation_response):
    """Models sometimes write "3" instead of "[3]" in the Reference column.

    A lone integer there can only be a source number, so it is read as one.
    """
    template = registry.resolve("mutation")
    answer = (
        "Organism\tGene Name\tMutation\tPhenotype\tAssertion\tReference\n"
        "M. tuberculosis\tkatG\tS315T\tINH resistance\tmeasured\t1\n"
    )
    result = extract(answer, template, mutation_response["sources"])
    assert result.rows[0].citations[0].pmid == "40580943"
    assert "no_citation" not in result.rows[0].flags


def test_bare_numbers_are_not_read_as_citations_elsewhere(mutation_response):
    """Outside a reference column an integer is a position or a dose."""
    assert parse_citations("3", mutation_response["sources"]) == []
    assert parse_citations(">32 mg/L", mutation_response["sources"], bare_numbers=True) == []


def test_out_of_range_bare_number_is_ignored(mutation_response):
    assert parse_citations("99", mutation_response["sources"], bare_numbers=True) == []
