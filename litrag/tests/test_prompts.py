"""Local prompt construction."""

from litrag.prompts import build_context, build_prompt, describe_subject


def test_prompt_uses_template_columns(registry, mutation_response):
    """The prompt is built from the server's declaration, not a second schema."""
    template = registry.resolve("mutation")
    prompt, _, _ = build_prompt(template, mutation_response["sources"], "M. tuberculosis")
    assert "\t".join(template.columns) in prompt
    assert template.label in prompt


def test_prompt_bounds_the_citable_range(registry, mutation_response):
    """Telling the model how many sources exist keeps markers in range."""
    sources = mutation_response["sources"]
    prompt, _, included = build_prompt(template=registry.resolve("mutation"),
                                       sources=sources, organism="M. tb")
    assert included == len(sources)
    assert f"1 to {len(sources)}" in prompt


def test_prompt_forbids_citing_bibliographies(registry, mutation_response):
    """A live run cited a paper found inside a chunk's reference list."""
    prompt, _, _ = build_prompt(registry.resolve("ppi-extraction"),
                                mutation_response["sources"], "SARS-CoV-2")
    assert "reference list" in prompt


def test_prompt_forbids_compound_rows(registry, mutation_response):
    """Live output packed 'katG, inhA' with two mutations into one row."""
    prompt, _, _ = build_prompt(registry.resolve("mutation"),
                                mutation_response["sources"], "M. tb")
    assert "One finding per row" in prompt


def test_hash_is_stable_and_content_sensitive(registry, mutation_response):
    template = registry.resolve("mutation")
    sources = mutation_response["sources"]
    a, ha, _ = build_prompt(template, sources, "M. tb", genes="katG")
    b, hb, _ = build_prompt(template, sources, "M. tb", genes="katG")
    c, hc, _ = build_prompt(template, sources, "M. tb", genes="inhA")
    assert a == b and ha == hb
    assert hc != ha, "a different query must produce a different prompt hash"


def test_context_truncates_from_the_tail(registry, mutation_response):
    """Sources arrive ranked, so the least relevant should be dropped first."""
    sources = mutation_response["sources"]
    context, included = build_context(sources, max_chars=1500)
    assert 0 < included < len(sources)
    assert "Source 1" in context
    assert f"Source {len(sources)}" not in context


def test_truncation_never_yields_an_empty_context(mutation_response):
    _context, included = build_context(mutation_response["sources"], max_chars=1)
    assert included == 1, "keep at least one source even under an absurd budget"


def test_prose_template_gets_prose_instructions(registry, mutation_response):
    prompt, _, _ = build_prompt(registry.resolve("literature-summary"),
                                mutation_response["sources"], "M. tb")
    assert "comprehensive summary" in prompt
    assert "TSV" not in prompt


def test_subject_clause_omits_blank_fields():
    assert describe_subject("M. tb") == ' for organism "M. tb"'
    assert "involving genes" in describe_subject("M. tb", genes="katG")
    assert "related to" in describe_subject("M. tb", other_terms="resistance")


def test_context_includes_identifiers(registry, mutation_response):
    prompt, _, _ = build_prompt(registry.resolve("mutation"),
                                mutation_response["sources"], "M. tb")
    assert "PMID:" in prompt


def test_assertion_column_gets_a_vocabulary(registry, mutation_response):
    """Left undefined, the column became a constant.

    Qwen echoed the instruction to report only what sources state and wrote
    "Stated" in every row -- no information at all. Llama wrote evidence types,
    the hosted path writes confidence grades, so the column meant three
    different things depending on who generated it.
    """
    from litrag.prompts import ASSERTION_VALUES
    prompt, _, _ = build_prompt(registry.resolve("mutation"),
                                mutation_response["sources"], "M. tb")
    assert "Assertion column must contain exactly one of" in prompt
    for value in ASSERTION_VALUES:
        assert value in prompt
    assert "not a confidence level" in prompt


def test_column_rules_apply_to_server_templates(registry):
    """Server templates publish column names but no guidance of their own, so
    the rule has to attach to the column, not to the template."""
    from litrag.prompts import column_rules
    for name in ("mutation", "ppi-extraction", "protein-function"):
        template = registry.resolve(name)
        assert column_rules(template.columns), f"{name} has an Assertion column"


def test_column_rules_skip_templates_without_the_column():
    """AST has no Assertion, so it must not be told how to fill one."""
    from litrag.prompts import column_rules
    assert column_rules(["Organism", "Strain", "Antibiotic", "MIC", "SIR"]) == []


def test_column_rule_lookup_is_normalized():
    from litrag.prompts import column_rules
    assert column_rules(["assertion"]) == column_rules([" Assertion "])


def test_prompt_demands_a_full_sweep_of_sources(registry, mutation_response):
    """Given 100 sources the model returned four rows, all from one paper's
    mutagenesis table, ignoring a mutation stated in passing elsewhere."""
    prompt, _, _ = build_prompt(registry.resolve("mutation"),
                                mutation_response["sources"], "dengue virus")
    assert "every numbered source" in prompt
    assert "in passing" in prompt


def test_mutation_column_accepts_prose_notation(registry, mutation_response):
    """The missed finding was written "NS1-53 glycine to aspartate", not G53D.

    A rule keyed to the column reaches the server's templates too.
    """
    from litrag.prompts import column_rules
    prompt, _, _ = build_prompt(registry.resolve("mutation"),
                                mutation_response["sources"], "dengue virus")
    assert "glycine to aspartate" in prompt
    assert "does not have to be written" in prompt
    assert column_rules(["Mutation"])


def test_prose_rule_only_applies_where_there_is_a_mutation_column():
    from litrag.prompts import column_rules
    rules = " ".join(column_rules(["Organism", "Strain", "Antibiotic", "MIC"]))
    assert "glycine to aspartate" not in rules
