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


def test_mutation_column_asks_for_standard_notation(registry, mutation_response):
    """The missed finding was written "NS1-53 glycine to aspartate", not G53D.

    An earlier version of this rule said to record prose "as the source words
    them", and the model duly answered with prose -- returning "S to A, W to A,
    D to A, T to A at positions 114, 115, 180, 301" as one value, which the
    comma splitter then chopped into seven fragments. The rule has to ask for
    notation and reject a set packed into one value.
    """
    from litrag.prompts import column_rules
    prompt, _, _ = build_prompt(registry.resolve("mutation"),
                                mutation_response["sources"], "dengue virus")
    assert "standard notation" in prompt
    assert "glycine to aspartate" in prompt and "G53D" in prompt
    assert "four rows" in prompt
    assert "as the source words it" in prompt, "prose is still a fallback"
    assert column_rules(["Mutation"])


def test_prose_rule_only_applies_where_there_is_a_mutation_column():
    from litrag.prompts import column_rules
    rules = " ".join(column_rules(["Organism", "Strain", "Antibiotic", "MIC"]))
    assert "glycine to aspartate" not in rules


def test_instructions_get_their_own_section(registry, mutation_response):
    """Guidance is layered on top of the schema rules, never in place of them."""
    template = registry.resolve("mutation")
    prompt, _, _ = build_prompt(template, mutation_response["sources"], "M. tb",
                                instructions="Report positions as A226, K128.")
    assert "Additional instructions from the user:" in prompt
    assert "Report positions as A226, K128." in prompt
    # The output shape the rest of the pipeline parses must survive.
    assert "\t".join(template.columns) in prompt
    # And guidance belongs with the instructions, not buried after the papers.
    assert prompt.index("A226") < prompt.index("--- LITERATURE CONTEXT ---")


def test_blank_instructions_add_no_section(registry, mutation_response):
    template = registry.resolve("mutation")
    plain, _, _ = build_prompt(template, mutation_response["sources"], "M. tb")
    blank, _, _ = build_prompt(template, mutation_response["sources"], "M. tb",
                               instructions="   \n  ")
    assert blank == plain


def test_instructions_change_the_prompt_hash(registry, mutation_response):
    sources = mutation_response["sources"]
    _, plain, _ = build_prompt(registry.resolve("mutation"), sources, "M. tb")
    _, guided, _ = build_prompt(registry.resolve("mutation"), sources, "M. tb",
                                instructions="Confirmed findings only.")
    assert plain != guided, "provenance must distinguish a guided run"
