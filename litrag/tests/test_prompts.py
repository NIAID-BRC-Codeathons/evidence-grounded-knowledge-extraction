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
