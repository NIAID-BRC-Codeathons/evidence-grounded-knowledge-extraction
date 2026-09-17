"""The dry run is a promise about what the real run will send.

Every number it prints has to be one the real path would produce, or it is
worse than printing nothing: a preview that quietly disagrees with the run it
previews sends the reader looking for a bug that is not there.
"""

import json

from litrag.cli import _dry_run
from litrag.llm import PRESETS
from litrag.pipeline import QuerySpec, plan_output_tokens
from litrag.templates import Template


def table_template(max_output_tokens=2500):
    return Template(
        id="mutation", label="Mutation", output="table",
        columns=["Organism", "Gene Name", "Mutation", "Reference"],
        max_output_tokens=max_output_tokens,
    )


def dry_run_output(capsys, top_k):
    _dry_run(
        QuerySpec(organism="Test organism", genes="abc", data_type="mutation",
                  top_k=top_k),
        table_template(),
        PRESETS["qwen"],
    )
    out = capsys.readouterr().out
    body, _, prompt = out.partition("--- prompt")
    return json.loads(body[body.index("{"):body.rindex("}") + 1]), prompt


def test_the_token_budget_matches_what_the_real_run_would_plan(capsys):
    """It was hardcoded to the template's declared cap, under-reporting a
    hundred-source run by sevenfold."""
    body, _ = dry_run_output(capsys, top_k=100)
    assert body["generate"]["max_tokens"] == plan_output_tokens(table_template(), 100)
    assert body["generate"]["max_tokens"] == 17000


def test_the_budget_still_reports_the_declared_floor_for_a_small_run(capsys):
    body, _ = dry_run_output(capsys, top_k=10)
    assert body["generate"]["max_tokens"] == 2500


def test_the_planned_source_count_is_labelled_as_planned(capsys):
    """The real count comes from retrieval, which has not run yet."""
    body, _ = dry_run_output(capsys, top_k=25)
    assert body["generate"]["max_tokens_basis"] == "planned for 25 sources"


def test_the_preview_never_promises_an_empty_source_range(capsys):
    """Built against no sources, the instructions read "1 to 0" -- a rule the
    real run would never send, telling the model no citation is valid."""
    _, prompt = dry_run_output(capsys, top_k=10)
    assert "1 to 0" not in prompt
    assert "Valid source numbers are 1 to 10." in prompt


def test_no_prompt_hash_is_offered_that_could_never_match(capsys):
    """The dry run cannot hash the real prompt: it has no literature in it. It
    used to print one anyway, which could never equal a run's _prompt_hash."""
    _, prompt = dry_run_output(capsys, top_k=10)
    assert "hash" not in prompt.split("\n")[0]
