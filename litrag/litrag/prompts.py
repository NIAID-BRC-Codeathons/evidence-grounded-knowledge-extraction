"""Prompt construction for local generation.

Generating locally means LitRAG owns the prompt, which the hosted path does not
expose. To keep that from becoming a second hardcoded schema, the prompt is
built from the server's template DECLARATION -- its label and columns, already
fetched from /v1/prompt-templates -- so a template added server-side still
produces the right table here.

What is lost is the server's template hash as a provenance anchor. It is
replaced by a hash of the prompt actually sent, recorded per row, so a local run
is reproducible on its own terms.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .templates import Template

# Rough chars-per-token for budgeting. Deliberately conservative: overshooting
# the context window costs a failed request, undershooting costs a little recall.
CHARS_PER_TOKEN = 3.5
# Headroom for the instructions and the model's own output.
RESERVED_TOKENS = 4000


def _source_block(index: int, source: Dict[str, Any]) -> str:
    meta = source.get("metadata", {}) or {}
    bits = [f"Source {index}"]
    title = meta.get("title")
    if title:
        bits.append(f"({title})")
    ident = meta.get("pmid") or meta.get("doi")
    if ident:
        label = "PMID" if meta.get("pmid") else "DOI"
        bits.append(f"[{label}: {ident}]")
    return " ".join(bits) + ":\n" + (source.get("content") or "")


def build_context(
    sources: Sequence[Dict[str, Any]],
    max_chars: Optional[int] = None,
) -> Tuple[str, int]:
    """Join sources into a context block, dropping the tail if over budget.

    Sources arrive ranked, so truncation removes the least relevant first. The
    number of sources actually included is returned, because citation markers
    must not point past it.
    """
    blocks: List[str] = []
    used = 0
    for index, source in enumerate(sources, start=1):
        block = _source_block(index, source)
        if max_chars is not None and used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block) + 7  # separator
    return "\n\n---\n\n".join(blocks), len(blocks)


# Rules attached to a column wherever it appears, including in the server's own
# templates, which publish column names but no guidance of their own.
#
# "Assertion" needs one badly. Left undefined, Qwen echoed the instruction to
# report only what sources state and wrote "Stated" in every row -- a constant,
# carrying no information. Llama produced evidence types, the hosted path
# produces confidence grades, so the column meant three things depending on who
# generated it.
#
# The vocabulary below is an evidence-provenance axis rather than a confidence
# one: whether a source measured, inferred, predicted, relayed, or disputed a
# claim is checkable against its text, whereas a model's self-rated confidence
# is not.
ASSERTION_VALUES = ("measured", "inferred", "predicted", "reported", "disputed")

COLUMN_RULES = {
    "method": (
        "Method is the experimental technique the source used — "
        "co-immunoprecipitation, mass spectrometry, yeast two-hybrid, reporter "
        "assay, microscopy, broth microdilution, site-directed mutagenesis, "
        "sequence-based prediction. It is never an evidence category: "
        '"measured", "reported", "inferred" and "predicted" belong in '
        'Assertion, not here. Write "N/A" when the source names no technique.'
    ),
    "mutation": (
        "Write every mutation in standard notation -- reference residue, "
        "position, variant residue, as in S315T or G53D. A source that spells "
        'it out still counts and still gets notation: "NS1-53 glycine to '
        'aspartate" is G53D, "position 315 serine to threonine" is S315T, '
        '"5\'UTR-57, C to T" is c.57T. Only when the source does not give all '
        "three parts, record the mutation as the source words it. Never write "
        'a set of mutations as one value such as "S, W, D and T to A at '
        'positions 114, 115, 180 and 301" -- that is four findings and needs '
        "four rows: S114A, W115A, D180A, T301A."
    ),
    "assertion": (
        "The Assertion column must contain exactly one of: "
        + ", ".join(ASSERTION_VALUES) + ". Use "
        '"measured" when the source itself performed the experiment that shows '
        'this; "inferred" when the source draws the conclusion indirectly from '
        'its own data; "predicted" when the result is computational or in '
        'silico only; "reported" when the source attributes it to other work '
        'rather than its own; "disputed" when the source contradicts it or '
        "fails to confirm it. Write nothing else in that column -- not a "
        "confidence level, and not a restatement of the finding."
    ),
}


def column_rules(columns: Sequence[str]) -> List[str]:
    """Rules for whichever of the known columns this template actually has."""
    rules = []
    for column in columns:
        rule = COLUMN_RULES.get(" ".join(str(column).split()).strip().lower())
        if rule and rule not in rules:
            rules.append(rule)
    return rules


def _table_instructions(template: Template, spec_bits: str, n_sources: int) -> str:
    columns = "\t".join(template.columns or [])
    rules = [
        "- Output the header row first, then one row per finding.",
        "- Separate columns with a single tab character.",
        '- Use "N/A" when a value is not stated.',
        f"- Cite the source number in brackets, e.g. [1], in the Reference column. "
        f"Valid source numbers are 1 to {n_sources}.",
        "- Cite only the numbered sources above. Do not cite papers mentioned "
        "inside a source's own reference list.",
        "- One finding per row. Never combine multiple genes, proteins, or "
        "variants into a single row.",
        "- Include only findings the sources actually state.",
        "- Work through every numbered source in turn. Do not stop once one "
        "source has yielded findings -- a later source usually reports "
        "different ones, and a source whose main topic is something else can "
        "still state a finding in passing.",
        "- Output no text before or after the table.",
    ]
    # What the template says a row cannot be read without. Stated as a rule so
    # the model omits such rows rather than emitting them for extract.py to
    # drop -- a dropped row still cost the tokens that wrote it, and at top_k
    # 40 those rows were most of the answer and pushed it into truncation.
    if template.required:
        required = ", ".join(template.required)
        rules.append(
            f"- Every row must name a real value in each of these columns: "
            f"{required}. These identify what the finding is about, so a row "
            f'missing any of them is not a partial finding -- write no row at '
            f'all rather than putting "N/A" in one of them.'
        )

    # Rules for particular columns, then any the template itself declares.
    rules.extend(f"- {line}" for line in column_rules(template.columns or []))
    rules.extend(f"- {line}" for line in template.guidance)

    return (
        f"Extract structured data about {template.label}{spec_bits}.\n\n"
        f"Return ONLY a TSV (tab-separated) table with exactly these columns:\n{columns}\n\n"
        "Rules:\n" + "\n".join(rules)
    )


def _prose_instructions(template: Template, spec_bits: str, n_sources: int) -> str:
    return (
        f"Provide a comprehensive summary{spec_bits}, based only on the "
        "literature context below.\n\n"
        f"Cite sources by number in brackets, e.g. [1], using only numbers 1 to "
        f"{n_sources}. Organize by gene or protein where several are discussed."
    )


def describe_subject(
    organism: str, genes: str = "", other_terms: str = ""
) -> str:
    """The "for organism X involving genes Y" clause shared by both prompt shapes."""
    bits = f' for organism "{organism}"'
    if genes:
        bits += f" involving genes/proteins: {genes}"
    if other_terms:
        bits += f", related to: {other_terms}"
    return bits


def build_prompt(
    template: Template,
    sources: Sequence[Dict[str, Any]],
    organism: str,
    genes: str = "",
    other_terms: str = "",
    max_context_tokens: Optional[int] = None,
    instructions: str = "",
) -> Tuple[str, str, int]:
    """Build the prompt, its hash, and how many sources it actually includes."""
    max_chars = None
    if max_context_tokens:
        budget = max(1, max_context_tokens - RESERVED_TOKENS)
        max_chars = int(budget * CHARS_PER_TOKEN)

    context, n_included = build_context(sources, max_chars)
    spec_bits = describe_subject(organism, genes, other_terms)

    base_instructions = (
        _table_instructions(template, spec_bits, n_included)
        if template.is_table
        else _prose_instructions(template, spec_bits, n_included)
    )
    # User-supplied guidance, layered on top of the schema/citation rules above
    # rather than replacing them -- it steers what gets extracted, not the
    # output shape the rest of the pipeline (extract.py, dedup.py) depends on.
    if instructions.strip():
        base_instructions += f"\n\nAdditional instructions from the user:\n{instructions.strip()}"

    prompt = f"{base_instructions}\n\n--- LITERATURE CONTEXT ---\n\n{context}"
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return prompt, digest, n_included
