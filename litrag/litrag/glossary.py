"""Plain-language definitions for the columns and flags a curator sees.

Column sets come from the templates, so they differ by data type and can change
when the operator adds one. Definitions are therefore keyed by column name and
looked up case-insensitively, with an empty result for anything unknown -- a new
column simply has no tooltip rather than the wrong one.

Kept in Python rather than in the page so the CLI, the API and the UI all
describe a column the same way.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# Keyed by lowercase column name.
COLUMNS: Dict[str, str] = {
    # -- subject ---------------------------------------------------------
    "organism": "The organism the finding is about, as named in the source.",
    "pathogen": "The pathogen the interaction was reported in, as named in the source.",
    "species": "The species the finding is about, as named in the source.",
    "gene name": (
        "Gene or protein the finding concerns. May fall outside the genes you "
        "asked for — those rows are kept and marked off_target_gene, since the "
        "finding is still real."
    ),
    "gene": "Gene the finding concerns, as named in the source.",
    "protein": "Protein the finding concerns, as named in the source.",
    "protein a": (
        "One partner in the interaction. Order carries no meaning: A–B and B–A "
        "are treated as the same interaction and merged."
    ),
    "protein b": (
        "The other partner in the interaction. Order carries no meaning: A–B and "
        "B–A are treated as the same interaction and merged."
    ),
    "strain": (
        "Strain or isolate designation as published (ATCC 25922, H37Rv, K-12 "
        "MG1655). N/A when the source reports a pooled result over many isolates "
        "rather than a named strain."
    ),

    # -- identifiers -----------------------------------------------------
    "genbank accession": (
        "Nucleotide or genome accession, recorded only when the source states it "
        "explicitly (e.g. CP012345). Never inferred from the organism or strain."
    ),
    "biosample": (
        "NCBI BioSample accession (e.g. SAMN01234567), recorded only when stated "
        "explicitly. Never inferred."
    ),
    "accession": "Database accession, recorded only when stated explicitly in the source.",

    # -- findings --------------------------------------------------------
    "mutation": (
        "The sequence change, shown as the source wrote it. Notation is "
        "normalized behind the scenes for merging, so Ser315Thr, S315T and "
        "katG-S315T count as one fact."
    ),
    "phenotype": "The observable effect attributed to the finding.",
    "function": "The molecular or biological function attributed to the gene or protein.",
    "interaction type": (
        "How the two partners relate. For host-virus rows this is one of six "
        "verbs — binds, cleaves, inhibits, activates, degrades, relocalizes — "
        "with the source's own verb kept when none of them fits. For "
        "protein-protein rows it is the source's own wording."
    ),
    "viral protein": (
        "The viral protein doing the acting. The direction is always viral "
        "protein onto host target, so a host protein never appears here."
    ),
    "host protein": (
        "What the viral protein acts on, named as the source names it — STAT1, "
        "TBK1, or a pathway such as type I interferon signalling."
    ),
    "host organism": (
        "The host the work was done in — human, mouse, or the cell line where "
        "that is all the source gives. Never assumed."
    ),
    "biological consequence": (
        "What follows for the host or the infection: reduced interferon "
        "signalling, blocked apoptosis, enhanced replication. N/A means the "
        "source reported the interaction without a consequence."
    ),
    "method": "The experimental method the source used to establish the finding.",
    "antibiotic": (
        "The antimicrobial tested. Salt forms and abbreviations are treated as "
        "the same drug when merging (Ampicillin sodium = Ampicillin)."
    ),
    "mic": (
        "Minimum inhibitory concentration, verbatim with its unit and any "
        "inequality (>32 mg/L, ≤0.125 mg/L). Units are never converted, and an "
        "MIC is never derived from an S/I/R call."
    ),
    "sir": (
        "Susceptible / Intermediate / Resistant, as the source reported it. "
        "Never derived from the MIC, and no breakpoint is applied here — the "
        "same MIC can be called S or R under CLSI versus EUCAST."
    ),
    "site": (
        "The modified residue and its position, as the source numbers it — "
        "N234, Asn234, T678. Numbering is never rewritten: it differs between "
        "isoforms, strains and constructs, so N234 and N235 are distinct sites "
        "even for the same residue in two papers."
    ),
    "glycosylation type": (
        "The linkage: N-linked (on Asn), O-linked (on Ser or Thr), or "
        "C-mannosylation (on Trp). Recorded only when the source states it — "
        "never inferred from the residue alone."
    ),
    "glycan": (
        "The glycan structure or composition as published — high-mannose, "
        "complex, hybrid, Man5GlcNAc2, core-fucosylated. Nomenclature is not "
        "converted. N/A means the glycan was not characterised, not that the "
        "site is unglycosylated."
    ),
    "effect": (
        "The functional consequence the source reports for this site — folding, "
        "receptor binding, antibody shielding. N/A when the source reports none."
    ),
    "assertion": (
        "Where the claim stands in the source: measured (the source ran the "
        "experiment), inferred (concluded indirectly from its own data), "
        "predicted (computational only), reported (attributed to other work), "
        "or disputed (contradicted or unconfirmed). Local models are held to "
        "these five; the hosted generator uses its own wording instead, such "
        "as \"High confidence\"."
    ),
    "reference": (
        "The citation marker the model emitted. Resolved to real papers in the "
        "Citations column; a marker matching no retrieved source is flagged "
        "rather than guessed at."
    ),
}

# Columns the UI adds to every table.
DERIVED: Dict[str, str] = {
    "support": (
        "How many extracted rows merged into this one. Higher means the same "
        "fact was stated more than once — check the citations to see whether "
        "that is several papers or one paper repeated."
    ),
    "citations": (
        "The papers backing this row, linked to PubMed or DOI. Each ¶ link jumps "
        "to the exact retrieved passage the claim came from."
    ),
    "flags": "Quality signals about this row. Hover any flag for what it means.",
}

# Values that appear in the Flags column.
FLAGS: Dict[str, str] = {
    "off_target_gene": (
        "This row is about a gene outside the ones you asked for. Kept, not "
        "dropped — the finding is real, it just was not what you searched for."
    ),
    "citation_not_in_sources": (
        "The reference names a paper that is not among the retrieved sources. "
        "Usually the model picked the name out of a chunk's own bibliography, so "
        "the attribution is unverified."
    ),
    "no_citation": "The source gave no reference for this row.",
    "unresolved_citation": (
        "The citation marker points past the end of the retrieved source list, "
        "so it could not be matched to a paper."
    ),
    "merged_variants": (
        "Rows merged into this one disagreed on this column. The fuller value is "
        "shown and the alternatives are listed underneath, so a single paper's "
        "wording is never presented as every source's finding."
    ),
    "method_not_a_technique": (
        "The Method cell held an evidence category rather than a technique — "
        '"reported" or "measured" instead of co-immunoprecipitation or mass '
        "spectrometry. It was cleared, since a method that was never stated is "
        "worse than none. The evidence category is in Assertion."
    ),
    "column_count_mismatch": (
        "The source row did not have one value per column, so the alignment "
        "shown is a reconstruction rather than a reading. It was fitted using "
        "the columns whose shape is known — the assertion vocabulary and the "
        "citation — but the rest could be off by one. Worth checking by hand."
    ),
    "compound_row": (
        "The source packed several findings into one row and they could not be "
        "split apart unambiguously. Worth reading by hand."
    ),
    "missing": "The column carrying the actual finding is empty for this row.",
    "evidence_free": (
        "This row names a subject but reports nothing about it. Shown only "
        "because 'Keep empty rows' is on; normally such rows are dropped."
    ),
}


def describe_column(name: Optional[str]) -> str:
    """The definition for a column, or an empty string if there is none."""
    if not name:
        return ""
    key = " ".join(str(name).split()).strip().lower()
    return COLUMNS.get(key) or DERIVED.get(key, "")


def describe_flag(flag: Optional[str]) -> str:
    """The definition for a flag value.

    Flags may be qualified by a column -- "merged_variants:Phenotype",
    "missing:MIC/SIR" -- so the prefix is what carries the meaning.
    """
    if not flag:
        return ""
    prefix = str(flag).split(":", 1)[0].strip()
    return FLAGS.get(prefix, "")


def for_columns(columns: Sequence[str]) -> Dict[str, str]:
    """Definitions for a template's columns, omitting any without one."""
    described = {}
    for column in columns:
        text = describe_column(column)
        if text:
            described[column] = text
    return described


def entries() -> List[Dict[str, str]]:
    """The whole glossary, for listing."""
    rows = [{"term": k, "kind": "column", "text": v} for k, v in COLUMNS.items()]
    rows += [{"term": k, "kind": "column", "text": v} for k, v in DERIVED.items()]
    rows += [{"term": k, "kind": "flag", "text": v} for k, v in FLAGS.items()]
    return rows
