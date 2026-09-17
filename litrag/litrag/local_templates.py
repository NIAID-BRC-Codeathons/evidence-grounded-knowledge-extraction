"""Data types defined by LitRAG rather than the server.

/v1/prompt-templates is read-only (POST returns 405) and its templates are
operator configuration, so a new data type cannot be added there. It can be
added here: local generation already builds its prompt from a template
declaration, so a declaration defined locally works the same way.

The consequence is that a local template only runs on a local generator
(--llm qwen, the default). The hosted path would reject the unknown template
id, and the pipeline says so rather than letting the server 404.

Extra templates can be dropped into ~/.config/litrag/templates.toml without
touching this file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

USER_TEMPLATES_PATH = Path.home() / ".config" / "litrag" / "templates.toml"

# Same slots as the server's templates, so both surfaces present one form.
_STANDARD_SLOTS = [
    {"name": "organism", "required": True, "max_len": 120,
     "label": "Organism of interest"},
    {"name": "genes", "required": False, "max_len": 200,
     "label": "Genes of interest"},
    {"name": "other_terms", "required": False, "max_len": 200,
     "label": "Other terms"},
]

BUILTIN: List[Dict[str, Any]] = [
    {
        "id": "ast",
        "label": "Antimicrobial Susceptibility Testing (AST)",
        "output": "table",
        "version": 1,
        # Reference is not optional: citation resolution keys off it, and an
        # AST value without a source is not curatable.
        "columns": [
            "Organism", "Strain", "GenBank Accession", "BioSample",
            "Antibiotic", "MIC", "SIR", "Reference",
        ],
        "max_output_tokens": 3000,
        "slots": _STANDARD_SLOTS,
        "guidance": [
            "One row per strain-antibiotic pair. Never combine several "
            "antibiotics or several strains into one row.",
            'Report the MIC exactly as published, including its unit and any '
            'inequality -- "2 µg/mL", ">32 mg/L", "0.25 mg/L". Do not convert units.',
            'Put only the categorical interpretation in SIR, as S, I, or R '
            '(or the published wording such as "Susceptible").',
            "Never infer one of MIC and SIR from the other, and never apply a "
            'breakpoint yourself. If a paper gives only one, put "N/A" in the other.',
            "Give GenBank Accession and BioSample only when the text states them "
            "explicitly (for example CP012345 or SAMN01234567). Do not guess them "
            "from the organism or strain name.",
            'Use the strain designation as published (for example "ATCC 25922", '
            '"H37Rv", "K-12 MG1655"). If a paper reports a pooled result over many '
            'isolates rather than a named strain, put "N/A" in Strain.',
        ],
    },
    # --- the project charter's three relation types --------------------------
    # The MVP names pathogen-host phenotype, pathogen-mechanism-disease and
    # biomarker-disease. The server serves none of them, so they are defined
    # here. Ids are deliberately short and are reused VERBATIM as the scorer's
    # matcher key, the gold-file key and the dedup identity key -- the existing
    # `ppi-extraction` / `ppi` split between the two codebases is the mistake
    # this avoids repeating.
    #
    # Every one carries Assertion and Reference: citation resolution keys off
    # Reference, and without Assertion the evidence-free check has nothing left
    # to inspect once subject and reference columns are removed.
    {
        "id": "phenotype",
        "label": "Pathogen-Host Phenotype",
        "output": "table",
        "version": 1,
        "columns": ["Pathogen", "Host", "Phenotype", "Assertion", "Reference"],
        "max_output_tokens": 3000,
        "slots": _STANDARD_SLOTS,
        "guidance": [
            "Report an observable effect in an infected host: a sign, a "
            "measurement, a survival or weight change, a tissue finding.",
            "One host per row. If a paper reports mice and ferrets, that is two "
            "rows, never one row saying \"mice and ferrets\".",
            'Give the host as published ("BALB/c mice", "ferret", "human"). Do '
            'not generalise a strain to its species or a species to "animal".',
            "Record the direction and magnitude when the paper states them "
            '("reduced weight gain of 15%%", "100%% lethality by day 6").',
            "A phenotype the authors looked for and did NOT find is still a "
            'finding: record it as stated ("no weight loss observed"), never as '
            "its opposite, and never omit the negation.",
            "Do not record a phenotype the paper only hypothesises or proposes "
            "for future work.",
        ],
    },
    {
        "id": "mechanism",
        "label": "Pathogen-Mechanism-Disease",
        "output": "table",
        "version": 1,
        "columns": ["Pathogen", "Mechanism", "Disease", "Assertion", "Reference"],
        "max_output_tokens": 3000,
        "slots": _STANDARD_SLOTS,
        "guidance": [
            "Record how the pathogen causes the disease, not merely that it is "
            "associated with it. A bare association is not a mechanism.",
            "Mechanism must name a molecular or cellular process -- what acts on "
            'what. "Disrupts epithelial tight junctions via protein Y" is a '
            'mechanism; "causes severe disease" is a restatement of the disease '
            "and must not be used.",
            "Name the responsible factor (a protein, toxin, gene or structure) "
            "whenever the paper identifies one.",
            "One mechanism per row. A pathogen acting through two distinct "
            "pathways produces two rows.",
            "Do not record a mechanism the authors propose without evidence, or "
            "one demonstrated only in a different pathogen.",
        ],
    },
    {
        "id": "biomarker",
        "label": "Biomarker-Disease",
        "output": "table",
        "version": 1,
        "columns": ["Biomarker", "Disease", "Association", "Assertion", "Reference"],
        "max_output_tokens": 3000,
        "slots": _STANDARD_SLOTS,
        "guidance": [
            "A biomarker is a measurable signal -- a gene, transcript, protein, "
            "metabolite, glycan or cell count -- linked to a disease state.",
            'Association must state the DIRECTION in the disease state: '
            '"increased", "decreased", "unchanged", or "no association". Add the '
            'clinical context after it when the paper gives one ("increased in '
            'severe cases").',
            '"Unchanged" and "no association" are real findings. Record them '
            "rather than dropping the row: a biomarker that failed to separate "
            "cases from controls is exactly what a curator needs to know.",
            "Give the biomarker as the paper names it, keeping the standard "
            'symbol where there is one ("IL-6", not "interleukin six").',
            "One biomarker and one disease per row.",
            "Do not infer a direction the paper does not state. If it reports "
            'only that a biomarker was "detected", the Association is "N/A".',
        ],
    },
]


def _hash(decl: Dict[str, Any]) -> str:
    """A stable hash of the declaration, so provenance works for local types too."""
    payload = json.dumps(decl, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _finalize(decl: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(decl)
    prepared.setdefault("output", "table")
    prepared.setdefault("version", 1)
    prepared.setdefault("slots", _STANDARD_SLOTS)
    prepared.setdefault("label", prepared["id"])
    prepared["source"] = "local"
    # Hash the declaration without the hash field itself.
    prepared["hash"] = _hash({k: v for k, v in prepared.items() if k != "hash"})
    return prepared


def load_user_templates(path: Path = USER_TEMPLATES_PATH) -> List[Dict[str, Any]]:
    """Read extra declarations from the user's config, if any.

    Shape mirrors the server's: [[templates]] with id, label, output, columns,
    and optional guidance lines.
    """
    if not path.is_file() or tomllib is None:
        return []
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        return []
    found = data.get("templates", [])
    return [t for t in found if isinstance(t, dict) and t.get("id")]


def declarations(path: Path = USER_TEMPLATES_PATH) -> List[Dict[str, Any]]:
    """Every locally-defined template declaration."""
    combined = list(BUILTIN)
    seen = {t["id"] for t in combined}
    for decl in load_user_templates(path):
        if decl["id"] not in seen:
            combined.append(decl)
            seen.add(decl["id"])
    return [_finalize(d) for d in combined]
