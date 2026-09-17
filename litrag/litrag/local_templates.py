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
    {
        "id": "glycosylation",
        "label": "Glycosylation Sites",
        "output": "table",
        "version": 1,
        "columns": [
            "Organism", "Protein", "Strain", "Site", "Glycosylation Type",
            "Glycan", "Method", "Effect", "Assertion", "Reference",
        ],
        "max_output_tokens": 3000,
        "slots": _STANDARD_SLOTS,
        "guidance": [
            "One row per glycosylation site. Never combine several sites, "
            "several proteins, or a range of positions into one row.",
            "Always name the protein the site is on — HA, NA, Spike, gp120. A "
            "site without a protein cannot be interpreted, so if a passage "
            "does not name one, do not report the site.",
            "Give the strain or isolate whenever the source names one, as it "
            "is written: A/California/07/2009 (H1N1), H3N2 A/Hong Kong/1/68, "
            'Wuhan-Hu-1. Use "N/A" when the source reports a site without '
            "tying it to a strain. Do not infer a strain from the organism.",
            'Write the site as the residue letter plus its position, as the '
            'source numbers it — N234, Asn234, T678. Include the residue; a '
            'bare number does not identify a site.',
            "Use the position numbering the source uses and never renumber. "
            "Glycosite numbering differs between isoforms, strains and "
            "constructs, so a renumbered position is a different site — which "
            "is why the strain matters.",
            'Glycosylation Type is N-linked, O-linked or C-mannosylation, and '
            'only when the source states it. Do not infer the type from the '
            'residue alone.',
            'Record the Glycan as published — high-mannose, complex, hybrid, '
            'Man5GlcNAc2, core-fucosylated. Do not convert between '
            'nomenclatures, and use "N/A" when the glycan was not characterised.',
            "Method is how the site was established: mass spectrometry, "
            "site-directed mutagenesis, lectin binding, cryo-EM or "
            "crystallography, or sequence-based prediction.",
            "Effect is the functional consequence the source reports for that "
            'site — folding, receptor binding, antibody shielding. Use "N/A" '
            "when none is reported; do not supply one from general knowledge.",
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
