"""Value normalization for deduplication.

A live katG query returned S315T, Ser315Thr, and katG-S315T as three separate
rows -- one fact in three notations. Exact-string dedup keeps all three, so
identity comparison happens on normalized values instead.
"""

from __future__ import annotations

import re
from typing import Optional

# Values the model uses for "nothing here". Canonicalized so a row of them is
# recognizable as empty regardless of which spelling came back.
NULL_TOKENS = {
    "", "n/a", "na", "n.a.", "none", "null", "-", "--", "not available",
    "not applicable", "not reported", "not specified", "unknown", "unspecified",
    "nd", "n/d",
}

AA_THREE_TO_ONE = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
    "ter": "*", "stop": "*", "sec": "U", "pyl": "O",
}

# Ser315Thr / p.Ser315Thr  ->  three-letter substitution
_THREE_LETTER_SUB = re.compile(
    r"^(?:p\.)?(" + "|".join(AA_THREE_TO_ONE) + r")(\d+)(" + "|".join(AA_THREE_TO_ONE) + r")$",
    re.IGNORECASE,
)
# S315T / p.S315T  ->  one-letter substitution
_ONE_LETTER_SUB = re.compile(r"^(?:p\.)?([A-Z])(\d+)([A-Z*])$", re.IGNORECASE)
# c-15t / g-10a  ->  ref base, position, alt base. The position may be negative
# (promoter numbering), so the sign must survive: c-15t and c15t are different
# sites. Nucleotide notation is conventionally lowercase, which is what
# distinguishes it from a one-letter amino-acid substitution like C15T.
_NUCLEOTIDE_SUB = re.compile(r"^([acgt])\s*(-?\d+)\s*([acgt])$", re.IGNORECASE)
# c.-15C>T / c.215A>G  ->  HGVS coding notation
_HGVS_CODING = re.compile(
    r"^c\.\s*(-?\d+)\s*([acgt])\s*>\s*([acgt])$", re.IGNORECASE
)
# c.-15T -- HGVS with the reference base omitted.
_HGVS_SHORT = re.compile(r"^c\.\s*(-?\d+)\s*([acgt])$", re.IGNORECASE)

_WHITESPACE = re.compile(r"\s+")
# Ser315->Thr, S315→T, Ser315 > Thr -- an arrow between position and alt allele.
_ARROW = re.compile(r"\s*(?:->|-->|→|⟶|>)\s*")
# "Ser315Thr (S315T)" -- a parenthetical restating the same variant in the other
# notation. Qwen emits these routinely; without stripping, the gloss defeats
# every substitution pattern and the row will not merge with its plain twin.
_GLOSS = re.compile(r"\s*[\(\[]([^)\]]*)[\)\]]\s*$")


def is_null(value: Optional[str]) -> bool:
    """True when a cell carries no information."""
    if value is None:
        return True
    return _WHITESPACE.sub(" ", str(value)).strip().lower() in NULL_TOKENS


def clean(value: Optional[str]) -> str:
    """Collapse whitespace and map every null spelling to a single empty string."""
    if value is None:
        return ""
    collapsed = _WHITESPACE.sub(" ", str(value)).strip()
    return "" if collapsed.lower() in NULL_TOKENS else collapsed


def normalize_text(value: Optional[str]) -> str:
    """Case- and punctuation-insensitive key for free-text comparison."""
    cleaned = clean(value).lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def normalize_gene(value: Optional[str]) -> str:
    """Gene/protein identity key.

    Strips the decorations that vary between papers -- "gene", "protein",
    italics markers -- so katG, katG gene, and KatG protein agree.
    """
    cleaned = clean(value)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s*\b(gene|protein|orf|cds)\b\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" .,;:_-*")
    return cleaned.lower()


def normalize_mutation(value: Optional[str], gene: Optional[str] = None) -> str:
    """Canonical form for a mutation, independent of notation.

    Ser315Thr, S315T, and katG-S315T all reduce to "S315T". Anything that does
    not parse as a known notation falls back to normalized text, which still
    collapses case and spacing differences.
    """
    cleaned = clean(value)
    if not cleaned:
        return ""

    token = cleaned.strip()

    # Collapse arrow notation, but not in HGVS c. form where ">" is part of the
    # grammar (c.-15C>T) and is handled by its own pattern below.
    if not token.lower().startswith("c."):
        token = _ARROW.sub("", token)

    # Drop a leading gene qualifier: "katG-S315T", "katG S315T", "katG:S315T".
    gene_key = normalize_gene(gene)
    if gene_key:
        token = re.sub(
            rf"^{re.escape(gene_key)}\s*[-_:. ]\s*", "", token, flags=re.IGNORECASE
        ).strip()
    # Or any leading word that is clearly a qualifier before a substitution.
    token = re.sub(
        r"^[A-Za-z][A-Za-z0-9]{1,9}\s*[-_:]\s*(?=(?:p\.)?[A-Za-z]{1,3}\d+[A-Za-z*]{1,3}$)",
        "",
        token,
    ).strip()

    nuc = _NUCLEOTIDE_SUB.match(token)
    if nuc:
        _ref, pos, alt = nuc.groups()
        # A1B is ambiguous: lowercase (a1b) is a base change by convention, and
        # a negative position is one necessarily -- proteins have no residue
        # -15. Uppercase at a positive position is an amino-acid substitution
        # and is left to the pattern below.
        if token.islower() or int(pos) < 0:
            return f"n{int(pos)}{alt.upper()}"

    gloss = _GLOSS.search(token)
    if gloss:
        head = _GLOSS.sub("", token).strip()
        for candidate in (head, gloss.group(1).strip()):
            if not candidate:
                continue
            parsed = normalize_mutation(candidate, gene)
            # Only accept a candidate that matched a real notation; otherwise
            # fall through and keep the full text.
            if parsed and parsed != normalize_text(candidate):
                return parsed

    three = _THREE_LETTER_SUB.match(token)
    if three:
        ref, pos, alt = three.groups()
        return f"{AA_THREE_TO_ONE[ref.lower()]}{int(pos)}{AA_THREE_TO_ONE[alt.lower()]}"

    one = _ONE_LETTER_SUB.match(token)
    if one:
        ref, pos, alt = one.groups()
        return f"{ref.upper()}{int(pos)}{alt.upper()}"

    hgvs = _HGVS_CODING.match(token)
    if hgvs:
        pos, _ref, alt = hgvs.groups()
        return f"n{int(pos)}{alt.upper()}"

    short = _HGVS_SHORT.match(token)
    if short:
        pos, alt = short.groups()
        return f"n{int(pos)}{alt.upper()}"

    return normalize_text(token)


# "Resistant", "R", "resistant (R)" all mean the same category. Without this a
# merge reports a disagreement that is only a difference in wording.
_SIR_WORDS = {
    "s": "S", "susceptible": "S", "sensitive": "S", "susceptible (s)": "S",
    "i": "I", "intermediate": "I", "intermediate (i)": "I",
    "susceptible dose dependent": "I", "sdd": "I",
    "r": "R", "resistant": "R", "resistant (r)": "R",
    "nonsusceptible": "R", "non-susceptible": "R",
}

# Salts and formulations do not change which drug was tested.
_ANTIBIOTIC_SUFFIX = re.compile(
    r"\s*\b(sodium|potassium|hydrochloride|hcl|sulfate|sulphate|succinate|"
    r"mesylate|tartrate|trihydrate|dihydrate|monohydrate|disc|disk)\b\s*$",
    re.IGNORECASE,
)


def normalize_sir(value: Optional[str]) -> str:
    """Canonical S / I / R, or normalized text when it is something else."""
    cleaned = clean(value)
    if not cleaned:
        return ""
    key = re.sub(r"[^\w\s()-]", "", cleaned).strip().lower()
    if key in _SIR_WORDS:
        return _SIR_WORDS[key]
    collapsed = normalize_text(cleaned)
    return _SIR_WORDS.get(collapsed, collapsed)


def normalize_antibiotic(value: Optional[str]) -> str:
    """Antibiotic identity, ignoring case, salt form, and a trailing dose."""
    cleaned = clean(value)
    if not cleaned:
        return ""
    # Drop a parenthetical abbreviation or concentration: "Ciprofloxacin (CIP)".
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned)
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _ANTIBIOTIC_SUFFIX.sub("", cleaned).strip()
    return normalize_text(cleaned)


# Asn234 / N234 / Asn-234 / N-234 / p.Asn234 -- a residue and its position.
_SITE = re.compile(
    r"^(?:p\.)?(" + "|".join(AA_THREE_TO_ONE) + r"|[A-Z])[\s\-]?(\d+)$",
    re.IGNORECASE,
)

_GLYCO_TYPES = {
    "n": "N-linked", "n linked": "N-linked", "nlinked": "N-linked",
    "n glycosylation": "N-linked", "n glycan": "N-linked",
    "n linked glycosylation": "N-linked", "asn linked": "N-linked",
    "o": "O-linked", "o linked": "O-linked", "olinked": "O-linked",
    "o glycosylation": "O-linked", "o glycan": "O-linked",
    "o linked glycosylation": "O-linked",
    "c mannosylation": "C-mannosylation", "c mannosylated": "C-mannosylation",
    "c linked": "C-mannosylation", "c mannose": "C-mannosylation",
}


def normalize_site(value: Optional[str]) -> str:
    """Canonical residue-and-position for a modification site.

    Asn234, N234 and Asn-234 are one site. The position is never rewritten:
    glycosite numbering differs between isoforms and strains, so N234 and N235
    must stay distinct even when they are the same residue in two constructs.
    """
    cleaned = clean(value)
    if not cleaned:
        return ""
    match = _SITE.match(cleaned.strip())
    if not match:
        return normalize_text(cleaned)
    residue, position = match.groups()
    letter = AA_THREE_TO_ONE.get(residue.lower(), residue.upper())
    return f"{letter}{int(position)}"


def normalize_glyco_type(value: Optional[str]) -> str:
    """Canonical N-linked / O-linked / C-mannosylation."""
    cleaned = clean(value)
    if not cleaned:
        return ""
    key = normalize_text(cleaned)
    if key in _GLYCO_TYPES:
        return _GLYCO_TYPES[key]
    key = re.sub(r"\b(glycosylation|glycosylated|glycan|site|linked)\b", " ", key).strip()
    key = _WHITESPACE.sub(" ", key)
    return _GLYCO_TYPES.get(key, normalize_text(cleaned))
