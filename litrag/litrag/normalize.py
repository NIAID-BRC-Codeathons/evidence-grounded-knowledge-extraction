"""Value normalization for deduplication.

A live katG query returned S315T, Ser315Thr, and katG-S315T as three separate
rows -- one fact in three notations. Exact-string dedup keeps all three, so
identity comparison happens on normalized values instead.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

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


# Gene symbols that collide with a null marker. Influenza's neuraminidase is
# "NA" and its NADH-dehydrogenase genes are "ND", both of which clean() would
# otherwise erase -- silently emptying the protein column of every NA row.
# "N/A" stays null: only the bare symbol is rescued.
SYMBOL_NOT_NULL = {"na", "nd"}


def clean_symbol(value: Optional[str]) -> str:
    """clean(), but keeps gene symbols that look like null markers.

    Use for gene and protein columns only. Elsewhere "NA" really does mean
    not-available and must keep nulling.
    """
    if value is None:
        return ""
    collapsed = _WHITESPACE.sub(" ", str(value)).strip()
    if collapsed.lower() in SYMBOL_NOT_NULL:
        return collapsed
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
    cleaned = clean_symbol(value)
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

    # Written out in words: "NS1-53 glycine to aspartate" is G53D, and must
    # merge with a row that wrote G53D.
    prose = _prose_notation(token)
    if prose:
        return prose

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


# Full residue names, for mutations written out in words rather than notation.
AA_FULL_TO_ONE = {
    "alanine": "A", "arginine": "R", "asparagine": "N",
    "aspartate": "D", "aspartic acid": "D", "aspartic-acid": "D",
    "cysteine": "C", "glutamine": "Q",
    "glutamate": "E", "glutamic acid": "E", "glutamic-acid": "E",
    "glycine": "G", "histidine": "H", "isoleucine": "I", "leucine": "L",
    "lysine": "K", "methionine": "M", "phenylalanine": "F", "proline": "P",
    "serine": "S", "threonine": "T", "tryptophan": "W", "tyrosine": "Y",
    "valine": "V", "selenocysteine": "U", "pyrrolysine": "O",
    "stop": "*", "termination": "*",
}
NT_FULL_TO_ONE = {
    "adenine": "A", "cytosine": "C", "guanine": "G",
    "thymine": "T", "thymidine": "T", "uracil": "U",
}

# Papers mix full names and three-letter codes: "glycine to aspartate" and
# "NS1-53 Gly-to-Asp" are the same substitution written two ways.
_AA_NAMES: Dict[str, str] = dict(AA_FULL_TO_ONE)
_AA_NAMES.update(AA_THREE_TO_ONE)

_AA_NAME = "|".join(sorted(_AA_NAMES, key=len, reverse=True))
_NT_NAME = "|".join(sorted(NT_FULL_TO_ONE, key=len, reverse=True))
# The separator may be hyphenated -- "Gly-to-Asp" -- as well as spaced.
_TO = r"(?:\s*[-–]?\s*(?:to|->|-->|→|>|into|by|for)\s*[-–]?\s*)"

# A leading gene, protein or region qualifier: "NS1-53 ...", "5'UTR-57, ...".
_QUALIFIED_POS = (
    r"(?:(?P<prefix>[A-Za-z0-9'\u2019/]+(?:\s+[A-Za-z0-9'\u2019/]+){0,2})"
    r"\s*[-–:]\s*)?(?P<pos>-?\d+)"
)

# "NS1-53 glycine to aspartate", "glycine 53 to aspartate", "Gly53 to Asp"
_PROSE_POS_FIRST = re.compile(
    rf"^{_QUALIFIED_POS}[\s,]*(?P<ref>{_AA_NAME})" + _TO + rf"(?P<alt>{_AA_NAME})$",
    re.IGNORECASE,
)
# "glycine 53 to aspartate", "glycine-53 to aspartate"
_PROSE_REF_FIRST = re.compile(
    rf"^(?P<ref>{_AA_NAME})\s*[-–]?\s*(?P<pos>-?\d+)" + _TO + rf"(?P<alt>{_AA_NAME})$",
    re.IGNORECASE,
)
# "position 315 serine to threonine", "codon 315, serine to threonine"
_PROSE_POSITION_WORD = re.compile(
    rf"^(?:at\s+)?(?:position|codon|residue|amino acid)\s*(?P<pos>-?\d+)[\s,]*"
    rf"(?P<ref>{_AA_NAME})" + _TO + rf"(?P<alt>{_AA_NAME})$",
    re.IGNORECASE,
)
# "glycine to aspartate at position 53"
_PROSE_TRAILING_POS = re.compile(
    rf"^(?P<ref>{_AA_NAME})" + _TO + rf"(?P<alt>{_AA_NAME})\s*"
    rf"(?:at\s+)?(?:position|codon|residue)?\s*(?P<pos>-?\d+)$",
    re.IGNORECASE,
)
# Nucleotides written out, or single letters after a UTR-style qualifier:
# "5'UTR-57, C to T", "57 cytosine to thymine"
_PROSE_NUCLEOTIDE = re.compile(
    rf"^{_QUALIFIED_POS}[\s,]*(?P<ref>{_NT_NAME}|[ACGTU])" + _TO + rf"(?P<alt>{_NT_NAME}|[ACGTU])$",
    re.IGNORECASE,
)

_PROSE_AA_PATTERNS = (
    _PROSE_POSITION_WORD, _PROSE_POS_FIRST, _PROSE_REF_FIRST, _PROSE_TRAILING_POS,
)
# A qualifier naming a non-coding region means the change is nucleotide.
_NONCODING = re.compile(r"utr|^5'?nc$|^3'?nc$|promoter|intron", re.IGNORECASE)
# What a successfully canonicalised mutation looks like: S315T, or n-15T.
_IS_NOTATION = re.compile(r"^(?:[A-Z]|n)-?\d+[A-Z*]$")


def _letter(name: str, table: Dict[str, str]) -> str:
    key = name.strip().lower()
    if key in table:
        return table[key]
    return key.upper() if len(key) == 1 else ""


def _prose_notation(token: str) -> str:
    """Canonical form for a mutation written in words, or "" if it is not one."""
    nucleotide = _PROSE_NUCLEOTIDE.match(token)
    if nucleotide:
        groups = nucleotide.groupdict()
        prefix = groups.get("prefix") or ""
        ref = _letter(groups["ref"], NT_FULL_TO_ONE)
        alt = _letter(groups["alt"], NT_FULL_TO_ONE)
        spelled_out = groups["ref"].lower() in NT_FULL_TO_ONE
        # Single letters are ambiguous -- "C to T" is Cys->Thr as readily as
        # cytosine->thymine -- so accept them only where the qualifier names a
        # non-coding region, or the names were spelled out in full.
        if ref and alt and (spelled_out or _NONCODING.search(prefix)):
            return f"n{int(groups['pos'])}{alt}"

    for pattern in _PROSE_AA_PATTERNS:
        match = pattern.match(token)
        if not match:
            continue
        groups = match.groupdict()
        ref = _letter(groups["ref"], _AA_NAMES)
        alt = _letter(groups["alt"], _AA_NAMES)
        if ref and alt:
            return f"{ref}{int(groups['pos'])}{alt}"
    return ""


def standard_notation(value: Optional[str], gene: Optional[str] = None) -> str:
    """Convert a mutation written in prose into standard notation.

    "NS1-53 glycine to aspartate" becomes G53D, "position 315 serine to
    threonine" becomes S315T, "5'UTR-57, C to T" becomes n57T. Returns "" when
    the text is not a recognisable mutation, so a caller can tell a real
    conversion from a guess.

    Reported against a dengue vaccine paper whose NS1-53 substitution was
    written out in words and so never matched anything written as G53D.
    """
    cleaned = clean(value)
    if not cleaned:
        return ""

    token = re.sub(r"\s+", " ", cleaned.strip(" .;")).strip()
    # Drop a leading gene qualifier when it repeats the gene we already know.
    gene_key = normalize_gene(gene)
    if gene_key:
        token = re.sub(rf"^{re.escape(gene_key)}\s*[-–:\s]\s*", "", token,
                       flags=re.IGNORECASE).strip()

    prose = _prose_notation(token)
    if prose:
        return prose

    # Already written in a notation the normalizer understands.
    canonical = normalize_mutation(cleaned, gene)
    return canonical if _IS_NOTATION.match(canonical) else ""


# The residue a linkage type implies. N-linked glycosylation occurs on
# asparagine by definition, so a bare position under it is an N. O-linked sits
# on serine or threonine and C-mannosylation on tryptophan -- only the last of
# those is unambiguous, so only N and W are filled in.
LINKAGE_RESIDUE = {"N-linked": "N", "C-mannosylation": "W"}


def site_with_residue(site: Optional[str], glyco_type: Optional[str] = None) -> str:
    """Add the residue letter to a site written as a bare position.

    Papers routinely list influenza HA glycosites as "42, 44, 50". The residue
    is not missing information under an N-linked heading -- it is asparagine by
    definition -- so it can be supplied. Returns "" when the site already names
    a residue, or when the linkage does not imply one (O-linked is Ser or Thr).
    """
    cleaned = clean(site)
    if not cleaned:
        return ""
    bare = re.fullmatch(r"(-?\d+)", cleaned.strip())
    if not bare:
        return ""
    residue = LINKAGE_RESIDUE.get(normalize_glyco_type(glyco_type))
    if not residue:
        return ""
    return f"{residue}{int(bare.group(1))}"
