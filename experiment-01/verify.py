"""Cite-or-refuse verification gate for slice 1 extraction rows.

A row proposed by the model carries a chunk_id and a verbatim quote. This
module decides whether that quote is actually supported by the chunk text,
and whether the row's other fields are well formed and on-spec, before the
row is allowed to reach a curator. See CONTRACT.md for the row shape, the
outcome values, and the fixed omit-reason set. Standard library only.
Prints nothing on import.
"""

import difflib
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUZZY_THRESHOLD = 0.92
MIN_QUOTE_LEN = 40

# Hedge and negation words checked against the quote itself, per CONTRACT.md.
# Kept visible at module level, not buried in a function.
HEDGE_WORDS = [
    "may",
    "might",
    "could",
    "suggests",
    "appears",
    "potentially",
    "unclear",
    "we hypothesize",
    "hypothesize",
    "hypothesized",
]

NEGATION_WORDS = [
    "not",
    "no evidence",
    "failed to",
]

DISPLAY_FIELDS = {
    "mutation": ["organism", "gene_name", "mutation", "phenotype", "assertion", "reference"],
    "ppi": ["pathogen", "protein_a", "protein_b", "interaction_type", "method", "assertion", "reference"],
}

OMIT_REASONS = {
    "unknown_chunk",
    "quote_mismatch",
    "schema_fail",
    "wrong_organism",
    "wrong_gene",
    "duplicate",
}

_ZERO_WIDTH_CHARS = ["​", "‌", "‍", "﻿"]

_CURLY_QUOTES = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
}

_DASHES = {
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
}

_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize(text):
    """Casefold, collapse whitespace, unify quotes/dashes, strip zero-width chars."""
    if text is None:
        return ""
    for zw in _ZERO_WIDTH_CHARS:
        text = text.replace(zw, "")
    for src, dst in _CURLY_QUOTES.items():
        text = text.replace(src, dst)
    for src, dst in _DASHES.items():
        text = text.replace(src, dst)
    text = text.casefold()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Quote support
# ---------------------------------------------------------------------------

def _fuzzy_best_ratio(norm_quote, norm_chunk):
    """Best SequenceMatcher ratio of norm_quote against any window of norm_chunk."""
    qlen = len(norm_quote)
    clen = len(norm_chunk)
    if qlen == 0 or clen == 0:
        return 0.0
    if clen <= qlen:
        return difflib.SequenceMatcher(None, norm_quote, norm_chunk, autojunk=False).ratio()
    step = max(1, qlen // 10)
    best = 0.0
    last_start = clen - qlen
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        window = norm_chunk[start:start + qlen]
        ratio = difflib.SequenceMatcher(None, norm_quote, window, autojunk=False).ratio()
        if ratio > best:
            best = ratio
            if best >= 1.0:
                break
    return best


def quote_supported(quote, chunk_text):
    """Try exact, then normalized, then fuzzy window matching.

    Returns (ok, method, ratio, offset). offset is the exact-match character
    position, or None otherwise. method is one of "exact", "normalized",
    "fuzzy", "below_threshold", "too_short".
    """
    if not quote or len(quote) < MIN_QUOTE_LEN:
        return False, "too_short", 0.0, None
    chunk_text = chunk_text or ""

    if quote in chunk_text:
        return True, "exact", 1.0, chunk_text.index(quote)

    norm_quote = normalize(quote)
    norm_chunk = normalize(chunk_text)
    if norm_quote and norm_quote in norm_chunk:
        return True, "normalized", 1.0, None

    ratio = _fuzzy_best_ratio(norm_quote, norm_chunk)
    if ratio >= FUZZY_THRESHOLD:
        return True, "fuzzy", ratio, None
    return False, "below_threshold", ratio, None


# ---------------------------------------------------------------------------
# Hedge detection
# ---------------------------------------------------------------------------

def _is_hedged(text):
    norm = " " + normalize(text) + " "
    for word in HEDGE_WORDS + NEGATION_WORDS:
        needle = " " + normalize(word) + " "
        if needle in norm:
            return True
    return False


# ---------------------------------------------------------------------------
# Row shape helpers
# ---------------------------------------------------------------------------

def _infer_data_type(row):
    if "gene_name" in row or "mutation" in row:
        return "mutation"
    if "pathogen" in row or "protein_a" in row:
        return "ppi"
    return None


def _missing_required_fields(row):
    data_type = _infer_data_type(row)
    missing = []
    fields = DISPLAY_FIELDS.get(data_type, [])
    if data_type is None:
        missing.append("data_type (unrecognized row shape)")
    for field in fields:
        if row.get(field) in (None, ""):
            missing.append(field)
    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        missing.append("evidence")
    else:
        for field in ("chunk_id", "passage"):
            if evidence.get(field) in (None, ""):
                missing.append("evidence.%s" % field)
    return missing, data_type


def _norm_field(value):
    return normalize(str(value)) if value not in (None, "") else ""


def _organism_gene_check(row, spec):
    """Return (omit_reason, detail) or (None, None) when the row is on-spec."""
    data_type = _infer_data_type(row)
    spec_organism = _norm_field(spec.get("organism", ""))
    gene = spec.get("gene", "")
    aliases = spec.get("aliases") or ([gene] if gene else [])
    norm_aliases = {_norm_field(a) for a in aliases if a}

    if data_type == "mutation":
        row_organism = _norm_field(row.get("organism"))
        if spec_organism and row_organism != spec_organism:
            return "wrong_organism", "organism %r does not match query organism %r" % (
                row.get("organism"), spec.get("organism"))
        row_gene = _norm_field(row.get("gene_name"))
        if norm_aliases and row_gene not in norm_aliases:
            return "wrong_gene", "gene_name %r does not match query gene/aliases %r" % (
                row.get("gene_name"), aliases)
    elif data_type == "ppi":
        row_pathogen = _norm_field(row.get("pathogen"))
        if spec_organism and row_pathogen != spec_organism:
            return "wrong_organism", "pathogen %r does not match query organism %r" % (
                row.get("pathogen"), spec.get("organism"))
        if norm_aliases:
            row_a = _norm_field(row.get("protein_a"))
            row_b = _norm_field(row.get("protein_b"))
            if row_a not in norm_aliases and row_b not in norm_aliases:
                return "wrong_gene", "neither protein_a %r nor protein_b %r matches query gene/aliases %r" % (
                    row.get("protein_a"), row.get("protein_b"), aliases)
    return None, None


def _duplicate_key(row):
    data_type = _infer_data_type(row)
    fields = [f for f in DISPLAY_FIELDS.get(data_type, []) if f != "assertion"]
    values = tuple(_norm_field(row.get(f)) for f in fields)
    evidence = row.get("evidence")
    chunk_id = evidence.get("chunk_id") if isinstance(evidence, dict) else None
    return (data_type, values, chunk_id)


def _chunk_text(chunk):
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return chunk.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Public verification API
# ---------------------------------------------------------------------------

def verify_row(row, chunks_by_id):
    """Verify one row against its cited chunk only (no spec, no dedupe).

    Returns (verdict, detail) where verdict is "cite", "qualify", or an
    omit reason from the fixed set. This is the schema and quote gate;
    verify_rows layers the spec-dependent and duplicate checks on top.
    """
    if not isinstance(row, dict):
        return "schema_fail", "row is not an object"

    missing, _data_type = _missing_required_fields(row)
    if missing:
        return "schema_fail", "missing required fields: " + ", ".join(missing)

    evidence = row["evidence"]
    chunk_id = evidence["chunk_id"]
    chunk = chunks_by_id.get(chunk_id)
    if chunk is None:
        return "unknown_chunk", "chunk_id %r not found in supplied papers" % (chunk_id,)

    quote = evidence["passage"]
    ok, method, ratio, offset = quote_supported(quote, _chunk_text(chunk))
    if not ok:
        return "quote_mismatch", "quote not supported (%s, ratio=%.2f)" % (method, ratio)

    evidence["passage_char_offset"] = offset if method == "exact" else None

    if _is_hedged(quote):
        return "qualify", "hedged quote (%s, ratio=%.2f)" % (method, ratio)
    return "cite", "verified quote (%s, ratio=%.2f)" % (method, ratio)


def verify_rows(rows, chunks_by_id, spec):
    """Apply schema, spec, duplicate, and quote checks in order to every row.

    Returns (accepted, omitted). Accepted rows carry outcome "cite" or
    "qualify" and have evidence.passage_char_offset filled on an exact
    match. Omitted rows carry outcome "omit", omit_reason, and detail.
    """
    accepted = []
    omitted = []
    seen = set()

    for row in rows:
        if not isinstance(row, dict):
            omitted.append({"outcome": "omit", "omit_reason": "schema_fail", "detail": "row is not an object"})
            continue

        missing, _data_type = _missing_required_fields(row)
        if missing:
            rec = dict(row)
            rec["outcome"] = "omit"
            rec["omit_reason"] = "schema_fail"
            rec["detail"] = "missing required fields: " + ", ".join(missing)
            omitted.append(rec)
            continue

        evidence = row["evidence"]
        chunk_id = evidence["chunk_id"]
        if chunk_id not in chunks_by_id:
            rec = dict(row)
            rec["outcome"] = "omit"
            rec["omit_reason"] = "unknown_chunk"
            rec["detail"] = "chunk_id %r not found in supplied papers" % (chunk_id,)
            omitted.append(rec)
            continue

        reason, detail = _organism_gene_check(row, spec)
        if reason:
            rec = dict(row)
            rec["outcome"] = "omit"
            rec["omit_reason"] = reason
            rec["detail"] = detail
            omitted.append(rec)
            continue

        dup_key = _duplicate_key(row)
        if dup_key in seen:
            rec = dict(row)
            rec["outcome"] = "omit"
            rec["omit_reason"] = "duplicate"
            rec["detail"] = "identical row already accepted in this run"
            omitted.append(rec)
            continue

        verdict, detail = verify_row(row, chunks_by_id)
        rec = dict(row)
        if verdict in ("cite", "qualify"):
            rec["outcome"] = verdict
            accepted.append(rec)
            seen.add(dup_key)
        else:
            rec["outcome"] = "omit"
            rec["omit_reason"] = verdict
            rec["detail"] = detail
            omitted.append(rec)

    return accepted, omitted


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CHUNK_ID = "chunk-pb2-0001"
    CHUNK_TEXT = (
        "The PB2 subunit of the influenza A virus polymerase complex plays a central role "
        "in host adaptation. Mutation E627K in PB2 increases polymerase activity in "
        "mammalian cells and is frequently observed in human-adapted avian influenza "
        "strains. Structural work indicates this substitution may enhance the interaction "
        "with host factor ANP32A in some cell lines, though the precise mechanism remains "
        "an open question for the field. Additional residues such as D701N have also been "
        "implicated in similar phenotypes across several viral lineages studied to date."
    )
    CHUNKS_BY_ID = {CHUNK_ID: CHUNK_TEXT}
    SPEC = {
        "organism": "Influenza A virus",
        "gene": "PB2",
        "aliases": ["PB2", "polymerase basic 2", "polymerase basic protein 2"],
        "data_type": "mutation",
    }

    results = []

    def check(label, condition):
        results.append((label, condition))
        print("PASS" if condition else "FAIL", "-", label)

    def make_row(quote, row_id="pb2__mutation__0001"):
        return {
            "row_id": row_id,
            "organism": "Influenza A virus",
            "gene_name": "PB2",
            "mutation": "E627K",
            "phenotype": "increased polymerase activity in mammalian cells",
            "assertion": "provisional, pending the leads",
            "reference": "Subbarao 1993, J Virol",
            "evidence": {
                "chunk_id": CHUNK_ID,
                "doc_id": "doc-1",
                "passage": quote,
                "passage_char_offset": None,
            },
        }

    # Case 1: verbatim quote accepted as cite.
    exact_quote = "Mutation E627K in PB2 increases polymerase activity in mammalian cells"
    assert exact_quote in CHUNK_TEXT, "fixture quote must appear verbatim in the chunk"
    verdict, detail = verify_row(make_row(exact_quote), CHUNKS_BY_ID)
    check("verbatim quote accepted as cite", verdict == "cite")

    # Case 2: same quote with one word changed, rejected as quote_mismatch.
    changed_quote = "Mutation E627K in PB2 abolishes polymerase activity in mammalian cells"
    verdict, detail = verify_row(make_row(changed_quote), CHUNKS_BY_ID)
    check("altered quote rejected as quote_mismatch (%s)" % detail, verdict == "quote_mismatch")

    # Case 3: same quote with different spacing and casing (a curly-quote unification
    # runs through the same normalize() path; this chunk has no literal quote marks
    # to vary), accepted as normalized.
    spaced_quote = "MUTATION  E627K  IN PB2   increases  polymerase   activity in mammalian cells"
    verdict, detail = verify_row(make_row(spaced_quote), CHUNKS_BY_ID)
    check("respaced/recased quote accepted as normalized (%s)" % detail,
          verdict == "cite" and "normalized" in detail)

    # Case 4: quote citing a chunk id that does not exist, rejected as unknown_chunk.
    row_bad_chunk = make_row(exact_quote)
    row_bad_chunk["evidence"]["chunk_id"] = "chunk-does-not-exist"
    verdict, detail = verify_row(row_bad_chunk, CHUNKS_BY_ID)
    check("unknown chunk id rejected as unknown_chunk", verdict == "unknown_chunk")

    # Case 5: a 20-character quote rejected as too_short.
    short_quote = "PB2 E627K increases "
    check("20-char quote length is under the floor", len(short_quote) < MIN_QUOTE_LEN)
    verdict, detail = verify_row(make_row(short_quote), CHUNKS_BY_ID)
    check("short quote rejected as too_short", verdict == "quote_mismatch" and "too_short" in detail)

    # Case 6: hedged sentence returns qualify rather than cite.
    hedged_quote = "this substitution may enhance the interaction with host factor ANP32A in some cell lines"
    assert hedged_quote in CHUNK_TEXT, "fixture hedged quote must appear verbatim in the chunk"
    verdict, detail = verify_row(make_row(hedged_quote), CHUNKS_BY_ID)
    check("hedged quote returns qualify", verdict == "qualify")

    # Case 7: the same row submitted twice returns duplicate on the second, via verify_rows.
    dup_rows = [make_row(exact_quote, row_id="pb2__mutation__0002"),
                make_row(exact_quote, row_id="pb2__mutation__0003")]
    accepted, omitted = verify_rows(dup_rows, CHUNKS_BY_ID, SPEC)
    check("first of two identical rows accepted",
          len(accepted) == 1 and accepted[0]["outcome"] == "cite")
    check("second of two identical rows omitted as duplicate",
          len(omitted) == 1 and omitted[0]["omit_reason"] == "duplicate")

    print()
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print("%d/%d checks passed" % (passed, total))
    if passed != total:
        raise SystemExit(1)
