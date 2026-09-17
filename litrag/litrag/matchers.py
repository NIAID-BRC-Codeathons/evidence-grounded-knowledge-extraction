"""Gold matchers for the three charter relation types.

MERGED, 2026-09-17. This logic now ALSO lives in experiment-01/evaluate.py and
experiment-01/schemas.py. Two copies exist on purpose -- this one is the tested
one (experiment-01 has a self-test, not a pytest suite), and evaluate.py needs
its own because it must run standalone and stdlib-only.

Duplication that nothing checks is duplication that diverges, so
tests/test_matchers.py compares the two copies on identical inputs and fails if
either is edited alone. If you change a matcher here, change it there.

The original merge instructions, kept for reference:

  1. Copy everything below the MERGE LINE into evaluate.py, after `ppi_match`.
  2. Delete the local `_ppi_token` here -- evaluate.py already defines an
     identical one at :96-99. Do not end up with two normalizers.
  3. Extend the existing registry:
         MATCHERS = {"mutation": mutation_match, "ppi": ppi_match,
                     "phenotype": phenotype_match, "mechanism": mechanism_match,
                     "biomarker": biomarker_match}
  4. Add the three DATA_TYPES entries from `SCHEMA_ADDITIONS` to
     experiment-01/schemas.py. Without them `new_row()` raises ValueError and
     `validate_row()` rejects every row, so nothing emits at all.

Stdlib only, matching evaluate.py. No imports from litrag: this file has to run
inside the other codebase.

Why matching these types is hard, stated up front. Two of the three slots in
each relation are entities and compare cleanly. The third -- Phenotype,
Mechanism, Association -- is free text a model wrote, and gold is free text a
human wrote. Exact equality scores ~0 and tells you nothing about the
extractor. So the entity slots are matched hard and the free-text slot softly,
which is a deliberate trade: recall is bounded by vocabulary overlap, and
precision can be inflated by coincidental word sharing. Both failure modes are
real and are named in the tests.
"""

import re

# ---------------------------------------------------------------- MERGE LINE

_PUNCT_RE = re.compile(r"[^a-z0-9]")


def _ppi_token(s):
    """Lowercase and strip punctuation. Duplicate of evaluate.py:96 -- see note."""
    if not s:
        return ""
    return _PUNCT_RE.sub("", s.lower())


# Words that carry no discriminating meaning in a biomedical phrase. "increased"
# and "decreased" are deliberately NOT here: direction is the finding.
_STOPWORDS = frozenset("""
a an the of in on at to and or with by via from for as is are was were be been
that this these those its their it results result showed shown demonstrate
demonstrated observed found study patients cases
""".split())

# Explicit negators only. "loss" is deliberately NOT here: "weight loss" is a
# positive finding, and treating it as a negation made "no weight loss" and
# "weight loss" both count as negated, so the guard never fired on the one pair
# it exists for.
_NEGATORS = frozenset("""
no not none never neither nor without absent absence failed unable
lack lacks lacking
""".split())

# Direction is separate from negation. An up/down clash is a disagreement even
# though neither side is negated: "increased virulence" and "reduced virulence"
# share every other word, and plain overlap would call them the same fact.
_DIRECTION_UP = frozenset("""
increased elevated higher raised upregulated enhanced enriched greater
""".split())
_DIRECTION_DOWN = frozenset("""
decreased reduced lower lowered downregulated impaired attenuated abolished
diminished suppressed depleted
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+")

# Suffixes stripped for overlap only, longest first. Without this, "disruption"
# and "disrupts" are different tokens and a correct match scores 0.43 -- below
# threshold purely on morphology.
_SUFFIXES = ("ational", "ation", "ition", "ings", "ion", "ing", "ed", "es", "s")
_MIN_STEM = 4


def _stem(word):
    """Crude suffix stripping. Deliberately conservative, not linguistic."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            return word[: -len(suffix)]
    return word


def _raw_tokens(text):
    """Lowercase word tokens with stopwords removed, unstemmed.

    Polarity is judged on these: stemming would turn "increased" into "increas"
    and stop it matching the direction vocabulary.
    """
    if not text:
        return frozenset()
    words = _WORD_RE.findall(str(text).lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def _tokens(text):
    """Stemmed tokens, for overlap comparison."""
    return frozenset(_stem(w) for w in _raw_tokens(text))


def _polarity(tokens):
    """(negated, direction) for a raw token set."""
    direction = None
    if tokens & _DIRECTION_UP:
        direction = "up"
    if tokens & _DIRECTION_DOWN:
        direction = "down" if direction is None else "both"
    return bool(tokens & _NEGATORS), direction


def _fuzzy(a, b, threshold=0.5):
    """Do two free-text slots state the same thing?

    Either token set containing the other counts as a match -- "reduced
    polymerase activity" against "reduced activity" is the same finding stated
    at two levels of detail. Otherwise stemmed Jaccard overlap must clear
    `threshold`.

    A negation mismatch or an up/down direction clash vetoes both routes.
    """
    raw_a, raw_b = _raw_tokens(a), _raw_tokens(b)
    if not raw_a or not raw_b:
        return False

    neg_a, dir_a = _polarity(raw_a)
    neg_b, dir_b = _polarity(raw_b)
    if neg_a != neg_b:
        return False
    if dir_a and dir_b and dir_a != dir_b:
        return False

    ta, tb = _tokens(a), _tokens(b)
    if ta <= tb or tb <= ta:
        return True
    return len(ta & tb) / len(ta | tb) >= threshold


# Direction vocabulary for biomarker associations. Collapsing synonyms to three
# buckets means "elevated" and "upregulated" agree, while "increased" and
# "decreased" stay distinct -- which a token overlap alone would not guarantee.
_ASSOC_BUCKETS = {
    "up": {"increased", "elevated", "upregulated", "higher", "raised",
           "overexpressed", "enriched", "positive"},
    "down": {"decreased", "reduced", "lower", "downregulated", "depleted",
             "suppressed", "diminished", "negative"},
    "none": {"unchanged", "unaffected", "none", "no", "similar", "comparable",
             "nonsignificant", "insignificant"},
}


def _assoc_bucket(text):
    """`up`, `down`, `none`, or None when the wording is unrecognised.

    Reads RAW tokens, not stemmed ones: the bucket vocabulary is spelled in full
    words, and stemming turns "elevated" into "elevat" so nothing matches.
    """
    tokens = _raw_tokens(text)
    for bucket, words in _ASSOC_BUCKETS.items():
        if tokens & words:
            return bucket
    return None


def _pair(row, first, second):
    return (_ppi_token(row.get(first, "")), _ppi_token(row.get(second, "")))


def phenotype_match(run_row, gold_row):
    """Same pathogen and host, and the same observable effect."""
    key = _pair(run_row, "pathogen", "host")
    if not any(key) or key != _pair(gold_row, "pathogen", "host"):
        return False
    return _fuzzy(run_row.get("phenotype", ""), gold_row.get("phenotype", ""))


def mechanism_match(run_row, gold_row):
    """Same pathogen and disease, reached by the same mechanism."""
    key = _pair(run_row, "pathogen", "disease")
    if not any(key) or key != _pair(gold_row, "pathogen", "disease"):
        return False
    return _fuzzy(run_row.get("mechanism", ""), gold_row.get("mechanism", ""))


def biomarker_match(run_row, gold_row):
    """Same biomarker and disease, and the same direction of association.

    Direction is compared by bucket when both sides use recognised wording,
    because it is the finding: a biomarker reported as increased in one row and
    decreased in another is not the same fact. Unrecognised wording falls back
    to free-text comparison rather than being treated as agreement.
    """
    key = _pair(run_row, "biomarker", "disease")
    if not any(key) or key != _pair(gold_row, "biomarker", "disease"):
        return False

    run_bucket = _assoc_bucket(run_row.get("association", ""))
    gold_bucket = _assoc_bucket(gold_row.get("association", ""))
    if run_bucket and gold_bucket:
        return run_bucket == gold_bucket
    return _fuzzy(run_row.get("association", ""), gold_row.get("association", ""))


CHARTER_MATCHERS = {
    "phenotype": phenotype_match,
    "mechanism": mechanism_match,
    "biomarker": biomarker_match,
}

# For experiment-01/schemas.py DATA_TYPES. Field names are the snake_case of the
# template columns, which is what litrag/envelope.py derives, so run rows and
# gold rows are read through one key set.
SCHEMA_ADDITIONS = {
    "phenotype": {
        "label": "Pathogen-Host Phenotype",
        "columns": ["Pathogen", "Host", "Phenotype", "Assertion", "Reference"],
        "fields": ["pathogen", "host", "phenotype", "assertion", "reference"],
    },
    "mechanism": {
        "label": "Pathogen-Mechanism-Disease",
        "columns": ["Pathogen", "Mechanism", "Disease", "Assertion", "Reference"],
        "fields": ["pathogen", "mechanism", "disease", "assertion", "reference"],
    },
    "biomarker": {
        "label": "Biomarker-Disease",
        "columns": ["Biomarker", "Disease", "Association", "Assertion", "Reference"],
        "fields": ["biomarker", "disease", "association", "assertion", "reference"],
    },
}
