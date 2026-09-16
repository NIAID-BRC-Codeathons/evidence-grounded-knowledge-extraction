"""Score a run file (extract.py's output envelope) and print a scorecard.

Single source of truth: extract/CONTRACT.md. Python standard library only.
Importable on its own, prints nothing on import.

Usage:
    python3 evaluate.py --fixture
    python3 evaluate.py out/PB2/PB2__mutation__<run_id>.json
    python3 evaluate.py out/PB2/PB2__mutation__<run_id>.json --gold gold/pb2.json
    python3 evaluate.py
        (no path: scans out/ for run files and scores each one found)
"""

import argparse
import copy
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR = os.path.join(HERE, "gold")
OUT_DIR = os.path.join(HERE, "out")

# --------------------------------------------------------------------------
# Notation normalization for mutation rows
# --------------------------------------------------------------------------

AA3TO1 = {
    "ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C",
    "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
    "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P",
    "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V",
}
AA1 = set(AA3TO1.values())

_MUT_RE = re.compile(r"^([A-Za-z]*)(\d+)([A-Za-z]*)$")


def _one_letter(code):
    """Turn a 1-letter or 3-letter amino acid code into its 1-letter form.

    Returns None if code is empty or not a recognized amino acid code.
    """
    if not code:
        return None
    code = code.strip()
    if len(code) == 1:
        up = code.upper()
        return up if up in AA1 else None
    if len(code) == 3:
        return AA3TO1.get(code.lower())
    return None


def normalize_mutation(gene_name, mutation_str):
    """Reduce a mutation notation to a canonical (gene, position, mutant_aa) key.

    Strips a leading gene-name prefix (e.g. "PB2-627K" with gene "PB2"),
    then accepts 1-letter or 3-letter amino acid codes on either side of the
    position number. The wild-type letter is dropped from the key, since some
    notations omit it (e.g. "627K"). Returns None if the string cannot be
    parsed into position + mutant amino acid.
    """
    if not gene_name or not mutation_str:
        return None
    gene = gene_name.strip()
    s = mutation_str.strip()

    if s.lower().startswith(gene.lower()):
        s = s[len(gene):]
        s = s.lstrip("-_: \t")

    m = _MUT_RE.match(s)
    if not m:
        return None
    _prefix, pos, suffix = m.groups()
    mutant = _one_letter(suffix)
    if mutant is None:
        return None
    return (gene.lower(), int(pos), mutant)


def mutation_key(row):
    """Match key for a mutation row: normalize_mutation(gene_name, mutation)."""
    return normalize_mutation(row.get("gene_name", ""), row.get("mutation", ""))


# --------------------------------------------------------------------------
# Normalization for PPI rows
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9]")


def _ppi_token(s):
    """Lowercase and strip all punctuation/whitespace for pair comparison."""
    if not s:
        return ""
    return _PUNCT_RE.sub("", s.lower())


def ppi_pair(row):
    return tuple(sorted([_ppi_token(row.get("protein_a", "")), _ppi_token(row.get("protein_b", ""))]))


def ppi_match(run_row, gold_row):
    """Unordered protein pair match, plus interaction type when both sides have one."""
    if ppi_pair(run_row) != ppi_pair(gold_row):
        return False
    if not ppi_pair(run_row)[0] and not ppi_pair(run_row)[1]:
        return False
    t_run = _ppi_token(run_row.get("interaction_type", ""))
    t_gold = _ppi_token(gold_row.get("interaction_type", ""))
    if t_run and t_gold:
        return t_run == t_gold
    return True


def mutation_match(run_row, gold_row):
    return mutation_key(run_row) is not None and mutation_key(run_row) == mutation_key(gold_row)


MATCHERS = {"mutation": mutation_match, "ppi": ppi_match}


# --------------------------------------------------------------------------
# No-gold metrics: computed from the run file alone
# --------------------------------------------------------------------------

def no_gold_metrics(envelope):
    """Metrics that require no gold set. Returns a dict of {name: (value, detail_str)}."""
    tally = envelope.get("tally", {})
    proposed = tally.get("proposed")
    omitted = envelope.get("omitted", [])
    refusals = envelope.get("refusals", [])
    rows = envelope.get("rows", [])
    retrieval = envelope.get("retrieval", {})

    metrics = {}

    quote_rejected = sum(1 for o in omitted if o.get("omit_reason") == "quote_mismatch")
    if proposed:
        metrics["unsupported_claim_rate"] = (
            quote_rejected / proposed,
            "{} / {}".format(quote_rejected, proposed),
        )
    else:
        metrics["unsupported_claim_rate"] = (None, "not run: tally.proposed is 0 or missing")

    reason_counts = {}
    for o in omitted:
        reason = o.get("omit_reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    metrics["omit_reasons"] = (reason_counts, "{} omitted rows".format(len(omitted)))

    metrics["refusal_count"] = (len(refusals), "{} refusal records".format(len(refusals)))

    papers = retrieval.get("papers")
    metrics["papers_searched"] = (papers, "retrieval.papers")

    if papers:
        metrics["rows_per_paper"] = (len(rows) / papers, "{} rows / {} papers".format(len(rows), papers))
    else:
        metrics["rows_per_paper"] = (None, "not run: retrieval.papers is 0 or missing")

    return metrics


# --------------------------------------------------------------------------
# Gold-set metrics
# --------------------------------------------------------------------------

def retrieved_pmids(envelope):
    manifest = envelope.get("retrieval", {}).get("paper_manifest", [])
    return {m.get("pubmed_id") for m in manifest if m.get("pubmed_id")}


def gold_metrics(envelope, gold):
    """Metrics that require a gold set. Returns a dict of {name: (value, detail_str)}."""
    metrics = {}

    data_type = envelope.get("run", {}).get("query", {}).get("data_type")
    gene = envelope.get("run", {}).get("query", {}).get("gene", "")
    manifest_pmids = retrieved_pmids(envelope)

    gold_pmids = set(gold.get("pmids", []))
    if gold_pmids:
        recall = len(gold_pmids & manifest_pmids) / len(gold_pmids)
        metrics["retrieval_recall"] = (
            recall,
            "{} / {} gold pmids retrieved".format(len(gold_pmids & manifest_pmids), len(gold_pmids)),
        )
    else:
        metrics["retrieval_recall"] = (None, "not run: no gold pmids for gene {!r}".format(gene))

    if data_type not in MATCHERS:
        metrics["precision"] = (None, "not run: unknown or missing data_type {!r}".format(data_type))
        metrics["recall"] = metrics["precision"]
        metrics["citation_correctness"] = metrics["precision"]
        return metrics

    matcher = MATCHERS[data_type]
    gold_rows = gold.get(data_type, [])
    restricted_gold = [g for g in gold_rows if g.get("pmid") in manifest_pmids]

    matched_gold_idx = set()
    tp_pairs = []  # (run_row, gold_row)
    fp = 0
    for row in envelope.get("rows", []):
        found = None
        for i, g in enumerate(restricted_gold):
            if i in matched_gold_idx:
                continue
            if matcher(row, g):
                found = i
                break
        if found is None:
            fp += 1
        else:
            matched_gold_idx.add(found)
            tp_pairs.append((row, restricted_gold[found]))

    tp = len(tp_pairs)
    fn = len(restricted_gold) - len(matched_gold_idx)

    if restricted_gold or tp or fp:
        if tp + fp:
            metrics["precision"] = (tp / (tp + fp), "{} correct / {} emitted, over {} retrieved gold rows".format(tp, tp + fp, len(restricted_gold)))
        else:
            metrics["precision"] = (None, "not run: no rows emitted")
        if tp + fn:
            metrics["recall"] = (tp / (tp + fn), "{} matched / {} gold rows in retrieved papers".format(tp, tp + fn))
        else:
            metrics["recall"] = (None, "not run: no gold rows fall within retrieved papers")
    else:
        metrics["precision"] = (None, "not run: no gold rows for data type {!r} in retrieved papers, and no rows emitted".format(data_type))
        metrics["recall"] = metrics["precision"]

    if tp_pairs:
        correct = 0
        for row, g in tp_pairs:
            row_pmid = row.get("evidence", {}).get("pubmed_id")
            if row_pmid == g.get("pmid"):
                correct += 1
        metrics["citation_correctness"] = (correct / len(tp_pairs), "{} / {} matched rows cite the gold pmid".format(correct, len(tp_pairs)))
    else:
        metrics["citation_correctness"] = (None, "not run: no rows matched a gold row")

    return metrics


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def print_scorecard(path, envelope, gold=None):
    print("== {} ==".format(path))
    nog = no_gold_metrics(envelope)
    print("-- no-gold metrics --")
    for name in ("unsupported_claim_rate", "refusal_count", "papers_searched", "rows_per_paper"):
        value, detail = nog[name]
        if value is None:
            print("{}: not run ({})".format(name, detail))
        else:
            print("{}: {} ({})".format(name, value, detail))
    reasons, detail = nog["omit_reasons"]
    print("omit_reasons: {} ({})".format(reasons if reasons else "none", detail))

    if gold is None:
        print("-- gold metrics: not run (no gold set found) --")
        return

    g = gold_metrics(envelope, gold)
    print("-- gold metrics --")
    for name in ("retrieval_recall", "precision", "recall", "citation_correctness"):
        value, detail = g[name]
        if value is None:
            print("{}: not run ({})".format(name, detail))
        else:
            print("{}: {} ({})".format(name, value, detail))


def load_gold(gene):
    path = os.path.join(GOLD_DIR, "{}.json".format(gene.lower()))
    if not os.path.isfile(path):
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def score_file(path, gold_path_override=None):
    with open(path, "r", encoding="utf-8") as f:
        envelope = json.load(f)
    gene = envelope.get("run", {}).get("query", {}).get("gene", "")
    if gold_path_override:
        with open(gold_path_override, "r", encoding="utf-8") as f:
            gold = json.load(f)
    else:
        gold, _ = load_gold(gene)
    print_scorecard(path, envelope, gold)


# --------------------------------------------------------------------------
# Gold set template
# --------------------------------------------------------------------------

def write_gold_template():
    template = {
        "gene": "PB2",
        "_comment": "Every pmid below is a placeholder. Fill in with the actual gold set by hand.",
        "pmids": ["<pmid-1>", "<pmid-2>"],
        "mutation": [
            {"gene_name": "PB2", "mutation": "<e.g. E627K>", "pmid": "<pmid>"}
        ],
        "ppi": [
            {
                "protein_a": "PB2",
                "protein_b": "<e.g. importin-alpha>",
                "interaction_type": "<e.g. binding, optional>",
                "pmid": "<pmid>",
            }
        ],
    }
    os.makedirs(GOLD_DIR, exist_ok=True)
    path = os.path.join(GOLD_DIR, "pb2.template.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
        f.write("\n")
    return path


# --------------------------------------------------------------------------
# Fixture mode
# --------------------------------------------------------------------------

def _fixture_mutation_run():
    return {
        "run": {"query": {"gene": "PB2", "data_type": "mutation"}},
        "retrieval": {"papers": 2, "paper_manifest": [
            {"doc_id": "d1", "pubmed_id": "24899203"},
            {"doc_id": "d2", "pubmed_id": "11111111"},
        ]},
        "rows": [
            {
                "row_id": "pb2__mutation__0001", "outcome": "cite",
                "gene_name": "PB2", "mutation": "PB2-627K",
                "evidence": {"chunk_id": "c1", "doc_id": "d1", "pubmed_id": "24899203"},
            },
            {
                "row_id": "pb2__mutation__0002", "outcome": "cite",
                "gene_name": "PB2", "mutation": "D701N",
                "evidence": {"chunk_id": "c2", "doc_id": "d2", "pubmed_id": "99999999"},
            },
            {
                "row_id": "pb2__mutation__0003", "outcome": "cite",
                "gene_name": "PB2", "mutation": "H274Y",
                "evidence": {"chunk_id": "c3", "doc_id": "d1", "pubmed_id": "24899203"},
            },
        ],
        "omitted": [
            {"row_id": "pb2__mutation__0004", "outcome": "omit", "omit_reason": "quote_mismatch", "detail": "no exact match"},
        ],
        "refusals": [],
        "tally": {"proposed": 4, "emitted": 3, "omitted": 1},
    }


def _fixture_ppi_run():
    return {
        "run": {"query": {"gene": "PB2", "data_type": "ppi"}},
        "retrieval": {"papers": 1, "paper_manifest": [
            {"doc_id": "d3", "pubmed_id": "17334375"},
        ]},
        "rows": [
            {
                "row_id": "pb2__ppi__0001", "outcome": "cite",
                "protein_a": "importin-alpha", "protein_b": "PB2", "interaction_type": "Binding",
                "evidence": {"chunk_id": "c4", "doc_id": "d3", "pubmed_id": "17334375"},
            },
        ],
        "omitted": [],
        "refusals": [],
        "tally": {"proposed": 1, "emitted": 1, "omitted": 0},
    }


def _fixture_gold():
    return {
        "gene": "PB2",
        "pmids": ["24899203", "17334375", "11111111", "55555555"],
        "mutation": [
            {"gene_name": "PB2", "mutation": "E627K", "pmid": "24899203"},
            {"gene_name": "PB2", "mutation": "D701N", "pmid": "11111111"},
            {"gene_name": "PB2", "mutation": "S714R", "pmid": "55555555"},
        ],
        "ppi": [
            {"protein_a": "PB2", "protein_b": "importin-alpha", "interaction_type": "binding", "pmid": "17334375"},
        ],
    }


def run_fixture():
    gold = _fixture_gold()
    mut_run = _fixture_mutation_run()
    ppi_run = _fixture_ppi_run()

    checks = []

    # Notation variant must match: E627K, Glu627Lys, PB2-627K, 627K all reduce to one key.
    variants = ["E627K", "Glu627Lys", "PB2-627K", "627K"]
    keys = [normalize_mutation("PB2", v) for v in variants]
    checks.append(("notation variants reduce to one key", keys[0], keys, len(set(keys)) == 1 and None not in keys))

    # Swapped protein pair must match.
    swap_match = ppi_match(
        {"protein_a": "importin-alpha", "protein_b": "PB2", "interaction_type": "Binding"},
        {"protein_a": "PB2", "protein_b": "importin-alpha", "interaction_type": "binding"},
    )
    checks.append(("swapped protein pair matches", True, swap_match, swap_match is True))

    mut_metrics = gold_metrics(mut_run, gold)
    ppi_metrics = gold_metrics(ppi_run, gold)

    def check(label, expected, actual):
        ok = expected == actual
        checks.append((label, expected, actual, ok))

    check("mutation precision", 2 / 3, mut_metrics["precision"][0])
    check("mutation recall", 1.0, mut_metrics["recall"][0])
    check("mutation citation correctness (wrong-pmid row counted as error)", 0.5, mut_metrics["citation_correctness"][0])
    check("mutation retrieval recall", 0.5, mut_metrics["retrieval_recall"][0])

    check("ppi precision", 1.0, ppi_metrics["precision"][0])
    check("ppi recall", 1.0, ppi_metrics["recall"][0])
    check("ppi citation correctness", 1.0, ppi_metrics["citation_correctness"][0])
    check("ppi retrieval recall", 0.25, ppi_metrics["retrieval_recall"][0])

    nog = no_gold_metrics(mut_run)
    check("unsupported claim rate", 0.25, nog["unsupported_claim_rate"][0])

    # False positive: H274Y row is absent from gold, must not be matched.
    fp_row = mut_run["rows"][2]
    matched_any = any(mutation_match(fp_row, g) for g in gold["mutation"])
    checks.append(("false-positive row matches no gold row", False, matched_any, matched_any is False))

    all_ok = True
    for entry in checks:
        label, expected, actual = entry[0], entry[1], entry[2]
        ok = entry[3] if len(entry) > 3 else (expected == actual)
        status = "PASS" if ok else "FAIL"
        print("{}: expected={} actual={} [{}]".format(label, expected, actual, status))
        if not ok:
            all_ok = False

    return all_ok


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score a PB2 extraction run file.")
    parser.add_argument("run_file", nargs="?", help="Path to a run envelope JSON file")
    parser.add_argument("--gold", help="Path to a gold set JSON file (overrides gold/<gene>.json lookup)")
    parser.add_argument("--fixture", action="store_true", help="Score the built-in fixture and exit")
    args = parser.parse_args()

    template_path = write_gold_template()

    if args.fixture:
        print("gold template written to {}".format(template_path))
        ok = run_fixture()
        sys.exit(0 if ok else 1)

    if args.run_file:
        score_file(args.run_file, args.gold)
        return

    if not os.path.isdir(OUT_DIR):
        print("no run files found: {} does not exist".format(OUT_DIR))
        return

    found = []
    for root, _dirs, files in os.walk(OUT_DIR):
        for name in files:
            if name.endswith(".json"):
                found.append(os.path.join(root, name))

    if not found:
        print("no run files found under {}".format(OUT_DIR))
        return

    for path in sorted(found):
        score_file(path, args.gold)


if __name__ == "__main__":
    main()
