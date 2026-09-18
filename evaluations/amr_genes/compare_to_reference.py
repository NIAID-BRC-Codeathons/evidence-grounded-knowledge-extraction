#!/usr/bin/env python3
"""Score LitRAG mutation output against the NCBI Reference Gene Catalog.

The catalog's POINT alleles (`gyrA_S81L`) are the reference set. Both sides are
reduced to a canonical form with `litrag.normalize`, so notation differences --
three-letter codes, `p.` prefixes, `Ter` and `*` stops -- do not read as
disagreements.

This scorer is specific to the Reference Gene Catalog: the `gene_MUTATION`
allele spelling and the `whitelisted_taxa` / `gene_family` columns are its
schema, not a general one. The notation handling it relies on is deliberately
not reimplemented here -- it lives in `litrag.normalize`, where it is tested and
where the rest of the tool uses it. A private copy would drift, and the
benchmark would stop measuring what LitRAG actually does.

Usage:
    python compare_to_reference.py <results.tsv> <queries.tsv> <out.json>
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from litrag.normalize import expand_mutation

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "data" / "filtered_rows.tsv"

Key = Tuple[str, str]           # (gene, mutation)
OrgKey = Tuple[str, str, str]   # (genus, gene, mutation)


def genus(name: str) -> str:
    """Leading genus token, lowercased. `whitelisted_taxa` writes
    `Acinetobacter_baumannii`; results write `Acinetobacter baumannii`."""
    cleaned = (name or "").strip().replace("_", " ")
    return cleaned.split()[0].lower() if cleaned else ""


def load_reference(path: Path) -> Tuple[Set[Key], Set[OrgKey], int, int]:
    """Reference mutations from the catalog's POINT alleles.

    Returns (gene+mutation keys, genus+gene+mutation keys, POINT rows seen,
    POINT rows that are not simple substitutions).
    """
    by_gene: Set[Key] = set()
    by_organism: Set[OrgKey] = set()
    seen = skipped = 0

    for row in csv.DictReader(path.open(), delimiter="\t"):
        # Only POINT carries a specific allele. POINT_DISRUPT names a gene whose
        # loss confers resistance -- any inactivating change counts, so there is
        # no substitution to match -- and AMR-SUSCEPTIBLE marks the wild type.
        if row.get("subtype", "").strip() != "POINT":
            continue
        allele = row["allele"].strip()
        gene = row["gene_family"].strip()
        if not allele or not gene:
            continue
        seen += 1
        # `gyrA_S81L` -> the mutation is whatever follows the last underscore.
        mutations = expand_mutation(allele.rsplit("_", 1)[-1], gene)
        if not mutations:
            skipped += 1
            continue
        for mutation in mutations:
            by_gene.add((gene.lower(), mutation))
            by_organism.add((genus(row["whitelisted_taxa"]), gene.lower(), mutation))

    return by_gene, by_organism, seen, skipped


def load_predictions(path: Path) -> Tuple[int, Set[Key], Set[OrgKey], int]:
    """Scorable predictions from a LitRAG results table."""
    by_gene: Set[Key] = set()
    by_organism: Set[OrgKey] = set()
    rows = unscorable = 0

    for row in csv.DictReader(path.open(), delimiter="\t"):
        rows += 1
        gene = (row.get("Gene Name") or "").strip()
        # `_standard_notation` is LitRAG's own canonical form when it produced
        # one; the raw cell is the fallback.
        mutations = expand_mutation(row.get("Mutation"), gene) or expand_mutation(
            row.get("_standard_notation"), gene
        )
        if not gene or not mutations:
            unscorable += 1
            continue
        for mutation in mutations:
            by_gene.add((gene.lower(), mutation))
            by_organism.add((genus(row.get("Organism", "")), gene.lower(), mutation))

    return rows, by_gene, by_organism, unscorable


def prf(predicted: Set, reference: Set) -> Dict[str, float]:
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    precision = tp / (tp + fp) if predicted else 0.0
    recall = tp / (tp + fn) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def queried_genes(path: Path) -> Set[str]:
    return {
        row["genes"].strip().lower()
        for row in csv.DictReader(path.open(), delimiter="\t")
        if row.get("genes", "").strip()
    }


def per_gene_recall(predicted: Set[Key], reference: Set[Key], limit: int = 15) -> List[Dict]:
    """Recall for the genes with the most reference alleles."""
    totals = Counter(gene for gene, _ in reference)
    found = Counter(gene for gene, _ in (predicted & reference))
    # Sort by allele count, then by name: `most_common` breaks ties by insertion
    # order, which follows set iteration and so varies between runs. A committed
    # scores.json must not diff against itself.
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [
        {"gene": gene, "reference": n, "found": found[gene],
         "recall": round(found[gene] / n, 4)}
        for gene, n in ranked
    ]


def score(results: Path, queries: Path, reference: Path = REFERENCE) -> Dict:
    ref_gene, ref_org, ref_seen, ref_skipped = load_reference(reference)
    n_rows, pred_gene, pred_org, unscorable = load_predictions(results)

    # Recall counts only genes that were actually queried: a gene never asked
    # about cannot fairly be called a miss.
    genes = queried_genes(queries)
    ref_gene_q = {(g, m) for g, m in ref_gene if g in genes}
    ref_org_q = {(o, g, m) for o, g, m in ref_org if g in genes}

    return {
        "results": results.name,
        "result_rows": n_rows,
        "unscorable_rows": unscorable,
        "scorable_predictions": len(pred_gene),
        "genes_in_predictions": len({g for g, _ in pred_gene}),
        "reference_allele_rows": ref_seen,
        "reference_skipped_non_substitution": ref_skipped,
        "reference_mutations": len(ref_gene),
        "reference_in_queried_genes": len(ref_gene_q),
        "gene_mutation": prf(pred_gene, ref_gene_q),
        "organism_gene_mutation": prf(pred_org, ref_org_q),
        "per_gene_recall": per_gene_recall(pred_gene, ref_gene_q),
    }


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    results, queries, out = (Path(a) for a in sys.argv[1:4])
    report = score(results, queries)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_gene_recall"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
