# AMR resistance mutations vs. the NCBI Reference Gene Catalog

How much of a curated resistance-mutation set can LitRAG recover from the
literature, and how does that change with the collection it searches?

Run 2026-09-18. Generator Qwen3.6-35B (local, thinking off), concurrency 4.

## Summary

| | Exp 1 `open-access` | Exp 2 `RefAMR` | Union |
| --- | ---: | ---: | ---: |
| Precision | 0.248 | **0.465** | — |
| Recall | **0.266** | 0.186 | **0.346** |
| F1 | 0.257 | 0.266 | — |

1. **The collections trade precision for recall almost evenly.** F1 is
   effectively the same, so the choice depends on which error costs more.
2. **They are strongly complementary.** Only 136 of the 444 reference mutations
   found were found by both; pooling lifts recall to 0.346. `-c all` is the
   obvious next run.
3. **`RefAMR` emits no PMIDs at all**, so any join back to PubMed silently
   returns nothing.

Precision is a **lower bound, not an error rate** — see [Reading the
scores](#reading-the-scores).

## Where the reference data come from

The NCBI **Reference Gene Catalog**, the curated gene set behind AMRFinderPlus.

| | |
| --- | --- |
| Browse | <https://www.ncbi.nlm.nih.gov/pathogens/refgene/> |
| Download | <https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/ReferenceGeneCatalog.txt> |
| Release used here | **`2026-08-07.1`** (the `db_version` column of every row in `data/filtered_rows.tsv`) |
| Terms | US Government work, public domain |

The catalog is reissued regularly and the FTP path above always serves the
latest release, so re-running this evaluation against a fresh download will not
reproduce these counts exactly. Pin the release by checking `db_version`.

### Which rows are the reference set

Resistance-conferring mutations live in the rows whose **`subtype` is `POINT`**.
Each carries an `allele` spelled `gene_MUTATION` — `gyrA_S81L`, `cirA_Q42Ter` —
and that is the answer key this evaluation scores against. The filtered set holds
1,501 of them.

Two neighbouring subtypes are present and deliberately excluded:

| Subtype | Rows | Why excluded |
| --- | ---: | --- |
| `POINT` | 1,501 | **The reference set.** |
| `POINT_DISRUPT` | 26 | The gene contributes to resistance when it is *lost* or mutated to inactivity. Any inactivating change counts, so there is no specific allele to match — the `allele` field is empty. |
| `AMR-SUSCEPTIBLE` | 26 | Marks the susceptible wild-type form, not a resistance determinant. |

`allele` is populated on exactly the `POINT` rows and empty on the other two, so
filtering on either is equivalent here; `compare_to_reference.py` tests `subtype`
because that is the meaning, not a coincidence of the data.

`POINT_DISRUPT` is worth understanding even though it is excluded — see
[Loss-of-function genes](#loss-of-function-genes-distort-the-worst-recall-numbers),
where it explains the lowest scores in the table.

## The question set

Catalog rows carrying both a `whitelisted_taxa` and a
`genbank_protein_accession` value were selected from
`ReferenceGeneCatalog.txt` — 1,553 of 11,456 rows, kept here as
`data/filtered_rows.tsv`. Their `gene_family` and `whitelisted_taxa` values give
**266 distinct gene/organism pairs** over 138 gene families and 30 organisms,
one query each (`queries.tsv`, identical in both experiments).

Fixed for every query: `other_terms = antimicrobial resistance`,
`data_type = mutation`, `top_k = 50`.

Two choices worth knowing:

- **One query per pair**, not per row: the 1,553 rows collapse to 266 pairs, and
  querying rows directly would repeat identical searches dozens of times. One
  query per organism with a gene list would cost 30 queries, but no result could
  then be attributed to the gene that retrieved it.
- **Underscores became spaces.** `Acinetobacter_baumannii` is sent as
  `Acinetobacter baumannii`, since retrieval matches literature text. Three
  values are genus-only (`Escherichia`, `Salmonella`, `Campylobacter`).

## Running them

```bash
# Experiment 1 -- default open-access collection
litrag batch experiment1_open_access/queries.tsv -f tsv -T mutation -k 50 \
    -o experiment1_open_access/results.tsv

# Experiment 2 -- identical but for -c RefAMR
litrag batch experiment2_refamr/queries.tsv -f tsv -T mutation -k 50 \
    -c RefAMR -o experiment2_refamr/results.tsv

# Score either one
python compare_to_reference.py <dir>/results.tsv <dir>/queries.tsv <dir>/scores.json
```

`RefAMR` holds 14,151 chunks against open-access's 47.6M — roughly 3,400x
smaller. Both runs completed all 266 queries with no failures.

## Results

| Measure | Exp 1 `open-access` | Exp 2 `RefAMR` |
| --- | ---: | ---: |
| Merged result rows | 2,137 | 1,017 |
| Rows per query (mean) | 8.2 | 4.3 |
| Queries returning nothing | 6 | 10 |
| Rows carrying a mutation | 2,062 (96%) | 947 (93%) |
| Scorable distinct predictions | 1,376 | 514 |
| Genes among those predictions | 115 | 89 |
| Mean `_n_support` | 4.40 | 3.28 |
| `off_target_gene` flags | 98 | 157 |
| Distinct PMIDs cited | 1,070 | **0** |
| Distinct DOIs cited | 19 | 526 |

Scored on gene + mutation, over the genes actually queried:

| Metric | Exp 1 `open-access` | Exp 2 `RefAMR` |
| --- | ---: | ---: |
| True positives | 341 | 239 |
| False positives | 1,035 | 275 |
| False negatives | 942 | 1,044 |
| **Precision** | 0.248 | **0.465** |
| **Recall** | **0.266** | 0.186 |
| **F1** | 0.257 | 0.266 |

Adding organism to the key (genus-level, so `Escherichia` matches
`Escherichia coli`) barely moves Experiment 1 — 0.248 → 0.246 — but costs
Experiment 2 a quarter of its precision, 0.465 → 0.340. With the higher
`off_target_gene` count, the small corpus more often returns a passage about a
different organism or gene than the one asked for, and the model attributes it
to the query rather than to the passage.

Recall by gene, for the genes with the most reference alleles. Rows are ordered
by Experiment 1 recall:

| Gene | Reference | Exp 1 found | Exp 1 recall | Exp 2 found | Exp 2 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `parC` | 57 | 23 | **40%** | 9 | 16% |
| `gyrA` | 118 | 46 | **39%** | 24 | 20% |
| `gyrB` | 47 | 14 | **30%** | 9 | 19% |
| `parE` | 43 | 11 | **26%** | 3 | 7% |
| `rpoB` | 109 | 27 | 25% | 27 | 25% |
| `ftsI` | 54 | 12 | **22%** | 11 | 20% |
| `pmrB` | 83 | 17 | 20% | 33 | **40%** |
| `folP` | 23 | 3 | 13% | 3 | 13% |
| `fusA` | 61 | 6 | **10%** | 4 | 7% |
| `oprD` | 32 | 1 | **3%** | 0 | 0% |
| `ampD` | 25 | 0 | 0% | 1 | **4%** |

`Reference` is one column for both runs: the queries are identical, so each gene
has the same denominator either way.

Experiment 2 is the weaker of the two on 7 of these 11 genes and ties on two
(`rpoB`, `folP`). It wins twice: `pmrB`, where it doubles Experiment 1 (40%
against 20%), and `ampD`, where it finds the one allele Experiment 1 misses
entirely.

The gap is not spread evenly. The four quinolone-target genes — `parC`, `gyrA`,
`gyrB`, `parE` — account for 49 of the 102 true positives separating the two runs
(94 found against 45), the largest single block of the aggregate recall
difference. `pmrB` alone pushes 16 back the other way.

79 of the 135 reference genes had at least one allele recovered in Experiment 1,
56 in Experiment 2. Each `scores.json` records only these top 15 genes by
reference count under `per_gene_recall`; for the full list, call
`compare_to_reference.per_gene_recall` with a larger `limit`.

## Method

The 1,501 `POINT` alleles described in [Where the reference data come
from](#where-the-reference-data-come-from) are the reference. Gene and mutation
are split apart and scored as a pair.

Both sides are canonicalized with `litrag.normalize.expand_mutation`, so
notation differences do not read as disagreements: `S315T`, `Ser315Thr` and
`p.His596Asn` reduce to one form, as do `C39Ter`, `Cys39*` and `C39*`. An allele
list expands to its members — `G288S/M/C` is three claims about position 288.

**Only simple amino-acid substitutions are scored.** Indels, frameshifts,
duplications and prose (`Deletion`, `Transposon insertion`, `overexpression`)
are excluded on both sides, because the two sources describe them too
differently for equality to mean agreement. The 1,501 alleles reduce in two
steps: 62 are dropped as non-substitutions, and the rest collapse to **1,283
distinct (gene, mutation) pairs** across 135 genes, since one allele is often
listed against several organisms or accessions.

Recall counts only reference entries whose gene was actually queried, so a gene
never asked about cannot be called a miss.

### On not reimplementing normalization

`compare_to_reference.py` imports `litrag.normalize` rather than carrying its
own notation handling. The first version of this scorer did carry a copy, and
both halves of that decision went wrong in ways worth recording:

- The private copy accepted **any** three uppercase letters as an amino-acid
  code, so insertion notations such as `ATH54ETR`, `D135DGD` and `Q517QDQ` were
  scored as substitutions. Six spurious reference entries came from that.
- Comparing the copy against the package exposed a real defect in
  `normalize_mutation`: its two substitution patterns each required both sides
  to use the same notation, so `C39Ter` and `S83Phe` fell through to lowercased
  text and stopped merging with their plain twins. That is fixed in the package
  and covered by its tests.

A private copy drifts, and a drifting copy stops measuring what the tool does.

## Reading the scores

**Precision is a lower bound, not an error rate.** The catalog is a curated
subset, not an exhaustive list of every resistance mutation in the literature,
so a prediction outside it is unconfirmed rather than wrong. The false positives
make this concrete: `23S rDNA 2576G>T` is a genuine, well-known
linezolid-resistance mutation, "wrong" here only because the catalog's POINT
entries are protein-level and this one is nucleotide-level. Others are
substitutions from mutagenesis studies (`acrB F136A`, `F178A`, `F610A`) — real
experimental results, but not resistance alleles anyone would catalog.

**Recall is the more trustworthy half**, because every reference allele
genuinely should have been findable.

**The comparison is narrower than the runs.** 619 of Experiment 1's rows and 265
of Experiment 2's fall outside the scorable substitution set, along with 62
reference alleles — roughly a quarter of what LitRAG produced is neither
credited nor penalized.

## Analysis

**`RefAMR`'s precision advantage is partly circular.** It was built from the
papers the catalog itself cites, so its passages are enriched for exactly the
alleles the catalog records, and it is being scored against a reference derived
from the same literature. Not evidence that `RefAMR` is a better corpus in
general.

**Its recall ceiling is structural.** Of the 425 PMIDs cited by the filtered
rows, only **120 (28.2%) yielded a PDF** — the rest are outside the PMC Open
Data bucket or have no PMC record at all (`data/pmid_status.tsv`). The corpus is
missing about 72% of the literature it was meant to represent, so most reference
alleles have no passage to retrieve. No prompt or `top_k` change recovers text
that was never indexed.

**The two are complementary — the most useful result here.** Of the 444
reference mutations either run found, 205 were found only by Experiment 1, 103
only by Experiment 2, and 136 by both. Only 208 of their 1,682 combined
predictions are shared. Pooling would beat either choice.

**`RefAMR` loses PMID traceability entirely.** Experiment 1 rows cite 1,070
distinct PMIDs; Experiment 2 rows cite none. Its citations carry DOIs (526
distinct, on 875 of 1,017 rows) and years, but no PMIDs, titles or authors — the
collection was indexed without PMID metadata. Results stay traceable through the
DOI, but any workflow joining back to PubMed by PMID returns nothing. Better
fixed in the collection than worked around downstream.

### Loss-of-function genes distort the worst recall numbers

The floor of the per-gene table — Experiment 1's `ampD` 0% and `oprD` 3%, and
`fptA`, which returned nothing at all in either run — is substantially a scoring
artifact rather than a retrieval failure.

All three are `POINT_DISRUPT` genes, along with `acrR`, `amrR`, `cirA`, `marR`,
`mexR`, `mexZ`, `mgrB`, `nalD`, `nfsA`, `nfsB`, `nfxB`, `ompC`, `ompF`,
`ompK35`, `ompK36`, `oqxR`, `ramR` and `uhpT`. Resistance there comes from
losing the gene, so the literature describes it the way the mechanism works:

```
oprD | premature stop mutation
oprD | loss-of-function mutation
oprD | insertion sequence (ISPa1635) truncating the gene
oprD | frameshift or premature stop
```

LitRAG is reporting these correctly. The scorer discards every one of them,
because they are not substitutions. Across Experiment 1, **50% of the mutation
values on `POINT_DISRUPT` genes are non-substitutions, against 19% elsewhere** —
so for exactly the genes scoring worst, half the output is thrown away before
matching begins.

These genes also carry `POINT` alleles (`ampD` has 25), which is why they appear
in the reference set at all. A fair assessment of loss-of-function resistance
needs a scoring mode that matches disruption claims against `POINT_DISRUPT`
rows, rather than requiring a residue-level match. That is a separate
evaluation, not a tweak to this one.

**Neither run approaches usable recall alone.** The best single result recovers
about a quarter of curated alleles and the union a third. The per-gene spread
(Experiment 1: `parC` 40%, `ampD` 0%) shows the gap is not uniform, and would
repay examining retrieval for the weak genes before tuning generation. The two
collections do not rank the genes the same way either — `pmrB` tops that table
for Experiment 2 but sits seventh of eleven for Experiment 1 — so a per-gene view
is worth keeping when comparing collections.

### Known problems in the output

- **Off-target genes.** 98 `off_target_gene` flags in Experiment 1 and 157 in
  Experiment 2; 15 gene names in Experiment 1's output were never queried.
- **Corrupted organism names.** `Clostridioides difficult` and
  `Streptoc pneumoniae` appear across 13 Experiment 1 rows — generation
  artifacts. (`Escherichia coli` under the genus-only `Escherichia` query is
  resolution, not error.)
- **Rows with no mutation.** 75 in Experiment 1, 70 in Experiment 2.
- **Citation integrity.** 13 Experiment 1 rows flagged `column_count_mismatch`
  and 13 `citation_not_in_sources`.
- **Corpus overlap.** Only 42 of Experiment 1's 1,070 cited PMIDs are among the
  425 in the catalog rows it was built from, so no claim that this run
  corroborates the catalog should rest on it.
- **`X` is not read as a stop.** `W138X` and `S319X` stay as written, so they do
  not match the catalog's `W138Ter`. `Ter` and `*` do agree. `X` is left alone
  deliberately: it means "any residue" as often as "stop", and guessing would
  invent matches. 5 Experiment 1 rows use it, against 87 `Ter` alleles in the
  reference.

## Files

| Path | Contents |
| --- | --- |
| `compare_to_reference.py` | Scorer; specific to the Reference Gene Catalog schema |
| `data/filtered_rows.tsv` | 1,553 catalog rows — query source and reference set |
| `data/pmids.txt` | 425 unique PMIDs cited by those rows |
| `data/pmids_downloaded.txt` | The 120 whose PDF was retrieved |
| `data/pmids_failed.txt` | The 305 with no PDF |
| `data/pmid_status.tsv` | Per-PMID outcome and reason |
| `experiment1_open_access/` | `queries.tsv`, `results.tsv`, `run.log`, `scores.json` |
| `experiment2_refamr/` | Same, against the `RefAMR` collection |

Article PDFs are not committed: licenses vary per article and PMC's terms do not
permit redistribution. `data/pmid_status.tsv` records what was retrievable, so
the corpus can be rebuilt with
[`pmc-downloader`](https://github.com/NIAID-BRC-Codeathons) from `data/pmids.txt`.
