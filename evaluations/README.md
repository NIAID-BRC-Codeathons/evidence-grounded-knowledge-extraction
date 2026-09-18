# Evaluations

Measurements of LitRAG output against a known answer, and comparisons between
run configurations. Each subdirectory is one evaluation: its inputs, its raw
results, the score it produced, and a write-up of what the numbers mean.

These are dated records of specific runs, not a regression suite. Nothing here
runs in CI. Re-running an evaluation against a later version of the tool is the
point, but the committed results stay as they were so a change in the numbers
is visible.

| Evaluation | What it measures |
|---|---|
| [`amr_genes/`](amr_genes/) | Resistance-mutation extraction scored against the NCBI Reference Gene Catalog, over two literature collections |

## Adding one

A new evaluation is a directory holding whatever it needs to be re-run and
understood: the query file, the results, a scoring script if the answer key has
its own schema, and a `README.md` covering aim, method, results and caveats.

Two conventions are worth keeping:

- **Name directories after the variable under test.** `experiment2_refamr`
  says what differed; `experiment2` does not.
- **Do not reimplement normalization.** Import `litrag.normalize`. A private
  copy drifts, and the evaluation stops measuring what the tool actually does —
  which has already happened once; see the note in `amr_genes/README.md`.

Large or non-redistributable inputs stay out. Record how to obtain them
instead: `amr_genes/` commits the PMID list and per-PMID retrieval status, not
the 213 MB of article PDFs.
