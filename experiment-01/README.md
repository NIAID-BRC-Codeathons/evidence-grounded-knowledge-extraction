# Experiment 01: evidence-grounded extraction with a quote gate

A command line tool and a small local web page that turn literature passages into structured claims, where every claim carries the sentence that supports it. Built at the NIAID-BRCs AI Codeathon 2.0, September 16, 2026.

The problem it addresses: a curator reads a paper to decide which claims are correct, and that reading is the slow step. This pipeline produces rows a curator can check in seconds, and refuses rather than guessing when nothing supports a claim.

## What it does

1. Search: RAGStack returns passages from PubMed Central with paper id, PMID, PMCID and text.
2. Extract: one model call per paper, through Argonne's Argo gateway. The model must quote a passage for every claim it makes.
3. Check: every quote is matched against the passage it cites. A row whose quote does not match is dropped, with a reason. Identifiers are filled in by code from the search results, never by the model, so a citation cannot be invented.
4. Review: a sample goes to a curator, who records a verdict and the seconds it took.

Two data types so far: `mutation` (gene, mutation, phenotype) and `ppi` (protein-protein and host-pathogen interactions).

## What makes it different from a plain RAG answer

RAGStack's own answer endpoint returns prose with bracket citations. Measured on September 16: one answer listed 11 PB2 mutations under a single `[1]` marker, and no sentence in it was traceable to a specific passage, although every mutation name it listed did appear in the cited sources. It is a paraphrase anchored in real papers, which a curator cannot check claim by claim.

This pipeline returns one row per claim with its own quote, so the check is per claim. `baseline.py` captures the RAGStack answers and scores both so the comparison is measured rather than asserted.

## Requirements

- Python 3.10 or newer. Standard library only, nothing to install.
- A connection to the Argonne network. Both services are unreachable from outside it.
- An Argo username for the model calls, of the form `ac.yourname`.
- A RAGStack credential, either of:
  - a read-only API key from the organizers, or
  - a BV-BRC login token, created with `p3-login <your bvbrc username>` from the BV-BRC command line tools. The token reaches more collections than the shared key, including the small `Dengue` test collection.

## Setup

Create a file named `.env` in the directory above this one, holding:

```
RAGSTACK_BASE=https://www.bv-brc.org/ragstack/hackathon/api
RAGSTACK_API_KEY=your-key-here
```

`.env` is gitignored. Never commit a credential.

Set your Argo username in the shell:

```bash
export ARGO_USER=ac.yourname
```

Credential selection is controlled by `RAGSTACK_AUTH`:

| Value | Behaviour |
|---|---|
| `auto` (default) | Use the BV-BRC login token when one exists, otherwise the API key |
| `token` | Always the login token, from `RAGSTACK_TOKEN_PATH` or the file `p3-login` writes in your home directory |
| `key` | Always the API key from `.env` |

The token goes in an `Authorization` header with no `Bearer` prefix. The service rejects a request carrying both an `Authorization` header and an API key, so only one is ever sent.

## Quick start, command line

```bash
cd experiment-01

# one gene, one data type, 5 papers
python3 extract.py --gene PB2 --data-type mutation --papers 5 --model gpt56luna

# see what a sweep costs before spending it
python3 batch.py --genes PB2,PB1,PA,HA --dry-run

# a real sweep: 8 genes, both data types, capped and resumable
python3 batch.py --genes PB2,PB1,PA,HA,NP,NA,M1,NS1 --data-types mutation,ppi \
  --papers 20 --workers 3 --cap 400 --model gpt56luna
```

Measured September 16: about 9 seconds per paper, 6 calls in flight by default, so 20 papers takes around 40 seconds and 20 papers run serially took 175.

## Quick start, web page

```bash
cd experiment-01
python3 serve.py --port 8765
```

Open `http://127.0.0.1:8765`. The server binds to localhost only and keeps the credential server-side, so the browser never holds it.

The page gives you a run form, a live job list with per-paper progress, the paper list as soon as retrieval finishes, results with each row beside its supporting passage, verdict capture, and a prompt panel where you can edit the extraction instructions and rerun on the same papers to compare what changed. A run started from an edited prompt is labelled in its output file, so a tuned run is never mistaken for a clean one.

## Checking the output

```bash
python3 verify.py                 # the quote gate against hand-made cases
python3 evaluate.py --fixture     # the scorer against a set with a known answer
python3 evaluate.py out/PB2/PB2__mutation__<timestamp>.json
python3 sample.py out/PB2/PB2__mutation__<timestamp>.json --seed 7
```

`sample.py` writes a curator sample. Open `spotcheck.html` in a browser, load that file, and work through it: each row shows its passage, you pick a verdict, and the page times you and exports the records.

Two checks worth running first, because they prove the gate rejects rather than waves things through:

- `verify.py` must reject a one-word-altered quote, a paraphrase, and a quote attributed to the wrong passage.
- An extraction with a made-up gene, for example `--gene ZZZ9`, must produce a refusal record rather than invented rows.

## Files

| File | Purpose |
|---|---|
| `CONTRACT.md` | Field names, record shapes and rules. The source of truth; code follows it |
| `ragstack.py` | Search, de-duplication, paper manifest, credential selection |
| `prompt.py` | System prompt and the per-paper extraction prompt |
| `extract.py` | One gene and one data type end to end |
| `batch.py` | Many genes crossed with data types, with concurrency, a call cap and resume |
| `verify.py` | The quote gate |
| `schemas.py` | Data type definitions and the row validator |
| `evaluate.py` | Scorer, plus a fixture that catches a scorer which always passes |
| `sample.py` | Curator sample builder |
| `spotcheck.html` | Offline review page |
| `serve.py`, `ui.html` | Local web server and page |
| `baseline.py` | Captures RAGStack's own answers for comparison |
| `argo.py` | Thin client for the Argo gateway, with each model family's parameter rules |
| `ONBOARDING.md` | A guided walk through for a first-time user |

Output lands in `out/`, one file per gene and data type, holding the rows, everything dropped and why, the refusals, the paper manifest and the run settings.

## Known limits

1. No gold set yet, so precision and recall cannot be computed. What can be computed today: the share of rows passing the quote check, refusal behaviour, and the comparison against the RAGStack baseline.
2. The model is not deterministic. Two identical runs gave 9 and 10 rows on the same 5 papers. Row numbering is deterministic given the same model output, verified by replacing the model with fixed text, but the model itself varies call to call. Report counts as a range, or run twice and treat claims found in both runs as the stable core.
3. Near-duplicate rows are not collapsed. One run returned the same mutation several times with different phenotype wording.
4. Row matching is literal, so a paper writing "position 265" instead of "N265S" counts as a different claim.
5. The `Dengue` collection carries no PMID, PMCID or title in its metadata. PMC ids are recovered from filenames where possible, and rows are marked as partially cited when an identifier is missing.
6. Two data types only. Protein function is not built.
7. No curator has been timed on a real sample, so no claim about time saved is measured.
