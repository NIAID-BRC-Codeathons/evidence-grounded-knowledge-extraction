# Extraction contract, slice 1

The single source of truth for field names, shapes, and rules. Every module in this folder is built against this file. If code and this file disagree, this file wins. Written September 16, 2026 from `Extraction_build_spec_PB2_September_16.md`.

Rules that hold everywhere:

- Python standard library only. Run with `python3` (3.10 on this laptop).
- The model never writes an identifier. It returns a `chunk_id` and a verbatim quote. Application code fills `pubmed_id`, `pmc_id`, `doi`, `title`, `year` from the retrieval metadata.
- A row with no verified quote is never emitted as `cite`.
- Every module is importable on its own and prints nothing on import.

## Data types

Two in slice 1. Display columns come from `Literature.js` `DATA_TYPES` (lines 17 to 34) and must not change, so the UI and the command line agree.

| Key | Display label | Display columns, in order |
|---|---|---|
| `mutation` | Mutation | Organism, Gene Name, Mutation, Phenotype, Assertion, Reference |
| `ppi` | Protein-Protein Interaction (PPI) | Pathogen, Protein A, Protein B, Interaction Type, Method, Assertion, Reference |

Field names in JSON are the snake_case form of the display column:

- `mutation`: `organism`, `gene_name`, `mutation`, `phenotype`, `assertion`, `reference`
- `ppi`: `pathogen`, `protein_a`, `protein_b`, `interaction_type`, `method`, `assertion`, `reference`

`assertion` carries the string `"provisional, pending the leads"` until the leads define it (execution plan open question 2). Never leave it blank.

Clarified September 16: the prompt asks the model to emit that exact string, and application code overwrites the field with it after parsing regardless of what the model returned. The value is ours, not the model's, so a model that invents an assertion cannot leak one into curator output. The same rule already applies to `pubmed_id`, `pmc_id`, `doi`, `title` and `year`.

Host-pathogen interaction is expressed as `ppi` in slice 1, with the host protein in `protein_b`. Whether it needs its own data type is open question 2.

## Row shape

```json
{
  "row_id": "pb2__mutation__0007",
  "outcome": "cite",
  "organism": "Influenza A virus",
  "gene_name": "PB2",
  "mutation": "E627K",
  "phenotype": "increased polymerase activity in mammalian cells",
  "assertion": "provisional, pending the leads",
  "reference": "Subbarao 1993, J Virol",
  "evidence": {
    "chunk_id": "...",
    "doc_id": "...",
    "pubmed_id": "24899203",
    "pmc_id": "PMC4136279",
    "doi": "10.1128/jvi.00422-14",
    "title": "PB2 Mutations D701N and S714R Promote Adaptation ...",
    "year": 2014,
    "passage": "the exact sentence copied from the chunk",
    "passage_char_offset": 812
  },
  "citation_partial": false,
  "failure_layer_tags": []
}
```

- `row_id`: `<gene lowercased>__<data type>__<4-digit counter>`, unique within a run.
- Required on every emitted row: `row_id`, `outcome`, every display field for its data type, and `evidence.chunk_id`, `evidence.doc_id`, `evidence.passage`.
- `citation_partial` is `true` when `pubmed_id` or `pmc_id` is absent from the retrieval metadata. Absent identifiers are `null`, never `""` or `"N/A"`.
- `passage_char_offset` is the character position of the quote inside the chunk text, or `null` when the match was not exact.
- `failure_layer_tags` stays an empty list in slice 1; it exists so the success picture's failure-layer tagging can be added without a schema change.

## Outcomes

Four values, from `preparation/Project_1_success_picture_September_12.html`:

| Value | Meaning | Where it lands |
|---|---|---|
| `cite` | Supported by a verified verbatim quote | `rows` |
| `qualify` | Supported but hedged or negated in the source, for example "may contribute" | `rows` |
| `omit` | Dropped by a check, with a reason | `omitted` |
| `refuse` | Nothing in the searched papers supports a row | `refusals` |

## Records

Refusal record:

```json
{
  "outcome": "refuse",
  "data_type": "mutation",
  "gene": "PB2",
  "reason": "no passage supported a row",
  "chunks_searched": 25,
  "papers_searched": 19
}
```

Omitted record: the row as the model proposed it, plus `"outcome": "omit"`, `"omit_reason"` from the fixed set `unknown_chunk`, `quote_mismatch`, `schema_fail`, `wrong_organism`, `wrong_gene`, `duplicate`, and `"detail"` with the specifics.

Run record:

```json
{
  "run_id": "20260916T1530Z-a1b2c3",
  "generated_at": "2026-09-16T15:30:00Z",
  "model": "gpt56luna",
  "prompt_sha256": "...",
  "prompt_edited": false,
  "query": {
    "organism": "Influenza A virus",
    "gene": "PB2",
    "aliases": ["PB2", "polymerase basic 2", "polymerase basic protein 2"],
    "additional_terms": "",
    "data_type": "mutation",
    "collection": "asm-semantic",
    "top_k": 25,
    "filters": {}
  }
}
```

`prompt_edited` marks a run as not comparable to golden runs, per the execution plan.

Verification record, one per spot-checked row, written by `sample.py` with blank values for the curator:

```json
{
  "row_id": "pb2__mutation__0007",
  "verdict": null,
  "opened_paper": null,
  "seconds": null,
  "note": ""
}
```

`verdict` is one of `correct`, `wrong entity`, `wrong relation or value`, `not supported by the quote`, `duplicate`.

## Output file

One file per gene per data type at `experiment-01/out/<gene>/<gene>__<dtype>__<run_id>.json`:

```json
{
  "schema_version": 1,
  "run": { "the run record above": "..." },
  "retrieval": {
    "chunks_returned": 25,
    "chunks_after_dedupe": 23,
    "duplicates_removed": 2,
    "papers": 19,
    "paper_manifest": [
      { "doc_id": "...", "pubmed_id": "...", "pmc_id": "...", "year": 2014, "chunks": 2 }
    ]
  },
  "rows": [],
  "omitted": [],
  "refusals": [],
  "tally": { "proposed": 0, "emitted": 0, "omitted": 0 }
}
```

`tally.emitted` plus `tally.omitted` equals `tally.proposed`, checked before the file is written. When `rows` is empty, `refusals` holds at least one record.

## RAGStack facts, measured September 16

- Two credentials, selected by `ragstack.py`'s credential-selection logic (`RAGSTACK_AUTH` env var: `token`, `key`, or `auto`, default `auto`):
  - Key: `.env` in the parent folder, keys `RAGSTACK_BASE` and `RAGSTACK_API_KEY`. Header `X-API-Key`. Tenant `hackathon-ro`, read-only, reaches `asm-semantic` and `open-access`.
  - Login token: written by `p3-login` into the user's home directory, or the path in `RAGSTACK_TOKEN_PATH` when set. Header `Authorization`, with no `Bearer ` prefix. Tenant is the signed-in BV-BRC user, `restricted_to` None, reaches `open-access`, `asm-semantic`, `Dengue`, and an empty index `ragstack_salesforce_sfr_embedding_mistral_4096_928f8ebe`.
  - `auto` mode uses the token when the file exists and is non-empty, otherwise the key. Never send `Authorization` and `X-API-Key` together; that is a 400. Never print or log either credential value, only report which mode was used.
- The collection id is exactly `Dengue`, capital D. Lowercase fails.
- `POST /v1/retrieve` body: `{"query": str, "collection": str, "top_k": int, "filters": {...}}`. Collection ids are validated live against `GET /v1/collections` for the active credential, not a hardcoded list, since the token and the key reach different collections.
- `filters` with `{"year": 2020}` returns only 2020 papers. Date range, PMID and PMCID filters are untested; treat an unsupported filter as a hard error, never as a silently ignored one.
- `top_k` 25 is the highest value tested.
- Each source: `chunk_id`, `doc_id`, `score` (0 to 1), `content`, and `metadata` with `title`, `journal`, `year`, `date`, `pmid`, `pmcid`, `doi`, `authors`, `prev_chunk_id`, `next_chunk_id`. Note the metadata keys are `pmid` and `pmcid`, while our JSON uses `pubmed_id` and `pmc_id`.
- Dengue metadata is sparse: no `pmid`, `pmcid`, or `title`. It does carry `filename` (e.g. `Dengue_PMC6428985.pdf`), `doi`, `doc_type`, `year`, and chunk ids. `ragstack.resolve_pmc_id()` recovers `pmc_id` from `filename` when `pmcid` is absent, and marks it `pmc_id_derived: true` so a curator can tell a supplied id from a recovered one. `pubmed_id` is never derived from a `pmc_id`; when no pmid exists the row keeps `pubmed_id` null and `citation_partial` true.
- Duplicate chunks occur. De-duplicate by `chunk_id`; count papers by `doc_id`.
- `GET /v1/chunks?ids=a,b,c` fetches chunks by id, up to 200 per request.

## Model calls

Go through `../argo.py`, never a new HTTP client:

```bash
python3 ../argo.py chat <model> - --system "<system text>" --max-tokens 20000
```

The prompt arrives on stdin. Call it as a subprocess, because `argo.py` exits the process on an HTTP error, which would kill a batch. Import `tls_context()` from `argo.py` for any direct HTTPS call, since the python.org build has no root certificates. Never send `temperature`. Keep `max_tokens` at or below 21000 for `claude*` models.

Slice 1 models: `gpt56luna` first, then the same inputs on `claudeopus5`.

## PB2 aliases

Fixed list for slice 1, reviewed by a lead before the gene sweep: `PB2`, `polymerase basic 2`, `polymerase basic protein 2`. Do not generate aliases at run time.
