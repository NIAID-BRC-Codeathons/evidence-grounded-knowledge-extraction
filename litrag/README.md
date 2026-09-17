# LitRAG

Automated literature curation and knowledge extraction over the
[BV-BRC](https://www.bv-brc.org) RAGStack literature API.

Give it an organism, some genes, and a data type; get back a deduplicated,
citation-resolved, reproducible table. One core pipeline, two front ends: a CLI
for batch curation and a standalone web UI for interactive work.

```
$ litrag query -O "Mycobacterium tuberculosis" -g "katG,inhA" \
      -t "isoniazid resistance" -T mutation -k 8 -f table

8 sources -> 7 rows | Qwen/Qwen3.6-35B-A3B (local) | mutation v2 | 7.5s

Organism                    Gene Name  Mutation                Phenotype                      n
--------------------------  ---------  ----------------------  -----------------------------  --
Mycobacterium tuberculosis  katG       Ser315Thr (S315T)       INH resistance (moderate...)   10
Mycobacterium tuberculosis  katG       Deletion                INH resistance (very high...)   3
Mycobacterium tuberculosis  katG       Ser315Asn (S315N)       INH resistance                  1
Mycobacterium tuberculosis  inhA       c-15t                   INH resistance (low level...)   4
Mycobacterium tuberculosis  inhA       Ser94Ala                INH resistance (moderate...)    1
```

That `n = 10` is ten spellings of one mutation — `S315T`, `Ser315Thr`,
`katG-S315T`, `Ser315→Thr`, `Ser315Thr (S315T)` — collapsed into a single fact
backed by seven distinct PMIDs, while `Ser315Asn` stays correctly separate.
Exact output varies between runs; the underlying models are nondeterministic.

## Install

With [uv](https://docs.astral.sh/uv/) (recommended) — no manual venv, no
activation step:

```bash
uv sync                 # creates .venv at the repo root from uv.lock
uv run litrag --help
```

`uv run` works from the repo root or from `litrag/`; it re-syncs the
environment first, so an edit to the source or to `pyproject.toml` is picked up
on the next invocation.

With pip, if you prefer:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.9+. The repo pins 3.12 for development in `.python-version`.

## Configure

The API key is resolved in this order — argument, environment, config file:

```bash
litrag query --api-key rk-...            # 1. explicit
export LITRAG_API_KEY=rk-...             # 2. environment
echo 'api_key = "rk-..."' > ~/.config/litrag/config.toml   # 3. file
```

Never commit a key. `.gitignore` already covers `config.toml` and `.env`.

## CLI

```bash
litrag templates            # data types this tenant offers, with their columns
litrag collections          # searchable corpora
litrag version              # client and server versions
litrag glossary             # what each output column and flag means

litrag query -O "SARS-CoV-2" -g "Spike,ACE2" -T ppi -f tsv -o ppi.tsv
litrag batch examples/queries.tsv -o curated.tsv -j 4
litrag serve --port 8080
```

### `query`

| Flag | Meaning |
|---|---|
| `-O, --organism` | Organism (required) |
| `-g, --genes` | Comma-separated genes/proteins |
| `-t, --other-terms` | Extra search terms |
| `-T, --type` | `ppi`, `protein-function`, `mutation`, `summary`, `ast` (see **Data types**) |
| `-k, --top-k` | Chunks to retrieve (1–100, default 10) |
| `-c, --collection` | Corpus id, comma-separated ids, or `all` (default: PubMed Central) |
| `-f, --format` | `table`, `tsv`, `csv`, `json`, `jsonl`, `md` |
| `-o, --output` | Write to a file instead of stdout |
| `--no-provenance` | Omit the `_`-prefixed provenance columns |
| `--keep-empty` | Keep evidence-free rows (flagged instead of dropped) |
| `--no-dedupe` | Do not merge duplicate facts |
| `--show-sources` | Print the retrieved papers |
| `--dry-run` | Print the request body and exit, without calling the API |

### `batch`

A batch file is TSV, CSV, JSON, or YAML with columns `organism`, `genes`,
`data_type`, `other_terms`, `top_k`, `collection`. Common aliases
(`species`, `gene`, `type`) are accepted. Only `organism` is required.

```tsv
organism	genes	data_type	other_terms
Mycobacterium tuberculosis	katG,inhA	mutation	isoniazid resistance
SARS-CoV-2	Spike,ACE2	ppi	viral entry
```

Every spec is validated before any request is sent, so a typo in row 40 does
not surface after 39 successful queries. Results merge into one table and
deduplicate across queries — a fact found by three searches becomes one row
with `_n_support = 3`.

Each completed query is appended to `<output>.progress.jsonl` as it finishes,
including its extracted rows. `--resume` skips what is already done and rebuilds
the complete table from the sidecar, so an interrupted run costs nothing to
finish.

## Web UI

```bash
litrag serve --port 8080
```

The API key stays in the server process and is never sent to the browser. The
page offers the same fields as the CLI, plus per-row citations as clickable
PMID/DOI links, expandable source cards, and TSV/CSV/JSON/Markdown download.

## Data types

Four come from the server (`/v1/prompt-templates`), versioned and hashed there.
One is defined by LitRAG:

| Type | Columns |
|---|---|
| `ppi` | Pathogen, Protein A, Protein B, Interaction Type, Method, Assertion, Reference |
| `mutation` | Organism, Gene Name, Mutation, Phenotype, Assertion, Reference |
| `protein-function` | Organism, Gene Name, Function, Assertion, Reference |
| `summary` | prose |
| `ast` *(local)* | Organism, Strain, GenBank Accession, BioSample, Antibiotic, MIC, SIR, Reference |
| `glycosylation` *(local)* | Organism, Protein, Strain, Site, Glycosylation Type, Glycan, Method, Effect, Assertion, Reference |
| `host-virus` *(local)* | Organism, Viral Protein, Interaction Type, Host Protein, Host Organism, Biological Consequence, Method, Assertion, Reference |

### Antimicrobial susceptibility testing (`ast`)

```bash
litrag query -O "Escherichia coli" -T ast -c all -k 12 \
    -t "complete genome antimicrobial susceptibility MIC BioSample"
```

```
Organism          Strain   BioSample       Antibiotic     MIC          SIR
----------------  -------  --------------  -------------  -----------  ---
Escherichia coli  1539851  SAMEA116133969  Ampicillin     >32 mg/L     R
Escherichia coli  1539851  SAMEA116133969  Ciprofloxacin  ≤0.125 mg/L  S
Escherichia coli  1539851  SAMEA116133969  Meropenem      0.5 mg/L     S
```

Aliases: `ast`, `amr`, `susceptibility`, `mic`.

**This type only runs on a local generator.** `/v1/prompt-templates` is
read-only — `POST` returns 405 — so a new data type cannot be added there. It is
defined in `litrag/local_templates.py` instead, which works because local
generation already builds its prompt from a template declaration. Selecting it
with `--llm server` fails with an explanation rather than a server 404, and the
UI disables the search button for that pairing.

**Rules specific to this type**, added to the prompt from the declaration:

- One row per strain-antibiotic pair.
- MIC recorded verbatim with its unit and inequality — `>32 mg/L`, `≤0.125 mg/L`,
  `2 µg/mL`. No unit conversion.
- **Neither MIC nor SIR is ever inferred from the other, and no breakpoint is
  applied.** A derived value would be fabricated data. If a paper gives only
  one, the other is `N/A`.
- Accession and BioSample only when stated explicitly — never guessed from the
  organism or strain name.

**Merging.** Identity is organism + strain + antibiotic; accession and BioSample
identify the strain rather than the measurement, so a paper that omits them
still merges with one that reports them. Salt forms and abbreviations are the
same drug (`Ampicillin sodium` = `Ampicillin`, `Ciprofloxacin (CIP)` =
`Ciprofloxacin`), and `R` = `Resistant` so wording differences are not reported
as conflicts. A genuine disagreement is: two papers calling the same
strain-drug pair S and R produce one row flagged `merged_variants:SIR` with both
values kept — which is what CLSI-versus-EUCAST breakpoint differences look like
in practice.

A row with neither an MIC nor an SIR names a strain and a drug without reporting
a result, and is dropped as evidence-free.

### Glycosylation sites (`glycosylation`)

```bash
litrag query -O "SARS-CoV-2" -g Spike -T glycosylation -k 12 \
    -t "N-linked glycosylation site glycan shield site-specific"
```

```
Organism    Protein  Site   Type      Glycan         Method                Effect              Assertion
----------  -------  -----  --------  -------------  --------------------  ------------------  ---------
SARS-CoV-2  Spike    N234   N-linked  oligomannose   mass spectrometry     antibody shielding  reported
SARS-CoV-2  Spike    N343   N-linked  fucosylated    molecular dynamics    structural stability predicted
SARS-CoV-2  Spike    N1194  N-linked  N-acetyl hex.  mass spectrometry                         reported
```

Aliases: `glycosylation`, `glyco`, `glycan`, `glycosite`. Local generator only,
for the same reason as `ast`.

**Site numbering is never rewritten.** It differs between isoforms, strains and
constructs, so `N234` and `N235` stay distinct even where two papers mean the
same residue. Notation is normalized for merging only: `N234`, `Asn234` and
`Asn-234` are one site.

The protein is always named — a site without one cannot be interpreted — and
the strain is recorded wherever a source gives it, as written
(`A/California/07/2009 (H1N1)`, `H3N2`). Papers often list sites as bare
positions (`42, 44, 50`); under an N-linked heading the residue is asparagine
by definition, so the cell shows `N42` while the source's wording is kept.

Identity is organism + protein + strain + site — numbering is strain-dependent,
so position 146 on H3N2 HA is not the same site as position 146 on H1N1 HA. Glycan, method and effect are things
observed *about* a site, not part of what identifies it, so two papers
characterising `N234` differently merge into one row with both glycans listed.
`N-linked` and `N-glycosylation` are the same linkage; `N-linked` versus
`O-linked` on one site is a real conflict and is flagged.

The linkage is recorded only when the source states it — never inferred from the
residue — and an empty Glycan means the glycan was not characterised, not that
the site is unglycosylated.

### Host-virus interactions (`host-virus`)

```bash
litrag query -O "SARS-CoV-2" -T host-virus -k 40 \
    -t "viral protein host interaction interferon antagonism"
```

```
Organism    Viral Protein  Interaction  Host Protein  Host      Biological Consequence
----------  -------------  -----------  ------------  --------  -----------------------------
SARS-CoV-2  ORF6           inhibits     STAT1         human     reduced interferon signalling
SARS-CoV-2  ORF6           binds        NUP98-RAE1    human     reduced interferon signalling
SARS-CoV-2  ORF6           degrades     TRIM25        human     reduced interferon signalling
```

Aliases: `host-virus`, `virus-host`, `hvi`, `host`.

The curated chain is **viral protein → interaction type → host protein →
biological consequence**. Interaction Type is a closed vocabulary — `binds`,
`cleaves`, `inhibits`, `activates`, `degrades`, `relocalizes` — which is what
makes the table queryable. Synonyms fold in (`sequesters` and `retains` are
`relocalizes`, `ubiquitinates` is `degrades`, `blocks` is `inhibits`), and a
verb that fits none of the six is kept as the source wrote it rather than filed
under the nearest.

The relation is **directional**, unlike a protein-protein pair: the viral
protein acts on the host target and the pair is never reordered. The verb is
part of the claim, so binding STAT1 and degrading it are two findings. The
consequence is not — two papers reporting different downstream effects of one
interaction merge, with both kept.

### Adding your own

Drop declarations into `~/.config/litrag/templates.toml` — no code change:

```toml
[[templates]]
id = "operon"
label = "Operon structure"
output = "table"
columns = ["Organism", "Operon", "Genes", "Reference"]
guidance = ["List genes in transcriptional order."]
```

They appear in `litrag templates`, the CLI `--type`, and the UI dropdown, and
run on any local generator. A server template always wins a name clash, so if
the operator later adds the same id, the hosted version takes over.

## Choosing collections

Both corpora are offered, and PubMed Central is the default:

```bash
litrag collections                                  # list them
litrag query ... -c all                             # search every corpus at once
litrag query ... -c asm-semantic                    # one corpus
litrag query ... -c open-access,asm-semantic        # an explicit set
```

| Id | Name | Size |
|---|---|---|
| `open-access` *(default)* | PubMed Central (open access) | 47.6M chunks |
| `asm-semantic` | ASM journals | 6.7M chunks |
| `Dengue`, `Influenza_2024_2025`, `Glyco` | Team-curated corpora | 382 – 3,064 chunks |
| `all` | Every corpus in one request | 54.3M chunks |

The list is read from the server, so corpora added by the team appear without a
code change. Empty indexes are filtered out: the registry also lists a raw
backing store with no state and no chunks, which cannot serve a query.

`all` is capped by the API at five collections per request. If more than five
are active, `all` fails and names them rather than silently searching a subset —
a quiet gap in coverage is worse for curation than an error.

`all` uses the API's `collections` array (capped at five), which stamps each
source with the corpus it came from — shown as a badge on every source card and
recorded in `_collection` as `open-access+asm-semantic`. A single collection is
still sent as the scalar `collection` field, so those responses stay identical
to a plain single-corpus request.

Unknown ids are rejected before any request is made, and the corpus set is part
of a query's identity, so `--resume` re-runs rather than reusing an answer from
a different corpus. ASM titles carry inline HTML, which is stripped before
display and export.

## Choosing a generator

Retrieval always comes from RAGStack. Generation is pluggable:

```bash
litrag query -O "M. tuberculosis" -g katG -T mutation            # qwen (default)
litrag query ... --llm llama                                     # Llama-4-Scout, direct
litrag query ... --llm server                                    # hosted /v1/query
litrag query ... --llm http://my-vllm:8000/v1 --llm-model My/Model
```

| Backend | Endpoint | Notes |
|---|---|---|
| `qwen` *(default)* | `mango.cels.anl.gov:8004` | Qwen3.6-35B, thinking disabled |
| `llama` | `mango.cels.anl.gov:8003` | Llama-4-Scout, same model the hosted path uses |
| `server` | RAGStack `/v1/query` | Server-side versioned prompt template |
| *URL* | any OpenAI-compatible server | Model auto-discovered from `/v1/models` |

**Why Qwen, and why with thinking off.** Benchmarked on identical retrieved
chunks (8 chunks, katG/inhA extraction):

| Config | Time | Rows | Output |
|---|---|---|---|
| Llama-4-Scout | 3.0s | 12 | Bare assertions: "High", "Moderate" |
| Qwen3.6 **thinking** | 29.4s | 3 | Spent 4758 tokens reasoning; collapsed findings into prose |
| Qwen3.6 **no-think** | 3.6s | **14** | Quantitative: "MIC ~6.4 mg/L", "≥19.2 mg/L" |

For schema-constrained extraction the reasoning budget crowds out the answer
without improving it, so `enable_thinking: false` is sent for Qwen. Override
with `--thinking` if you want to see for yourself.

**The hosted endpoint cannot select a model.** `/v1/query` accepts an `llm`
field, but passing one returns HTTP 200 with an empty answer and zero sources —
no error. LitRAG never sends it. Choosing a generator is therefore why the local
path splits retrieval from generation.

**What that costs.** On the local path LitRAG owns the prompt, so the server's
versioned template hash no longer describes what was sent. The prompt is still
built from the server's template *declaration* — its label and columns — so a
template added server-side works everywhere without code changes. Provenance
records `_generator`, `_llm_endpoint`, and a `_prompt_hash` of the exact prompt,
alongside the declaration's id/version/hash. One upside: because the prompt is
ours, **View request** can actually show it, which the hosted path cannot.

## What it does to the data

The model's raw table is not the output. Five things happen to it first.

**Server-driven templates.** Data types, their columns, and their input slots
come from `GET /v1/prompt-templates`. Nothing about them is hardcoded, so a
template added server-side appears in the CLI and UI with no code change.
Slots are validated locally, turning a server 422 into a message before the call.

**Evidence-free rows are dropped.** A live PPI query returned
`SARS-CoV-2  NSP13  Spike  N/A  N/A  N/A  N/A` — the question restated as a
finding. Rows with nothing outside their subject columns are removed and counted
in the summary. `--keep-empty` retains them, flagged.

**Citations are resolved to real papers, and to the passage behind them.**
`[1]` maps to the first retrieved source, expanded to PMID, PMCID, DOI, journal,
year, and first author. Chunks of one paper collapse to one citation, but every
supporting passage is kept: three chunks of one paper are three pieces of
evidence, and the chunk is what the model actually read. When the model cites by name instead of by
number, matching falls back to the **first author only** — `X et al.` means X is
first, and a looser rule mis-attributes claims. A reference that matches nothing
is flagged `citation_not_in_sources` rather than guessed at; in practice this
catches the model citing a paper it read in another paper's bibliography.

**Duplicate facts merge.** One katG response returned `S315T`, `Ser315Thr`, and
`katG-S315T` as three rows. Identity is computed on normalized values —
three-letter and one-letter amino acids, arrow notation, HGVS, gene-prefixed
forms, and parenthetical glosses like `Ser315Thr (S315T)` — so those become one row with `_n_support = 3` and the union of their
citations. Protein interactions are matched unordered, since A–B is B–A.
Promoter positions keep their sign: `c-15t` and `c15t` stay distinct.

**Disagreement stays visible.** When merged rows differ on a non-identity
column, every value is listed in the cell, separated by `; `, and the row is
flagged `merged_variants:<column>`. Merging must not present one paper's
qualifier as every source's finding. Flat formats (TSV, CSV, Markdown) join the
same way; JSON keeps the representative value and a structured `variants` list,
since it can represent both.

### What the columns mean

Every column header in the web UI carries its definition: hover it, or tab to it
and the tooltip opens. Flag chips explain themselves the same way. The
definitions live in `litrag/glossary.py`, so the CLI serves the same text:

```bash
litrag glossary              # every column and flag
litrag glossary mic          # one term
litrag glossary off_target_gene
```

A column with no definition simply gets no tooltip, so a template added
server-side never shows a wrong one.

### The Assertion column

Server templates publish column names but no value vocabulary, so `Assertion`
was undefined and each generator invented its own meaning: Qwen echoed the
instruction to report only what sources state and wrote `Stated` in every row,
Llama wrote evidence types, and the hosted path writes confidence grades like
`High confidence`.

On a local generator the column is now constrained to one of five values,
describing where the claim stands in its source:

| Value | Meaning |
|---|---|
| `measured` | The source ran the experiment that shows this |
| `inferred` | The source concludes it indirectly from its own data |
| `predicted` | Computational or in silico only |
| `reported` | The source attributes it to other work, not its own |
| `disputed` | The source contradicts it or fails to confirm it |

This is an evidence-provenance axis rather than a confidence one: whether a
source measured or merely relayed a claim is checkable against its text, while a
model's self-rated confidence is not.

The rule attaches to the column, so it applies to the server's templates too —
they carry no guidance of their own. **The hosted path is unaffected**: its
prompt belongs to the operator, and it still emits its own wording.

### Mutations written in prose

Papers often write a substitution out in words. A dengue vaccine paper states
*"NS1-53 glycine to aspartate"*, which matches nothing searching for `G53D`.

The prompt asks for standard notation whenever the source gives a reference
residue, a position and a variant residue, so most values arrive as `G53D`
already. Anything still written in prose is converted, and the cell shows the
notation alone — a column is for comparing values, and prose is not comparable.
The source's wording is not lost: it is on hover in the UI and in `_as_written`
in exports, next to `_standard_notation`.

| Written | Standard |
|---|---|
| `NS1-53 glycine to aspartate` | `G53D` |
| `NS1-53 Gly-to-Asp` | `G53D` |
| `NS3-250 glutamate to valine` | `E250V` |
| `5' UTR-57, C to T` | `n57T` |
| `position 315 serine to threonine` | `S315T` |
| `serine to threonine at position 315` | `S315T` |

The converted form is also the dedup identity, so a row written in prose merges
with one that used notation instead of sitting beside it as a separate fact.

Only a genuine conversion is reported: a value already in notation gets no
badge, and anything that is not a recognisable mutation (`katG deletion`,
`Multiple mutations (codons 315, 316)`) is left alone rather than guessed at.
A bare `C to T` is read as a base change only where the qualifier names a
non-coding region, since `C to T` is Cys→Thr as readily as cytosine→thymine.

### Row flags

| Flag | Meaning |
|---|---|
| `off_target_gene` | Gene outside the requested set — kept, since the finding is still real |
| `citation_not_in_sources` | Reference names a paper that is not among the retrieved sources |
| `no_citation` | No reference given |
| `unresolved_citation` | Marker points past the end of the source list |
| `merged_variants:<col>` | Merged rows disagreed on that column; see `variants` |
| `column_count_mismatch` | The row had the wrong number of cells; the alignment is a reconstruction |
| `compound_row` | Several facts in one row that could not be split unambiguously |
| `missing:<col>` | The column carrying the actual finding is empty (for `ast`, `missing:MIC/SIR`) |
| `missing:<col>` | The column carrying the actual finding is empty (for `ast`, `missing:MIC/SIR`) |
| `evidence_free` | Only present with `--keep-empty` |

### Passage-level provenance

Every claim links back to the passage it came from, not just the paper.

In the UI, a `[3]` inside an assertion is a link: clicking it scrolls to that
retrieved passage and highlights it. Each citation also carries `¶` links, one
per supporting passage, so a paper cited through two different chunks shows
`¶1 ¶2`. Source cards display their chunk id and character span.

In exports, `_chunk_ids` and `_markers` accompany `_pmids`, and JSON nests the
full references:

```json
{
  "Mutation": "S315T",
  "citations": [{
    "pmid": "19578178", "journal": "J Antimicrob Chemother",
    "chunks": [
      {"marker": 3, "chunk_id": "e641b7e2-…", "start_char": 0,    "end_char": 2110},
      {"marker": 5, "chunk_id": "f09b6a14-…", "start_char": 3596, "end_char": 5400}
    ]
  }],
  "chunk_ids": ["e641b7e2-…", "f09b6a14-…"]
}
```

Those ids go straight to the API's `/v1/chunks?ids=` to fetch the passage text
back, along with its `prev_chunk_id` / `next_chunk_id` neighbours.

### How retrieval works

Retrieval defaults to `fused`: the query runs under both `hybrid` and `bm25` and
the two rankings are merged with reciprocal rank fusion. `--retrieval-mode`
takes `fused`, `hybrid`, `vector` or `bm25`.

Dense retrieval ranks a chunk by what it is *about*, which loses passages that
mention an identifier in passing. Measured on the Dengue corpus, searching
`dengue virus NS1 Mutation`:

| Mode | Rank of a chunk stating "NS1-53 glycine to aspartate" |
|---|---|
| `hybrid` (was the default) | absent from the top 100 |
| `vector` | absent from the top 100 |
| `bm25` | **12** |
| `fused` | **69** |

Gene names, mutation codes and accessions are exactly the literal tokens BM25
matches and dense similarity does not, so a curation tool should not rely on
either alone. Fusion costs one extra retrieval call, around 0.3s, and leaves the
top of a good ranking unchanged.

Only the local path fuses. `/v1/query` retrieves server-side under a single
mode, and is sent `hybrid`.

### Retrieval depth

`--top-k` goes to 100, and the slider with it. A deeper retrieval finds more,
but the retrieved context and the answer compete for one context window, so both
are budgeted: the answer is sized first (scaled to the number of sources, capped
at 20k tokens), and the prompt gets what is left.

Neither kind of loss is allowed to pass silently.

- If the answer still hits its limit, the run is flagged **truncated** — a
  cut-off table is missing rows and must not read as a complete result.
- If the retrieval does not fit the model's window, the tail is dropped and the
  run reports how many of the requested sources were actually shown. Only those
  are kept as sources, so a citation marker can never point at a passage the
  model never saw.

Window sizes differ: Qwen holds 131k tokens and takes all 100 sources; Llama
holds 60k and shows about 79 of them, which it now says. At `--top-k 100` expect
roughly 20s rather than 2s.

The hosted path budgets its own context and is unaffected — it accepts
`top_k: 100` but returns about as many rows as it does at 10.

### Provenance

Every row carries `_`-prefixed columns recording what produced it: template id,
version and hash, model, collection, `top_k`, retrieval mode, UTC timestamp, the
server's request id, and — on the local path — the generator, its endpoint, and
the prompt hash. A table can be traced back to the exact configuration
that generated it. Use `--no-provenance` for a clean table.

## Notes on this deployment

- The knowledge-graph backend reports `{"backend": "disabled"}`, so LitRAG never
  sets `use_graph`. The original demo's toggle had no effect.
- `/v1/models/available` returns an empty list and the `llm` request field
  silently yields an empty answer, so the hosted path is fixed to
  Llama-4-Scout. Use `--llm` to pick a model.
- Llama-4-Scout sometimes emits the table transposed — one line per field
  rather than per record. LitRAG detects and rotates it back.
- Two collections are available; see **Choosing collections** above.
- Only read-only endpoints are used. Ingest, collection management, grading, and
  admin endpoints exist on the API but are deliberately out of scope.

## Development

```bash
uv run pytest       # no network required
```

`uv sync` installs the `dev` dependency group, so pytest is already present.
To add or change a dependency, edit `pyproject.toml` (or use `uv add <pkg>` /
`uv add --dev <pkg>`) and commit the updated `uv.lock`.

With pip:

```bash
pip install -e ".[dev]"
pytest
```

Tests run against recorded API responses in `tests/fixtures/`. Every
data-quality case is pinned to something a live query actually returned.

## Relationship to Literature.js

This started from a BV-BRC Dojo widget. The API turned out to already provide
much of what that widget did by hand:

| Literature.js | LitRAG |
|---|---|
| Four data types hardcoded in the source | Fetched from `/v1/prompt-templates` |
| Prompts assembled client-side | `template` + `template_vars` sent to the server |
| `/rag/retrieve` then a separate CopilotApi call | One `/v1/query` call |
| Knowledge-graph toggle | Removed; backend is disabled |
| Model dropdown (dead — no choice existed) | Real: `--llm` / UI selector, defaulting to Qwen3.6 |
| Rendered title/DOI/year only | Also PMID, PMCID, journal, authors |
| No collection choice | All corpora, plus `all` for a combined search |
| Raw model table shown as-is | Filtered, citation-resolved, deduplicated |

**View / Edit Prompt** partly survives. On the hosted path it cannot: the server
owns the prompt text and `/v1/prompt-templates` returns declarations only,
"never its `system`/`user` bodies" — so **View request** shows the exact JSON
payload plus the template id, version, and hash. On the local path the prompt is
LitRAG's own, so it is shown in full with its hash. Editing it is still not
offered; a recorded hash is reproducible in a way an ad-hoc edit is not.

## License

MIT
