# Onboarding: how to use this and how to check it

Start here. Ten minutes end to end, and by the end you will have extracted claims from real papers and checked one yourself.

## What this thing is

You give it an organism, a gene, and a question type. It searches PubMed Central, reads the passages it finds, and returns one row per claim. Every row carries the paper it came from and the exact sentence that supports it. When nothing supports a claim, it says so instead of guessing.

The point is not the extraction. The point is that a curator can check a row in seconds instead of reading a paper in twenty minutes.

## Two ways in

| You want to | Use |
|---|---|
| Click around, look at results, try prompt changes | The web page |
| Run many genes at once and leave it working | The command line |

Both use the same code underneath, so they produce identical output files.

## Start the web page

```bash
cd experiment-01
python3 serve.py --port 8765
```

Leave that terminal open. It prints one line per request, which tells you what the page is doing. Ctrl-C stops it.

Then open http://127.0.0.1:8765 in your browser.

If the browser says it cannot connect, the server is not running. That terminal has to stay open, and the port in the address has to match the one you started.

## Your first run, one minute

1. Leave organism as Influenza A virus.
2. Type PB2 in the gene box.
3. Leave data type as mutation.
4. Leave the collection as PubMed Central open access.
5. Set papers to 3, so this run is quick and cheap.
6. Leave the model as `gpt56luna`, the cheap one.
7. Press Run.

A job appears in the job list and moves from running to done in about 20 to 30 seconds. Click it.

## What you are looking at

The results table has one row per claim. For a mutation run the columns are the gene, the mutation, the phenotype, and the reference.

Click any row. You see the full passage from the paper, with the sentence that supports the claim highlighted, plus the title, the year, and a PubMed link.

That is the whole idea: the claim and its evidence on one screen.

Above the table is the run summary:

- Papers: how many papers were read
- Proposed: how many rows the model offered
- Emitted: how many survived the checks and are shown to you
- Omitted: how many were thrown out, and why
- Refusals: cases where nothing in the papers supported an answer

## How to check that it is actually working

Four checks, in order of how convincing they are.

### Check one, does the quote match the paper

Click a row, read the highlighted sentence, then click the PubMed link and find that sentence in the paper. If it is there, the row is real.

This is the check the whole design exists to make fast. Do it on three or four rows.

### Check two, does it refuse when it should

Run again with the gene set to `ZZZ9`, which does not exist. Papers 2 is enough.

Expected: zero rows and a refusal saying nothing supported a row. If it invents rows for a made-up gene, stop and tell me, because that is the failure mode that matters most.

### Check three, does it reject a bad quote

From the command line:

```bash
cd experiment-01
python3 verify.py
```

This runs the quote checker against hand-made cases and prints one line each. It should pass a verbatim quote, reject a quote with one word changed, reject a paraphrase, and reject a quote attributed to the wrong passage. All nine lines say PASS.

### Check four, does the scorer work

```bash
python3 evaluate.py --fixture
```

Twelve lines, all PASS. This scores a small made-up set where the right answer is known, including a deliberately wrong citation. It exists so a scorer that always reports success gets caught.

## Changing the prompt and seeing what happens

Open the Prompt panel on the page. You see the real instructions sent to the model.

Edit them, for example add a line requiring the quoted sentence to contain the mutation code itself, then press Rerun. The rerun uses the same papers, so the comparison is fair, and it only costs model calls.

The compare view then shows rows only in the original, rows only after your change, and rows in both.

One rule the page enforces for you: a run started from an edited prompt is labelled as edited in its output file. Tuned runs must never be presented as clean runs, or your accuracy numbers stop meaning anything.

## Running a lot at once

```bash
# see the cost before spending it
python3 batch.py --genes PB2,PB1,PA,HA --dry-run

# the real sweep: 8 genes, both question types, capped
python3 batch.py --genes PB2,PB1,PA,HA,NP,NA,M1,NS1 --data-types mutation,ppi \
  --papers 20 --workers 3 --cap 400 --model gpt56luna
```

Measured on September 16: about 9 seconds per paper, 3 papers at a time. So 8 genes with both types and 20 papers each is roughly 320 calls and about 16 minutes.

Two things that make a long run safe:

- `--cap` stops the run at a set number of model calls, so it cannot run away.
- Rerunning the same command skips work already finished, so an interrupted sweep continues instead of starting over.

Tell the organizers before a sweep this size. Every call is paid by Argonne and logged against your username.

## Where your results live

```
experiment-01/out/<GENE>/<GENE>__<type>__<timestamp>.json   one file per gene and question type
experiment-01/out/manifest.jsonl                            what has been run, used for resume
experiment-01/out/batch_summary_<timestamp>.json            totals for a sweep
experiment-01/out/spotcheck/                                curator samples
experiment-01/out/baseline/                                 what the existing tool answers, for comparison
```

Each result file holds the rows you saw, everything that was thrown out and why, the refusals, and the run settings including which model was used.

## Spot-checking properly

When you want a real verification number rather than a glance:

```bash
python3 sample.py out/PB2/PB2__mutation__<timestamp>.json --seed 7
```

That draws a random sample and writes it to `out/spotcheck/`. Open `spotcheck.html` in a browser, load that file, and work through it. Each row shows its passage, you pick a verdict, and the page times you. It exports the verdicts and reports the median seconds per row.

Median seconds per row is the number this project is judged on, because it is the curator time you are saving.

## When something looks wrong

| Symptom | Likely cause |
|---|---|
| Browser cannot connect | The server is not running, or the port does not match |
| Every call hangs or times out | You are off the Argonne-auth network. Argo and RAGStack are only reachable from inside it |
| A search returns nothing | The gene name has no match in that collection. Try an alias, for example NEP for NS2 |
| `unknown collection` | The key can reach PubMed Central open access and the ASM corpus, and nothing else yet |
| The same claim appears several times | A known issue: the model splits one finding into near-duplicate rows. Being fixed |
| Rows look right but nothing is scored | Precision and recall need the gold set, which does not exist yet |

## What this cannot do yet

1. No gold set, so precision and recall cannot be computed. What can be computed today: how many rows pass the quote check, how often it refuses, and how it compares to the existing tool.
2. Two question types only, mutation and protein interaction. Protein function is not built.
3. The dengue and influenza collections are not reachable by your key.
4. Near-duplicate rows are not yet collapsed.
5. Nobody has timed a curator on a real sample, so the time-saved claim is not measured.
