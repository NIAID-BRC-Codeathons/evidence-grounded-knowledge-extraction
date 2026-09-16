#!/usr/bin/env python3
"""Draw a random spot-check sample from a run file written by extract.py.

See CONTRACT.md for the row shape, the outcome values, and the verification
record. This module draws a reproducible random sample of accepted rows
(outcome "cite" or "qualify", from the run file's "rows" list), omitted
rows, and refusal records, then writes one sample file a curator can load
into spotcheck.html.

Python standard library only. Imports schemas.py and ragstack.py, both of
which are standard library only. Prints nothing on import; only __main__
prints.
"""

import argparse
import json
import os
import sys
import urllib.parse
from random import Random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ragstack  # noqa: E402
import schemas  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "out", "spotcheck")

DEFAULT_ACCEPTED = 20
DEFAULT_OMITTED = 10
DEFAULT_REFUSED = 5


def pubmed_link(pubmed_id):
    """Build a PubMed link from a pubmed_id, or None when absent.

    Built once here, in Python, so spotcheck.html never has to assemble a
    URL itself: the curator page only ever inserts a link it was handed.
    """
    if not pubmed_id:
        return None
    return "https://pubmed.ncbi.nlm.nih.gov/{}/".format(
        urllib.parse.quote(str(pubmed_id), safe="")
    )


def load_run_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        envelope = json.load(handle)
    problems = schemas.check_tally(envelope)
    for problem in problems:
        print("warning: run file tally problem: {}".format(problem), file=sys.stderr)
    return envelope


def infer_gene_dtype(envelope):
    """Prefer run.query; fall back to parsing a row_id if query is missing."""
    query = (envelope.get("run") or {}).get("query") or {}
    gene = query.get("gene")
    dtype = query.get("data_type")
    if gene and dtype:
        return gene, dtype

    for bucket in ("rows", "omitted"):
        for row in envelope.get(bucket, []):
            row_id = row.get("row_id") or ""
            parts = row_id.split("__")
            if len(parts) == 3:
                gene = gene or parts[0]
                dtype = dtype or parts[1]
    return gene or "unknown", dtype or "unknown"


def sample_list(items, n, rng):
    """All items when there are n or fewer, else a random sample of size n."""
    if len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def collect_chunk_ids(rows):
    ids = set()
    for row in rows:
        evidence = row.get("evidence") if isinstance(row, dict) else None
        if isinstance(evidence, dict) and evidence.get("chunk_id"):
            ids.add(evidence["chunk_id"])
    return ids


def fetch_chunk_data(chunk_ids, offline):
    """Return (text_by_id, metadata_by_id). Never raises: a fetch failure

    (no network, missing .env, RAGStack down) degrades to empty maps and a
    stderr warning rather than aborting the draw. build_sample_entry() then
    falls back to the stored passage as the displayed chunk text.
    """
    text_by_id = {}
    metadata_by_id = {}
    if not chunk_ids or offline:
        return text_by_id, metadata_by_id
    try:
        env = ragstack.load_env()
        fetched = ragstack.fetch_chunks(sorted(chunk_ids), env=env)
        for chunk in fetched:
            if not isinstance(chunk, dict):
                continue
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if not chunk_id:
                continue
            text_by_id[chunk_id] = chunk.get("content")
            metadata_by_id[chunk_id] = chunk.get("metadata") or {}
    except ragstack.RAGStackError as error:
        print(
            "warning: could not fetch full chunk text from RAGStack ({}); "
            "falling back to the stored passage only".format(error),
            file=sys.stderr,
        )
    except Exception as error:  # defensive: a fetch problem must not crash the draw
        print(
            "warning: unexpected error fetching chunk text ({}); "
            "falling back to the stored passage only".format(error),
            file=sys.stderr,
        )
    return text_by_id, metadata_by_id


def build_sample_entry(kind, row, index, gene, dtype, text_by_id, metadata_by_id):
    entry = {"kind": kind, "row": row}

    evidence = row.get("evidence") if isinstance(row, dict) else None
    if isinstance(evidence, dict):
        chunk_id = evidence.get("chunk_id")
        fetched_text = text_by_id.get(chunk_id)
        entry["chunk_text"] = fetched_text if fetched_text is not None else evidence.get("passage")
        entry["chunk_text_source"] = "ragstack" if fetched_text is not None else "passage_only"
        entry["passage_char_offset"] = evidence.get("passage_char_offset")
        entry["pubmed_link"] = pubmed_link(evidence.get("pubmed_id"))
        entry["paper_metadata"] = metadata_by_id.get(chunk_id) or {}
        row_id_value = row.get("row_id")
    else:
        entry["chunk_text"] = None
        entry["chunk_text_source"] = "unavailable"
        entry["passage_char_offset"] = None
        entry["pubmed_link"] = None
        entry["paper_metadata"] = {}
        row_id_value = row.get("row_id") if isinstance(row, dict) else None

    if not row_id_value:
        # Refusal records carry no row_id (CONTRACT.md's refusal record has
        # none). Synthesize a stable one, keyed on the draw order rather
        # than object identity, so the same seed reproduces the same id.
        row_id_value = "{}__{}__refusal__{:04d}".format(gene.lower(), dtype, index)

    entry["verification"] = schemas.verification_record(row_id_value)
    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Draw a random spot-check sample from an extraction run file."
    )
    parser.add_argument("run_file", help="path to a run file written by extract.py")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed, recorded in the output (default 42)"
    )
    parser.add_argument(
        "--accepted", type=int, default=DEFAULT_ACCEPTED,
        help="accepted rows to sample, outcome cite or qualify (default 20)",
    )
    parser.add_argument(
        "--omitted", type=int, default=DEFAULT_OMITTED, help="omitted rows to sample (default 10)"
    )
    parser.add_argument(
        "--refused", type=int, default=DEFAULT_REFUSED,
        help="refusal records to sample (default 5)",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="output directory (default extract/out/spotcheck)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="skip the RAGStack chunk-text fetch; use the stored passage as chunk text instead",
    )
    args = parser.parse_args()

    envelope = load_run_file(args.run_file)
    gene, dtype = infer_gene_dtype(envelope)

    rng = Random(args.seed)

    accepted_rows = envelope.get("rows", [])
    omitted_rows = envelope.get("omitted", [])
    refusal_rows = envelope.get("refusals", [])

    sampled_accepted = sample_list(accepted_rows, args.accepted, rng)
    sampled_omitted = sample_list(omitted_rows, args.omitted, rng)
    sampled_refused = sample_list(refusal_rows, args.refused, rng)

    chunk_ids = collect_chunk_ids(sampled_accepted) | collect_chunk_ids(sampled_omitted)
    text_by_id, metadata_by_id = fetch_chunk_data(chunk_ids, args.offline)

    samples = []
    for index, row in enumerate(sampled_accepted):
        samples.append(build_sample_entry("accepted", row, index, gene, dtype, text_by_id, metadata_by_id))
    for index, row in enumerate(sampled_omitted):
        samples.append(build_sample_entry("omitted", row, index, gene, dtype, text_by_id, metadata_by_id))
    for index, row in enumerate(sampled_refused):
        samples.append(build_sample_entry("refused", row, index, gene, dtype, text_by_id, metadata_by_id))

    output = {
        "gene": gene,
        "data_type": dtype,
        "run_id": (envelope.get("run") or {}).get("run_id"),
        "source_run_file": os.path.abspath(args.run_file),
        "seed": args.seed,
        "counts_available": {
            "accepted": len(accepted_rows),
            "omitted": len(omitted_rows),
            "refused": len(refusal_rows),
        },
        "counts_drawn": {
            "accepted": len(sampled_accepted),
            "omitted": len(sampled_omitted),
            "refused": len(sampled_refused),
        },
        "samples": samples,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "{}__{}__sample.json".format(gene.lower(), dtype))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        "Drawn: {} accepted, {} omitted, {} refused (seed={})".format(
            len(sampled_accepted), len(sampled_omitted), len(sampled_refused), args.seed
        )
    )
    print("Wrote {}".format(out_path))


if __name__ == "__main__":
    main()
