#!/usr/bin/env python3
"""Loop extract.run_extraction over subjects crossed with terms crossed with
data types, with real concurrency, a hard call budget, resume-by-manifest,
and a final summary.

Same code path as extract.py (imported, not duplicated). Append-only
manifest at out/manifest.jsonl records each completed combination so a
rerun skips it unless --force.

Subjects and terms both accept a comma list or a file path. A subjects file
with two columns is read as explicit subject/term pairs instead of a cross
product, which is how a sweep names its own pairings.

Usage:
    python3 batch.py --organism "Influenza A virus" --genes PB2,PA,NP --papers 5 --cap 40
    python3 batch.py --organisms "Influenza A virus,Dengue virus" --genes PB2,NS1
    python3 batch.py --organisms pairs.tsv --dry-run
    python3 batch.py --organism "Influenza A virus" --genes genes.txt --workers 4
"""

import argparse
import datetime
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import extract
import ragstack
import schemas

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = extract.DEFAULT_OUT_DIR
DEFAULT_MANIFEST = os.path.join(DEFAULT_OUT_DIR, "manifest.jsonl")
DEFAULT_WORKERS = 3

# Two independent levels of parallelism multiply here: --workers combos
# running at once, and, inside each combo, extract.run_extraction's own
# per-paper concurrency. Their product is the real total of in-flight
# model calls, capped here so the two never multiply into a runaway.
MAX_TOTAL_IN_FLIGHT_CALLS = 8


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def combo_key(organism, gene, data_type, model, collection_str):
    return "|".join([organism, gene, data_type, model, collection_str])


def _read_rows(path):
    """Read a list file into rows of columns. Splits on tab first, then on
    comma, so both a .tsv and a .csv work. Blank lines and lines starting
    with '#' are ignored.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            separator = "\t" if "\t" in line else ","
            cells = [c.strip() for c in line.split(separator)]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
    return rows


def load_list(raw):
    """A comma list or a path to a one-name-per-line file. Returns a list of
    names, order preserved, duplicates dropped.
    """
    if raw is None:
        return []
    if os.path.isfile(raw):
        names = [row[0] for row in _read_rows(raw)]
    else:
        names = [n.strip() for n in raw.split(",") if n.strip()]
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


# Kept as the old name so anything importing it still works.
def load_genes(raw):
    return load_list(raw)


def load_pairs(subjects_raw, terms_raw, default_subject=None):
    """Work out the (subject, term) pairs this sweep should run.

    Three input shapes, all of which the sweep needs:

    - A two-column subjects file: each line is one explicit subject/term
      pair, so a sweep can name its own pairings instead of taking every
      combination. --genes is then not required, and is rejected if given,
      because the file already decides the terms.
    - A subjects comma list (or one-column file) plus a terms list: the full
      cross product of the two.
    - No subjects at all: the single --organism value crossed with the terms
      list, which is the original behaviour.

    Returns (pairs, mode). Raises ValueError with a plain message on a bad
    combination of inputs.
    """
    if subjects_raw and os.path.isfile(subjects_raw):
        rows = _read_rows(subjects_raw)
        if any(len(row) >= 2 for row in rows):
            if not all(len(row) >= 2 for row in rows):
                raise ValueError(
                    "%s mixes one-column and two-column lines: a pairs file needs "
                    "a subject and a term on every line" % subjects_raw
                )
            if terms_raw:
                raise ValueError(
                    "%s is a two-column pairs file, which already names the terms, "
                    "so --genes must not be given as well" % subjects_raw
                )
            pairs = []
            for row in rows:
                pair = (row[0], row[1])
                if pair not in pairs:
                    pairs.append(pair)
            return pairs, "pairs_file"

    subjects = load_list(subjects_raw) if subjects_raw else []
    if not subjects:
        if not (default_subject or "").strip():
            raise ValueError("a subject is required: pass --organism or --organisms")
        subjects = [default_subject.strip()]

    terms = load_list(terms_raw)
    if not terms:
        raise ValueError("--genes must name at least one term")

    pairs = [(subject, term) for subject in subjects for term in terms]
    return pairs, "cross"


def load_data_types(raw):
    if raw is None:
        return sorted(schemas.DATA_TYPES)
    parts = [d.strip() for d in raw.split(",") if d.strip()]
    if not parts:
        return sorted(schemas.DATA_TYPES)
    unknown = [d for d in parts if d not in schemas.DATA_TYPES]
    if unknown:
        sys.exit(
            "unknown data type(s) %r: --data-types can name %s"
            % (unknown, sorted(schemas.DATA_TYPES))
        )
    seen = []
    for d in parts:
        if d not in seen:
            seen.append(d)
    return seen


def load_completed(manifest_path):
    """Return the set of combo keys with a 'done' entry in the manifest.
    A missing or unreadable manifest is treated as empty, never an error.
    """
    completed = set()
    if not os.path.exists(manifest_path):
        return completed
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") == "done":
                completed.add(entry.get("combo_key"))
    return completed


def append_manifest(manifest_path, entry):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def print_results_table(results):
    header = ("organism", "gene", "data_type", "status", "proposed", "emitted", "omitted",
              "refusals", "calls", "seconds")
    rows = [header]
    for r in results:
        rows.append((
            r.get("organism", ""), r["gene"], r["data_type"], r["status"],
            str(r.get("proposed", "")), str(r.get("emitted", "")),
            str(r.get("omitted", "")), str(r.get("refusals", "")),
            str(r.get("model_calls", "")), str(r.get("elapsed_seconds", "")),
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def totals_from_results(results):
    totals = {
        "combos": len(results),
        "done": 0,
        "failed": 0,
        "skipped": 0,
        "budget_exhausted": 0,
        "proposed": 0,
        "emitted": 0,
        "omitted": 0,
        "refusals": 0,
        "model_calls": 0,
    }
    for r in results:
        status = r["status"]
        if status in totals:
            totals[status] += 1
        for field in ("proposed", "emitted", "omitted", "refusals", "model_calls"):
            totals[field] += r.get(field, 0) or 0
    return totals


class BudgetTracker:
    """Thread-safe cap on total model calls across the whole batch.

    Each worker reserves a papers-processed budget before it starts (an
    optimistic worst case of one model call per paper, matching how
    extract.run_extraction estimates its own cap), then settles that
    reservation against the actual model_calls spent once the combination
    finishes. Retries can spend more than one call per paper, so a
    settlement can push remaining below zero; the next reservation then
    sees the cap as exhausted, which is the intended fail-closed behavior.
    """

    def __init__(self, cap):
        self.remaining = cap  # None means unlimited
        self.lock = threading.Lock()

    def reserve(self, requested_papers):
        with self.lock:
            if self.remaining is None:
                return requested_papers
            if self.remaining <= 0:
                return None
            papers = self.remaining if requested_papers is None else min(requested_papers, self.remaining)
            self.remaining -= papers
            return papers

    def settle(self, reserved_papers, actual_calls):
        if self.remaining is None:
            return
        with self.lock:
            self.remaining -= (actual_calls - reserved_papers)


def project_batch(num_combos, papers, cap):
    """Best-effort projected model call count for --dry-run. Worst case is
    one call per paper (retries can spend more); with no --papers cap the
    true count depends on retrieval and is unknown ahead of time.
    """
    if papers is not None:
        raw_total = num_combos * papers
    else:
        raw_total = None

    if cap is not None:
        if raw_total is None:
            return cap, "capped at --cap %d; per-combo paper count is not fixed, so the true total may be less" % cap
        total = min(raw_total, cap)
        note = ("capped at --cap %d (worst case would be %d)" % (cap, raw_total)
                if cap < raw_total else "under --cap %d" % cap)
        return total, note

    if raw_total is not None:
        return raw_total, "no --cap set, worst case one call per paper"

    return None, "unbounded: pass --papers or --cap to bound this projection"


def run_combo(ns, env, organism, gene, data_type, collections_list, collection_str, budget,
               manifest_path, state_lock, state, concurrency):
    """Run one subject/term/data-type combination. Returns a result dict.
    Never raises: a failed combination is recorded and reported, not thrown.
    """
    label = "%s / %s / %s" % (organism, gene, data_type)
    key = combo_key(organism, gene, data_type, ns.model, collection_str)
    reserved = budget.reserve(ns.papers)
    if reserved is None:
        with state_lock:
            print("skip (budget exhausted): %s" % label)
        return {"organism": organism, "gene": gene, "data_type": data_type,
                "status": "budget_exhausted"}

    start = time.time()
    try:
        out_path, summary = extract.run_extraction(
            organism=organism,
            gene=gene,
            data_type=data_type,
            model=ns.model,
            collection_ids=collections_list,
            top_k=ns.top_k,
            papers=reserved,
            year=ns.year,
            out_dir=ns.out_dir,
            env=env,
            concurrency=concurrency,
            additional_terms=ns.additional_terms,
        )
    except Exception as error:  # a bad combo must not end the batch
        budget.settle(reserved, 0)
        with state_lock:
            print("FAILED: %s: %s" % (label, error))
            append_manifest(manifest_path, {
                "combo_key": key, "organism": organism, "gene": gene,
                "data_type": data_type, "model": ns.model,
                "collection": collection_str, "status": "failed",
                "error": str(error), "timestamp": _now_iso(),
            })
        return {"organism": organism, "gene": gene, "data_type": data_type,
                "status": "failed"}

    budget.settle(reserved, summary["model_calls"])

    with state_lock:
        state["combos_done"] += 1
        state["calls_used"] += summary["model_calls"]
        print(
            "done: %s  papers=%d emitted=%d omitted=%d refusals=%d "
            "seconds=%.1f  [progress %d/%d, calls used %d]"
            % (label, summary["papers_processed"], summary["emitted"],
               summary["omitted"], summary["refusals"], summary["elapsed_seconds"],
               state["combos_done"], state["total_combos"], state["calls_used"])
        )
        for notice in summary.get("notices") or []:
            print("  notice [%s]: %s" % (notice["code"], notice["text"]))
        append_manifest(manifest_path, {
            "combo_key": key, "organism": organism, "gene": gene,
            "data_type": data_type, "model": ns.model,
            "collection": collection_str, "status": "done",
            "out_path": out_path, "proposed": summary["proposed"],
            "emitted": summary["emitted"], "omitted": summary["omitted"],
            "refusals": summary["refusals"], "model_calls": summary["model_calls"],
            "notices": [n["code"] for n in (summary.get("notices") or [])],
            "timestamp": _now_iso(),
        })

    return {
        "organism": organism, "gene": gene, "data_type": data_type, "status": "done",
        "proposed": summary["proposed"], "emitted": summary["emitted"],
        "omitted": summary["omitted"], "refusals": summary["refusals"],
        "model_calls": summary["model_calls"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "notices": summary.get("notices") or [],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", default=None,
                         help="comma-separated term list, or a path to a file of "
                              "terms, one per line. Not required when --organisms "
                              "is a two-column pairs file")
    parser.add_argument("--data-types", default=None,
                         help="comma list, default both (%s)" % ", ".join(sorted(schemas.DATA_TYPES)))
    parser.add_argument("--organism", default=None,
                         help="a single subject, crossed with every term")
    parser.add_argument("--organisms", default=None,
                         help="comma-separated subject list, or a file path. A "
                              "one-column file is a subject list crossed with the "
                              "terms; a two-column file (tab or comma separated) "
                              "is read as explicit subject/term pairs instead")
    parser.add_argument("--additional-terms", default="",
                         help="extra words added to every retrieval query")
    parser.add_argument("--model", default="gpt56luna")
    parser.add_argument("--collection", default=None,
                         help="old alias for --collections, kept working")
    parser.add_argument("--collections", default=None,
                         help="comma list of collections to search, e.g. "
                              "asm-semantic,open-access; validated live against "
                              "what the active credential can reach")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--papers", type=int, default=None,
                         help="cap papers processed per combination")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=None,
                         help="defaults to <out-dir>/manifest.jsonl")
    parser.add_argument("--cap", type=int, default=None,
                         help="total model calls allowed across the whole batch")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help="thread pool size, over combinations (default 3)")
    parser.add_argument("--concurrency", type=int, default=None,
                         help="per-combo model-call concurrency, passed to "
                              "extract.run_extraction (default %d). Clamped so "
                              "--workers x concurrency stays at or below %d total "
                              "in-flight model calls."
                              % (extract.DEFAULT_CONCURRENCY, MAX_TOTAL_IN_FLIGHT_CALLS))
    parser.add_argument("--force", action="store_true",
                         help="rerun combinations already marked done in the manifest")
    parser.add_argument("--dry-run", action="store_true",
                         help="print the combinations and projected model calls; call no model")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        pairs, pair_mode = load_pairs(args.organisms, args.genes, args.organism)
    except ValueError as error:
        sys.exit(str(error))
    data_types = load_data_types(args.data_types)

    # Validate and normalize collections once, up front, so an unknown
    # collection id fails fast with a clear error before any work starts,
    # rather than per combination.
    try:
        collections_list = extract.parse_collections(
            collection=args.collection, collections=args.collections
        )
    except ValueError as error:
        sys.exit(str(error))
    collection_str = ",".join(collections_list)

    combos = [
        (organism, gene, data_type)
        for organism, gene in pairs
        for data_type in data_types
    ]

    if args.dry_run:
        print("=== dry run: %d combination(s) ===" % len(combos))
        print("subject/term source: %s (%d pair(s))" % (pair_mode, len(pairs)))
        print("collections: %s" % collection_str)
        for organism, gene, data_type in combos:
            aliases, aliases_source = extract.aliases_for_term(gene)
            print("  %-26s %-16s %-10s model=%s papers=%s aliases=%s (%s)" % (
                organism, gene, data_type, args.model,
                args.papers if args.papers is not None else "uncapped",
                "|".join(aliases), aliases_source,
            ))
        total, note = project_batch(len(combos), args.papers, args.cap)
        print("projected model calls: %s (%s)" % (
            total if total is not None else "unknown", note
        ))
        return 0

    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.jsonl")
    completed = load_completed(manifest_path) if not args.force else set()
    env = ragstack.load_env()

    to_run = []
    results = []
    for organism, gene, data_type in combos:
        key = combo_key(organism, gene, data_type, args.model, collection_str)
        if key in completed:
            print("skip (already done): %s / %s / %s" % (organism, gene, data_type))
            results.append({"organism": organism, "gene": gene,
                            "data_type": data_type, "status": "skipped"})
            continue
        to_run.append((organism, gene, data_type))

    budget = BudgetTracker(args.cap)
    state_lock = threading.Lock()
    state = {"combos_done": 0, "calls_used": 0, "total_combos": len(to_run)}

    workers = max(1, args.workers)
    requested_concurrency = args.concurrency if args.concurrency is not None else extract.DEFAULT_CONCURRENCY
    concurrency = max(1, min(requested_concurrency, MAX_TOTAL_IN_FLIGHT_CALLS // workers))
    print(
        "in-flight model calls: workers=%d x concurrency=%d = %d (cap %d)"
        % (workers, concurrency, workers * concurrency, MAX_TOTAL_IN_FLIGHT_CALLS)
    )

    batch_start = time.time()
    if to_run:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    run_combo, args, env, organism, gene, data_type, collections_list,
                    collection_str, budget, manifest_path, state_lock, state,
                    concurrency,
                )
                for organism, gene, data_type in to_run
            ]
            for future in as_completed(futures):
                results.append(future.result())
    batch_elapsed = round(time.time() - batch_start, 1)

    # Restore combo order (gene, then data type) for the printed table and
    # the summary file, since as_completed() finishes them out of order.
    order = {combo: i for i, combo in enumerate(combos)}
    results.sort(key=lambda r: order.get(
        (r.get("organism"), r["gene"], r["data_type"]), 1 << 30
    ))

    totals = totals_from_results(results)
    totals["batch_elapsed_seconds"] = batch_elapsed

    print()
    print("=== batch results ===")
    print_results_table(results)
    print()
    print("totals: combos=%d done=%d failed=%d skipped=%d budget_exhausted=%d "
          "proposed=%d emitted=%d omitted=%d refusals=%d model_calls=%d elapsed=%ss"
          % (totals["combos"], totals["done"], totals["failed"], totals["skipped"],
             totals["budget_exhausted"], totals["proposed"], totals["emitted"],
             totals["omitted"], totals["refusals"], totals["model_calls"],
             totals["batch_elapsed_seconds"]))

    summary_path = os.path.join(
        args.out_dir, "batch_summary_%s.json" % datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "generated_at": _now_iso(),
            "subject_term_pairs": [list(p) for p in pairs],
            "subject_term_source": pair_mode,
            "additional_terms": args.additional_terms,
            "model": args.model,
            "collections": collections_list,
            "data_types": data_types,
            "workers": args.workers,
            "concurrency": concurrency,
            "cap": args.cap,
            "results": results,
            "totals": totals,
        }, handle, indent=2)
        handle.write("\n")
    print("summary written: %s" % summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
