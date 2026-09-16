#!/usr/bin/env python3
"""One gene, one data type, end to end: retrieve, prompt, call the model,
verify, write the output envelope. See CONTRACT.md for the row and record
shapes this follows. Python standard library only.

Usage:
    python3 extract.py --gene PB2 --data-type mutation
    python3 extract.py --gene PB2 --data-type ppi --papers 3
"""

import argparse
import collections
import datetime
import json
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ragstack
import schemas
import verify
import prompt as prompt_mod

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ARGO_PATH = os.path.join(os.path.dirname(THIS_DIR), "argo.py")
DEFAULT_OUT_DIR = os.path.join(THIS_DIR, "out")

# PB2 aliases are the only fixed list in the contract (slice 1). Any other
# gene gets no invented aliases, per "Do not generate aliases at run time."
PB2_ALIASES = ["PB2", "polymerase basic 2", "polymerase basic protein 2"]

# What a fixed KNOWN_COLLECTIONS tuple used to hold is now a live fact: a
# token and a key reach different collections (per CONTRACT.md's RAGStack
# facts, measured September 16), so validation calls GET /v1/collections
# for the active credential instead of checking against a hardcoded list.
# See _known_collections() below. Cached per (base, auth mode) so a batch
# of many gene/data-type combinations does not refetch it per combination.
_COLLECTIONS_CACHE = {}


def _known_collections(env):
    """Live-fetch (and cache for the process) the collection ids reachable
    by the active credential. Cache key is (base URL, auth mode), since a
    token and a key can reach different sets even against the same base.
    """
    cache_key = (env.get("RAGSTACK_BASE"), ragstack.auth_mode(env))
    if cache_key not in _COLLECTIONS_CACHE:
        _COLLECTIONS_CACHE[cache_key] = tuple(ragstack.list_collections(env=env))
    return _COLLECTIONS_CACHE[cache_key]

# Query paraphrase terms unioned with the gene/alias for retrieval. Not part
# of the contract's fixed vocabulary; chosen to match the data type without
# inventing new identifiers or claims.
DATA_TYPE_QUERY_TERMS = {
    "mutation": ["mutation", "polymerase activity", "host adaptation", "phenotype"],
    "ppi": ["protein interaction", "binding", "protein-protein interaction"],
}

MODEL_TIMEOUT_SECONDS = 180
MODEL_MAX_TOKENS = 20000
MODEL_RETRIES = 2  # one attempt plus one retry

# Per-paper model calls are independent (one call per paper, never grouped
# into one prompt), so they run through a thread pool. Measured September
# 16: about 9 seconds per call, strictly serial. 6 is a starting point that
# keeps a single run well under any per-key rate limit; --concurrency
# overrides it.
DEFAULT_CONCURRENCY = 6


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def make_run_id():
    return _now_utc().strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def aliases_for_gene(gene):
    if gene.upper() == "PB2":
        return list(PB2_ALIASES)
    return [gene]


def extract_json(text):
    """Pull one JSON object out of a model reply, tolerating a code fence or
    stray prose before/after it. Returns None when nothing parses.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def call_model(model, prompt_text, system):
    """One subprocess call to ../argo.py. Returns (reply_text, error)."""
    cmd = [
        sys.executable,
        ARGO_PATH,
        "chat",
        model,
        "-",
        "--system",
        system,
        "--max-tokens",
        str(MODEL_MAX_TOKENS),
    ]
    try:
        result = subprocess.run(
            cmd,
            input=prompt_text,
            capture_output=True,
            text=True,
            timeout=MODEL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, "timed out after %ds" % MODEL_TIMEOUT_SECONDS

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()[:300]
        return None, "exit %d: %s" % (result.returncode, stderr)

    output = (result.stdout or "").strip()
    if not output:
        return None, "empty response"
    return output, None


def call_model_with_retry(model, prompt_text, system, retries=MODEL_RETRIES):
    """Call the model, retrying once on failure. A bad paper must not end
    the run: this always returns, never raises.
    """
    attempts = 0
    last_err = None
    for _ in range(retries):
        attempts += 1
        reply, err = call_model(model, prompt_text, system)
        if reply is not None:
            return reply, attempts, None
        last_err = err
    return None, attempts, last_err


def convert_model_row(model_row, data_type, gene, counter, chunks_by_id):
    """Build a full contract row from one model-proposed row. Application
    code fills pubmed_id, pmc_id, doi, title, year, row_id, and overwrites
    assertion; the model's values for those are ignored.
    """
    row = schemas.new_row(data_type)
    row["row_id"] = schemas.row_id(gene, data_type, counter)

    for field in schemas.DATA_TYPES[data_type]["fields"]:
        if field == "assertion":
            row[field] = schemas.ASSERTION_PLACEHOLDER
        else:
            value = model_row.get(field) if isinstance(model_row, dict) else None
            row[field] = value

    chunk_id = model_row.get("chunk_id") if isinstance(model_row, dict) else None
    quote = model_row.get("quote") if isinstance(model_row, dict) else None
    chunk = chunks_by_id.get(chunk_id) if chunk_id else None
    meta = (chunk or {}).get("metadata") or {}

    pubmed_id = meta.get("pmid") or None
    pmc_id, pmc_id_derived = ragstack.resolve_pmc_id(meta)
    row["evidence"] = {
        "chunk_id": chunk_id,
        "doc_id": chunk.get("doc_id") if chunk else None,
        "pubmed_id": pubmed_id,
        "pmc_id": pmc_id,
        "pmc_id_derived": pmc_id_derived,
        "doi": meta.get("doi") or None,
        "title": meta.get("title") or None,
        "year": meta.get("year") or None,
        "passage": quote,
        "passage_char_offset": None,
    }
    row["citation_partial"] = pubmed_id is None or pmc_id is None
    row["failure_layer_tags"] = []
    return row


def parse_collections(collection=None, collections=None, env=None):
    """Normalize the --collection / --collections inputs into an ordered,
    de-duplicated list of collection ids.

    `collections` is the new, preferred flag; `collection` is the old alias,
    still honored so nothing that calls it breaks. When neither is given,
    falls back to the single default collection. Validates against a live
    GET /v1/collections for the active credential (see _known_collections),
    not a hardcoded list, since a token and a key reach different
    collections. `env` defaults to ragstack.load_env() when not supplied.
    Raises ValueError naming what is actually reachable when an unknown id
    is requested.
    """
    raw = collections if collections is not None else collection
    if raw is None:
        raw = "open-access"

    if isinstance(raw, str):
        parts = [c.strip() for c in raw.split(",") if c.strip()]
    else:
        parts = [c.strip() for c in raw if c and str(c).strip()]
    if not parts:
        parts = ["open-access"]

    seen = []
    for c in parts:
        if c not in seen:
            seen.append(c)

    env = env or ragstack.load_env()
    known = _known_collections(env)
    unknown = [c for c in seen if c not in known]
    if unknown:
        raise ValueError(
            "unknown collection(s) %r: this credential (auth mode=%s) currently "
            "reaches %s"
            % (unknown, ragstack.auth_mode(env), ", ".join(known))
        )
    return seen


def collect_multi_collection(organism, gene, aliases, data_type_terms, collections_list,
                              top_k, filters, env):
    """Call ragstack.collect once per collection (the live API's retrieve
    takes one collection per request), then merge and dedupe across
    collections. Returns (kept, manifest, stats).

    stats["by_collection"] carries the per-collection chunk/paper counts the
    CONTRACT retrieval block wants. Each manifest entry also carries a
    "collections" list naming which collection(s) contributed a kept chunk
    for that paper, which is the "recording which collection each kept
    source came from" the multi-collection support needs; this is an
    additive key, not a change to the contract's fixed paper_manifest
    fields (doc_id, pubmed_id, pmc_id, year, chunks).
    """
    all_sources = []
    chunks_returned = 0
    duplicates_removed = 0
    queries_issued = 0
    by_collection = {}
    doc_collections = {}

    for coll in collections_list:
        kept, coll_manifest, stats = ragstack.collect(
            organism=organism,
            gene=gene,
            aliases=aliases,
            data_type_terms=data_type_terms,
            collection=coll,
            top_k=top_k,
            filters=filters,
            env=env,
        )
        all_sources.extend(kept)
        chunks_returned += stats["chunks_returned"]
        duplicates_removed += stats["duplicates_removed"]
        queries_issued += stats["queries_issued"]
        by_collection[coll] = {
            "chunks_returned": stats["chunks_returned"],
            "chunks_after_dedupe": stats["chunks_after_dedupe"],
            "papers": stats["papers"],
        }
        for entry in coll_manifest:
            doc_collections.setdefault(entry["doc_id"], set()).add(coll)

    kept, cross_collection_dupes = ragstack.dedupe(all_sources)
    manifest = ragstack.paper_manifest(kept)
    for entry in manifest:
        entry["collections"] = sorted(doc_collections.get(entry["doc_id"], []))

    stats = {
        "queries_issued": queries_issued,
        "chunks_returned": chunks_returned,
        "chunks_after_dedupe": len(kept),
        "duplicates_removed": duplicates_removed + cross_collection_dupes,
        "papers": len(manifest),
        "by_collection": by_collection,
    }
    return kept, manifest, stats


def run_extraction(organism, gene, data_type, model, collection=None, top_k=25,
                    papers=None, year=None, out_dir=None, env=None, collection_ids=None,
                    concurrency=DEFAULT_CONCURRENCY):
    """Run one gene, one data type, end to end. Returns (out_path, summary).

    `collection` is the old comma-separated-string flag, kept working as an
    alias. `collection_ids` is the new, preferred parameter and can be a
    comma-separated string or a list; it wins when both are given and
    differ. Named `collection_ids` rather than `collections` to avoid
    shadowing the stdlib `collections` module used below in this function.
    See parse_collections() for the normalization and the error on an
    unrecognized collection id.
    `papers` caps the number of papers sent to the model (for smoke runs);
    None means no cap.
    `concurrency` bounds how many per-paper model calls run at once, through
    a thread pool. Each call still covers exactly one paper (never grouped
    into one prompt); only the scheduling is concurrent. Results are
    collected as they complete but rows are numbered in manifest (paper)
    order afterward, so row_id assignment never depends on completion
    order and two runs of the same input produce identical output apart
    from timing.
    """
    start_time = time.time()
    out_dir = out_dir or DEFAULT_OUT_DIR
    env = env or ragstack.load_env()

    if data_type not in schemas.DATA_TYPES:
        raise ValueError("unknown data type: %r" % (data_type,))

    aliases = aliases_for_gene(gene)
    collections_list = parse_collections(
        collection=collection, collections=collection_ids, env=env
    )
    collection_str = ",".join(collections_list)
    filters = {"year": year} if year else {}
    data_type_terms = DATA_TYPE_QUERY_TERMS[data_type]

    kept, manifest, stats = collect_multi_collection(
        organism, gene, aliases, data_type_terms, collections_list, top_k, filters, env
    )

    by_doc = collections.OrderedDict()
    for source in kept:
        by_doc.setdefault(source.get("doc_id"), []).append(source)
    doc_ids = list(by_doc.keys())
    if papers is not None:
        doc_ids = doc_ids[:papers]

    chunks_by_id = {}
    for doc_id in doc_ids:
        for source in by_doc[doc_id]:
            chunks_by_id[source.get("chunk_id")] = source

    spec = {
        "organism": organism,
        "gene": gene,
        "aliases": aliases,
        "additional_terms": "",
        "data_type": data_type,
    }

    proposed_rows = []
    failed_papers = []
    model_calls = 0
    counter = 0

    def _call_one_paper(doc_id):
        paper_chunks = by_doc[doc_id]
        paper_prompt = prompt_mod.build_prompt(spec, paper_chunks)
        reply, attempts, err = call_model_with_retry(model, paper_prompt, prompt_mod.SYSTEM)
        return doc_id, reply, attempts, err

    # The per-paper calls are independent, so they are scheduled through a
    # thread pool and collected as they complete. Nothing below this point
    # depends on completion order: results land in results_by_doc keyed by
    # doc_id, and are then walked in doc_ids order (the manifest/paper
    # order) so row_id numbering is deterministic.
    results_by_doc = {}
    workers = max(1, concurrency or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_call_one_paper, doc_id) for doc_id in doc_ids]
        for future in as_completed(futures):
            doc_id, reply, attempts, err = future.result()
            model_calls += attempts
            results_by_doc[doc_id] = (reply, err)

    for doc_id in doc_ids:
        reply, err = results_by_doc[doc_id]
        if reply is None:
            failed_papers.append({"doc_id": doc_id, "reason": err})
            continue

        parsed = extract_json(reply)
        if parsed is None or "rows" not in parsed:
            failed_papers.append({"doc_id": doc_id, "reason": "no parseable JSON rows in reply"})
            continue

        model_rows = parsed.get("rows") or []
        for model_row in model_rows:
            counter += 1
            proposed_rows.append(
                convert_model_row(model_row, data_type, gene, counter, chunks_by_id)
            )

    accepted, omitted = verify.verify_rows(proposed_rows, chunks_by_id, spec)

    final_accepted = []
    for row in accepted:
        problems = schemas.validate_row(data_type, row)
        if problems:
            omitted.append(schemas.omitted_record(row, "schema_fail", "; ".join(problems)))
        else:
            final_accepted.append(row)

    refusals = []
    if not final_accepted:
        refusals.append(
            schemas.refusal_record(
                data_type, gene, "no passage supported a row",
                chunks_searched=len(chunks_by_id), papers_searched=len(doc_ids),
            )
        )

    run_id = make_run_id()
    generated_at = _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
    # The prompt body differs per paper (different chunks), so this hashes
    # the fixed SYSTEM instruction rather than any one paper's prompt. See
    # the report for why this reading of prompt_sha256 was chosen.
    prompt_hash = prompt_mod.prompt_sha256(prompt_mod.SYSTEM)

    run = schemas.run_record(
        run_id=run_id,
        generated_at=generated_at,
        model=model,
        prompt_sha256=prompt_hash,
        prompt_edited=False,
        query={
            "organism": organism,
            "gene": gene,
            "aliases": aliases,
            "additional_terms": "",
            "data_type": data_type,
            "collection": collection_str,
            "collections": collections_list,
            "top_k": top_k,
            "filters": filters,
            "concurrency": workers,
        },
    )

    retrieval = {
        "chunks_returned": stats["chunks_returned"],
        "chunks_after_dedupe": stats["chunks_after_dedupe"],
        "duplicates_removed": stats["duplicates_removed"],
        "papers": stats["papers"],
        "paper_manifest": manifest,
        "collections_queried": collections_list,
        "by_collection": stats["by_collection"],
    }

    envelope = schemas.output_envelope(run, retrieval, final_accepted, omitted, refusals)
    tally_problems = schemas.check_tally(envelope)
    if tally_problems:
        raise RuntimeError("tally check failed: " + "; ".join(tally_problems))

    gene_dir = os.path.join(out_dir, gene)
    os.makedirs(gene_dir, exist_ok=True)
    out_path = os.path.join(gene_dir, "%s__%s__%s.json" % (gene, data_type, run_id))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(envelope, handle, indent=2)
        handle.write("\n")

    elapsed = time.time() - start_time
    omit_reasons = collections.Counter(rec.get("omit_reason") for rec in omitted)

    summary = {
        "out_path": out_path,
        "papers_returned": stats["papers"],
        "papers_processed": len(doc_ids),
        "chunks_returned": stats["chunks_returned"],
        "chunks_after_dedupe": stats["chunks_after_dedupe"],
        "chunks_processed": len(chunks_by_id),
        "proposed": envelope["tally"]["proposed"],
        "emitted": envelope["tally"]["emitted"],
        "omitted": envelope["tally"]["omitted"],
        "omit_reasons": dict(omit_reasons),
        "refusals": len(refusals),
        "failed_papers": failed_papers,
        "model_calls": model_calls,
        "elapsed_seconds": round(elapsed, 1),
        "collections_queried": collections_list,
        "by_collection": stats["by_collection"],
        "concurrency": workers,
    }
    return out_path, summary


def print_summary(gene, data_type, model, summary):
    print("--- %s / %s / %s ---" % (gene, data_type, model))
    print("collections:       %s" % ", ".join(summary.get("collections_queried") or []))
    for coll, coll_stats in (summary.get("by_collection") or {}).items():
        print("  %-14s chunks %d (after dedupe %d)  papers %d" % (
            coll + ":", coll_stats["chunks_returned"], coll_stats["chunks_after_dedupe"],
            coll_stats["papers"],
        ))
    print("papers returned:   %d" % summary["papers_returned"])
    print("papers processed:  %d" % summary["papers_processed"])
    print("chunks returned:   %d (after dedupe %d, processed %d)" % (
        summary["chunks_returned"], summary["chunks_after_dedupe"], summary["chunks_processed"]
    ))
    print("proposed:          %d" % summary["proposed"])
    print("emitted:           %d" % summary["emitted"])
    print("omitted:           %d" % summary["omitted"])
    if summary["omit_reasons"]:
        for reason, count in sorted(summary["omit_reasons"].items()):
            print("  omitted %-16s %d" % (reason + ":", count))
    print("refusals:          %d" % summary["refusals"])
    if summary["failed_papers"]:
        print("failed papers:     %d" % len(summary["failed_papers"]))
        for failure in summary["failed_papers"]:
            print("  %s: %s" % (failure["doc_id"], failure["reason"]))
    print("model calls:       %d" % summary["model_calls"])
    print("concurrency:       %d" % summary["concurrency"])
    print("elapsed seconds:   %s" % summary["elapsed_seconds"])
    print("output:            %s" % summary["out_path"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organism", default="Influenza A virus")
    parser.add_argument("--gene", default="PB2")
    parser.add_argument("--data-type", choices=sorted(schemas.DATA_TYPES), required=True)
    parser.add_argument("--model", default="gpt56luna")
    parser.add_argument("--collection", default=None,
                         help="old alias for --collections, kept working")
    parser.add_argument("--collections", default=None,
                         help="comma list of collections to search, e.g. "
                              "asm-semantic,open-access,Dengue; validated live "
                              "against what the active credential can reach")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--papers", type=int, default=None,
                         help="cap the number of papers processed, for smoke runs")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                         help="how many per-paper model calls run at once "
                              "(default %d); one call per paper either way"
                              % DEFAULT_CONCURRENCY)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_path, summary = run_extraction(
        organism=args.organism,
        gene=args.gene,
        data_type=args.data_type,
        model=args.model,
        collection=args.collection,
        collection_ids=args.collections,
        top_k=args.top_k,
        papers=args.papers,
        year=args.year,
        out_dir=args.out_dir,
        concurrency=args.concurrency,
    )
    print_summary(args.gene, args.data_type, args.model, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
