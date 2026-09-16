#!/usr/bin/env python3
"""Local server for the extraction pipeline: a run form, a job queue, and a
results/spot-check viewer. Standard library only (http.server, json,
threading, queue). Binds to 127.0.0.1 only, so the RAGStack API key never
leaves this machine and the browser never calls RAGStack directly.

Imports extract.py's row/record building blocks (retrieval, prompt build,
model call, row conversion) rather than extract.run_extraction() itself: the
job runner needs to report progress and check for a stop request between
papers, and run_extraction() is one opaque call with no hook for either. See
CONTRACT.md for the row/record shapes and RAGStack facts this relies on.

Usage:
    python3 serve.py --port 8765
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import batch  # noqa: E402
import extract  # noqa: E402
import prompt as prompt_mod  # noqa: E402
import ragstack  # noqa: E402
import schemas  # noqa: E402
import verify  # noqa: E402

STATIC_UI = os.path.join(THIS_DIR, "ui.html")
STATIC_SPOTCHECK = os.path.join(THIS_DIR, "spotcheck.html")
OUT_DIR = extract.DEFAULT_OUT_DIR
VERDICTS_DIR = os.path.join(OUT_DIR, "verdicts")

# Static per the contract's model table (Extraction_build_spec, line 27):
# gpt56luna first, then claudeopus5. Not a live Argo lookup on purpose --
# argo.py's "models" subcommand hits a different gateway than RAGStack and
# a hackathon server has no business depending on a second live network
# call just to populate a dropdown.
MODELS = [
    {"id": "gpt56luna", "label": "gpt56luna"},
    {"id": "claudeopus5", "label": "claudeopus5"},
]

DEFAULT_MAX_CONCURRENT = 2

# A batch multiplies two independent levels of parallelism: how many jobs
# (gene/data-type combinations) run at once, and how many per-paper model
# calls each running job fires at once. The two must not multiply into a
# runaway against RAGStack/the model gateway, so a batch always caps their
# product here, the same 8 that batch.py's CLI path enforces.
MAX_TOTAL_IN_FLIGHT_CALLS = 8

# Default paper cap when the caller gives no papers value (or an empty/zero
# one) and has not explicitly opted into no_paper_limit. Every run is bounded
# by default; unlimited is an explicit choice, never a side effect of an
# empty form field.
DEFAULT_PAPERS_CAP = 5

# Measured on this event, September 16: about 9 seconds per model call
# (one call per paper). Used only to project a run's cost before an
# unlimited run starts; never used to bound anything.
SECONDS_PER_PAPER = 9

# A small, realistic sample passage so the prompt panel always has real text
# to show, even before any job has run. Not retrieved data -- built once,
# never sent to RAGStack or the model.
_FAKE_PROMPT_CHUNK = {
    "chunk_id": "chunk-example001",
    "content": (
        "The E627K substitution in PB2 increased polymerase activity in "
        "mammalian cells at 33C compared to the avian-signature 627E."
    ),
    "metadata": {
        "pmid": "24899203",
        "pmcid": "PMC4136279",
        "year": 2014,
        "title": "Example paper used only to preview the prompt template",
    },
}


def _default_prompt_body(data_type=None):
    """The per-paper prompt body built from the sample passage above, so
    the prompt panel has something real to show before any job exists or
    while a job's retrieval is not yet cached.
    """
    if data_type not in schemas.DATA_TYPES:
        data_type = "mutation"
    spec = {
        "organism": "Influenza A virus",
        "gene": "PB2",
        "aliases": ["PB2"],
        "additional_terms": "",
        "data_type": data_type,
    }
    return prompt_mod.build_prompt(spec, [_FAKE_PROMPT_CHUNK])


# ---------------------------------------------------------------------------
# Shared retrieval helper
# ---------------------------------------------------------------------------


def _retrieve_for_params(params):
    """Run the retrieval phase only (RAGStack collect across collections,
    grouped by paper). No model calls. Shared by the run job, which then
    reads each retrieved paper, and by the /api/estimate endpoint, which
    only needs the paper count and a cost projection before an unlimited
    run is allowed to start.
    """
    env = ragstack.load_env()
    gene = params["gene"]
    data_type = params["data_type"]
    aliases = extract.aliases_for_gene(gene)
    collections_list = extract.parse_collections(
        collection=params["collection"], env=env
    )
    filters = {"year": params["year"]} if params.get("year") else {}
    data_type_terms = extract.DATA_TYPE_QUERY_TERMS[data_type]

    kept, manifest, stats = extract.collect_multi_collection(
        params["organism"], gene, aliases, data_type_terms,
        collections_list, params["top_k"], filters, env,
    )

    by_doc = {}
    for source in kept:
        by_doc.setdefault(source.get("doc_id"), []).append(source)
    doc_ids = list(by_doc.keys())

    chunks_by_id = {}
    for doc_id in doc_ids:
        for source in by_doc[doc_id]:
            chunks_by_id[source.get("chunk_id")] = source

    spec = {
        "organism": params["organism"],
        "gene": gene,
        "aliases": aliases,
        "additional_terms": "",
        "data_type": data_type,
    }
    return {
        "doc_ids": doc_ids,
        "by_doc": by_doc,
        "chunks_by_id": chunks_by_id,
        "manifest": manifest,
        "stats": stats,
        "spec": spec,
        "collections_list": collections_list,
    }


def _paper_rows_from_by_doc(doc_ids, by_doc):
    """Build the browser-facing paper list: title, year, pubmed_id, and a
    per-paper status the run loop updates as it goes. Shown the moment
    retrieval finishes, before any model call.
    """
    rows = []
    for doc_id in doc_ids:
        chunks = by_doc.get(doc_id) or []
        meta = (chunks[0].get("metadata") if chunks else None) or {}
        rows.append({
            "doc_id": doc_id,
            "title": meta.get("title"),
            "year": meta.get("year"),
            "pubmed_id": meta.get("pmid"),
            "status": "waiting",
            "rows": None,
        })
    return rows


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------


class JobStore:
    """Thread-safe job registry plus a bounded worker pool.

    A job is one call to extract.run_extraction. Jobs are submitted to a
    queue.Queue and picked up by a fixed pool of worker threads, so
    max_concurrent actually bounds concurrency rather than just labeling it.
    """

    def __init__(self, max_concurrent=DEFAULT_MAX_CONCURRENT):
        self._lock = threading.Lock()
        self._jobs = {}
        self._retrieval_cache = {}  # job_id -> {doc_ids, by_doc, chunks_by_id, spec}
        self._queue = queue.Queue()
        self._max_concurrent = max_concurrent
        self._workers = []
        for _ in range(max_concurrent):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self._workers.append(thread)

    def _worker_loop(self):
        while True:
            job_id = self._queue.get()
            try:
                job = self.get(job_id)
                if job and job.get("type") == "rerun":
                    self._run_rerun_job(job_id)
                else:
                    self._run_job(job_id)
            except Exception:  # a bad job must not kill the worker thread
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is not None:
                        job["status"] = "failed"
                        job["error"] = "internal error: " + traceback.format_exc(limit=3)
            finally:
                self._queue.task_done()

    def submit(self, params):
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "type": "run",
            "status": "queued",
            "stage": "queued",
            "params": params,
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "papers_done": 0,
            "papers_total": None,
            "papers": [],
            "cancel_requested": False,
            "error": None,
            "out_path": None,
            "summary": None,
            "parent_job_id": None,
            "prompt_edited": False,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job_id

    def request_stop(self, job_id):
        """Ask a queued or running job to stop between papers. Returns True
        when the request was accepted (the job will end as "stopped" and
        keep whatever rows it already produced), False when the job is
        already finished and there is nothing to stop.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job["status"] not in ("queued", "running"):
                return False
            job["cancel_requested"] = True
            return True

    def _is_cancelled(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.get("cancel_requested"))

    def _set_paper_status(self, job_id, index, status, rows=None):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            papers = job.get("papers")
            if papers and 0 <= index < len(papers):
                papers[index]["status"] = status
                if rows is not None:
                    papers[index]["rows"] = rows

    def _run_papers_parallel(self, job_id, doc_ids, by_doc, spec, model, data_type, gene,
                              chunks_by_id, concurrency, system_text=None,
                              prompt_body_override=None):
        """Read every paper in doc_ids through the model, in parallel, one
        call per paper (never grouped into one prompt). Shared by a fresh
        run and a rerun, which differ only in which system text/prompt body
        they use.

        A call's own future flips that paper's status to "reading" the
        moment it actually starts (not merely when it is submitted), and to
        "done" or "refused" the moment it finishes, so the browser sees
        live per-paper status, not just a status jump at the end.

        A stop request (checked after every completion) cancels every
        future that has not started yet, so no new paper begins. Futures
        already running are left alone and their rows are kept; the
        ThreadPoolExecutor's queue is FIFO, so the papers actually started
        are always doc_ids[0:k] for some k, but the code below does not
        rely on that and instead keys off which doc_ids actually produced a
        result.

        Rows are collected as they complete but only assembled into
        proposed_rows afterward, walking doc_ids in its fixed (manifest)
        order, so row_id numbering never depends on completion timing.

        Returns (proposed_rows, failed_papers, model_calls, stopped,
        processed_count).
        """
        system_text = system_text or prompt_mod.SYSTEM
        workers = max(1, concurrency or extract.DEFAULT_CONCURRENCY)

        def _task(index, doc_id):
            self._set_paper_status(job_id, index, "reading")
            paper_chunks = by_doc[doc_id]
            paper_prompt = prompt_body_override or prompt_mod.build_prompt(spec, paper_chunks)
            reply, attempts, err = extract.call_model_with_retry(model, paper_prompt, system_text)
            return index, doc_id, reply, attempts, err

        results_by_doc = {}
        model_calls = 0
        completed = 0
        stopped = False

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_task, index, doc_id): (index, doc_id)
                for index, doc_id in enumerate(doc_ids)
            }
            for future in as_completed(futures):
                try:
                    index, doc_id, reply, attempts, err = future.result()
                except CancelledError:
                    continue

                model_calls += attempts
                completed += 1

                if reply is None:
                    status = "refused"
                    parsed_rows = None
                else:
                    parsed = extract.extract_json(reply)
                    if parsed is None or "rows" not in parsed:
                        status = "refused"
                        parsed_rows = None
                        err = err or "no parseable JSON rows in reply"
                    else:
                        parsed_rows = parsed.get("rows") or []
                        status = "done"

                results_by_doc[doc_id] = {"err": err, "rows": parsed_rows, "status": status}
                self._set_paper_status(
                    job_id, index, status,
                    rows=(len(parsed_rows) if parsed_rows is not None else None),
                )
                self._set(job_id, papers_done=completed, papers_total=len(doc_ids))

                if not stopped and self._is_cancelled(job_id):
                    stopped = True
                    # Stop scheduling new papers: cancel every future that
                    # has not started running yet. A future already running
                    # cannot be cancelled and is left to finish.
                    for other_future in futures:
                        if not other_future.done():
                            other_future.cancel()

        proposed_rows = []
        failed_papers = []
        counter = 0
        for doc_id in doc_ids:
            entry = results_by_doc.get(doc_id)
            if entry is None:
                continue  # never started (cancelled before it began)
            if entry["status"] == "refused":
                failed_papers.append({"doc_id": doc_id, "reason": entry["err"]})
                continue
            for model_row in entry["rows"]:
                counter += 1
                proposed_rows.append(
                    extract.convert_model_row(model_row, data_type, gene, counter, chunks_by_id)
                )

        processed_count = len(results_by_doc)
        return proposed_rows, failed_papers, model_calls, stopped, processed_count

    def submit_rerun(self, parent_job_id, system_text, prompt_body_text, model_override=None):
        """Queue a rerun of parent_job_id's cached retrieval with an edited
        system prompt (and optionally a verbatim per-paper prompt body).
        Raises ValueError if the parent has no cached retrieval to reuse.
        """
        with self._lock:
            parent = self._jobs.get(parent_job_id)
            cache = self._retrieval_cache.get(parent_job_id)
        if parent is None:
            raise ValueError("no such job: %s" % parent_job_id)
        if cache is None:
            raise ValueError(
                "job %s has no cached retrieval to rerun against "
                "(only a finished, non-rerun job can be rerun)" % parent_job_id
            )
        job_id = uuid.uuid4().hex[:12]
        params = dict(parent["params"])
        if model_override:
            params["model"] = model_override
        # A rerun replays cached retrieval, so the paper list is already
        # known at submit time, not just after a retrieval step. Expose it
        # immediately rather than waiting for the worker to start.
        paper_rows = _paper_rows_from_by_doc(cache["doc_ids"], cache["by_doc"])
        job = {
            "id": job_id,
            "type": "rerun",
            "status": "queued",
            "stage": "queued",
            "params": params,
            "queued_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "papers_done": 0,
            "papers_total": len(cache["doc_ids"]),
            "papers": paper_rows,
            "cancel_requested": False,
            "error": None,
            "out_path": None,
            "summary": None,
            "parent_job_id": parent_job_id,
            "prompt_edited": True,
            "rerun_system": system_text,
            "rerun_prompt_body": prompt_body_text or None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job_id

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def get_cached_retrieval(self, job_id):
        with self._lock:
            return self._retrieval_cache.get(job_id)

    def list_all(self):
        with self._lock:
            return [dict(job) for job in self._jobs.values()]

    def _set(self, job_id, **fields):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def _run_job(self, job_id):
        """Run one extraction with real, visible progress: retrieve first
        and publish the paper list before any model call, then read papers
        one at a time so the job's progress fields change per paper, and
        check for a stop request between papers so a cancel keeps whatever
        rows are already in hand. This calls extract.py's building blocks
        directly (the same ones _run_rerun_job already uses) instead of the
        single opaque extract.run_extraction() call, because that call has
        no hook to report progress or to stop mid-run.
        """
        job = self.get(job_id)
        if job is None:
            return
        params = job["params"]
        self._set(job_id, status="running", started_at=time.time(), stage="retrieving")

        try:
            retrieval = _retrieve_for_params(params)
        except Exception as error:
            self._set(job_id, status="failed", error=str(error),
                      finished_at=time.time(), stage="failed")
            return

        doc_ids = retrieval["doc_ids"]
        by_doc = retrieval["by_doc"]
        manifest = retrieval["manifest"]
        stats = retrieval["stats"]
        spec = retrieval["spec"]
        collections_list = retrieval["collections_list"]
        gene = params["gene"]
        data_type = params["data_type"]
        model = params["model"]

        no_paper_limit = bool(params.get("no_paper_limit"))
        papers_cap = params.get("papers")
        if not no_paper_limit and papers_cap is not None:
            doc_ids = doc_ids[:papers_cap]

        chunks_by_id = {}
        for doc_id in doc_ids:
            for source in by_doc[doc_id]:
                chunks_by_id[source.get("chunk_id")] = source

        # Paper list goes out now, before the first model call.
        paper_rows = _paper_rows_from_by_doc(doc_ids, by_doc)
        with self._lock:
            job_obj = self._jobs.get(job_id)
            if job_obj is not None:
                job_obj["papers"] = paper_rows
                job_obj["papers_total"] = len(doc_ids)
                job_obj["papers_done"] = 0
                job_obj["stage"] = "reading" if doc_ids else "checking"

        concurrency = params.get("concurrency") or extract.DEFAULT_CONCURRENCY

        proposed_rows, failed_papers, model_calls, stopped, processed_count = (
            self._run_papers_parallel(
                job_id, doc_ids, by_doc, spec, model, data_type, gene,
                chunks_by_id, concurrency,
            )
        )

        self._set(job_id, stage="checking")
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
            refusals.append(schemas.refusal_record(
                data_type, gene, "no passage supported a row",
                chunks_searched=len(chunks_by_id), papers_searched=len(doc_ids),
            ))

        self._set(job_id, stage="writing")

        run_id = extract.make_run_id()
        generated_at = extract._now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
        prompt_hash = prompt_mod.prompt_sha256(prompt_mod.SYSTEM)
        filters = {"year": params["year"]} if params.get("year") else {}
        collection_str = ",".join(collections_list)

        run = schemas.run_record(
            run_id=run_id, generated_at=generated_at, model=model,
            prompt_sha256=prompt_hash, prompt_edited=False,
            query={
                "organism": params["organism"], "gene": gene, "aliases": spec["aliases"],
                "additional_terms": "", "data_type": data_type,
                "collection": collection_str, "collections": collections_list,
                "top_k": params["top_k"], "filters": filters,
                "concurrency": concurrency,
            },
        )
        retrieval_block = {
            "chunks_returned": stats["chunks_returned"],
            "chunks_after_dedupe": stats["chunks_after_dedupe"],
            "duplicates_removed": stats["duplicates_removed"],
            "papers": stats["papers"],
            "paper_manifest": manifest,
            "collections_queried": collections_list,
            "by_collection": stats["by_collection"],
        }
        envelope = schemas.output_envelope(run, retrieval_block, final_accepted, omitted, refusals)
        if stopped:
            envelope["stopped"] = True
            envelope["stopped_note"] = (
                "This run was stopped by the user after %d of %d retrieved papers. "
                "The rows above come only from the papers processed before the stop."
            ) % (processed_count, len(doc_ids))

        tally_problems = schemas.check_tally(envelope)
        if tally_problems:
            self._set(job_id, status="failed", error="tally check failed: " + "; ".join(tally_problems),
                      finished_at=time.time(), stage="failed")
            return

        gene_dir = os.path.join(OUT_DIR, gene)
        os.makedirs(gene_dir, exist_ok=True)
        suffix = "__stopped" if stopped else ""
        out_path = os.path.join(gene_dir, "%s__%s__%s%s.json" % (gene, data_type, run_id, suffix))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2)
            handle.write("\n")

        omit_reasons = {}
        for rec in omitted:
            reason = rec.get("omit_reason") or "unknown"
            omit_reasons[reason] = omit_reasons.get(reason, 0) + 1

        summary = {
            "out_path": out_path,
            "papers_returned": stats["papers"],
            "papers_processed": processed_count,
            "proposed": envelope["tally"]["proposed"],
            "emitted": envelope["tally"]["emitted"],
            "omitted": envelope["tally"]["omitted"],
            "omit_reasons": omit_reasons,
            "refusals": len(refusals),
            "failed_papers": failed_papers,
            "model_calls": model_calls,
            "concurrency": concurrency,
            "stopped": stopped,
        }

        final_status = "stopped" if stopped else "done"
        self._set(job_id, status=final_status, out_path=out_path, summary=summary,
                  papers_done=processed_count, papers_total=len(doc_ids),
                  finished_at=time.time(), stage=final_status)

        # Cache the retrieval already fetched above so a rerun can replay
        # these same papers with zero new RAGStack calls -- no second
        # retrieval pass is needed, unlike the previous implementation.
        with self._lock:
            self._retrieval_cache[job_id] = {
                "doc_ids": doc_ids,
                "by_doc": by_doc,
                "chunks_by_id": chunks_by_id,
                "spec": spec,
            }

    def _run_rerun_job(self, job_id):
        job = self.get(job_id)
        if job is None:
            return
        parent_id = job["parent_job_id"]
        cache = self.get_cached_retrieval(parent_id)
        if cache is None:
            self._set(job_id, status="failed", error="parent retrieval no longer cached",
                       finished_at=time.time())
            return

        params = job["params"]
        system_text = job["rerun_system"] or prompt_mod.SYSTEM
        prompt_body_override = job.get("rerun_prompt_body")
        spec = cache["spec"]
        by_doc = cache["by_doc"]
        doc_ids = cache["doc_ids"]
        chunks_by_id = cache["chunks_by_id"]

        self._set(job_id, status="running", started_at=time.time(),
                  stage="reading" if doc_ids else "checking")

        concurrency = params.get("concurrency") or extract.DEFAULT_CONCURRENCY

        proposed_rows, failed_papers, model_calls, stopped, processed_count = (
            self._run_papers_parallel(
                job_id, doc_ids, by_doc, spec, params["model"], params["data_type"],
                params["gene"], chunks_by_id, concurrency,
                system_text=system_text, prompt_body_override=prompt_body_override,
            )
        )

        self._set(job_id, stage="checking")
        accepted, omitted = verify.verify_rows(proposed_rows, chunks_by_id, spec)
        final_accepted = []
        for row in accepted:
            problems = schemas.validate_row(params["data_type"], row)
            if problems:
                omitted.append(schemas.omitted_record(row, "schema_fail", "; ".join(problems)))
            else:
                final_accepted.append(row)

        refusals = []
        if not final_accepted:
            refusals.append(schemas.refusal_record(
                params["data_type"], params["gene"], "no passage supported a row",
                chunks_searched=len(chunks_by_id), papers_searched=len(doc_ids),
            ))

        run_id = extract.make_run_id()
        generated_at = extract._now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")
        prompt_hash = prompt_mod.prompt_sha256(system_text + "\n" + (prompt_body_override or ""))

        run = schemas.run_record(
            run_id=run_id, generated_at=generated_at, model=params["model"],
            prompt_sha256=prompt_hash, prompt_edited=True,
            query={
                "organism": params["organism"], "gene": params["gene"],
                "aliases": spec["aliases"], "additional_terms": "",
                "data_type": params["data_type"], "collection": params["collection"],
                "top_k": params["top_k"], "filters": {"year": params["year"]} if params.get("year") else {},
                "concurrency": concurrency,
            },
        )
        retrieval = {
            "chunks_returned": len(chunks_by_id),
            "chunks_after_dedupe": len(chunks_by_id),
            "duplicates_removed": 0,
            "papers": len(doc_ids),
            "paper_manifest": ragstack.paper_manifest(list(chunks_by_id.values())),
        }
        envelope = schemas.output_envelope(run, retrieval, final_accepted, omitted, refusals)
        envelope["prompt_edited_note"] = (
            "This run used an edited prompt, not prompt.py's default SYSTEM text. "
            "Per the execution plan, it is not comparable to golden-dataset runs."
        )
        if stopped:
            envelope["stopped"] = True
            envelope["stopped_note"] = (
                "This rerun was stopped by the user after %d of %d cached papers. "
                "The rows above come only from the papers processed before the stop."
            ) % (processed_count, len(doc_ids))

        self._set(job_id, stage="writing")

        tally_problems = schemas.check_tally(envelope)
        if tally_problems:
            self._set(job_id, status="failed", error="tally check failed: " + "; ".join(tally_problems),
                       finished_at=time.time(), stage="failed")
            return

        gene_dir = os.path.join(OUT_DIR, params["gene"])
        os.makedirs(gene_dir, exist_ok=True)
        suffix = "__stopped" if stopped else ""
        out_path = os.path.join(gene_dir, "%s__%s__%s__rerun%s.json" % (params["gene"], params["data_type"], run_id, suffix))
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2)
            handle.write("\n")

        summary = {
            "out_path": out_path,
            "papers_processed": processed_count,
            "proposed": envelope["tally"]["proposed"],
            "emitted": envelope["tally"]["emitted"],
            "omitted": envelope["tally"]["omitted"],
            "refusals": len(refusals),
            "failed_papers": failed_papers,
            "model_calls": model_calls,
            "concurrency": concurrency,
            "stopped": stopped,
        }
        final_status = "stopped" if stopped else "done"
        self._set(job_id, status=final_status, out_path=out_path, summary=summary,
                   papers_done=processed_count, papers_total=len(doc_ids),
                   finished_at=time.time(), stage=final_status)
        self._retrieval_cache_copy_for_rerun(job_id, cache)

    def _retrieval_cache_copy_for_rerun(self, job_id, cache):
        # A rerun can itself be rerun again (e.g. tweak further), replaying
        # the same cached papers as its parent -- no new retrieval either.
        with self._lock:
            self._retrieval_cache[job_id] = cache


JOBS = JobStore()


def job_view(job):
    """Trim a job dict to what the browser needs: a stage-aware progress
    line ("reading paper 3 of 5", not just the word "running"), the paper
    list as soon as retrieval knows it, and whether a stop can still be
    requested.
    """
    if job is None:
        return None
    started = job.get("started_at")
    finished = job.get("finished_at")
    now = time.time()
    if finished:
        elapsed = finished - (started or job["queued_at"])
    elif started:
        elapsed = now - started
    else:
        elapsed = 0.0

    status = job["status"]
    stage = job.get("stage") or status
    papers_done = job.get("papers_done") or 0
    papers_total = job.get("papers_total")

    if status == "queued":
        progress_line = "queued"
    elif stage == "retrieving":
        progress_line = "retrieving passages"
    elif stage == "reading" and papers_total:
        progress_line = "read %d of %d papers" % (papers_done, papers_total)
    elif stage == "checking":
        progress_line = "checking quotes"
    elif stage == "writing":
        progress_line = "writing results"
    elif status == "done":
        progress_line = "done (%d papers)" % papers_done if papers_total else "done"
    elif status == "stopped":
        progress_line = "stopped (%d papers kept)" % papers_done
    elif status == "failed":
        progress_line = "failed"
    else:
        progress_line = status

    summary = job.get("summary") or {}
    tally = {
        "proposed": summary.get("proposed"),
        "emitted": summary.get("emitted"),
        "omitted": summary.get("omitted"),
        "omit_reasons": summary.get("omit_reasons"),
        "refusals": summary.get("refusals"),
    }

    return {
        "id": job["id"],
        "type": job.get("type", "run"),
        "status": status,
        "stage": stage,
        "params": job["params"],
        "progress_line": progress_line,
        "papers_done": papers_done,
        "papers_total": papers_total,
        "papers": job.get("papers") or [],
        "elapsed_seconds": round(elapsed, 1),
        "tally": tally,
        "out_path": job.get("out_path"),
        "error": job.get("error"),
        "parent_job_id": job.get("parent_job_id"),
        "prompt_edited": bool(job.get("prompt_edited")),
        "stoppable": status in ("queued", "running"),
    }


# ---------------------------------------------------------------------------
# Batch queueing (genes x data types, respecting max_concurrent and a cap)
# ---------------------------------------------------------------------------


def queue_batch(genes, data_types, params_common, max_concurrent, cap):
    """Submit one job per (gene, data_type) combination. The JobStore's fixed
    worker pool already enforces max_concurrent; the total-call cap is
    approximated by capping the papers processed per job once the estimated
    remaining budget runs low, mirroring batch.py's own budget logic.

    Each queued job also fires its own per-paper model calls in parallel
    (extract.DEFAULT_CONCURRENCY by default). max_concurrent jobs times
    that per-job concurrency is the real total of in-flight model calls, so
    the per-job concurrency passed to every job here is clamped to keep
    that product at or below MAX_TOTAL_IN_FLIGHT_CALLS, and the effective
    number is printed once, up front.
    """
    requested_concurrency = params_common.get("concurrency") or extract.DEFAULT_CONCURRENCY
    max_concurrent = max(1, max_concurrent)
    job_concurrency = max(1, min(requested_concurrency, MAX_TOTAL_IN_FLIGHT_CALLS // max_concurrent))
    print(
        "batch in-flight model calls: max_concurrent jobs=%d x concurrency per job=%d "
        "= %d (cap %d)" % (
            max_concurrent, job_concurrency, max_concurrent * job_concurrency,
            MAX_TOTAL_IN_FLIGHT_CALLS,
        )
    )

    job_ids = []
    remaining = cap
    no_paper_limit = bool(params_common.get("no_paper_limit"))
    for gene in genes:
        for data_type in data_types:
            papers = params_common.get("papers")
            job_no_paper_limit = no_paper_limit
            if remaining is not None:
                # An explicit total call cap always wins over no_paper_limit:
                # the user asked for a hard ceiling on this batch, so an
                # individual job never gets to ignore it.
                if remaining <= 0:
                    break
                papers = remaining if papers is None else min(papers, remaining)
                job_no_paper_limit = False
            job_id = JOBS.submit({
                "organism": params_common["organism"],
                "gene": gene,
                "data_type": data_type,
                "model": params_common["model"],
                "collection": params_common["collection"],
                "top_k": params_common["top_k"],
                "papers": papers,
                "no_paper_limit": job_no_paper_limit,
                "year": params_common["year"],
                "concurrency": job_concurrency,
            })
            job_ids.append(job_id)
            if remaining is not None:
                # Optimistic accounting (one model call per paper, worst
                # case): corrected once real summaries land, same spirit as
                # batch.py's own remaining_budget bookkeeping.
                remaining -= papers if papers is not None else 0
    return job_ids


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "ExtractCurator/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ----------------------------------------------------------

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=400):
        self.log_message("error %s: %s", self.path, message)
        self._send_json({"error": message}, status=status)

    def _send_file(self, path, content_type):
        try:
            with open(path, "rb") as handle:
                body = handle.read()
        except OSError as error:
            self._send_error_json("could not read %s: %s" % (path, error), status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    # -- routing ------------------------------------------------------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                self._send_file(STATIC_UI, "text/html; charset=utf-8")
            elif path == "/static/spotcheck.html":
                self._send_file(STATIC_SPOTCHECK, "text/html; charset=utf-8")
            elif path == "/api/collections":
                self._handle_collections()
            elif path == "/api/models":
                self._send_json({"models": MODELS})
            elif path == "/api/prompt":
                self._handle_prompt(urllib.parse.parse_qs(parsed.query))
            elif path.startswith("/api/compare/"):
                self._handle_compare(path[len("/api/compare/"):])
            elif path == "/api/jobs":
                self._send_json({"jobs": [job_view(j) for j in JOBS.list_all()]})
            elif path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/"):]
                job = JOBS.get(job_id)
                if job is None:
                    self._send_error_json("no such job: %s" % job_id, status=404)
                else:
                    self._send_json(job_view(job))
            elif path.startswith("/api/result/"):
                job_id = path[len("/api/result/"):]
                self._handle_result(job_id)
            else:
                self._send_error_json("not found: %s" % path, status=404)
        except Exception as error:
            self._send_error_json("internal error: %s" % error, status=500)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/run":
                self._handle_run()
            elif path == "/api/batch":
                self._handle_batch()
            elif path == "/api/verdicts":
                self._handle_verdicts()
            elif path == "/api/rerun":
                self._handle_rerun()
            elif path == "/api/estimate":
                self._handle_estimate()
            elif path.startswith("/api/jobs/") and path.endswith("/stop"):
                job_id = path[len("/api/jobs/"):-len("/stop")].strip("/")
                self._handle_stop(job_id)
            else:
                self._send_error_json("not found: %s" % path, status=404)
        except json.JSONDecodeError as error:
            self._send_error_json("body is not valid JSON: %s" % error, status=400)
        except Exception as error:
            self._send_error_json("internal error: %s" % error, status=500)

    # -- handlers -----------------------------------------------------------

    def _handle_collections(self):
        try:
            env = ragstack.load_env()
            base = env["RAGSTACK_BASE"].rstrip("/")
            mode = ragstack.auth_mode(env)
            data = ragstack._request("GET", base + "/v1/collections", env)
        except ragstack.RAGStackError as error:
            self._send_error_json("could not reach RAGStack: %s" % error, status=502)
            return
        raw = data.get("collections", data if isinstance(data, list) else [])
        collections = [
            {"id": entry.get("id"), "label": entry.get("label") or entry.get("id")}
            for entry in raw
            if isinstance(entry, dict) and entry.get("id")
        ]
        # auth_mode names which credential is active (token or key), never
        # the credential value, so the page can show the user which one is
        # supplying the collections listed here (e.g. Dengue only appears
        # when the token is active).
        self._send_json({"collections": collections, "auth_mode": mode})

    def _handle_run(self):
        body = self._read_json_body()
        try:
            params = _params_from_body(body)
        except ValueError as error:
            self._send_error_json(str(error), status=400)
            return
        job_id = JOBS.submit(params)
        self._send_json({"job_id": job_id})

    def _handle_stop(self, job_id):
        job = JOBS.get(job_id)
        if job is None:
            self._send_error_json("no such job: %s" % job_id, status=404)
            return
        ok = JOBS.request_stop(job_id)
        if not ok:
            self._send_error_json(
                "job %s cannot be stopped (status: %s)" % (job_id, job["status"]), status=400)
            return
        self._send_json({"stopping": True, "job_id": job_id})

    def _handle_estimate(self):
        """Retrieval-only preview: how many papers this query would return
        and what an uncapped run would cost, so the page can show that
        before a no_paper_limit run is allowed to start. No model calls.
        """
        body = self._read_json_body()
        try:
            params = _params_from_body(body, require_gene=True, require_data_type=True)
        except ValueError as error:
            self._send_error_json(str(error), status=400)
            return
        try:
            retrieval = _retrieve_for_params(params)
        except Exception as error:
            self._send_error_json("could not estimate: %s" % error, status=502)
            return
        papers_total = len(retrieval["doc_ids"])
        self._send_json({
            "papers_total": papers_total,
            "projected_calls": papers_total,
            "projected_seconds": papers_total * SECONDS_PER_PAPER,
            "seconds_per_paper": SECONDS_PER_PAPER,
        })

    def _handle_batch(self):
        body = self._read_json_body()
        genes_raw = body.get("genes")
        if isinstance(genes_raw, str):
            genes = [g.strip() for g in genes_raw.split(",") if g.strip()]
        elif isinstance(genes_raw, list):
            genes = [str(g).strip() for g in genes_raw if str(g).strip()]
        else:
            genes = []
        if not genes:
            self._send_error_json("genes must be a non-empty list or comma string", status=400)
            return

        data_types_raw = body.get("data_types")
        if isinstance(data_types_raw, list) and data_types_raw:
            data_types = [d for d in data_types_raw if d in schemas.DATA_TYPES]
        else:
            data_types = sorted(schemas.DATA_TYPES)
        if not data_types:
            self._send_error_json("no valid data_types given", status=400)
            return

        try:
            common = _params_from_body(body, require_gene=False, require_data_type=False)
        except ValueError as error:
            self._send_error_json(str(error), status=400)
            return

        max_concurrent = body.get("max_concurrent") or DEFAULT_MAX_CONCURRENT
        cap = body.get("cap")
        try:
            max_concurrent = int(max_concurrent)
            cap = int(cap) if cap is not None else None
        except (TypeError, ValueError):
            self._send_error_json("max_concurrent and cap must be integers", status=400)
            return

        job_ids = queue_batch(genes, data_types, common, max_concurrent, cap)
        self._send_json({"job_ids": job_ids})

    def _handle_result(self, job_id):
        job = JOBS.get(job_id)
        if job is None:
            # The job store is in memory only. If the server process was
            # restarted since this job ran, its record (and any progress)
            # is gone -- that is a different, more explainable situation
            # than "you typed the wrong id", so say so.
            self._send_error_json(
                "job %s is not known to this server. If the server was "
                "restarted since that job ran, its in-memory record is "
                "gone; check out/<gene>/ for a file it may have written "
                "before the process stopped." % job_id,
                status=404,
            )
            return
        out_path = job.get("out_path")
        if not out_path:
            status = job["status"]
            if status == "failed":
                self._send_error_json(
                    "this run failed before it produced results: %s"
                    % (job.get("error") or "unknown error"),
                    status=409,
                )
            elif status in ("queued", "running"):
                self._send_error_json(
                    "this run is still in progress (status: %s, %s) and has no results yet"
                    % (status, job_view(job)["progress_line"]),
                    status=404,
                )
            else:
                # A job can end up here (finished status, no out_path) only
                # if it was interrupted before it could write its file --
                # for example the server process was stopped mid-run.
                self._send_error_json(
                    "this run was interrupted before it finished, so it has no results "
                    "(status: %s)" % status,
                    status=404,
                )
            return
        try:
            with open(out_path, "r", encoding="utf-8") as handle:
                envelope = json.load(handle)
        except OSError as error:
            self._send_error_json("could not read result file: %s" % error, status=500)
            return
        self._send_json(envelope)

    def _handle_verdicts(self):
        body = self._read_json_body()
        job_id = body.get("job_id")
        records = body.get("verdicts") or body.get("verification_records")
        if not job_id or not isinstance(records, list):
            self._send_error_json("body must carry job_id and a verdicts list", status=400)
            return
        os.makedirs(VERDICTS_DIR, exist_ok=True)
        out_path = os.path.join(VERDICTS_DIR, "%s__verdicts.json" % job_id)
        payload = {
            "job_id": job_id,
            "saved_at": time.time(),
            "verdicts": records,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        self._send_json({"saved": True, "path": out_path})

    def _handle_prompt(self, query):
        """Always return a real, non-empty prompt. With no job_id (or a job
        with nothing cached yet) this is a template built from a small
        sample passage, not a live job's papers -- `source` says which one
        the caller is looking at, so the panel never renders blank and
        never quietly passes off a template as a real run.
        """
        job_id = (query.get("job_id") or [None])[0]
        data_type = (query.get("data_type") or [None])[0]
        default_body = _default_prompt_body(data_type)
        result = {
            "system": prompt_mod.SYSTEM,
            "default_system": prompt_mod.SYSTEM,
            "per_paper_prompt": default_body,
            "default_per_paper_prompt": default_body,
            "doc_id": None,
            "source": "default",
            "prompt_note": (
                "showing the default prompt template, built from a sample passage "
                "(no job selected yet, or this job's papers are not cached)"
            ),
        }
        if job_id:
            job = JOBS.get(job_id)
            if job is None:
                self._send_error_json("no such job: %s" % job_id, status=404)
                return
            cache = JOBS.get_cached_retrieval(job_id)
            if cache and cache["doc_ids"]:
                first_doc = cache["doc_ids"][0]
                paper_chunks = cache["by_doc"][first_doc]
                built = prompt_mod.build_prompt(cache["spec"], paper_chunks)
                result["per_paper_prompt"] = built
                result["default_per_paper_prompt"] = built
                result["doc_id"] = first_doc
                result["source"] = "job"
                result["prompt_note"] = (
                    "showing the exact prompt sent for job %s, built from paper %s"
                    % (job_id, first_doc)
                )
            else:
                result["prompt_note"] = (
                    "no cached retrieval for job %s yet (still running, failed, "
                    "or caching did not complete); showing the default prompt "
                    "template instead" % job_id
                )
        self._send_json(result)

    def _handle_rerun(self):
        body = self._read_json_body()
        job_id = body.get("job_id")
        system_text = body.get("system")
        prompt_body = body.get("prompt_body")
        model_override = body.get("model")
        if not job_id or not system_text:
            self._send_error_json("job_id and system are required", status=400)
            return
        try:
            new_job_id = JOBS.submit_rerun(job_id, system_text, prompt_body, model_override)
        except ValueError as error:
            self._send_error_json(str(error), status=400)
            return
        self._send_json({"job_id": new_job_id, "parent_job_id": job_id})

    def _handle_compare(self, job_id):
        job = JOBS.get(job_id)
        if job is None:
            self._send_error_json("no such job: %s" % job_id, status=404)
            return
        parent_id = job.get("parent_job_id")
        if not parent_id:
            self._send_error_json("job %s is not a rerun (no parent_job_id)" % job_id, status=400)
            return
        parent = JOBS.get(parent_id)
        if parent is None or not parent.get("out_path"):
            self._send_error_json("parent job %s has no result to compare against" % parent_id, status=404)
            return
        if not job.get("out_path"):
            self._send_error_json("job %s has no result yet (status: %s)" % (job_id, job["status"]), status=404)
            return
        try:
            with open(parent["out_path"], "r", encoding="utf-8") as handle:
                parent_envelope = json.load(handle)
            with open(job["out_path"], "r", encoding="utf-8") as handle:
                rerun_envelope = json.load(handle)
        except OSError as error:
            self._send_error_json("could not read result files: %s" % error, status=500)
            return

        data_type = (rerun_envelope.get("run") or {}).get("query", {}).get("data_type")
        comparison = _compare_envelopes(parent_envelope, rerun_envelope, data_type)
        self._send_json(comparison)


def _row_compare_key(row, data_type):
    if data_type == "ppi":
        pair = sorted([
            (row.get("protein_a") or "").strip().lower(),
            (row.get("protein_b") or "").strip().lower(),
        ])
        return "ppi:" + "|".join(pair)
    # mutation, and any other/unknown data type: gene plus the claimed value.
    gene = (row.get("gene_name") or "").strip().lower()
    value = (row.get("mutation") or "").strip().lower()
    return "mutation:%s:%s" % (gene, value)


def _compare_envelopes(parent_envelope, rerun_envelope, data_type):
    parent_rows = parent_envelope.get("rows") or []
    rerun_rows = rerun_envelope.get("rows") or []

    parent_by_key = {_row_compare_key(r, data_type): r for r in parent_rows}
    rerun_by_key = {_row_compare_key(r, data_type): r for r in rerun_rows}

    only_parent_keys = set(parent_by_key) - set(rerun_by_key)
    only_rerun_keys = set(rerun_by_key) - set(parent_by_key)
    both_keys = set(parent_by_key) & set(rerun_by_key)

    def omit_reasons(envelope):
        counts = {}
        for rec in envelope.get("omitted") or []:
            reason = rec.get("omit_reason") or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    return {
        "parent_job": (parent_envelope.get("run") or {}).get("run_id"),
        "rerun_job": (rerun_envelope.get("run") or {}).get("run_id"),
        "only_in_original": [parent_by_key[k] for k in sorted(only_parent_keys)],
        "only_in_rerun": [rerun_by_key[k] for k in sorted(only_rerun_keys)],
        "in_both": [parent_by_key[k] for k in sorted(both_keys)],
        "tally": {
            "original": {
                "proposed": (parent_envelope.get("tally") or {}).get("proposed"),
                "emitted": (parent_envelope.get("tally") or {}).get("emitted"),
                "omitted": (parent_envelope.get("tally") or {}).get("omitted"),
                "omit_reasons": omit_reasons(parent_envelope),
                "refusals": len(parent_envelope.get("refusals") or []),
            },
            "rerun": {
                "proposed": (rerun_envelope.get("tally") or {}).get("proposed"),
                "emitted": (rerun_envelope.get("tally") or {}).get("emitted"),
                "omitted": (rerun_envelope.get("tally") or {}).get("omitted"),
                "omit_reasons": omit_reasons(rerun_envelope),
                "refusals": len(rerun_envelope.get("refusals") or []),
            },
        },
    }


def _params_from_body(body, require_gene=True, require_data_type=True):
    organism = body.get("organism") or "Influenza A virus"
    gene = body.get("gene")
    if require_gene and not gene:
        raise ValueError("gene is required")
    data_type = body.get("data_type")
    if require_data_type:
        if data_type not in schemas.DATA_TYPES:
            raise ValueError("data_type must be one of %s" % sorted(schemas.DATA_TYPES))

    collections_list = body.get("collections")
    if isinstance(collections_list, list) and collections_list:
        collection = ",".join(str(c) for c in collections_list)
    else:
        collection = body.get("collection") or "asm-semantic"

    top_k = body.get("top_k", 25)
    papers_raw = body.get("papers")
    no_paper_limit = bool(body.get("no_paper_limit"))
    year = body.get("year")
    model = body.get("model") or "gpt56luna"
    concurrency_raw = body.get("concurrency")

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise ValueError("top_k must be an integer")

    if concurrency_raw in (None, ""):
        concurrency = extract.DEFAULT_CONCURRENCY
    else:
        try:
            concurrency = int(concurrency_raw)
        except (TypeError, ValueError):
            raise ValueError("concurrency must be an integer")
        if concurrency <= 0:
            concurrency = extract.DEFAULT_CONCURRENCY

    # Every run is bounded by default. An empty or zero papers value means
    # the default cap, never unlimited -- unlimited is only ever the
    # explicit no_paper_limit choice, which must be ticked on the page.
    if no_paper_limit:
        papers = None
    else:
        if papers_raw in (None, ""):
            papers = DEFAULT_PAPERS_CAP
        else:
            try:
                papers = int(papers_raw)
            except (TypeError, ValueError):
                raise ValueError("papers must be an integer or null")
            if papers <= 0:
                papers = DEFAULT_PAPERS_CAP

    if year is not None and year != "":
        try:
            year = int(year)
        except (TypeError, ValueError):
            raise ValueError("year must be an integer or null")
    else:
        year = None

    return {
        "organism": organism,
        "gene": gene,
        "data_type": data_type,
        "model": model,
        "collection": collection,
        "top_k": top_k,
        "papers": papers,
        "no_paper_limit": no_paper_limit,
        "year": year,
        "concurrency": concurrency,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    global JOBS
    JOBS = JobStore(max_concurrent=args.max_concurrent)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("Serving on http://127.0.0.1:%d (Ctrl+C to stop)" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
