"""Baseline capture: RAGStack's own /v1/query answer-with-citations endpoint.

Purpose: RAGStack already answers questions with citations, and UI v01 is
built on that pattern. This module runs the same handful of questions
against /v1/query, saves the raw response unchanged, and computes the same
kind of mechanical citation checks our own pipeline computes, so the day-3
comparison is measured rather than asserted. See CONTRACT.md for our row
shape and quote-gate rules; the endpoint itself is not in CONTRACT.md, it
was measured live on 2026-09-16 (see the task note this module was built
from): POST $RAGSTACK_BASE/v1/query, body {"query": str, "collection": str,
"top_k": int}, returns {"answer": str, "sources": [...], "rewritten_queries":
[...]}. `sources` has the same shape as /v1/retrieve.

Reuses ragstack.py's load_env and _request rather than a new HTTP client
(ragstack.py has no /v1/query wrapper yet, since CONTRACT.md never asked
for one, so this imports the private _request helper directly instead of
duplicating the urllib/tls_context/retry machinery it already has).

Standard library only. Prints nothing on import.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ragstack import RAGStackError, _request, load_env  # noqa: E402
from verify import quote_supported  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "baseline")

DEFAULT_COLLECTION = "asm-semantic"
DEFAULT_TOP_K = 25

# Two or three per data type, visible at module level per the task spec.
# Keyed by data type so the gene under test can be swapped without touching
# the question wording.
QUESTIONS = {
    "mutation": [
        "Which PB2 mutations increase polymerase activity in mammalian cells?",
        "Which PB2 mutations are associated with adaptation of influenza A virus to mammalian hosts?",
    ],
    "ppi": [
        "Which host proteins interact with PB2?",
        "What is known about the interaction between PB2 and ANP32A?",
    ],
}

_MARKER_RE = re.compile(r"\[(\d+)\]")
# Simple sentence splitter: break after ., ?, or ! followed by whitespace.
# No NLP dependency is available (standard library only), so this is a
# heuristic, not a parser. It will over-split on abbreviations and
# under-split on run-on sentences; documented as a known limitation below.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Entity extraction patterns, one per data type. Both are heuristic: they
# catch the concrete tokens a domain reader would treat as checkable facts
# (a mutation code, a gene/protein symbol), not every noun phrase.
_ENTITY_PATTERNS = {
    "mutation": re.compile(r"\b[A-Z]\d{2,4}[A-Z]\b"),
    "ppi": re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b"),
}
# Common capitalized words that match the ppi pattern's shape but are not
# gene/protein symbols. Filtered out so entity_presence_share is not
# diluted by sentence-initial capitals.
_PPI_STOPWORDS = {
    "THE", "IT", "IS", "IN", "ON", "OF", "TO", "AS", "AT", "BY", "OR",
    "AND", "A", "AN", "PB2",
}


def query(question, collection=DEFAULT_COLLECTION, top_k=DEFAULT_TOP_K, env=None):
    """POST /v1/query. Returns (response_dict, seconds_elapsed)."""
    env = env or load_env()
    base = env["RAGSTACK_BASE"].rstrip("/")
    payload = {"query": question, "collection": collection, "top_k": top_k}
    started = time.time()
    data = _request("POST", base + "/v1/query", env, payload)
    elapsed = time.time() - started
    if not isinstance(data, dict) or "answer" not in data:
        raise RAGStackError(
            f"Unrecognized /v1/query response shape: "
            f"{list(data.keys()) if isinstance(data, dict) else type(data)}"
        )
    return data, elapsed


def split_sentences(answer):
    """Heuristic sentence split. See _SENTENCE_SPLIT_RE for the known limitation."""
    pieces = _SENTENCE_SPLIT_RE.split((answer or "").strip())
    return [p.strip() for p in pieces if p.strip()]


def _cited_marker_numbers(sentence):
    return sorted({int(n) for n in _MARKER_RE.findall(sentence)})


def citation_stats(answer, sources):
    """Mechanical, checkable facts about one answer against its source list.

    Citation markers are assumed 1-indexed into `sources`, i.e. [1] refers
    to sources[0]. This is the common convention for this kind of RAG
    endpoint; CONTRACT.md does not define it because /v1/query is outside
    slice 1's contracted surface, so it is an assumption, stated here and
    in the returned dict as `marker_convention`.
    """
    all_markers = [int(n) for n in _MARKER_RE.findall(answer or "")]
    distinct_markers = sorted(set(all_markers))
    valid_markers = [n for n in distinct_markers if 1 <= n <= len(sources)]
    invalid_markers = [n for n in distinct_markers if n not in valid_markers]

    return {
        "marker_convention": "1-indexed into sources, [1] == sources[0] (assumption, not contracted)",
        "citation_marker_count": len(all_markers),
        "distinct_citation_markers": distinct_markers,
        "distinct_sources_cited": len(valid_markers),
        "markers_with_no_matching_source": invalid_markers,
        "answer_char_count": len(answer or ""),
    }


def sentence_grounding(answer, sources):
    """Per-sentence quote-grounding check, matching our own quote-gate logic.

    For each sentence, look at the bracket markers it carries, map each to
    a source (1-indexed, see citation_stats), and test with
    verify.quote_supported whether the sentence's own text is supported by
    that source's content. A sentence with no citation marker cannot be
    grounded by definition and is recorded with method "no_citation"
    rather than run through quote_supported.

    Every sentence is treated as a claim sentence: there is no reliable
    standard-library way to tell a factual claim from connective tissue
    ("In summary," "Overall,"), so this over-counts unsupported claims by
    counting transitional sentences too. Documented as a known limitation.
    """
    sentences = split_sentences(answer)
    results = []
    for sentence in sentences:
        markers = _cited_marker_numbers(sentence)
        valid = [n for n in markers if 1 <= n <= len(sources)]
        if not valid:
            results.append({
                "sentence": sentence,
                "cited_markers": markers,
                "grounded": False,
                "method": "no_citation",
                "best_ratio": 0.0,
            })
            continue
        best_ok, best_method, best_ratio = False, "quote_mismatch", 0.0
        for n in valid:
            source_text = sources[n - 1].get("content", "")
            # Strip the bracket markers themselves before matching; they
            # are not part of the source prose.
            clean_sentence = _MARKER_RE.sub("", sentence).strip()
            ok, method, ratio, _offset = quote_supported(clean_sentence, source_text)
            if ok:
                best_ok, best_method, best_ratio = True, method, ratio
                break
            if ratio > best_ratio:
                best_method, best_ratio = method, ratio
        results.append({
            "sentence": sentence,
            "cited_markers": markers,
            "grounded": best_ok,
            "method": best_method,
            "best_ratio": round(best_ratio, 4),
        })
    return results


def no_citation_sentences(sentence_results):
    return [r for r in sentence_results if r["method"] == "no_citation"]


def extract_entities(sentence, data_type):
    """Concrete, checkable tokens a sentence asserts: mutation codes for
    mutation questions, gene/protein symbols for ppi questions. Citation
    markers are stripped first so a bracket number is never mistaken for
    part of an entity. Heuristic, not a named-entity recognizer.
    """
    clean = _MARKER_RE.sub("", sentence)
    pattern = _ENTITY_PATTERNS.get(data_type, _ENTITY_PATTERNS["mutation"])
    found = sorted(set(pattern.findall(clean)))
    if data_type == "ppi":
        found = [e for e in found if e not in _PPI_STOPWORDS]
    return found


def entity_presence(sentence, valid_markers, sources, data_type):
    """For each entity a sentence asserts, whether it appears verbatim
    (exact substring, case-sensitive since mutation codes and gene symbols
    are case-sensitive) anywhere in the sentence's own cited sources.

    This is deliberately looser than quote_supported: it checks that the
    named fact exists in the cited material somewhere, not that the whole
    sentence's wording does. A sentence can fail verbatim traceability
    (quote_supported on the full sentence) while every entity it names
    still passes this check, which is exactly the paraphrase-anchored-in-
    real-sources case this metric exists to surface.
    """
    entities = extract_entities(sentence, data_type)
    if not entities:
        return {"entities": [], "present": [], "absent": []}
    cited_texts = [sources[n - 1].get("content", "") for n in valid_markers]
    present, absent = [], []
    for entity in entities:
        (present if any(entity in text for text in cited_texts) else absent).append(entity)
    return {"entities": entities, "present": present, "absent": absent}


def run_one(gene, data_type, question, collection, top_k, env):
    response, elapsed = query(question, collection=collection, top_k=top_k, env=env)
    record = score_response(gene, data_type, question, collection, top_k, response)
    record["seconds"] = round(elapsed, 3)
    return record


def score_response(gene, data_type, question, collection, top_k, response):
    """Compute every derived metric from a raw /v1/query response. Split
    out from run_one so a previously saved raw_response can be rescored
    without another live call, e.g. after a metric definition changes.
    """
    answer = response.get("answer", "")
    sources = response.get("sources", []) or []
    stats = citation_stats(answer, sources)
    sentence_results = sentence_grounding(answer, sources)

    entities_total = 0
    entities_present = 0
    for result in sentence_results:
        valid_markers = [n for n in result["cited_markers"] if 1 <= n <= len(sources)]
        ep = entity_presence(result["sentence"], valid_markers, sources, data_type)
        result["entities"] = ep
        entities_total += len(ep["entities"])
        entities_present += len(ep["present"])

    grounded_count = sum(1 for r in sentence_results if r["grounded"])
    no_cite_count = len(no_citation_sentences(sentence_results))
    total = len(sentence_results)
    verbatim_traceable_share = round(grounded_count / total, 4) if total else None
    not_verbatim_traceable_share = round(1 - verbatim_traceable_share, 4) if total else None
    entity_presence_share = round(entities_present / entities_total, 4) if entities_total else None

    record = {
        "gene": gene,
        "data_type": data_type,
        "question": question,
        "collection": collection,
        "top_k": top_k,
        "raw_response": response,
        "citation_stats": stats,
        "sentence_grounding": sentence_results,
        "summary": {
            "sentences_total": total,
            "sentences_grounded": grounded_count,
            "sentences_no_citation": no_cite_count,
            # Named for what the quote gate actually measures: whether a
            # sentence's own wording is traceable verbatim to a cited
            # source. Not a truth or support judgment, see run_compare.
            "verbatim_traceable_share": verbatim_traceable_share,
            "not_verbatim_traceable_share": not_verbatim_traceable_share,
            "entities_total": entities_total,
            "entities_present": entities_present,
            "entity_presence_share": entity_presence_share,
        },
    }
    return record


def rescore_file(path, data_type=None):
    """Reload a saved baseline file's raw_response and recompute every
    derived field with the current metric definitions, leaving
    raw_response byte-for-byte untouched. Overwrites the file in place.
    """
    record = _load_json(path)
    gene = record.get("gene", "PB2")
    dtype = data_type or record.get("data_type", "mutation")
    rescored = score_response(
        gene, dtype, record.get("question", ""),
        record.get("collection", DEFAULT_COLLECTION),
        record.get("top_k", DEFAULT_TOP_K),
        record["raw_response"],
    )
    rescored["seconds"] = record.get("seconds")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rescored, handle, indent=2, ensure_ascii=False)
    return rescored


def write_record(record, gene, data_type):
    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    unique = uuid.uuid4().hex[:6]
    path = os.path.join(OUT_DIR, f"{gene.lower()}__{data_type}__{timestamp}-{unique}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    return path


def print_comparison_block(gene, data_type, records):
    print(f"Baseline (/v1/query) -- {gene} / {data_type}")
    print("-" * 60)
    for record in records:
        summary = record["summary"]
        stats = record["citation_stats"]
        print(f"Q: {record['question']}")
        print(f"  seconds: {record['seconds']}")
        print(f"  answer_char_count: {stats['answer_char_count']}")
        print(f"  citation_marker_count: {stats['citation_marker_count']}")
        print(f"  distinct_sources_cited: {stats['distinct_sources_cited']}")
        print(f"  markers_with_no_matching_source: {stats['markers_with_no_matching_source']}")
        print(f"  sentences_total (measured): {summary['sentences_total']}")
        print(f"  sentences_grounded (measured): {summary['sentences_grounded']}")
        print(f"  sentences_no_citation (measured): {summary['sentences_no_citation']}")
        print(f"  verbatim_traceable_share (measured): {summary['verbatim_traceable_share']}")
        print(f"  not_verbatim_traceable_share (measured): {summary['not_verbatim_traceable_share']}")
        print(f"  entities_total / entities_present (measured): "
              f"{summary['entities_total']} / {summary['entities_present']}")
        print(f"  entity_presence_share (measured): {summary['entity_presence_share']}")
        print()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_matching_baseline(gene, data_type):
    pattern = os.path.join(OUT_DIR, f"{gene.lower()}__{data_type}__*.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def run_compare(our_run_path, baseline_path=None):
    our_run = _load_json(our_run_path)
    run_meta = our_run.get("run", {})
    query_meta = run_meta.get("query", {})
    gene = query_meta.get("gene", "PB2")
    data_type = query_meta.get("data_type", "mutation")

    if baseline_path is None:
        baseline_path = _find_matching_baseline(gene, data_type)
        if baseline_path is None:
            print(f"No baseline file found under {OUT_DIR} for {gene}/{data_type}.")
            print("Run baseline.py for that gene/data type first, or pass --baseline explicitly.")
            return

    baseline_record = _load_json(baseline_path)

    tally = our_run.get("tally", {})
    proposed = tally.get("proposed", 0)
    emitted = tally.get("emitted", 0)
    our_pass_rate = round(emitted / proposed, 4) if proposed else None

    baseline_summary = baseline_record.get("summary", {})
    baseline_total = baseline_summary.get("sentences_total", 0)
    baseline_traceable = baseline_summary.get("verbatim_traceable_share")
    baseline_entity_share = baseline_summary.get("entity_presence_share")

    print("Comparison: our pipeline vs baseline /v1/query")
    print("-" * 60)
    print("Units differ: ours is per row (a proposed fact with its own")
    print("citation), the baseline's is per sentence (prose that can bundle")
    print("many facts under one bracket marker). Neither share below is a")
    print("truth measure without the gold set: both only test whether a")
    print("claim is traceable to a cited source, not whether it is correct.")
    print()
    print(f"Our run: {our_run_path}")
    print(f"  rows proposed: {proposed}")
    print(f"  rows emitted (passed quote gate): {emitted}")
    print(f"  quote-gate pass rate (emitted/proposed): {our_pass_rate}")
    print()
    print(f"Baseline run: {baseline_path}")
    print(f"  question: {baseline_record.get('question')}")
    print(f"  sentences total: {baseline_total}")
    print(f"  verbatim_traceable_share (sentence wording matches a cited source): {baseline_traceable}")
    print(f"  entity_presence_share (named entities appear in a cited source): {baseline_entity_share}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", default="PB2")
    parser.add_argument("--dtype", choices=sorted(QUESTIONS), default=None,
                         help="Restrict to one data type. Default: run all.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--compare", metavar="OUR_RUN_JSON",
                         help="Compare one of our run files against a matching baseline file.")
    parser.add_argument("--baseline", metavar="BASELINE_JSON",
                         help="With --compare, use this baseline file instead of the newest match.")
    parser.add_argument("--rescore", nargs="+", metavar="BASELINE_JSON",
                         help="Recompute derived metrics on already-saved baseline file(s) "
                              "from their stored raw_response, without a live call.")
    args = parser.parse_args()

    if args.rescore:
        for path in args.rescore:
            rescored = rescore_file(path)
            summary = rescored["summary"]
            print(f"{path}")
            print(f"  verbatim_traceable_share: {summary['verbatim_traceable_share']}")
            print(f"  entity_presence_share: {summary['entity_presence_share']} "
                  f"({summary['entities_present']}/{summary['entities_total']})")
        return

    if args.compare:
        run_compare(args.compare, args.baseline)
        return

    env = load_env()
    data_types = [args.dtype] if args.dtype else sorted(QUESTIONS)

    for data_type in data_types:
        records = []
        for question in QUESTIONS[data_type]:
            record = run_one(args.gene, data_type, question, args.collection, args.top_k, env)
            path = write_record(record, args.gene, data_type)
            record["_written_to"] = path
            records.append(record)
        print_comparison_block(args.gene, data_type, records)
        for record in records:
            print(f"  saved: {record['_written_to']}")
        print()


if __name__ == "__main__":
    main()
