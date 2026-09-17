"""How much of the literature to read, and in what order.

Retrieval ranks *chunks*, not papers. A `top_k` of 10 returns ten ~500-character
fragments, which in practice come from about eight different papers -- one
fragment each. Two things follow, and neither is fixable by prompting:

- The fragment that matched the query is often the Introduction, where a topic is
  merely mentioned. The finding lives in the Results, in a chunk nobody
  retrieved. Measured on a live katG query, the chunk *after* the top hit
  contained the resistance result that the hit itself only alluded to.
- Recall is capped before the model runs. Paper nine is not judged and rejected;
  it is never looked at.

So this module widens the input. `expand` walks each hit's neighbours to recover
the surrounding section, `order` puts a paper's passages back into reading order,
and `pack` splits the result into batches that can be sent concurrently.

Measured against the live API (2026-09-17):

    open-access  k=100 hops=2  ->  285 passages,  64 papers
    Influenza    k=100 hops=3  ->  503 passages,  15 papers (~the whole corpus)
    Dengue       k=100 hops=3  ->  252 passages,   2 papers (~the whole corpus)

against 10 passages from 8 papers today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

# /v1/retrieve rejects anything above this: "top_k must be <= 100".
MAX_TOP_K = 100

# Char budget per LLM call. ~60k chars is ~17k tokens by the project's
# conservative 3.5 chars/token estimate, which leaves generous room for the
# instruction block and a long table in a 32k-token context -- the smallest
# context among the models this tool targets. Bigger batches mean fewer calls
# but a longer tail: one slow batch holds up the whole query.
DEFAULT_BATCH_CHARS = 60_000

# Upper bound on concurrent generation calls for ONE query. Beyond this the
# shared vLLM server stops going faster and starts queueing -- measured on
# mango: 6 batches took 30.2s at concurrency 4, 24.9s at 6, 23.6s at 8. The
# curve is already flat by 8, so more would only crowd out other users.
MAX_CONCURRENCY = 8

# Refuse rather than launch an unbounded fan-out. At ~1,700 chars per passage
# this is roughly 3,400 passages, well past the point where a single query
# should be silently spending this much of a shared GPU. Raised from 30 when
# chunk coverage became the goal: completing every matched paper on a broad
# open-access query is ~48 batches by measurement, and a refusal there would
# reject exactly the run the coverage work exists to make possible.
MAX_BATCHES = 60

# Chunks one document may contribute in "full" mode. Set from measurement, not
# taste: the two papers in the Dengue collection are ~191 chunks each, and an
# earlier cap of 120 stopped mid-paper on both -- so "full" completed nothing
# and the coverage line said so. A cap that silently prevents completion is
# worse than no completion mode at all.
MAX_CHUNKS_PER_DOC = 250

# Ceiling across all documents in one run. This is now the ONLY thing that
# stops completion: there is no paper cap, so papers are finished in score order
# until this budget is spent. Sized from measurement -- an open-access paper ran
# ~17 chunks in the katG run, so 1,500 covers ~85 papers, comfortably more than
# the ~69 a top_k of 100 surfaces. Small curated corpora (Dengue at 382 chunks)
# still fit entirely underneath it.
MAX_COMPLETION_CHUNKS = 1_500

# Measured against both mango models on real prompts: 3.98 chars/token on
# Qwen3.6, 4.40 on Llama-4-Scout. prompts.CHARS_PER_TOKEN is 3.5, which
# understates chars per token and therefore OVERstates the token cost -- the
# safe direction, and kept deliberately.
MEASURED_CHARS_PER_TOKEN = 3.5

# Share of a model's context a batch of passages may occupy. The rest is for
# the instruction block and the model's own table, which for a full batch runs
# to several thousand tokens.
CONTEXT_SHARE = 0.6


def batch_chars_for(context_limit: Optional[int],
                    default: int = DEFAULT_BATCH_CHARS) -> int:
    """Largest batch that fits this model, in characters.

    Context windows differ by more than 2x across the two models on one host
    (Llama-4-Scout 60,000 tokens, Qwen3.6 131,072), so a fixed character budget
    is a guess that happens to be safe today. Sizing from the server's declared
    limit makes it a check. Never grows past `default`: bigger batches also mean
    fewer of them, and fewer batches measurably LOSES findings, because output
    tokens are capped per call.
    """
    if not context_limit:
        return default
    usable = int(context_limit * CONTEXT_SHARE * MEASURED_CHARS_PER_TOKEN)
    return max(4_000, min(default, usable))

# Depth presets. "standard" is today's behaviour and must stay that way: it is
# the control arm for any comparison, so it may not acquire expansion by
# accident.
STANDARD = "standard"
ADAPTIVE = "adaptive"
FULL = "full"
DEPTHS = (STANDARD, ADAPTIVE, FULL)


@dataclass(frozen=True)
class RetrievalPlan:
    """How wide to search and how far to read around each hit."""

    top_k: int
    hops: int
    batch_chars: int = DEFAULT_BATCH_CHARS
    depth: str = STANDARD

    @property
    def expands(self) -> bool:
        return self.hops > 0


def plan(count: int, depth: str = STANDARD, top_k: int = 10,
         batch_chars: int = DEFAULT_BATCH_CHARS) -> RetrievalPlan:
    """Choose depth from the size of the collection being searched.

    `count` is the collection's chunk count, which the registry already knows
    before a query runs. The asymmetry is deliberate: a small curated corpus can
    be read almost in full, while PubMed Central cannot be read at all, so on the
    big corpora the budget buys breadth (more distinct papers) and on the small
    ones it buys depth (more of each paper).
    """
    if depth not in DEPTHS:
        raise ValueError(f"unknown depth '{depth}'. Choose from: {', '.join(DEPTHS)}")

    if depth == STANDARD:
        # Exactly what the caller asked for: no expansion, and `batch_chars=0`
        # meaning do not pack at all. Anything else would split a large top_k
        # into several calls and quietly stop being the control arm -- which is
        # precisely what a test caught here.
        return RetrievalPlan(top_k=min(top_k, MAX_TOP_K), hops=0,
                             batch_chars=0, depth=depth)

    if depth == FULL:
        # hops=0 on purpose: `full` reads whole DOCUMENTS (see
        # complete_documents), which is a different traversal from widening the
        # window around every hit. Breadth-first expansion provably fails to
        # reach a fact buried mid-paper.
        return RetrievalPlan(top_k=MAX_TOP_K, hops=0, batch_chars=batch_chars,
                             depth=depth)

    # ADAPTIVE. top_k is the API maximum everywhere, because breadth and depth
    # come from different levers and only one of them is top_k: expansion walks
    # outward within the papers already found, so it adds passages but almost no
    # new papers. Measured on open-access, top_k=50 with one hop reached 95
    # passages across 44 papers -- the same 44 papers as top_k=50 alone. Lowering
    # top_k here to save time would cost breadth and buy nothing back.
    #
    # What varies is how far to read INTO each paper, and that is what the
    # corpus size should decide.
    hops = 1
    if count and count < 10_000:
        # Dengue (382) and Influenza_2024_2025 (3,064): few enough papers that
        # three hops covers most of each one -- effectively reading the corpus.
        hops = 3
    elif count and count < 1_000_000:
        hops = 2
    # asm-semantic (6.7M) and open-access (47.6M) keep one hop: at ~285 passages
    # for two, the wall-clock and token cost outrun the marginal context.
    return RetrievalPlan(max(MAX_TOP_K, min(top_k, MAX_TOP_K)), hops,
                         batch_chars, depth)


def dedupe_chunks(sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop repeated chunks, keeping the first (highest-scoring) occurrence.

    Nothing did this before. The API can return the same chunk more than once --
    notably when several collections are searched at once -- and every duplicate
    was being paid for twice: once in context tokens, and again as a second
    apparent source supporting the same claim.
    """
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for source in sources:
        key = source.get("chunk_id")
        if key is None:
            # No id to dedupe on: keep it rather than guess. Losing a passage is
            # worse than carrying a possible duplicate.
            unique.append(source)
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _neighbour_ids(sources: Sequence[Dict[str, Any]], held: set) -> List[str]:
    """Ids adjacent to these chunks that we do not already have."""
    wanted: List[str] = []
    for source in sources:
        meta = source.get("metadata") or {}
        for key in ("prev_chunk_id", "next_chunk_id"):
            neighbour = meta.get(key)
            if neighbour and neighbour not in held and neighbour not in wanted:
                wanted.append(neighbour)
    return wanted


def expand(client: Any, sources: Sequence[Dict[str, Any]],
           collection: Optional[str], hops: int = 1) -> List[Dict[str, Any]]:
    """Pull the chunks either side of each hit, `hops` times outward.

    One bulk request per hop -- 182 ids in a single call took 0.26s against the
    live API, so there is no reason to page. Chunks with no recorded neighbour
    are normal (about 3 in 10 on open-access), not an error.

    `collection` must be passed through. The endpoint scopes lookups to a
    collection and returns an EMPTY LIST rather than an error when the parameter
    is missing, so omitting it looks exactly like "this corpus has no
    neighbours".
    """
    held = {s.get("chunk_id") for s in sources if s.get("chunk_id")}
    collected = list(sources)
    frontier = list(sources)

    for _ in range(max(0, hops)):
        wanted = _neighbour_ids(frontier, held)
        if not wanted:
            break
        fetched = client.chunks(wanted, collection=collection) or []
        asked = set(wanted)
        frontier = []
        for chunk in fetched:
            key = chunk.get("chunk_id") or chunk.get("id")
            # Only accept what we asked for. An unexpected id would otherwise
            # become a root for the next hop and pull in a whole unrelated
            # document, which the model would then cite as though it were a
            # neighbour of a real hit.
            if not key or key in held or key not in asked:
                continue
            held.add(key)
            normalised = _as_source(chunk, key)
            collected.append(normalised)
            frontier.append(normalised)
        if not frontier:
            break
    return collected


def complete_documents(client: Any, sources: Sequence[Dict[str, Any]],
                       collection: Optional[str], max_docs: Optional[int] = None,
                       max_chunks_per_doc: int = MAX_CHUNKS_PER_DOC,
                       max_total: int = MAX_COMPLETION_CHUNKS
                       ) -> List[Dict[str, Any]]:
    """Read the best-ranked papers end to end, instead of widening every paper.

    `expand` walks outward from every hit at once, which spreads a fixed budget
    thinly across all of them. Measured against a known answer -- the PDK-53
    vaccine mutation NS1 G53D -- that never arrives: the paper containing it WAS
    retrieved, but the sentence sits 16 chunks away from the chunk that matched,
    and four hops of breadth-first expansion (481 passages across 77 papers)
    still missed it. Walking that one document to completion found it in 36
    chunks.

    So this is the known-item strategy: pick the papers most likely to hold the
    answer and read all of them, rather than skimming the neighbourhood of
    everything. Breadth still comes from top_k; this buys depth.

    There is no paper cap by default. `max_docs` used to be 8, which made the
    paper count -- not the chunk budget -- the thing that stopped a run: the katG
    query finished 8 papers using 136 of the 500 chunks it was allowed, leaving
    61 matched papers represented by a single fragment each. The goal is that
    every chunk of a matched paper is reviewed, so papers are now completed in
    score order until `max_total` is spent, and that budget is the only limit.
    `max_docs` survives for tests that need a small deterministic run.
    """
    by_doc: Dict[Any, List[Dict[str, Any]]] = {}
    for source in sources:
        by_doc.setdefault(source.get("doc_id"), []).append(source)

    def best(group):
        scores = [s.get("score") for s in group if isinstance(s.get("score"), (int, float))]
        return max(scores) if scores else -1.0

    ranked = sorted(by_doc.items(), key=lambda kv: -best(kv[1]))
    if max_docs is not None:
        ranked = ranked[:max_docs]
    held = {s.get("chunk_id") for s in sources if s.get("chunk_id")}
    collected = list(sources)

    for _, group in ranked:
        if len(collected) - len(sources) >= max_total:
            break
        frontier = list(group)
        gathered = len(group)
        while frontier and gathered < max_chunks_per_doc:
            if len(collected) - len(sources) >= max_total:
                break
            wanted = _neighbour_ids(frontier, held)[:max_chunks_per_doc]
            if not wanted:
                break
            fetched = client.chunks(wanted, collection=collection) or []
            asked = set(wanted)
            frontier = []
            for chunk in fetched:
                key = chunk.get("chunk_id") or chunk.get("id")
                if not key or key in held or key not in asked:
                    continue
                held.add(key)
                source = _as_source(chunk, key)
                collected.append(source)
                frontier.append(source)
                gathered += 1
    return collected


def _as_source(chunk: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """Shape a fetched chunk like a retrieved one.

    Deliberately no `score`: this chunk was never ranked against the query, it
    was pulled in because it sits next to something that was. Inventing a score
    would let a neighbour outrank a real hit in any later sort.
    """
    source = dict(chunk)
    source["chunk_id"] = chunk_id
    source.setdefault("metadata", {})
    source["expanded"] = True
    return source


def order(sources: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group by paper, best paper first, passages in reading order within it.

    Retrieval returns chunks ranked individually, so a paper's Methods can arrive
    before its Abstract and with another paper's text in between. Presenting a
    document's passages contiguously and in order is what lets the model read a
    section rather than a shuffle of fragments.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for source in sources:
        groups.setdefault(source.get("doc_id"), []).append(source)

    def best_score(group: List[Dict[str, Any]]) -> float:
        scores = [s.get("score") for s in group if isinstance(s.get("score"), (int, float))]
        return max(scores) if scores else -1.0

    def position(source: Dict[str, Any]) -> tuple:
        index = (source.get("metadata") or {}).get("chunk_index")
        # Chunks with no index sort last, stably, rather than colliding at 0 and
        # scrambling the ones that do have an index.
        if isinstance(index, (int, float)):
            return (0, index)
        return (1, 0)

    ordered: List[Dict[str, Any]] = []
    for _, group in sorted(groups.items(), key=lambda kv: -best_score(kv[1])):
        ordered.extend(sorted(group, key=position))
    return ordered


def pack(sources: Sequence[Dict[str, Any]],
         max_chars: int = DEFAULT_BATCH_CHARS) -> List[List[Dict[str, Any]]]:
    """Split into batches under a char budget, keeping papers whole.

    A document is only split across batches when it exceeds the budget on its
    own. Splitting a paper means the model sees half its evidence in one call and
    half in another, and can reconcile neither -- the cost of an occasional
    under-full batch is much lower.

    `max_chars=0` disables packing entirely: everything goes in one batch. That
    is what the "standard" depth uses to stay identical to the original
    single-call behaviour.
    """
    if not sources:
        return []
    if not max_chars:
        return [list(sources)]

    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    used = 0

    for group in _by_document(sources):
        size = sum(len(s.get("content") or "") for s in group)
        if current and used + size > max_chars:
            batches.append(current)
            current, used = [], 0
        if size > max_chars and not current:
            # One document bigger than a whole batch: split it, but only this
            # one, and only because the alternative is dropping passages.
            for piece in _split(group, max_chars):
                batches.append(piece)
            continue
        current.extend(group)
        used += size

    if current:
        batches.append(current)
    return batches


def _by_document(sources: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Consecutive runs of the same doc_id, preserving the given order."""
    groups: List[List[Dict[str, Any]]] = []
    for source in sources:
        if groups and groups[-1][0].get("doc_id") == source.get("doc_id"):
            groups[-1].append(source)
        else:
            groups.append([source])
    return groups


def _split(group: Sequence[Dict[str, Any]], max_chars: int) -> List[List[Dict[str, Any]]]:
    pieces: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    used = 0
    for source in group:
        size = len(source.get("content") or "")
        if current and used + size > max_chars:
            pieces.append(current)
            current, used = [], 0
        current.append(source)
        used += size
    if current:
        pieces.append(current)
    return pieces


def is_complete(chunks: Sequence[Dict[str, Any]]) -> bool:
    """True when these chunks are a whole document, with nothing missing.

    Answers "did we actually read the whole paper, or a window out of it?" --
    which a passage count alone cannot. Two conditions, both necessary:

    - Both ends are present: the lowest chunk records no `prev_chunk_id` and the
      highest records no `next_chunk_id`. That is how the corpus marks the start
      and end of a document.
    - No gap in between: the chunk_index values form an unbroken run. A window
      that happens to include the first and last chunk but nothing in the middle
      is not a complete read.

    Unknown rather than optimistic: a document whose chunks carry no
    `chunk_index` cannot be verified, so it is reported incomplete.
    """
    indexed = [c for c in chunks
               if isinstance((c.get("metadata") or {}).get("chunk_index"), int)]
    if not indexed or len(indexed) != len(chunks):
        return False

    ordered = sorted(indexed, key=lambda c: c["metadata"]["chunk_index"])
    first, last = ordered[0].get("metadata") or {}, ordered[-1].get("metadata") or {}
    if first.get("prev_chunk_id") or last.get("next_chunk_id"):
        return False

    span = ordered[-1]["metadata"]["chunk_index"] - ordered[0]["metadata"]["chunk_index"]
    return span + 1 == len(ordered)


def available_passages(chunks: Sequence[Dict[str, Any]]) -> int:
    """How many passages this document holds, as far as we can tell.

    The corpus exposes no per-document chunk total, so this is derived. A
    complete document answers exactly: it is however many chunks we hold. An
    incomplete one answers with a floor, because `chunk_index` is 0-based and
    the highest index we have seen proves the document runs at least that far.
    The floor can understate a paper whose tail we never fetched, which is why
    `summarise` reports whether the total is exact rather than implying it is.
    """
    if not chunks:
        return 0
    if is_complete(chunks):
        return len(chunks)
    indices = [(c.get("metadata") or {}).get("chunk_index") for c in chunks]
    highest = max((i for i in indices if isinstance(i, int)), default=None)
    if highest is None:
        return len(chunks)
    return max(highest + 1, len(chunks))


def summarise(sources: Sequence[Dict[str, Any]],
              collection_count: int = 0) -> Dict[str, int]:
    """Counts worth showing a user: how much was actually read.

    `n_passages` over `n_passages_available` is the headline: every chunk of a
    matched paper should be reviewed, so the question is what share of them the
    model actually saw. `n_papers_complete` stays because it answers a different
    question -- whether any single paper was read end to end -- and a fact buried
    mid-paper is invisible to a window, however many windows there are.

    `passages_available_exact` is false when some matched paper was left
    incomplete, because the denominator is then a floor rather than a total.
    """
    by_doc: Dict[Any, List[Dict[str, Any]]] = {}
    for source in sources:
        by_doc.setdefault(source.get("doc_id"), []).append(source)

    papers = [(doc, group) for doc, group in by_doc.items() if doc is not None]
    complete = sum(1 for _, group in papers if is_complete(group))
    return {
        "n_passages": len(sources),
        "n_papers": len(papers),
        "n_papers_complete": complete,
        "n_passages_available": sum(available_passages(g) for _, g in papers),
        "passages_available_exact": complete == len(papers),
        "n_expanded": sum(1 for s in sources if s.get("expanded")),
        "n_chars": sum(len(s.get("content") or "") for s in sources),
        # Share of the whole corpus seen. Meaningful for a small curated
        # collection, vanishingly small for PubMed Central -- which is itself
        # worth showing rather than hiding.
        "collection_chunks": collection_count,
    }
