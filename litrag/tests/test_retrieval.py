"""Chunk coverage: what gets fetched, and what the coverage number claims.

The module's job is to put every chunk of a matched paper in front of the model.
These tests pin the two halves of that: completion stops on the chunk budget and
nothing else, and the reported denominator never pretends to know a total it
cannot know.
"""

from litrag import retrieval


class Corpus:
    """Documents of a fixed length, wired into prev/next chains.

    `doc-3` means three chunks, `c-doc-3-0` through `c-doc-3-2`. One `chunks`
    call answers any mix of ids, which is how the real bulk endpoint behaves.
    """

    def __init__(self):
        self.calls = 0

    def chunks(self, ids, collection=None):
        self.calls += 1
        out = []
        for cid in ids:
            _, doc, size, index = cid.split("-")
            size, index = int(size), int(index)
            if index < 0 or index >= size:
                continue
            out.append(_chunk(f"{doc}-{size}", size, index))
        return out


def _chunk(doc_id, size, index, score=None):
    name = doc_id.split("-")[0]
    source = {
        "chunk_id": f"c-{name}-{size}-{index}",
        "doc_id": doc_id,
        "content": f"{doc_id} passage {index}",
        "metadata": {
            "chunk_index": index,
            "prev_chunk_id": f"c-{name}-{size}-{index - 1}" if index else None,
            "next_chunk_id": (f"c-{name}-{size}-{index + 1}"
                              if index < size - 1 else None),
        },
    }
    if score is not None:
        source["score"] = score
    return source


def _hits(*specs):
    """One retrieved hit per document: (name, size, index, score)."""
    return [_chunk(f"{name}-{size}", size, index, score)
            for name, size, index, score in specs]


# -- completion is bounded by the chunk budget, not by a paper count ---------

def test_completion_has_no_paper_cap():
    """Twelve matched papers, twelve completed.

    The old default stopped at eight, which left the rest of the matched
    literature represented by a single fragment each -- the failure the coverage
    work exists to remove.
    """
    hits = _hits(*[(f"d{i}", 4, 0, 0.9 - i / 100) for i in range(12)])
    out = retrieval.complete_documents(Corpus(), hits, "open-access")

    by_doc = {}
    for source in out:
        by_doc.setdefault(source["doc_id"], []).append(source)
    assert len(by_doc) == 12
    assert all(len(group) == 4 for group in by_doc.values())


def test_budget_stops_completion_and_best_scoring_papers_win():
    """With room for some papers and not others, score decides which."""
    hits = _hits(("top", 10, 0, 0.99), ("mid", 10, 0, 0.50), ("low", 10, 0, 0.10))
    out = retrieval.complete_documents(Corpus(), hits, "open-access", max_total=9)

    counts = {}
    for source in out:
        counts[source["doc_id"]] = counts.get(source["doc_id"], 0) + 1
    assert counts["top-10"] == 10, "the best paper is finished first"
    assert counts["low-10"] < 10, "the budget ran out before the worst paper"
    assert sum(counts.values()) <= len(hits) + 9 + 2, "overshoot is one frontier"


def test_max_docs_still_works_when_a_caller_asks_for_it():
    """The cap survives for tests and callers that want a small fixed run."""
    hits = _hits(*[(f"d{i}", 3, 0, 0.9 - i / 100) for i in range(6)])
    out = retrieval.complete_documents(Corpus(), hits, "open-access", max_docs=2)

    completed = {s["doc_id"] for s in out if s.get("expanded")}
    assert len(completed) == 2


# -- the denominator: exact when known, a floor when not ---------------------

def test_available_is_exact_for_a_complete_document():
    chunks = [_chunk("d-3", 3, i) for i in range(3)]
    assert retrieval.is_complete(chunks)
    assert retrieval.available_passages(chunks) == 3


def test_available_is_a_floor_for_a_partial_document():
    """Chunk 7 of a paper of unknown length proves at least eight passages."""
    chunks = [_chunk("d-20", 20, 7)]
    assert not retrieval.is_complete(chunks)
    assert retrieval.available_passages(chunks) == 8


def test_available_falls_back_to_what_is_held_without_an_index():
    chunks = [{"chunk_id": "a", "doc_id": "d", "content": "x", "metadata": {}},
              {"chunk_id": "b", "doc_id": "d", "content": "y", "metadata": {}}]
    assert retrieval.available_passages(chunks) == 2


def test_summarise_reports_coverage_and_whether_it_is_exact():
    complete = [_chunk("done-2", 2, i) for i in range(2)]
    partial = [_chunk("part-20", 20, 9)]

    whole = retrieval.summarise(complete)
    assert whole["n_passages"] == whole["n_passages_available"] == 2
    assert whole["passages_available_exact"] is True

    mixed = retrieval.summarise(complete + partial)
    assert mixed["n_passages"] == 3
    assert mixed["n_passages_available"] == 12, "2 known plus a floor of 10"
    assert mixed["passages_available_exact"] is False, \
        "one unfinished paper makes the total a floor, and it must say so"
