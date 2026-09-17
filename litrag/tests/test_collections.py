"""Corpus selection."""

import pytest

from litrag.collections import (ALL, PREFERRED_DEFAULT, CollectionRegistry,
                                clean_title)

PAYLOAD = {
    "collections": [
        {"id": "asm-semantic", "label": "ASM papers", "count": 6_718_269,
         "chunk_method": "semantic", "state": "active"},
        {"id": "open-access", "label": "PubMed Central open access",
         "count": 47_625_155, "chunk_method": "fixed_token", "state": "active"},
    ],
    "default": "open-access",
}


@pytest.fixture
def registry():
    return CollectionRegistry.from_payload(PAYLOAD)


def test_every_active_collection_is_listed(registry):
    assert registry.ids == ["asm-semantic", "open-access"]
    assert len(registry) == 2


def test_pubmed_central_is_the_default(registry):
    assert registry.default == PREFERRED_DEFAULT == "open-access"


def test_default_holds_even_when_the_server_prefers_another():
    payload = dict(PAYLOAD, default="asm-semantic")
    assert CollectionRegistry.from_payload(payload).default == "open-access"


def test_falls_back_to_server_default_when_pmc_is_absent():
    payload = {"collections": [PAYLOAD["collections"][0]], "default": "asm-semantic"}
    assert CollectionRegistry.from_payload(payload).default == "asm-semantic"


def test_inactive_collections_are_excluded():
    """An archived corpus cannot serve a query, so it must not be offered."""
    payload = {
        "collections": PAYLOAD["collections"] + [
            {"id": "archived-one", "count": 10, "state": "archived"},
        ],
        "default": "open-access",
    }
    assert "archived-one" not in CollectionRegistry.from_payload(payload).ids


def test_ids_get_readable_names(registry):
    """"open-access" does not tell a curator it means PubMed Central."""
    assert registry.get("open-access").display_name == "PubMed Central (open access)"
    assert registry.get("asm-semantic").display_name == "ASM journals"


def test_unknown_id_keeps_its_own_name():
    payload = {"collections": [{"id": "future-corpus", "count": 5}], "default": None}
    assert CollectionRegistry.from_payload(payload).get("future-corpus").display_name == "future-corpus"


def test_size_is_human_readable(registry):
    assert registry.get("open-access").size_note == "47.6M chunks"
    assert registry.total_count == 54_343_424


def test_resolve_defaults_to_pmc(registry):
    assert registry.resolve(None) == ["open-access"]
    assert registry.resolve("") == ["open-access"]


def test_resolve_all_expands_to_every_collection(registry):
    assert set(registry.resolve(ALL)) == set(registry.ids)


def test_resolve_accepts_a_list(registry):
    assert registry.resolve("open-access,asm-semantic") == ["open-access", "asm-semantic"]
    assert registry.resolve("open-access, asm-semantic") == ["open-access", "asm-semantic"]


def test_resolve_deduplicates(registry):
    assert registry.resolve("open-access,open-access") == ["open-access"]


def test_unknown_id_is_rejected_with_the_alternatives(registry):
    """Catch it here rather than as a 404 midway through a batch."""
    with pytest.raises(ValueError, match="unknown collection"):
        registry.resolve("pubmed")
    try:
        registry.resolve("pubmed")
    except ValueError as exc:
        assert "open-access" in str(exc) and ALL in str(exc)


def test_clean_title_strips_markup():
    """ASM records carry inline HTML that would otherwise reach TSV output."""
    assert clean_title("Defects in <i>Mycobacterium</i> smegmatis") == \
        "Defects in Mycobacterium smegmatis"
    assert clean_title("<b>A</b>   <i>B</i>") == "A B"
    assert clean_title(None) == ""
    assert clean_title("Plain title") == "Plain title"


def test_empty_indexes_are_not_offered():
    """The registry lists a raw backing store with no state and no chunks.

    An empty index cannot serve a query, so offering it would only produce a
    confusing empty result.
    """
    payload = {
        "collections": PAYLOAD["collections"] + [
            {"id": "ragstack_raw_backing_store", "count": 0, "state": None},
        ],
        "default": "open-access",
    }
    registry = CollectionRegistry.from_payload(payload)
    assert "ragstack_raw_backing_store" not in registry.ids


def test_all_refuses_to_exceed_the_api_cap():
    """Searching 5 of 6 corpora and saying nothing would be a silent coverage
    gap, which is worse for curation than an error."""
    from litrag.collections import MAX_PER_REQUEST
    payload = {
        "collections": [
            {"id": f"corpus-{i}", "count": 100, "state": "active"}
            for i in range(MAX_PER_REQUEST + 1)
        ],
        "default": "corpus-0",
    }
    registry = CollectionRegistry.from_payload(payload)
    with pytest.raises(ValueError, match="at most"):
        registry.resolve(ALL)
    # An explicit selection within the cap still works.
    assert len(registry.resolve("corpus-0,corpus-1")) == 2


def test_all_works_at_exactly_the_cap():
    from litrag.collections import MAX_PER_REQUEST
    payload = {
        "collections": [
            {"id": f"corpus-{i}", "count": 100, "state": "active"}
            for i in range(MAX_PER_REQUEST)
        ],
        "default": "corpus-0",
    }
    assert len(CollectionRegistry.from_payload(payload).resolve(ALL)) == MAX_PER_REQUEST
