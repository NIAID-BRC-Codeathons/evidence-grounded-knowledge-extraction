"""The corpora available to search.

The server publishes these through /v1/collections, so nothing here is
hardcoded. This module only adds what the raw list lacks: readable names (an id
like "open-access" does not say "PubMed Central"), and the notion of searching
several at once, which the API supports through its `collections` array.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# Selecting every corpus at once. Not a collection id -- expanded before the
# request is built.
ALL = "all"

# The API caps a multi-collection request at five (QueryRequest.collections).
MAX_PER_REQUEST = 5

# Preferred default, by id. PubMed Central open access is the broadest corpus
# and the one a curator wants unless they say otherwise.
PREFERRED_DEFAULT = "open-access"

# Readable names for ids that do not explain themselves.
DISPLAY_NAMES = {
    "open-access": "PubMed Central (open access)",
    "asm-semantic": "ASM journals",
}


@dataclass(frozen=True)
class Collection:
    id: str
    label: str = ""
    count: int = 0
    chunk_method: str = ""
    server_default: bool = False

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.id, self.id)

    @property
    def size_note(self) -> str:
        if self.count >= 1_000_000:
            return f"{self.count / 1_000_000:.1f}M chunks"
        if self.count:
            return f"{self.count:,} chunks"
        return ""

    @classmethod
    def from_payload(cls, data: Dict[str, Any], default_id: Optional[str]) -> "Collection":
        return cls(
            id=data["id"],
            label=data.get("label", ""),
            count=int(data.get("count") or 0),
            chunk_method=data.get("chunk_method") or "",
            server_default=data.get("id") == default_id,
        )


class CollectionRegistry:
    """Collections this tenant can search."""

    def __init__(self, collections: Sequence[Collection], server_default: Optional[str] = None):
        self._collections = list(collections)
        self._server_default = server_default

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "CollectionRegistry":
        default_id = payload.get("default")
        items = [
            Collection.from_payload(c, default_id)
            for c in payload.get("collections", [])
            # Archived or restoring corpora cannot serve a query, and neither
            # can an empty index -- the registry lists a raw backing store with
            # no state and no chunks, which is not something to offer a user.
            if (c.get("state") or "active") == "active" and (c.get("count") or 0) > 0
        ]
        return cls(items, default_id)

    @classmethod
    def fetch(cls, client: Any) -> "CollectionRegistry":
        return cls.from_payload(client.collections())

    def __iter__(self):
        return iter(self._collections)

    def __len__(self) -> int:
        return len(self._collections)

    @property
    def ids(self) -> List[str]:
        return [c.id for c in self._collections]

    @property
    def total_count(self) -> int:
        return sum(c.count for c in self._collections)

    @property
    def default(self) -> Optional[str]:
        """PubMed Central when present, else whatever the server prefers."""
        if PREFERRED_DEFAULT in self.ids:
            return PREFERRED_DEFAULT
        if self._server_default in self.ids:
            return self._server_default
        return self.ids[0] if self._collections else None

    def get(self, collection_id: str) -> Optional[Collection]:
        for collection in self._collections:
            if collection.id == collection_id:
                return collection
        return None

    def resolve(self, value: Optional[str]) -> List[str]:
        """Turn a --collection value into the ids to search.

        Accepts a single id, a comma-separated list, "all", or nothing (the
        default). Unknown ids are rejected here rather than becoming a 404 from
        the server halfway through a batch.
        """
        if not value or not str(value).strip():
            default = self.default
            return [default] if default else []

        raw = [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]
        if any(part.lower() == ALL for part in raw):
            every = list(self.ids)
            if len(every) > MAX_PER_REQUEST:
                # Searching 5 of 6 corpora and saying nothing would be a silent
                # gap in coverage, which is worse for curation than an error.
                raise ValueError(
                    f"'{ALL}' covers {len(every)} collections but the API allows "
                    f"at most {MAX_PER_REQUEST} per request. Choose up to "
                    f"{MAX_PER_REQUEST} of: {', '.join(every)}"
                )
            return every

        known = set(self.ids)
        unknown = [part for part in raw if part not in known]
        if unknown:
            raise ValueError(
                f"unknown collection(s): {', '.join(unknown)}. "
                f"Available: {', '.join(self.ids)}, or '{ALL}'"
            )
        return list(dict.fromkeys(raw))


_TAG = re.compile(r"<[^>]+>")


def clean_title(title: Optional[str]) -> str:
    """Strip markup from a title.

    ASM records carry inline HTML -- "Isoniazid Activation Defects in
    Recombinant <i>Mycobacterium</i>..." -- which would otherwise be escaped
    and shown literally in the UI and written into TSV output.
    """
    if not title:
        return ""
    return re.sub(r"\s+", " ", _TAG.sub("", str(title))).strip()
