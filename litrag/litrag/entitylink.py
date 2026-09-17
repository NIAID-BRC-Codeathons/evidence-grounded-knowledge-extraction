"""Entity linking: free-text organism names to NCBI Taxonomy identifiers.

The fourth evaluation axis the project charter requires, and the one the domain
primer calls the usual accuracy bottleneck. The other three axes score what the
model said; this one scores whether we know what it was talking about.

Split in two on purpose:

    resolve()  touches the network and writes a cache
    score()    reads the cache and never touches the network

so the scorer stays offline, reproducible and fast, and a missing cache reports
as `not run` rather than as zero. A check that could not run must never look
like a check that failed.

Measured against live NCBI E-utilities while writing this:

    Mycobacterium tuberculosis  -> 1773      full binomials resolve cleanly
    SARS-CoV-2                  -> 2697049
    influenza A virus           -> 11320
    M. tuberculosis             -> NOTHING   every abbreviation fails
    Mtb                         -> NOTHING
    mouse                       -> 10090, 10088   ambiguous, species + genus
    BALB/c mice                 -> 10095, 10088   resolves to the wrong thing

Abbreviations are not a rare edge: papers and models use them constantly. They
cannot be rescued by searching the epithet alone (NCBI returns nothing for a
bare "tuberculosis"), so there are two mechanisms here, both deliberate and
both bounded:

  1. A small hand-written alias table. Hand-curated, never model-generated,
     the same rule experiment-01/synonyms.json follows.
  2. Initial expansion against names already in the cache: once
     "Mycobacterium tuberculosis" is known, "M. tuberculosis" resolves by
     matching the genus initial and the epithet. This costs no extra requests
     and improves as the cache fills.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# NCBI asks for 3 requests/second without a key, and for identification.
RATE_LIMIT_S = 0.34
USER_AGENT = "litrag-entitylink/0.1 (NIAID BRC codeathon; via NCBI E-utilities)"

DEFAULT_CACHE = Path("entity_cache.json")

# Hand-written, deliberately short. Anything here is a claim a human made.
ALIASES: Dict[str, str] = {
    "mtb": "Mycobacterium tuberculosis",
    "m. tb": "Mycobacterium tuberculosis",
    "m tb": "Mycobacterium tuberculosis",
    "tb": "Mycobacterium tuberculosis",
    "sars cov 2": "SARS-CoV-2",
    "sars-cov2": "SARS-CoV-2",
    "hiv": "Human immunodeficiency virus",
    "mice": "Mus musculus",
    "mouse": "Mus musculus",
    "murine": "Mus musculus",
    "rat": "Rattus norvegicus",
    "rats": "Rattus norvegicus",
    "human": "Homo sapiens",
    "humans": "Homo sapiens",
    "ferret": "Mustela putorius furo",
    "ferrets": "Mustela putorius furo",
}

# Strain and condition words that qualify a host without changing the species.
# "BALB/c mice" is Mus musculus; keeping the strain in the query resolves to
# the wrong record (10095, not 10090).
_QUALIFIERS = re.compile(
    r"\b(aged|adult|young|infant|juvenile|wild[- ]?type|wt|transgenic|knockout|ko|"
    r"balb/?c|c57bl/?6[a-z]*|c3h|dba|129s[a-z0-9]*|nod|scid|nsg|athymic|nude)\b",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")
_ABBREV = re.compile(r"^([a-z])\.?\s+([a-z][a-z-]+)$")


def normalize_surface(surface: str) -> str:
    """Lowercase, strip strain and condition qualifiers, collapse whitespace."""
    text = _QUALIFIERS.sub(" ", str(surface or "").lower())
    text = text.replace("_", " ")
    return _WS.sub(" ", text).strip(" .,;:")


@dataclass
class Resolution:
    """What one surface string resolved to."""

    surface: str
    taxid: Optional[str] = None
    scientific_name: str = ""
    rank: str = ""
    # (taxid, name, rank), root first.
    lineage: List[List[str]] = field(default_factory=list)
    source: str = ""
    ambiguous: bool = False
    note: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.taxid)

    def ancestor_at(self, rank: str) -> Optional[str]:
        """The taxid of this entity's ancestor at `rank`, if any.

        Used for partial credit. Comparing lineage ids is exact, where comparing
        names would guess -- "Mycobacterium tuberculosis" and "Mycobacterium
        bovis" share a genus that no string comparison of the two would reveal.
        """
        if self.rank == rank and self.taxid:
            return self.taxid
        for taxid, _name, entry_rank in self.lineage:
            if entry_rank == rank:
                return taxid
        return None


class EntityLinker:
    """Resolves surfaces to taxids, backed by a JSON cache on disk."""

    def __init__(self, cache_path: Path = DEFAULT_CACHE, offline: bool = False):
        self.cache_path = Path(cache_path)
        self.offline = offline
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_request = 0.0
        if self.cache_path.is_file():
            try:
                self._cache = json.loads(self.cache_path.read_text("utf-8"))
            except ValueError:
                self._cache = {}

    # -- cache ------------------------------------------------------------

    @property
    def cached_names(self) -> List[str]:
        return [e.get("scientific_name", "") for e in self._cache.values()
                if e.get("scientific_name")]

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True) + "\n", "utf-8")

    # -- network ----------------------------------------------------------

    def _get(self, path: str, params: Dict[str, str]) -> str:
        elapsed = time.monotonic() - self._last_request
        if elapsed < RATE_LIMIT_S:
            time.sleep(RATE_LIMIT_S - elapsed)
        url = f"{EUTILS}/{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", "replace")
        finally:
            self._last_request = time.monotonic()

    def _esearch(self, term: str) -> List[str]:
        raw = self._get("esearch.fcgi", {
            "db": "taxonomy", "term": term, "retmode": "json", "retmax": "5"})
        try:
            return json.loads(raw)["esearchresult"]["idlist"]
        except (ValueError, KeyError):
            return []

    def _efetch(self, taxid: str) -> Tuple[str, str, List[List[str]]]:
        """(scientific_name, rank, lineage) for one taxid."""
        xml = self._get("efetch.fcgi", {
            "db": "taxonomy", "id": taxid, "retmode": "xml"})
        name = _first(xml, "ScientificName")
        rank = _first(xml, "Rank")
        lineage = []
        for block in re.findall(r"<Taxon>(.*?)</Taxon>", xml, re.DOTALL)[1:]:
            entry_id = _first(block, "TaxId")
            entry_name = _first(block, "ScientificName")
            entry_rank = _first(block, "Rank")
            if entry_id and entry_rank:
                lineage.append([entry_id, entry_name, entry_rank])
        return name, rank, lineage

    # -- resolution -------------------------------------------------------

    def _query_forms(self, normalized: str) -> List[str]:
        """Query strings to try, best first."""
        forms = []
        alias = ALIASES.get(normalized)
        if alias:
            forms.append(alias)

        # "m. tuberculosis" -> a cached full name with that initial and epithet.
        match = _ABBREV.match(normalized)
        if match:
            initial, epithet = match.group(1), match.group(2)
            for name in self.cached_names:
                parts = name.lower().split()
                if (len(parts) >= 2 and parts[0].startswith(initial)
                        and parts[1] == epithet):
                    forms.append(name)
                    break

        forms.append(normalized)
        seen, ordered = set(), []
        for form in forms:
            if form and form.lower() not in seen:
                seen.add(form.lower())
                ordered.append(form)
        return ordered

    def resolve_taxid(self, taxid: str) -> Optional[Resolution]:
        """A Resolution for a known taxid, so gold can be compared by lineage.

        Cached under a `taxid:` prefix, which cannot collide with a surface key
        because normalize_surface strips a bare number to itself, never to
        something containing a colon.
        """
        if not taxid:
            return None
        key = f"taxid:{taxid}"
        if key in self._cache:
            data = dict(self._cache[key])
            data["surface"] = key
            return Resolution(**data)
        if self.offline:
            return None

        name, rank, lineage = self._efetch(taxid)
        if not name:
            return None
        result = Resolution(surface=key, taxid=taxid, scientific_name=name,
                            rank=rank, lineage=lineage, source="ncbi_taxonomy")
        stored = asdict(result)
        stored.pop("surface", None)
        self._cache[key] = stored
        return result

    def resolve(self, surface: str) -> Resolution:
        normalized = normalize_surface(surface)
        if not normalized:
            return Resolution(surface=surface, note="empty surface")

        if normalized in self._cache:
            data = dict(self._cache[normalized])
            data["surface"] = surface
            return Resolution(**data)

        if self.offline:
            return Resolution(surface=surface, note="not in cache (offline)")

        result = Resolution(surface=surface)
        for form in self._query_forms(normalized):
            ids = self._esearch(form)
            if not ids:
                continue
            result.ambiguous = len(ids) > 1
            # Prefer a species-rank hit: "mouse" returns both Mus musculus and
            # the genus Mus, and the species is what a curator means.
            chosen = None
            for taxid in ids[:3]:
                name, rank, lineage = self._efetch(taxid)
                if chosen is None:
                    chosen = (taxid, name, rank, lineage)
                if rank == "species":
                    chosen = (taxid, name, rank, lineage)
                    break
            if chosen:
                result.taxid, result.scientific_name, result.rank, result.lineage = chosen
                result.source = "ncbi_taxonomy"
                if form != normalized:
                    result.note = f"matched via '{form}'"
                break

        if not result.resolved:
            result.note = "no NCBI taxonomy match"

        stored = asdict(result)
        stored.pop("surface", None)
        self._cache[normalized] = stored
        return result


def _first(xml: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.DOTALL)
    return match.group(1).strip() if match else ""


# -- scoring ---------------------------------------------------------------

EXACT, SAME_SPECIES, SAME_GENUS, WRONG = 1.0, 0.5, 0.25, 0.0


def credit(predicted: Resolution, gold_taxid: str,
           gold: Optional[Resolution] = None) -> float:
    """Partial credit for one linked mention.

    A boolean would be the wrong instrument. Resolving "Mycobacterium
    tuberculosis" to its subspecies is a near miss a curator can fix in
    seconds; resolving it to a different genus is a different organism. Scoring
    both as 0 hides which kind of wrong a system is.
    """
    if not predicted.resolved or not gold_taxid:
        return WRONG
    if predicted.taxid == gold_taxid:
        return EXACT
    if gold is not None:
        for rank, value in (("species", SAME_SPECIES), ("genus", SAME_GENUS)):
            mine, theirs = predicted.ancestor_at(rank), gold.ancestor_at(rank)
            if mine and theirs and mine == theirs:
                return value
        return WRONG
    # Without a resolved gold entity, only the predicted side's lineage is
    # available: gold sitting in our ancestry means we were too general.
    if any(entry[0] == gold_taxid for entry in predicted.lineage):
        return SAME_GENUS
    return WRONG


def score_mentions(surfaces: Sequence[str], gold_links: Dict[str, str],
                   linker: EntityLinker) -> Dict[str, Any]:
    """Coverage and accuracy for a run's entity mentions.

    Two numbers, never one. A system that links 12% of mentions perfectly would
    report 1.0 accuracy, which is true and useless. Coverage says how much it
    attempted; accuracy says how well it did on what it attempted.
    """
    if not linker._cache:
        return {"status": "not run",
                "detail": f"no entity cache at {linker.cache_path}"}

    total = 0
    resolved = 0
    scored: List[float] = []
    misses: List[Dict[str, str]] = []

    for surface in surfaces:
        if not str(surface or "").strip():
            continue
        total += 1
        prediction = linker.resolve(surface)
        if prediction.resolved:
            resolved += 1
        else:
            misses.append({"surface": surface, "why": prediction.note})
            continue

        gold_taxid = gold_links.get(normalize_surface(surface))
        if not gold_taxid:
            continue  # not in gold: counts for coverage, not for accuracy
        # The gold side's lineage is what makes "right genus, wrong species"
        # distinguishable from "wrong organism", so resolve it too.
        value = credit(prediction, gold_taxid, linker.resolve_taxid(gold_taxid))
        scored.append(value)
        if value < EXACT:
            misses.append({"surface": surface,
                           "why": f"credit {value}: got {prediction.taxid} "
                                  f"({prediction.rank}), gold {gold_taxid}"})

    return {
        "status": "ok",
        "mentions": total,
        "entity_link_coverage": round(resolved / total, 3) if total else 0.0,
        "entity_linking_accuracy": round(sum(scored) / len(scored), 3) if scored else None,
        "scored_against_gold": len(scored),
        "misses": misses[:20],
    }
