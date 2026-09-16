"""RAGStack client: retrieve, fetch_chunks, dedupe, paper_manifest, collect.

Standard library only. Reuses tls_context() from ../argo.py for HTTPS,
because the python.org build on this laptop has no root certificates.
Prints nothing on import.
"""

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argo import tls_context  # noqa: E402

DEFAULT_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)
DEFAULT_TOKEN_PATH = os.path.join(os.path.expanduser("~"), ".patric_token")
CHUNKS_BATCH_SIZE = 200
REQUEST_TIMEOUT = 60
PMC_ID_RE = re.compile(r"PMC\d+")
# Lowered from 20 to 4: a high retry ceiling was hiding a real outage as a
# long hang. The IncompleteRead truncation this retries has only been
# observed through this session's local sandbox proxy; a direct curl call
# against the same endpoint did not reproduce it, so it is not attributed
# to RAGStack itself.
MAX_TRANSIENT_RETRIES = 4


class RAGStackError(Exception):
    """Raised on any non-200 response, timeout, non-JSON body, or filter mismatch."""


def _verify_year(source, expected):
    return source.get("metadata", {}).get("year") == expected


def _verify_journal(source, expected):
    return source.get("metadata", {}).get("journal") == expected


def _verify_pmid(source, expected):
    return source.get("metadata", {}).get("pmid") == expected


def _verify_pmcid(source, expected):
    return source.get("metadata", {}).get("pmcid") == expected


def _verify_doi(source, expected):
    return source.get("metadata", {}).get("doi") == expected


# Filter keys we can actually verify against the response metadata. A
# filter key that is not in this map is unverifiable, so retrieve() treats
# it as unsupported and raises before the request ever goes out, rather
# than trusting the server to have honored it.
VERIFIABLE_FILTERS = {
    "year": _verify_year,
    "journal": _verify_journal,
    "pmid": _verify_pmid,
    "pmcid": _verify_pmcid,
    "doi": _verify_doi,
}


def load_env(path=None):
    """Parse the .env file into a dict, without shelling out.

    Returns a dict with at least RAGSTACK_BASE. RAGSTACK_API_KEY is no
    longer required here: a caller with a working login token (see
    _read_token) never needs the key. Raises only when RAGSTACK_BASE is
    missing, or when neither a key nor a token is available at all. Never
    prints or logs the value of either credential.
    """
    env_path = path or DEFAULT_ENV_PATH
    values = {}
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            values[key] = value
    if "RAGSTACK_BASE" not in values:
        raise RAGStackError(f"{env_path} is missing RAGSTACK_BASE")
    if "RAGSTACK_API_KEY" not in values and not _read_token():
        raise RAGStackError(
            f"{env_path} has no RAGSTACK_API_KEY and no login token was found "
            f"at {_token_path()}: no credential is available"
        )
    return values


def _token_path():
    """Where the login token is read from: RAGSTACK_TOKEN_PATH if set,
    else the p3-login token file in the user's home directory.
    """
    return os.environ.get("RAGSTACK_TOKEN_PATH") or DEFAULT_TOKEN_PATH


def _read_token(path=None):
    """Read and return the login token, stripped of trailing whitespace, or
    None when the file is missing or empty. Never logs the value.
    """
    token_path = path or _token_path()
    try:
        with open(token_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return None
    content = content.rstrip()
    return content or None


def auth_mode(env=None):
    """Decide which credential mode is active: 'token' or 'key'. Reads the
    RAGSTACK_AUTH environment variable ('token', 'key', or 'auto', default
    'auto'). Never returns or logs the credential value itself, only the
    mode name.

    - 'token': use the login token; raises clearly if it is missing.
    - 'key': always use the API key.
    - 'auto': use the token when the token file exists and is non-empty,
      otherwise fall back to the key.
    """
    requested = os.environ.get("RAGSTACK_AUTH", "auto").strip().lower()
    if requested not in ("auto", "token", "key"):
        raise RAGStackError(
            f"RAGSTACK_AUTH={requested!r} is not one of: token, key, auto"
        )

    if requested == "token":
        if not _read_token():
            raise RAGStackError(
                f"RAGSTACK_AUTH=token but no login token was found at "
                f"{_token_path()}"
            )
        return "token"

    if requested == "key":
        return "key"

    # auto
    return "token" if _read_token() else "key"


def build_auth_header(env=None):
    """Build the single auth header for the active credential mode. Returns
    (headers_dict, mode). Sends exactly one of Authorization (no "Bearer "
    prefix, per the login-token API) or X-API-Key, never both. Never logs
    or includes the credential value in an exception message; only the
    mode is ever reported.
    """
    mode = auth_mode(env)
    if mode == "token":
        token = _read_token()
        if not token:
            raise RAGStackError(
                f"no login token found at {_token_path()} (mode=token)"
            )
        return {"Authorization": token}, "token"

    api_key = (env or {}).get("RAGSTACK_API_KEY")
    if not api_key:
        raise RAGStackError(
            "no RAGSTACK_API_KEY available for the key credential (mode=key)"
        )
    return {"X-API-Key": api_key}, "key"


def resolve_pmc_id(metadata):
    """Return (pmc_id, derived) from source/paper metadata.

    Prefers the supplied `pmcid`. When absent, derives it by matching
    PMC<digits> in `filename` (e.g. "Dengue_PMC6428985.pdf"), which is the
    only recovery path this makes: it never invents a pmid. `derived` is
    True only when the id came from the filename match rather than the
    metadata's own pmcid field. This is the single place both
    paper_manifest() and extract.py's evidence filling call, so the two
    cannot disagree on a derived id.
    """
    metadata = metadata or {}
    pmcid = metadata.get("pmcid") or None
    if pmcid:
        return pmcid, False
    filename = metadata.get("filename") or ""
    match = PMC_ID_RE.search(filename)
    if match:
        return match.group(0), True
    return None, False


def _request(method, url, env, payload=None):
    """Make one HTTPS call and return parsed JSON. `env` supplies the
    credential (via build_auth_header); never logs either credential
    value, only reports which mode was used on error.
    """
    headers, mode = build_auth_header(env)
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            **headers,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; ragstack-client/1.0)",
            "Connection": "close",
        },
    )
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT, context=tls_context()
            ) as response:
                status = response.getcode()
                body = response.read()
            break
        except urllib.error.HTTPError as error:
            body_text = error.read().decode(errors="replace")[:500]
            raise RAGStackError(
                f"HTTP {error.code} from {url} (auth mode={mode}): {body_text}"
            ) from error
        except urllib.error.URLError as error:
            raise RAGStackError(f"Cannot reach RAGStack ({error.reason})") from error
        except TimeoutError as error:
            raise RAGStackError(f"Timed out calling {url}") from error
        except http.client.IncompleteRead as error:
            # Transient proxy/connection drop mid-body. Retry a bounded
            # number of times before surfacing it as a hard error.
            attempt += 1
            if attempt > MAX_TRANSIENT_RETRIES:
                raise RAGStackError(
                    f"Response from {url} was truncated after {attempt} attempts "
                    f"({error}). This truncation has only been seen through this "
                    "session's local sandbox proxy; rerun outside the sandbox "
                    "before suspecting RAGStack itself."
                ) from error
            time.sleep(min(0.5 * attempt, 3.0))

    if status != 200:
        raise RAGStackError(f"Unexpected status {status} from {url}: {body[:500]!r}")

    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise RAGStackError(
            f"Response from {url} was not valid JSON: {body[:500]!r}"
        ) from error


def retrieve(query, collection="asm-semantic", top_k=25, filters=None, env=None):
    """POST /v1/retrieve. Verifies a requested filter actually held on every
    returned source; fails closed (raises) rather than silently returning
    unfiltered results.

    A filter key that is not in VERIFIABLE_FILTERS is unverifiable, which
    this treats as unsupported: it raises before the request is sent,
    rather than trusting the server (or a client-side blind spot) to have
    honored a key we have no way to check.
    """
    filters = filters or {}
    unverifiable = [key for key in filters if key not in VERIFIABLE_FILTERS]
    if unverifiable:
        raise RAGStackError(
            f"Unsupported filter key(s) {unverifiable}: not in the set this "
            f"client can verify against response metadata ({sorted(VERIFIABLE_FILTERS)}). "
            "An unverifiable filter is treated as unsupported, never as silently ignored."
        )

    env = env or load_env()
    base = env["RAGSTACK_BASE"].rstrip("/")
    payload = {
        "query": query,
        "collection": collection,
        "top_k": top_k,
        "filters": filters,
    }
    data = _request("POST", base + "/v1/retrieve", env, payload)

    sources = data.get("sources", data if isinstance(data, list) else [])
    if isinstance(data, dict) and "sources" not in data:
        # Unknown response shape; treat as a hard error rather than guessing.
        raise RAGStackError(f"Unrecognized /v1/retrieve response shape: {list(data.keys())}")

    for key, expected in filters.items():
        verify = VERIFIABLE_FILTERS[key]
        for source in sources:
            if not verify(source, expected):
                raise RAGStackError(
                    f"Filter {key}={expected!r} was not honored: got "
                    f"{source.get('metadata', {}).get(key)!r} on "
                    f"chunk_id={source.get('chunk_id')!r}"
                )

    return sources


def fetch_chunks(ids, env=None):
    """GET /v1/chunks?ids=... , batching at CHUNKS_BATCH_SIZE (200) per request."""
    env = env or load_env()
    base = env["RAGSTACK_BASE"].rstrip("/")
    ids = list(ids)
    chunks = []
    for start in range(0, len(ids), CHUNKS_BATCH_SIZE):
        batch = ids[start : start + CHUNKS_BATCH_SIZE]
        query_string = urllib.parse.urlencode({"ids": ",".join(batch)})
        url = f"{base}/v1/chunks?{query_string}"
        data = _request("GET", url, env)
        batch_chunks = data.get("chunks", data if isinstance(data, list) else [])
        chunks.extend(batch_chunks)
    return chunks


def dedupe(sources):
    """Remove repeats by chunk_id. Returns (kept, duplicates_removed)."""
    seen = set()
    kept = []
    duplicates_removed = 0
    for source in sources:
        chunk_id = source.get("chunk_id")
        if chunk_id in seen:
            duplicates_removed += 1
            continue
        seen.add(chunk_id)
        kept.append(source)
    return kept, duplicates_removed


def paper_manifest(sources):
    """One entry per doc_id: pubmed_id, pmc_id, year, chunk count.

    Contract field names (pubmed_id, pmc_id) differ from the API's metadata
    keys (pmid, pmcid). Absent values are None, never "" or "N/A". pmc_id
    falls back to a filename-derived id via resolve_pmc_id() when the
    metadata carries no pmcid (sparse collections such as Dengue); the
    additive "pmc_id_derived" key records whether that happened, so a
    curator can tell a supplied id from a recovered one. pubmed_id is never
    derived, only ever taken from metadata.
    """
    by_doc = {}
    for source in sources:
        doc_id = source.get("doc_id")
        metadata = source.get("metadata", {})
        if doc_id not in by_doc:
            pmc_id, pmc_id_derived = resolve_pmc_id(metadata)
            by_doc[doc_id] = {
                "doc_id": doc_id,
                "pubmed_id": metadata.get("pmid") or None,
                "pmc_id": pmc_id,
                "pmc_id_derived": pmc_id_derived,
                "year": metadata.get("year") or None,
                "chunks": 0,
            }
        by_doc[doc_id]["chunks"] += 1
    return list(by_doc.values())


def list_collections(env=None):
    """GET /v1/collections and return the list of reachable collection ids
    for the active credential. Used to live-validate a requested collection
    id instead of trusting a hardcoded list, since a token and a key reach
    different collections.
    """
    env = env or load_env()
    base = env["RAGSTACK_BASE"].rstrip("/")
    data = _request("GET", base + "/v1/collections", env)
    raw = data.get("collections", data if isinstance(data, list) else [])
    return [
        entry.get("id")
        for entry in raw
        if isinstance(entry, dict) and entry.get("id")
    ]


def collect(
    organism,
    gene,
    aliases,
    data_type_terms,
    collection="asm-semantic",
    top_k=25,
    filters=None,
    env=None,
):
    """Union several query paraphrases from aliases x topic terms, dedupe by
    chunk_id, return (kept_sources, manifest, stats). Never pads with
    off-topic queries: only alias/topic paraphrases are issued.
    """
    env = env or load_env()
    aliases = list(aliases) or [gene]
    data_type_terms = list(data_type_terms) or [""]

    queries = []
    for alias in aliases:
        for term in data_type_terms:
            query = f"{organism} {alias} {term}".strip()
            if query not in queries:
                queries.append(query)

    all_sources = []
    chunks_returned = 0
    for query in queries:
        sources = retrieve(
            query, collection=collection, top_k=top_k, filters=filters, env=env
        )
        chunks_returned += len(sources)
        all_sources.extend(sources)

    kept, duplicates_removed = dedupe(all_sources)
    manifest = paper_manifest(kept)

    stats = {
        "queries_issued": len(queries),
        "chunks_returned": chunks_returned,
        "chunks_after_dedupe": len(kept),
        "duplicates_removed": duplicates_removed,
        "papers": len(manifest),
    }
    return kept, manifest, stats


if __name__ == "__main__":
    env = load_env()

    print("RAGStack self-check")
    print("-" * 40)
    print(f"Auth mode: {auth_mode(env)}")

    kept, manifest, stats = collect(
        organism="Influenza A virus",
        gene="PB2",
        aliases=["PB2", "polymerase basic 2", "polymerase basic protein 2"],
        data_type_terms=["mutation", "adaptation", "phenotype"],
        collection="asm-semantic",
        top_k=25,
        env=env,
    )

    distinct_papers = {entry["doc_id"] for entry in manifest}
    with_pmid = sum(1 for entry in manifest if entry["pubmed_id"])

    print(f"Distinct papers (doc_id) for PB2 mutation collect: {len(distinct_papers)}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Chunks returned (pre-dedupe): {stats['chunks_returned']}")
    print(f"Chunks after dedupe: {stats['chunks_after_dedupe']}")
    print(f"Papers carrying a pmid: {with_pmid} / {len(manifest)}")
    missing_chunk_id = [s for s in kept if not s.get("chunk_id")]
    print(f"Sources missing chunk_id: {len(missing_chunk_id)} / {len(kept)}")

    if len(distinct_papers) >= 20:
        print("PASS: 20 or more distinct papers")
    else:
        print(f"FAIL: only {len(distinct_papers)} distinct papers (need 20+)")

    print()
    print("Bad filter check (expect an exception):")
    try:
        retrieve(
            "PB2 mutation",
            collection="asm-semantic",
            top_k=5,
            filters={"year": "not-a-real-year-value-9999"},
            env=env,
        )
        print("FAIL: bad filter did not raise")
    except RAGStackError as error:
        print(f"PASS: bad filter raised RAGStackError: {error}")

    print()
    print("Unknown filter key check (expect a raise before any request):")
    try:
        retrieve(
            "PB2 mutation",
            collection="asm-semantic",
            top_k=3,
            filters={"not_a_real_filter": "x"},
            env=env,
        )
        print("FAIL: unknown filter key did not raise")
    except RAGStackError as error:
        print(f"PASS: unknown filter key raised RAGStackError: {error}")

    print()
    print("Year=2020 filter check (expect only 2020 papers back):")
    try:
        year_sources = retrieve(
            "PB2 mutation",
            collection="asm-semantic",
            top_k=25,
            filters={"year": 2020},
            env=env,
        )
        years = {source.get("metadata", {}).get("year") for source in year_sources}
        if years and years == {2020}:
            print(f"PASS: all {len(year_sources)} sources are year 2020")
        elif not year_sources:
            print("PASS: zero sources returned, nothing to violate the filter")
        else:
            print(f"FAIL: returned years {years}, not all 2020")
    except RAGStackError as error:
        print(f"FAIL: year=2020 filter raised unexpectedly: {error}")

    print()
    print("Verification helper check (mismatched pair must raise directly):")
    fake_source = {"chunk_id": "fake-0001", "metadata": {"year": 1999}}
    try:
        if not _verify_year(fake_source, 2020):
            raise RAGStackError(
                f"Filter year=2020 was not honored: got "
                f"{fake_source['metadata']['year']!r} on chunk_id={fake_source['chunk_id']!r}"
            )
        print("FAIL: mismatched year pair did not raise")
    except RAGStackError as error:
        print(f"PASS: mismatched year pair raised RAGStackError: {error}")
