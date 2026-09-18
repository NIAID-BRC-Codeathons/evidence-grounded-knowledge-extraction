"""Batch curation runs.

A curation job is many queries, and a long one should survive an interruption.
Each completed query is appended to a progress sidecar as it finishes, so
--resume picks up exactly where the run stopped rather than re-billing the
queries that already succeeded.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .client import ApiError, RagStackClient
from .llm import LlmClient, LlmEndpoint, LlmError
from .extract import Row
from .pipeline import QuerySpec, RunResult, run_query
from .templates import TemplateError, TemplateRegistry

# Accepted column names in a batch file, mapped to QuerySpec fields.
FIELD_ALIASES = {
    "organism": "organism", "species": "organism", "pathogen": "organism",
    "genes": "genes", "gene": "genes", "gene_name": "genes", "gene_names": "genes",
    "other_terms": "other_terms", "other": "other_terms", "terms": "other_terms",
    "keywords": "other_terms",
    "data_type": "data_type", "type": "data_type", "datatype": "data_type",
    "template": "data_type",
    "top_k": "top_k", "topk": "top_k", "k": "top_k",
    "collection": "collection",
    "query_id": "query_id", "id": "query_id",
}


@dataclass
class BatchFailure:
    query_id: str
    organism: str
    error: str


@dataclass
class BatchOutcome:
    runs: List[RunResult] = field(default_factory=list)
    failures: List[BatchFailure] = field(default_factory=list)
    skipped: int = 0
    # Rows recovered from the progress sidecar for queries that were skipped.
    restored_rows: List[Row] = field(default_factory=list)
    restored_columns: List[str] = field(default_factory=list)
    restored_template_id: Optional[str] = None

    @property
    def n_total(self) -> int:
        return len(self.runs) + len(self.failures) + self.skipped


def _coerce(spec_fields: Dict[str, Any], defaults: "BatchDefaults") -> QuerySpec:
    top_k = spec_fields.get("top_k")
    try:
        top_k_value = int(top_k) if str(top_k or "").strip() else defaults.top_k
    except (TypeError, ValueError):
        top_k_value = defaults.top_k

    return QuerySpec(
        organism=str(spec_fields.get("organism", "")).strip(),
        genes=str(spec_fields.get("genes", "") or "").strip(),
        other_terms=str(spec_fields.get("other_terms", "") or "").strip(),
        data_type=str(spec_fields.get("data_type") or defaults.data_type).strip(),
        top_k=top_k_value,
        collection=(str(spec_fields.get("collection") or "").strip() or defaults.collection),
        collections=list(defaults.collections),
        retrieval_mode=defaults.retrieval_mode,
        keep_empty=defaults.keep_empty,
        no_dedupe=defaults.no_dedupe,
        query_id=str(spec_fields.get("query_id") or "").strip(),
        backend=defaults.backend,
        # Batch-wide, not per row: one curation job asks one question of the
        # literature, so the guidance that shapes the answer is a property of
        # the run rather than of any single organism/gene pair.
        instructions=defaults.instructions,
    )


@dataclass
class BatchDefaults:
    data_type: str = "literature-summary"
    top_k: int = 10
    collection: Optional[str] = None
    collections: List[str] = field(default_factory=list)
    retrieval_mode: str = "hybrid"
    keep_empty: bool = False
    no_dedupe: bool = False
    backend: str = "server"
    instructions: str = ""


def load_specs(path: Path, defaults: Optional[BatchDefaults] = None) -> List[QuerySpec]:
    """Read query specs from TSV, CSV, JSON, or YAML."""
    defaults = defaults or BatchDefaults()
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".json"}:
        raw = json.loads(text)
        records = raw.get("queries", raw) if isinstance(raw, dict) else raw
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML batch files") from exc
        raw = yaml.safe_load(text)
        records = raw.get("queries", raw) if isinstance(raw, dict) else raw
    else:
        delimiter = "," if suffix == ".csv" else "\t"
        # Sniff, because a .txt or mislabelled file is common enough to handle.
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if delimiter not in first_line:
            delimiter = "," if "," in first_line else "\t"
        records = list(csv.DictReader(text.splitlines(), delimiter=delimiter))

    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a list of query rows")

    specs: List[QuerySpec] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: row {index} is not a mapping")
        normalized: Dict[str, Any] = {}
        for key, value in record.items():
            if key is None:
                continue
            field_name = FIELD_ALIASES.get(str(key).strip().lower().replace(" ", "_"))
            if field_name:
                normalized[field_name] = value
        spec = _coerce(normalized, defaults)
        if not spec.organism:
            raise ValueError(
                f"{path}: row {index} has no organism "
                f"(columns seen: {', '.join(str(k) for k in record)})"
            )
        specs.append(spec)
    return specs


def _progress_record(run: RunResult) -> Dict[str, Any]:
    """A completed query, complete enough to rebuild its rows without re-asking."""
    record = dict(run.summary())
    record["columns"] = run.columns
    record["template_id"] = run.template.id
    record["is_table"] = run.extraction.is_table
    record["answer"] = "" if run.extraction.is_table else run.extraction.raw_answer
    record["rows"] = [row.to_dict() for row in run.rows]
    return record


def restore_rows(
    done: Dict[str, Dict[str, Any]]
) -> "tuple[List[Row], List[str]]":
    """Rebuild rows and columns from a progress sidecar."""
    rows: List[Row] = []
    columns: List[str] = []
    for record in done.values():
        if not record.get("is_table", True):
            continue
        for column in record.get("columns", []):
            if column not in columns:
                columns.append(column)
        for raw in record.get("rows", []):
            rows.append(Row.from_dict(raw))
    return rows, columns


def restored_template_id(done: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """The template of restored rows, when they all share one."""
    ids = {r.get("template_id") for r in done.values() if r.get("is_table", True)}
    ids.discard(None)
    return ids.pop() if len(ids) == 1 else None


def load_progress(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the sidecar written by a previous run."""
    if not path.is_file():
        return {}
    done: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        query_id = record.get("query_id")
        if query_id:
            done[query_id] = record
    return done


def run_batch(
    client: RagStackClient,
    registry: TemplateRegistry,
    specs: Sequence[QuerySpec],
    concurrency: int = 4,
    progress_path: Optional[Path] = None,
    resume: bool = False,
    on_done: Optional[Callable[[QuerySpec, Optional[RunResult], Optional[str]], None]] = None,
    endpoint: Optional[LlmEndpoint] = None,
) -> BatchOutcome:
    """Run many queries concurrently, recording progress as they complete."""
    outcome = BatchOutcome()
    already: Dict[str, Dict[str, Any]] = {}
    if resume and progress_path:
        already = load_progress(progress_path)

    pending = []
    wanted: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        query_id = spec.identity()
        if query_id in already:
            outcome.skipped += 1
            wanted[query_id] = already[query_id]
            continue
        pending.append(spec)

    if wanted:
        outcome.restored_rows, outcome.restored_columns = restore_rows(wanted)
        outcome.restored_template_id = restored_template_id(wanted)

    if not pending:
        return outcome

    # One LLM client for the whole batch so connections are pooled across
    # concurrent workers rather than reopened per query.
    llm_client = LlmClient(endpoint) if endpoint is not None else None

    handle = None
    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        handle = progress_path.open("a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {
                pool.submit(run_query, client, registry, spec, endpoint, llm_client): spec
                for spec in pending
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    run = future.result()
                except (ApiError, TemplateError, LlmError, ValueError) as exc:
                    outcome.failures.append(
                        BatchFailure(spec.identity(), spec.organism, str(exc))
                    )
                    if on_done:
                        on_done(spec, None, str(exc))
                    continue

                outcome.runs.append(run)
                if handle:
                    handle.write(json.dumps(_progress_record(run), ensure_ascii=False) + "\n")
                    handle.flush()
                if on_done:
                    on_done(spec, run, None)
    finally:
        if handle:
            handle.close()
        if llm_client is not None:
            llm_client.close()

    return outcome
