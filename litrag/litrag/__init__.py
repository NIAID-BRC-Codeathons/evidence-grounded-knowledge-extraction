"""LitRAG -- automated literature curation and knowledge extraction.

Built on the BV-BRC RAGStack literature API. One core pipeline, two surfaces:
a CLI for batch curation and a web UI for interactive exploration.
"""

from .client import ApiError, QueryResult, RagStackClient
from .config import Config, ConfigError, load_config
from .extract import Citation, Extraction, Row, extract
from .pipeline import QuerySpec, RunResult, merge_runs, run_query
from .templates import Template, TemplateError, TemplateRegistry

__version__ = "0.1.0"

__all__ = [
    "ApiError", "QueryResult", "RagStackClient",
    "Config", "ConfigError", "load_config",
    "Citation", "Extraction", "Row", "extract",
    "QuerySpec", "RunResult", "run_query", "merge_runs",
    "Template", "TemplateError", "TemplateRegistry",
    "__version__",
]
