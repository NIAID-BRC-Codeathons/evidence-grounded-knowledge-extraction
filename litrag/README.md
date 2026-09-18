# LitRAG

Automated literature curation and knowledge extraction over the
[BV-BRC](https://www.bv-brc.org) RAGStack literature API.

Give it an organism, some genes, and a data type; get back a deduplicated,
citation-resolved, reproducible table. One core pipeline, two front ends: a CLI
for batch curation and a standalone web UI for interactive work.

## Install

Commands run from the repository root, which is the uv workspace this package belongs to.

```bash
uv sync                 # creates .venv at the repository root from uv.lock
uv run litrag --help
```

With pip:

```bash
pip install -e litrag
```

## Documentation and license

Full documentation lives at the repository root:

- [README](../README.md): what LitRAG does, quick start, data types, and every output column and flag
- [Architecture](../ARCHITECTURE.md): how the package is wired, for maintainers
- [Walkthrough](../WALKTHROUGH.md): what a curated table is and how to audit a claim in it, no installation needed
- [License](../LICENSE): Apache License 2.0
