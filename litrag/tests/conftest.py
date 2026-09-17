import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def registry():
    from litrag.templates import TemplateRegistry
    return TemplateRegistry.from_declarations(load("templates.json")["templates"])


@pytest.fixture(scope="session")
def mutation_response():
    return load("mutation_katg.json")


@pytest.fixture(scope="session")
def ppi_response():
    return load("ppi_sars_cov2.json")
