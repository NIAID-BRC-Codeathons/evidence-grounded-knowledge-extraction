"""CLI option handling that does not need a server."""

from pathlib import Path

import pytest
import typer

from litrag.cli import _instructions


def test_typed_instructions_pass_through():
    assert _instructions("Confirmed findings only.", None) == "Confirmed findings only."


def test_a_file_supplies_instructions_too_long_to_type(tmp_path):
    path = tmp_path / "guidance.txt"
    body = "Report positions as A226, K128.\n" * 200
    path.write_text(body, encoding="utf-8")
    assert _instructions("", path) == body.strip()


def test_trailing_newline_does_not_perturb_the_identity_hash(tmp_path):
    """A file and the same text typed inline must hash to the same query."""
    path = tmp_path / "guidance.txt"
    path.write_text("Confirmed findings only.\n", encoding="utf-8")
    assert _instructions("", path) == _instructions("Confirmed findings only.", None)


def test_both_sources_is_an_error(tmp_path):
    """Silently picking one of two given sources would be the wrong guess."""
    path = tmp_path / "guidance.txt"
    path.write_text("From the file.", encoding="utf-8")
    with pytest.raises(typer.Exit) as excinfo:
        _instructions("Typed inline.", path)
    assert excinfo.value.exit_code == 2


def test_an_empty_file_is_an_error(tmp_path):
    """Naming a file and getting an unguided run back is a mistake, not a default."""
    path = tmp_path / "guidance.txt"
    path.write_text("  \n\n", encoding="utf-8")
    with pytest.raises(typer.Exit) as excinfo:
        _instructions("", path)
    assert excinfo.value.exit_code == 2


def test_a_binary_file_is_reported_not_raised(tmp_path):
    path = tmp_path / "guidance.bin"
    path.write_bytes(b"\xff\xfe\x00not text")
    with pytest.raises(typer.Exit) as excinfo:
        _instructions("", path)
    assert excinfo.value.exit_code == 2


def test_an_unreadable_path_exits_cleanly(tmp_path):
    """Typer screens most bad paths, but the read itself must still not traceback."""
    with pytest.raises(typer.Exit) as excinfo:
        _instructions("", Path(tmp_path))
    assert excinfo.value.exit_code == 2
