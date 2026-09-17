"""Credential resolution."""

import pytest

from litrag.config import DEFAULT_BASE_URL, ConfigError, load_config


def test_explicit_argument_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LITRAG_API_KEY", "from-env")
    config = load_config(api_key="explicit", config_path=tmp_path / "none.toml")
    assert config.api_key == "explicit"


def test_environment_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("LITRAG_API_KEY", "from-env")
    assert load_config(config_path=tmp_path / "none.toml").api_key == "from-env"


def test_config_file_is_last_resort(tmp_path, monkeypatch):
    monkeypatch.delenv("LITRAG_API_KEY", raising=False)
    path = tmp_path / "config.toml"
    path.write_text('api_key = "from-file"\nbase_url = "https://example.test"\n')
    config = load_config(config_path=path)
    assert config.api_key == "from-file"
    assert config.base_url == "https://example.test"


def test_missing_key_explains_every_option(tmp_path, monkeypatch):
    monkeypatch.delenv("LITRAG_API_KEY", raising=False)
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path=tmp_path / "none.toml")
    message = str(excinfo.value)
    assert "--api-key" in message and "LITRAG_API_KEY" in message


def test_default_base_url(tmp_path, monkeypatch):
    monkeypatch.delenv("LITRAG_BASE_URL", raising=False)
    assert load_config(api_key="k", config_path=tmp_path / "n.toml").base_url == DEFAULT_BASE_URL


def test_trailing_slash_is_trimmed(tmp_path):
    config = load_config(api_key="k", base_url="https://x.test/api/",
                         config_path=tmp_path / "n.toml")
    assert config.base_url == "https://x.test/api"


def test_key_is_redacted_for_display(tmp_path):
    config = load_config(api_key="rk-test-0000-SECRETSECRETSECRET-tail",
                         config_path=tmp_path / "n.toml")
    assert "SECRET" not in config.redacted_key
    assert config.redacted_key.startswith("rk-test-")
