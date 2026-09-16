"""API key and endpoint resolution.

The key is never read from source. Resolution order is explicit argument, then
environment, then user config file -- so a CI run and a laptop both work without
either one leaking a credential into the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 and below
    tomllib = None  # type: ignore[assignment]

DEFAULT_BASE_URL = "https://www.bv-brc.org/ragstack/hackathon/api"
CONFIG_PATH = Path.home() / ".config" / "litrag" / "config.toml"

ENV_API_KEY = "LITRAG_API_KEY"
ENV_BASE_URL = "LITRAG_BASE_URL"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or unreadable."""


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 300.0

    @property
    def redacted_key(self) -> str:
        """The key with its middle removed, safe to print in logs and errors."""
        if len(self.api_key) <= 12:
            return "***"
        return f"{self.api_key[:8]}...{self.api_key[-4:]}"


def _read_config_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    if tomllib is None:  # pragma: no cover
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc


def load_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 300.0,
    config_path: Optional[Path] = None,
) -> Config:
    """Resolve configuration from argument, environment, then config file."""
    path = config_path if config_path is not None else CONFIG_PATH
    file_cfg = _read_config_file(path)

    resolved_key = api_key or os.environ.get(ENV_API_KEY) or file_cfg.get("api_key")
    resolved_url = (
        base_url
        or os.environ.get(ENV_BASE_URL)
        or file_cfg.get("base_url")
        or DEFAULT_BASE_URL
    )

    if not resolved_key:
        raise ConfigError(
            "No API key found. Provide one of:\n"
            "  --api-key <key>\n"
            f"  export {ENV_API_KEY}=<key>\n"
            f"  echo 'api_key = \"<key>\"' > {path}"
        )

    return Config(
        api_key=str(resolved_key),
        base_url=str(resolved_url).rstrip("/"),
        timeout=timeout,
    )
