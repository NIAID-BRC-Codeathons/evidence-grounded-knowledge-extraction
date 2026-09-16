"""OpenAI-compatible generation backends.

The hosted /v1/query endpoint always generates with Llama-4-Scout -- its `llm`
parameter is accepted but silently returns an empty answer with no sources, so
model choice is not available there. Pointing at a vLLM server directly is the
way to pick a generator.

Benchmarked on identical retrieved chunks (8 chunks, katG/inhA extraction):

    Llama-4-Scout           3.0s   12 rows   bare assertions ("High")
    Qwen3.6 thinking       29.4s    3 rows   burned 4758 tokens reasoning
    Qwen3.6 no-think        3.6s   14 rows   quantitative ("MIC ~6.4 mg/L")

Hence Qwen is the default and its thinking is disabled: for schema-constrained
extraction the reasoning budget crowds out the answer without improving it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


class LlmError(RuntimeError):
    """A generation request failed."""


@dataclass(frozen=True)
class LlmEndpoint:
    """A generation backend."""

    name: str
    base_url: str
    model: Optional[str] = None
    # Qwen3 exposes a thinking mode through the chat template. False disables it.
    thinking: Optional[bool] = None
    label: str = ""

    def described(self) -> str:
        return self.label or f"{self.name} ({self.base_url})"


# `server` is the hosted RAGStack path and has no endpoint of its own.
SERVER = "server"

PRESETS: Dict[str, LlmEndpoint] = {
    "qwen": LlmEndpoint(
        name="qwen",
        base_url="http://mango.cels.anl.gov:8004/v1",
        model="Qwen/Qwen3.6-35B-A3B",
        thinking=False,
        label="Qwen3.6-35B (local, thinking off)",
    ),
    "llama": LlmEndpoint(
        name="llama",
        base_url="http://mango.cels.anl.gov:8003/v1",
        model="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
        label="Llama-4-Scout (local)",
    ),
}

DEFAULT_BACKEND = "qwen"


def resolve_endpoint(
    spec: Optional[str],
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
) -> Optional[LlmEndpoint]:
    """Turn a --llm value into an endpoint.

    Returns None for the hosted server path. Accepts a preset name or a bare
    URL, so an endpoint that is not in PRESETS can still be used.
    """
    choice = (spec or DEFAULT_BACKEND).strip()
    if choice.lower() in {SERVER, "hosted", "ragstack"}:
        return None

    preset = PRESETS.get(choice.lower())
    if preset is None:
        if "://" not in choice:
            options = ", ".join([SERVER] + sorted(PRESETS))
            raise LlmError(f"unknown backend '{choice}'. Choose from: {options}, or give a URL")
        preset = LlmEndpoint(name="custom", base_url=choice.rstrip("/"))

    return LlmEndpoint(
        name=preset.name,
        base_url=preset.base_url,
        model=model or preset.model,
        thinking=preset.thinking if thinking is None else thinking,
        label=preset.label,
    )


@dataclass
class Completion:
    text: str
    model: str
    endpoint: str
    reasoning: str = ""
    finish_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    truncated: bool = False


class LlmClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        endpoint: LlmEndpoint,
        timeout: float = 600.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.endpoint = endpoint
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LlmClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def available_models(self) -> List[str]:
        try:
            response = self._client.get(f"{self.endpoint.base_url}/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LlmError(f"could not list models at {self.endpoint.base_url}: {exc}") from exc
        return [m["id"] for m in response.json().get("data", [])]

    def resolve_model(self) -> str:
        """The model to request, discovered from the server when unset."""
        if self.endpoint.model:
            return self.endpoint.model
        models = self.available_models()
        if not models:
            raise LlmError(f"no models served at {self.endpoint.base_url}")
        return models[0]

    def complete(
        self,
        prompt: str,
        max_tokens: int = 2500,
        temperature: float = 0.0,
    ) -> Completion:
        model = self.resolve_model()
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.endpoint.thinking is not None:
            # vLLM passes this through to the chat template.
            body["chat_template_kwargs"] = {"enable_thinking": self.endpoint.thinking}

        started = time.monotonic()
        try:
            response = self._client.post(
                f"{self.endpoint.base_url}/chat/completions", json=body
            )
        except httpx.HTTPError as exc:
            raise LlmError(f"generation request to {self.endpoint.base_url} failed: {exc}") from exc

        if response.status_code >= 400:
            raise LlmError(
                f"generation failed with HTTP {response.status_code} "
                f"at {self.endpoint.base_url}: {response.text[:300]}"
            )

        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise LlmError(f"no completion returned by {self.endpoint.base_url}")

        choice = choices[0]
        message = choice.get("message", {}) or {}
        finish = choice.get("finish_reason", "")

        return Completion(
            text=(message.get("content") or "").strip(),
            model=payload.get("model", model),
            endpoint=self.endpoint.base_url,
            reasoning=(message.get("reasoning") or message.get("reasoning_content") or ""),
            finish_reason=finish,
            usage=payload.get("usage", {}) or {},
            elapsed_s=time.monotonic() - started,
            truncated=finish == "length",
        )
