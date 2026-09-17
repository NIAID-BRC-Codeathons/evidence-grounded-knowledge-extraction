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

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    # Which parameter dialect the request body must speak. The vLLM boxes take
    # plain OpenAI; Argo diverges per model family (see _apply_argo_params).
    policy: str = "openai"
    # Auth headers, if the backend needs any. The vLLM servers need none.
    headers: Optional[Dict[str, str]] = None

    def described(self) -> str:
        return self.label or f"{self.name} ({self.base_url})"


# `server` is the hosted RAGStack path and has no endpoint of its own.
SERVER = "server"

ARGO_BASE_URL = "https://apps.inside.anl.gov/argoapi/v1"

# Argo parameter rules, measured on the gateway and recorded in the codeathon
# quickstart. Each exists because omitting it produced a live failure:
#   - o-series take max_completion_tokens and reject max_tokens.
#   - Claude models on /v1/chat/completions cap max_tokens at 21000 (500 above).
#   - gemini35flash spends its budget on internal reasoning and returns an empty
#     candidate below 2048.
#   - temperature is never sent: claudesonnet5 and every GPT-5.x reject any value
#     other than 1, and omitting it is accepted by every family.
ARGO_O_SERIES = {"gpto1", "gpto3", "gpto3mini", "gpto4mini"}
ARGO_CLAUDE_MAX_TOKENS = 21000
ARGO_MIN_TOKENS = {"gemini35flash": 2048}

# Retry policy, mirroring the hosted client (client.py:18-20).
#
# 500 is deliberately absent. A vLLM or Argo 500 is usually deterministic -- a bad
# parameter for that model family, or a model that is simply broken on the gateway
# -- so retrying costs four times the latency and still fails. Transient conditions
# announce themselves as 429 or as a 502/503/504 from the fronting proxy.
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_S = 1.0
RETRY_STATUS = {429, 502, 503, 504}

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
    "argo": LlmEndpoint(
        name="argo",
        base_url=ARGO_BASE_URL,
        # Deliberately unpinned: Argo serves 30+ chat models and the caller must
        # say which. resolve_model() would otherwise pick models[0], and Argo's
        # call id lives in internal_id rather than id, so discovery is unsafe here.
        model=None,
        policy="argo",
        label="Argo gateway (Argonne)",
    ),
}

DEFAULT_BACKEND = "qwen"


def argo_username() -> str:
    """The Argo credential: an Argonne collaborator username, not a password.

    ARGO_USER wins. Failing that, walk up from the working directory for the
    codeathon helper script, so a checkout inside the workspace works unconfigured.
    """
    user = os.environ.get("ARGO_USER", "").strip()
    if user:
        return user

    directory = Path.cwd().resolve()
    for candidate in [directory, *directory.parents]:
        helper = candidate / ".claude" / "argo-user.sh"
        if helper.is_file() and os.access(helper, os.X_OK):
            try:
                found = subprocess.run(
                    [str(helper)], capture_output=True, text=True, timeout=10
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                break
            if found:
                return found
            break

    raise LlmError(
        "Argo needs a username. Set ARGO_USER to your Argonne collaborator "
        "username (for example ac.jdoe). It is an identifier, not a password -- "
        "never put the domain password here."
    )


def _apply_argo_params(body: Dict[str, Any], model: str, max_tokens: int) -> None:
    """Rewrite an OpenAI-shaped body into what Argo actually accepts.

    Mutates `body` in place. Temperature is dropped entirely rather than set,
    because "not sent" and "sent as the default" are different requests here.
    """
    body.pop("temperature", None)
    if model in ARGO_O_SERIES:
        body.pop("max_tokens", None)
        body["max_completion_tokens"] = max_tokens
        return

    limit = max_tokens
    if model.startswith("claude"):
        limit = min(limit, ARGO_CLAUDE_MAX_TOKENS)
    floor = ARGO_MIN_TOKENS.get(model)
    if floor:
        limit = max(limit, floor)
    body["max_tokens"] = limit


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

    resolved_model = model or preset.model
    headers = preset.headers
    if preset.policy == "argo":
        if not resolved_model:
            raise LlmError(
                "the argo backend needs an explicit model, e.g. --llm-model gpt56sol. "
                "Argo serves many models and none is a safe default."
            )
        # Resolved at endpoint-build time so a missing username fails before a
        # batch starts rather than on the first query inside the worker pool.
        headers = {"Authorization": f"Bearer {argo_username()}"}

    return LlmEndpoint(
        name=preset.name,
        base_url=preset.base_url,
        model=resolved_model,
        thinking=preset.thinking if thinking is None else thinking,
        label=preset.label,
        policy=preset.policy,
        headers=headers,
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
        retries: int = RETRY_ATTEMPTS,
        backoff_s: float = RETRY_BACKOFF_S,
    ) -> None:
        self.endpoint = endpoint
        self.retries = max(1, retries)
        self.backoff_s = max(0.0, backoff_s)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout, headers=endpoint.headers or None
        )
        # An injected client (tests, or a caller pooling connections) keeps its own
        # headers, so auth has to ride on the request instead.
        self._request_headers = endpoint.headers if not self._owns_client else None

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

    def _post_with_retry(self, body: Dict[str, Any]) -> httpx.Response:
        """POST the completion, retrying transport errors and transient statuses.

        The hosted RagStackClient has had backoff since the start (client.py:18-20);
        this path had none, so a single pod hiccup surfaced as a hard failure after
        a potentially very long timeout. Same shape, same status allowlist.
        """
        url = f"{self.endpoint.base_url}/chat/completions"
        last_error: Optional[Exception] = None

        for attempt in range(self.retries):
            try:
                response = self._client.post(
                    url, json=body, headers=self._request_headers
                )
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code not in RETRY_STATUS or attempt == self.retries - 1:
                    return response
                last_error = LlmError(f"HTTP {response.status_code}")

            if attempt < self.retries - 1 and self.backoff_s:
                time.sleep(self.backoff_s * (2 ** attempt))

        raise LlmError(
            f"generation request to {self.endpoint.base_url} failed after "
            f"{self.retries} attempts: {last_error}"
        ) from last_error

    def complete(
        self,
        prompt: str,
        max_tokens: int = 2500,
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> Completion:
        """Generate once.

        `system` adds a system role. The original single-user-message shape is
        preserved when it is None, so existing callers and recorded prompt hashes
        are unaffected.
        """
        model = self.resolve_model()
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.endpoint.policy == "argo":
            _apply_argo_params(body, model, max_tokens)
        if self.endpoint.thinking is not None:
            # vLLM passes this through to the chat template.
            body["chat_template_kwargs"] = {"enable_thinking": self.endpoint.thinking}

        started = time.monotonic()
        response = self._post_with_retry(body)

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
