"""Local generation backends."""

import httpx
import pytest

from litrag.llm import (DEFAULT_BACKEND, PRESETS, Completion, LlmClient,
                        LlmEndpoint, LlmError, resolve_endpoint)


def make_client(handler, endpoint=None):
    endpoint = endpoint or PRESETS["qwen"]
    return LlmClient(endpoint, client=httpx.Client(transport=httpx.MockTransport(handler)))


def chat_response(content, reasoning=None, finish="stop", model="Qwen/Qwen3.6-35B-A3B"):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return httpx.Response(200, json={
        "model": model,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"completion_tokens": 42},
    })


def test_qwen_is_the_default_backend():
    endpoint = resolve_endpoint(None)
    assert DEFAULT_BACKEND == "qwen"
    assert endpoint.base_url == "http://mango.cels.anl.gov:8004/v1"
    assert endpoint.model == "Qwen/Qwen3.6-35B-A3B"


def test_qwen_disables_thinking_by_default():
    """Benchmarked: thinking cost 8x the time and produced a quarter the rows."""
    assert resolve_endpoint("qwen").thinking is False


def test_server_backend_resolves_to_none():
    """None means the hosted /v1/query path, not a local endpoint."""
    assert resolve_endpoint("server") is None
    assert resolve_endpoint("hosted") is None


def test_bare_url_is_accepted():
    endpoint = resolve_endpoint("http://elsewhere.test:8000/v1")
    assert endpoint.base_url == "http://elsewhere.test:8000/v1"
    assert endpoint.model is None


def test_unknown_backend_lists_choices():
    with pytest.raises(LlmError, match="unknown backend"):
        resolve_endpoint("gpt9")


def test_model_and_thinking_can_be_overridden():
    endpoint = resolve_endpoint("qwen", model="Other/Model", thinking=True)
    assert endpoint.model == "Other/Model" and endpoint.thinking is True


def test_thinking_flag_is_sent_to_the_server():
    captured = {}

    def handler(request):
        captured.update(request.read() and __import__("json").loads(request.read()))
        return chat_response("OK")

    make_client(handler).complete("hi")
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["temperature"] == 0.0


def test_no_thinking_kwarg_when_unset():
    """Llama has no thinking mode; sending the kwarg would be meaningless."""
    captured = {}

    def handler(request):
        import json as _json
        captured.update(_json.loads(request.content))
        return chat_response("OK", model="llama")

    make_client(handler, PRESETS["llama"]).complete("hi")
    assert "chat_template_kwargs" not in captured


def test_reasoning_is_separated_from_content():
    """Qwen returns reasoning in its own field, so it never pollutes the table."""
    completion = make_client(lambda r: chat_response("A\tB", reasoning="thinking...")).complete("p")
    assert completion.text == "A\tB"
    assert completion.reasoning == "thinking..."


def test_length_finish_is_reported_as_truncated():
    """A table cut off mid-row is worse than a short one -- callers must know."""
    completion = make_client(lambda r: chat_response("partial", finish="length")).complete("p")
    assert completion.truncated is True


def test_model_is_discovered_when_unset():
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "Discovered/Model"}]})
        return chat_response("ok", model="Discovered/Model")

    endpoint = LlmEndpoint(name="custom", base_url="http://x.test/v1")
    assert make_client(handler, endpoint).resolve_model() == "Discovered/Model"


def test_http_error_is_wrapped():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(LlmError, match="HTTP 500"):
        make_client(handler).complete("p")


def test_empty_choices_is_an_error():
    def handler(request):
        return httpx.Response(200, json={"choices": [], "model": "m"})

    with pytest.raises(LlmError, match="no completion"):
        make_client(handler).complete("p")
