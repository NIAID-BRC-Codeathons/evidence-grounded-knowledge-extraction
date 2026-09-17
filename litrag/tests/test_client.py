"""RagStackClient's retry and error shaping, tested directly.

This path had no tests of its own. It was exercised incidentally through
test_batch and test_pipeline, which only ever fed it a 200 -- so the retry
loop, the status allowlist, the non-retryable branch and the request-id
propagation had never been run under test at all.

`time.sleep` is patched out throughout. The real backoff is 1.5**attempt,
about seven seconds across four attempts, which would dominate the suite.
Patching it keeps the attempt COUNT honest while removing only the waiting.
"""

import httpx
import pytest

from litrag.client import (BACKOFF_BASE, MAX_ATTEMPTS, RETRY_STATUS, ApiError,
                           RagStackClient)
from litrag.config import Config

BASE = "https://example.invalid"


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    slept = []
    monkeypatch.setattr("litrag.client.time.sleep", lambda s: slept.append(s))
    return slept


def client_for(handler):
    return RagStackClient(
        Config(api_key="k", base_url=BASE),
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE),
    )


def responder(statuses, payload=None, headers=None):
    """Answer with each status in turn, then 200 for anything after."""
    calls = []

    def handler(request):
        calls.append(request)
        index = len(calls) - 1
        if index < len(statuses):
            return httpx.Response(statuses[index], text="upstream said no",
                                  headers=headers or {})
        return httpx.Response(200, json=payload or {"ok": True},
                              headers=headers or {})

    return handler, calls


# --- retry behaviour ----------------------------------------------------------

def test_transient_status_is_retried_then_succeeds():
    handler, calls = responder([503])
    result = client_for(handler).health()
    assert result == {"ok": True}
    assert len(calls) == 2, "one failure then one success"


def test_retries_are_exhausted_and_then_raise():
    handler, calls = responder([503] * (MAX_ATTEMPTS + 2))
    with pytest.raises(ApiError) as excinfo:
        client_for(handler).health()
    assert len(calls) == MAX_ATTEMPTS, "must stop at the attempt cap, not loop forever"
    assert excinfo.value.status == 503


def test_backoff_grows_between_attempts(no_sleeping):
    handler, _ = responder([503] * (MAX_ATTEMPTS + 2))
    with pytest.raises(ApiError):
        client_for(handler).health()
    assert no_sleeping == [BACKOFF_BASE ** n for n in range(1, MAX_ATTEMPTS)]
    assert no_sleeping == sorted(no_sleeping), "backoff must not shrink"


@pytest.mark.parametrize("status", sorted(RETRY_STATUS))
def test_every_allowlisted_status_retries(status):
    handler, calls = responder([status])
    client_for(handler).health()
    assert len(calls) == 2


def test_non_retryable_status_fails_immediately():
    """A 400 is the caller's fault; retrying cannot change the answer."""
    handler, calls = responder([400])
    with pytest.raises(ApiError) as excinfo:
        client_for(handler).health()
    assert len(calls) == 1, "must not retry a client error"
    assert excinfo.value.status == 400


def test_a_timeout_is_retried():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.TimeoutException("too slow", request=request)
        return httpx.Response(200, json={"ok": True})

    assert client_for(handler).health() == {"ok": True}
    assert len(calls) == 2


def test_transport_error_exhausts_and_raises():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError):
        client_for(handler).health()
    assert len(calls) == MAX_ATTEMPTS


# --- error shaping ------------------------------------------------------------

def test_request_id_is_propagated_into_the_error():
    """Support cannot trace a failure without the server's request id."""
    handler, _ = responder([400], headers={"x-request-id": "req-abc123"})
    with pytest.raises(ApiError) as excinfo:
        client_for(handler).health()
    assert excinfo.value.request_id == "req-abc123"
    assert "req-abc123" in str(excinfo.value)


def test_server_detail_is_surfaced_in_the_message():
    def handler(request):
        return httpx.Response(422, json={"detail": "collection 'nope' is unknown"})

    with pytest.raises(ApiError) as excinfo:
        client_for(handler).health()
    assert "collection 'nope' is unknown" in str(excinfo.value)


def test_detail_is_not_duplicated_when_it_repeats_the_message():
    def handler(request):
        return httpx.Response(400, json={"detail": "bad"})

    with pytest.raises(ApiError) as excinfo:
        client_for(handler).health()
    assert str(excinfo.value).count("bad") == 1


def test_success_after_retry_still_returns_the_request_id():
    handler, _ = responder([503], headers={"x-request-id": "req-xyz"})
    # health() returns only the payload, so reach through _request for the id.
    payload, request_id, elapsed = client_for(handler)._request("GET", "/v1/health")
    assert payload == {"ok": True}
    assert request_id == "req-xyz"
    assert elapsed >= 0
