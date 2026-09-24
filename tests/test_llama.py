from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from model_broker.llama import LlamaAdapter, LlamaError


def adapter(handler: httpx.AsyncBaseTransport) -> LlamaAdapter:
    """Build an adapter that sends every router request through one mock transport."""
    return LlamaAdapter("http://router.test/", transport=handler, poll_interval_seconds=0)


def models(*states: tuple[str, str]) -> dict[str, object]:
    """Build the documented router model-list shape from model ID and status pairs."""
    return {"data": [{"id": model_id, "status": {"value": state}} for model_id, state in states]}


def test_model_states_returns_router_ids_and_status_values() -> None:
    """The adapter reads model state from the router without assuming a local catalogue.

    The router is free to report any configured model ID. The broker will later compare this
    observation with its separately parsed preset, but this HTTP boundary must first preserve the
    router's own IDs and status values exactly.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        """Serve a model-list response for the adapter's read-only observation request."""
        assert request.method == "GET"
        assert request.url.path == "/models"
        return httpx.Response(
            200, json=models(("flash--cuda1", "loaded"), ("cpu--cpu", "unloaded"))
        )

    assert asyncio.run(adapter(httpx.MockTransport(handler)).model_states()) == {
        "flash--cuda1": "loaded",
        "cpu--cpu": "unloaded",
    }


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"data": {}},
        {"data": ["not an object"]},
        {"data": [{"id": "flash--cuda1", "status": {}}]},
        {"data": [{"id": "flash--cuda1", "status": {"value": "loaded"}}] * 2},
    ],
)
def test_model_states_rejects_malformed_router_documents(document: object) -> None:
    """A malformed state document is an adapter failure, not an empty model list.

    Treating a broken response as no loaded models would let a future scheduler make unsafe
    replacement decisions. The adapter therefore rejects missing fields, wrong container types,
    and duplicate model IDs before returning any state.
    """

    async def handler(_: httpx.Request) -> httpx.Response:
        """Return the malformed response selected by this parameterised test case."""
        return httpx.Response(200, json=document)

    with pytest.raises(LlamaError, match="router GET /models response"):
        asyncio.run(adapter(httpx.MockTransport(handler)).model_states())


def test_load_and_unload_use_explicit_control_then_observed_state() -> None:
    """Loading and unloading use the router control API and wait for its observed result.

    The adapter does not send inference to an unloaded model and hope that router autoload fixes
    it. Each action posts exactly one model ID to its control endpoint, then observes GET
    /models until the state is safe for the scheduler to act on.
    """
    requests: list[tuple[str, str, object | None]] = []
    states = iter(["loading", "loaded", "unloading", "unloaded"])

    async def handler(request: httpx.Request) -> httpx.Response:
        """Record each control call and progress the model state on each observation poll."""
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.method == "GET":
            return httpx.Response(200, json=models(("flash--cuda1", next(states))))
        return httpx.Response(200)

    async def exercise() -> None:
        """Run one explicit load followed by one explicit unload through the adapter."""
        router = adapter(httpx.MockTransport(handler))
        await router.load("flash--cuda1")
        await router.unload("flash--cuda1")

    asyncio.run(exercise())

    assert requests == [
        ("POST", "/models/load", {"model": "flash--cuda1"}),
        ("GET", "/models", None),
        ("GET", "/models", None),
        ("POST", "/models/unload", {"model": "flash--cuda1"}),
        ("GET", "/models", None),
        ("GET", "/models", None),
    ]


def test_wait_for_state_reports_timeout_and_missing_model() -> None:
    """A worker that never reaches the requested state cannot silently remain reserved.

    The very short timeout keeps this test immediate. It checks a known model stuck loading and
    a model absent from the router response, because the future scheduler must distinguish a slow
    transition from an incompatible or changed router catalogue.
    """

    async def loading(_: httpx.Request) -> httpx.Response:
        """Keep the expected model in loading state for the timeout branch."""
        return httpx.Response(200, json=models(("flash--cuda1", "loading")))

    async def absent(_: httpx.Request) -> httpx.Response:
        """Return a valid but unrelated router model list for the missing-model branch."""
        return httpx.Response(200, json=models(("other--cuda0", "loaded")))

    timed_out = LlamaAdapter(
        "http://router.test",
        transport=httpx.MockTransport(loading),
        state_timeout_seconds=0.001,
        poll_interval_seconds=0,
    )
    missing = LlamaAdapter(
        "http://router.test", transport=httpx.MockTransport(absent), state_timeout_seconds=1
    )

    with pytest.raises(LlamaError, match="did not become"):
        asyncio.run(timed_out.wait_for_state("flash--cuda1", "loaded"))
    with pytest.raises(LlamaError, match="does not list"):
        asyncio.run(missing.wait_for_state("flash--cuda1", "loaded"))


def test_chat_completions_forwards_only_the_admitted_payload_and_keeps_stream_open() -> None:
    """The adapter forwards one admitted chat request through the fixed private router URL.

    The future scheduler will decide whether this request may run before calling the adapter.
    This test checks the adapter's narrower responsibility: preserve the JSON body, use the
    OpenAI chat endpoint, and keep the response client open while the caller reads streaming
    bytes. It does not forward caller-supplied router URLs or credentials.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        """Verify the proxied request and return a small SSE-shaped response body."""
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content) == {"model": "flash--cuda1", "stream": True}
        assert request.headers["accept"] == "application/json, text/event-stream"
        return httpx.Response(200, content=b"data: example\n\n")

    async def request() -> bytes:
        """Read the full test stream while the adapter-managed response is still open."""
        async with adapter(httpx.MockTransport(handler)).chat_completions(
            {"model": "flash--cuda1", "stream": True}
        ) as response:
            return b"".join([part async for part in response.aiter_bytes()])

    assert b"data: example" in asyncio.run(request())


@pytest.mark.parametrize("value", ["", "http://router.test"])
def test_adapter_rejects_invalid_timeouts(value: str) -> None:
    """The adapter rejects empty URLs and non-positive request or state timeouts early.

    A bad adapter configuration should fail during broker setup rather than leave a request
    waiting forever for an impossible router operation. A zero polling interval remains allowed
    for deterministic tests and immediate rechecks.
    """
    if value:
        with pytest.raises(ValueError, match="request_timeout_seconds"):
            LlamaAdapter(value, request_timeout_seconds=0)
    else:
        with pytest.raises(ValueError, match="URL"):
            LlamaAdapter(value)
