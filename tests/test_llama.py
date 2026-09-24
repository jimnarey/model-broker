from __future__ import annotations

import asyncio
import itertools
import json
import math
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import httpx
import pytest

from model_broker.llama import LlamaAdapter, LlamaError, ModelState

MODEL = "qwen2.5-coder-7b-instruct-q4_k_m--cuda0"
LOADING = {"value": "loading"}
LOADED = {"value": "loaded"}
UNLOADED = {"value": "unloaded"}
# What the pinned router (llama-cpp:upstream-eafe15a5) reports after a worker fails to start.
FAILED = {"value": "unloaded", "exit_code": 1, "failed": True}
SUCCESS = httpx.Response(200, json={"success": True})
NOT_RUNNING = httpx.Response(
    400,
    json={
        "error": {"code": 400, "message": "model is not running", "type": "invalid_request_error"}
    },
)

Handler = Callable[[httpx.Request], httpx.Response]
Requests = list[tuple[str, str, Any]]


def model_entry(model_id: str, status: dict[str, Any]) -> dict[str, Any]:
    """One entry in the shape the pinned router returns from GET /models."""
    return {
        "id": model_id,
        "aliases": [],
        "tags": [],
        "object": "model",
        "owned_by": "llamacpp",
        "created": 1790255370,
        "status": {**status, "args": ["/app/llama-server"], "preset": f"[{model_id}]\n"},
    }


def router(
    statuses: Iterable[dict[str, Any]], control: httpx.Response = SUCCESS
) -> tuple[Handler, Requests]:
    """A mock router: each GET /models reports MODEL with the next status; POSTs get control.

    Returns the handler and the list of (method, path, JSON body) it received.
    """
    requests: Requests = []
    remaining = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.method == "GET":
            document = {"object": "list", "data": [model_entry(MODEL, next(remaining))]}
            return httpx.Response(200, json=document)
        return control

    return handler, requests


def run[T](
    handler: Handler,
    action: Callable[[LlamaAdapter], Awaitable[T]],
    state_timeout_seconds: float = 60.0,
) -> T:
    """Run ``action`` against an adapter whose client sends everything to ``handler``."""

    async def main() -> T:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://router.test"
        ) as client:
            adapter = LlamaAdapter(client, state_timeout_seconds, poll_interval_seconds=0)
            return await action(adapter)

    return asyncio.run(main())


# Model states


def test_model_states_reads_value_and_failure_from_the_real_shape() -> None:
    """Each entry becomes a ModelState; extra router fields (args, preset...) are ignored.

    ``b`` has the failed-load status: its value is still ``unloaded``, so only the ``failed``
    flag and exit code distinguish it from a model that was never loaded.
    """
    document = {"data": [model_entry("a", LOADED), model_entry("b", FAILED)]}

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=document)

    assert run(handler, LlamaAdapter.model_states) == {
        "a": ModelState("loaded"),
        "b": ModelState("unloaded", failed=True, exit_code=1),
    }


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(500), "GET /models failed"),
        (httpx.Response(200, text="<html>"), "GET /models failed"),
        (httpx.Response(200, json=[]), "no data list"),
        (httpx.Response(200, json={"data": {}}), "no data list"),
        (httpx.Response(200, json={"data": ["a"]}), "invalid model entry"),
        (httpx.Response(200, json={"data": [{"id": "a", "status": {}}]}), "invalid model entry"),
        (
            httpx.Response(200, json={"data": [model_entry("a", LOADED)] * 2}),
            "repeats model 'a'",
        ),
    ],
    ids=[
        "http-error",
        "not-json",
        "not-object",
        "data-not-list",
        "entry-not-object",
        "no-status-value",
        "duplicate-id",
    ],
)
def test_model_states_rejects_unusable_responses(response: httpx.Response, error: str) -> None:
    """A broken /models response is an error, never an empty or partial list of models.

    Reading it as "nothing loaded" would let the scheduler load over a running worker.
    """
    with pytest.raises(LlamaError, match=error):
        run(lambda _: response, LlamaAdapter.model_states)


# Load and unload


def test_load_posts_the_model_then_polls_until_loaded() -> None:
    """Load sends one explicit control request, then polls GET /models until ``loaded``.

    Setup: the router reports loading, then loaded. Expect exactly one POST with the model ID
    followed by two polls; the adapter never relies on autoload.
    """
    handler, requests = router([LOADING, LOADED])

    run(handler, lambda llama: llama.load(MODEL))

    assert requests == [
        ("POST", "/models/load", {"model": MODEL}),
        ("GET", "/models", None),
        ("GET", "/models", None),
    ]


def test_load_fails_as_soon_as_the_router_reports_a_failed_load() -> None:
    """A failed load is reported on the poll that shows it, not after the state timeout.

    Setup: loading, then the failed-load status, with a 60 s state timeout. Expect a
    LlamaError naming the exit code after the second poll. Waiting for ``loaded`` alone would
    keep polling for 60 s and then report only "last state was 'unloaded'".
    """
    handler, requests = router([LOADING, FAILED])

    with pytest.raises(LlamaError, match=r"failed to load \(exit code 1\)"):
        run(handler, lambda llama: llama.load(MODEL), state_timeout_seconds=60)
    assert len(requests) == 3


def test_load_reports_a_refused_request() -> None:
    """The router's 404 for an unknown model is an error, with no polling afterwards."""
    handler, requests = router([], control=httpx.Response(404, text="File Not Found"))

    with pytest.raises(LlamaError, match=r"refused to load .*: 404 File Not Found"):
        run(handler, lambda llama: llama.load("no-such-model"))
    assert len(requests) == 1


def test_unload_posts_the_model_then_polls_until_unloaded() -> None:
    handler, requests = router([LOADED, UNLOADED])

    run(handler, lambda llama: llama.unload(MODEL))

    assert requests == [
        ("POST", "/models/unload", {"model": MODEL}),
        ("GET", "/models", None),
        ("GET", "/models", None),
    ]


def test_unload_of_a_model_that_is_not_running_succeeds() -> None:
    """Unloading an already-unloaded model is success, since the model is where it should be.

    Setup: the router answers 400 "model is not running" (as the pinned router does) and
    lists the model as unloaded. This happens when reconciling after a worker crash.
    """
    handler, requests = router([UNLOADED], control=NOT_RUNNING)

    run(handler, lambda llama: llama.unload(MODEL))

    assert [path for _, path, _ in requests] == ["/models/unload", "/models"]


def test_unload_refusal_for_a_loaded_model_is_an_error() -> None:
    """If the router refuses to unload a model it still reports as loaded, that is a failure."""
    handler, _ = router([LOADED], control=NOT_RUNNING)

    with pytest.raises(LlamaError, match="refused to unload"):
        run(handler, lambda llama: llama.unload(MODEL))


def test_wait_times_out_with_the_last_state() -> None:
    """A model stuck in ``loading`` past the state timeout is an error naming that state."""
    handler, _ = router(itertools.repeat(LOADING))

    with pytest.raises(LlamaError, match=r"did not become 'loaded'.*last state was 'loading'"):
        run(
            handler,
            lambda llama: llama.wait_for_state(MODEL, "loaded"),
            state_timeout_seconds=0.001,
        )


def test_wait_for_a_model_the_router_does_not_list() -> None:
    """A model missing from the router's list means the router has a different catalogue."""
    handler, _ = router([LOADED])

    with pytest.raises(LlamaError, match="does not list model 'other--cuda1'"):
        run(handler, lambda llama: llama.wait_for_state("other--cuda1", "loaded"))


# Chat completions


def chat(handler: Handler, payload: dict[str, Any]) -> tuple[int, bytes]:
    """Send a chat request through the adapter; return the status and full streamed body."""

    async def action(llama: LlamaAdapter) -> tuple[int, bytes]:
        async with llama.chat_completions(payload) as response:
            return response.status_code, b"".join([part async for part in response.aiter_bytes()])

    return run(handler, action)


def test_chat_forwards_the_payload_and_streams_the_response() -> None:
    """The payload is posted unchanged to /v1/chat/completions and the body can be streamed.

    The request must also carry the inference timeout, which has no read limit, rather than
    the client's shorter control timeout: prompt processing on a long context can take minutes
    before the first byte.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"data: example\n\n")

    assert chat(handler, {"model": MODEL, "stream": True}) == (200, b"data: example\n\n")
    request = seen[0]
    assert (request.method, request.url.path) == ("POST", "/v1/chat/completions")
    assert json.loads(request.content) == {"model": MODEL, "stream": True}
    assert request.headers["accept"] == "application/json, text/event-stream"
    assert request.extensions["timeout"]["read"] is None


def test_chat_passes_router_errors_back_unchanged() -> None:
    """A router error response is returned to the caller, not raised.

    Setup: the router answers 400 as it does for an unloaded model. Expect the same status
    and body, so the broker can relay the router's OpenAI-style error to its client.
    """
    body = b'{"error":{"code":400,"message":"model is not loaded"}}'

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=body)

    assert chat(handler, {"model": MODEL}) == (400, body)


def test_chat_connection_failure_is_a_llama_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LlamaError, match="POST /v1/chat/completions failed"):
        chat(handler, {"model": MODEL})


# Configuration


@pytest.mark.parametrize(
    ("state_timeout", "poll_interval", "error"),
    [
        (0.0, 0.1, "state_timeout_seconds must be positive"),
        (math.nan, 0.1, "state_timeout_seconds must be positive"),
        (60.0, -1.0, "poll_interval_seconds must not be negative"),
    ],
    ids=["zero-state-timeout", "nan-state-timeout", "negative-poll"],
)
def test_adapter_rejects_invalid_timings(
    state_timeout: float, poll_interval: float, error: str
) -> None:
    """Unusable timings fail when the adapter is created, not during a load."""
    with pytest.raises(ValueError, match=error):
        LlamaAdapter(httpx.AsyncClient(), state_timeout, poll_interval)
