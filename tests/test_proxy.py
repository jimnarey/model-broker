"""Tests for the public API: /v1/models and the scheduled, proxied chat endpoint.

Each test starts the real application against FakeRouter. Most send requests through
httpx's ASGI transport; the disconnect test drives the ASGI app directly, because that
transport cannot leave part-way through a response.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from model_broker.application import Settings, create_app
from model_broker.events import ClientDisconnected, Event, ProxyFailed
from model_broker.scheduler import Scheduler

from .fake_router import CATALOGUE, CHAT_REPLY, CUDA1, MOE_CUDA1, STREAM_CHUNKS, FakeRouter

PRESET = Path(__file__).parent / "fixtures" / "models-preset.ini"
SETTINGS = Settings("http://router.test", PRESET, idle_protect_seconds=0, max_wait_seconds=0)
MESSAGES = [{"role": "user", "content": "hi"}]


def scheduler_of(app: FastAPI) -> Scheduler:
    return app.state.scheduler


def serve[T](
    router: FakeRouter,
    action: Callable[[httpx.AsyncClient, FastAPI], Awaitable[T]],
    settings: Settings = SETTINGS,
) -> T:
    """Start the broker against ``router`` and run ``action`` with a client and the app."""

    async def main() -> T:
        app = create_app(settings, httpx.MockTransport(router.handle))
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://b") as client:
                return await action(client, app)

    return asyncio.run(main())


def post_chat(router: FakeRouter, body: dict[str, Any], **settings: Any) -> httpx.Response:
    async def action(client: httpx.AsyncClient, _: FastAPI) -> httpx.Response:
        return await client.post("/v1/chat/completions", json=body)

    return serve(router, action, replace(SETTINGS, **settings))


# /v1/models


def test_models_lists_the_catalogue_in_openai_shape() -> None:
    async def action(client: httpx.AsyncClient, _: FastAPI) -> Any:
        return (await client.get("/v1/models")).json()

    listing = serve(FakeRouter(), action)
    assert listing["object"] == "list"
    assert {model["id"] for model in listing["data"]} == set(CATALOGUE)
    assert all(model["object"] == "model" for model in listing["data"])


def test_one_model_or_openai_not_found() -> None:
    async def action(client: httpx.AsyncClient, _: FastAPI) -> tuple[Any, httpx.Response]:
        found = (await client.get(f"/v1/models/{CUDA1}")).json()
        return found, await client.get("/v1/models/no-such-model--cuda0")

    found, missing = serve(FakeRouter(), action)
    assert found["id"] == CUDA1
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"


# Proxying


def test_chat_loads_the_model_and_relays_the_reply() -> None:
    """An unloaded model is loaded, the request is forwarded unchanged, and the router's
    JSON reply comes back as it was sent. The admission ends with the response."""
    router = FakeRouter()
    body = {"model": CUDA1, "messages": MESSAGES, "temperature": 0.2}

    async def action(client: httpx.AsyncClient, app: FastAPI) -> httpx.Response:
        response = await client.post("/v1/chat/completions", json=body)
        assert scheduler_of(app).loaded[CUDA1].active == 0
        return response

    response = serve(router, action)
    assert (response.status_code, response.json()) == (200, CHAT_REPLY)
    assert router.calls == [("load", CUDA1)]
    assert router.chat_payloads == [body]


def test_a_stream_is_relayed_as_a_stream() -> None:
    """A streamed reply keeps its SSE content type and arrives chunk for chunk."""
    response = post_chat(FakeRouter(), {"model": CUDA1, "messages": MESSAGES, "stream": True})

    assert response.headers["content-type"] == "text/event-stream"
    assert response.content == b"".join(STREAM_CHUNKS)


def test_router_errors_reach_the_client_unchanged() -> None:
    """A router error, such as a prompt too long for the context, belongs to the client."""
    router = FakeRouter()
    error = {"error": {"code": 400, "message": "the request exceeds the available context"}}
    router.chat_error = (400, error)

    response = post_chat(router, {"model": CUDA1, "messages": MESSAGES})
    assert (response.status_code, response.json()) == (400, error)


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"model": "no-such-model--cuda0", "messages": MESSAGES}, 404, "model_not_found"),
        ({"messages": MESSAGES}, 400, "missing_model"),
        ({"model": 7, "messages": MESSAGES}, 400, "missing_model"),
    ],
    ids=["unknown-model", "no-model", "model-not-a-string"],
)
def test_requests_the_broker_cannot_route_are_refused(
    body: dict[str, Any], status: int, code: str
) -> None:
    response = post_chat(FakeRouter(), body)
    assert (response.status_code, response.json()["error"]["code"]) == (status, code)


def test_a_body_that_is_not_json_is_refused() -> None:
    async def action(client: httpx.AsyncClient, _: FastAPI) -> httpx.Response:
        return await client.post("/v1/chat/completions", content=b"{not json")

    response = serve(FakeRouter(), action)
    assert (response.status_code, response.json()["error"]["code"]) == (400, "invalid_json")


def test_busy_resources_give_503_with_retry_after() -> None:
    """A conflicting model within its protection window refuses the request with 503.

    CUDA1 is adopted at startup with a 60 s window; MOE_CUDA1 needs GPU1 too. Retry-After
    tells the client when the window ends.
    """
    response = post_chat(
        FakeRouter(loaded=(CUDA1,)),
        {"model": MOE_CUDA1, "messages": MESSAGES},
        idle_protect_seconds=60,
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "resource_busy"
    assert response.headers["retry-after"] == "60"


def test_an_unfit_router_gives_503() -> None:
    router = FakeRouter()
    router.props["models_autoload"] = True

    response = post_chat(router, {"model": CUDA1, "messages": MESSAGES})
    assert (response.status_code, response.json()["error"]["code"]) == (503, "router_unavailable")


def test_a_failed_load_gives_502() -> None:
    router = FakeRouter()
    router.fail.add(("load", CUDA1))

    response = post_chat(router, {"model": CUDA1, "messages": MESSAGES})
    assert (response.status_code, response.json()["error"]["code"]) == (502, "model_load_failed")


def test_a_broken_router_stream_still_releases_the_admission(
    events: Callable[[], list[Event]],
) -> None:
    """If the router's stream breaks part-way, the client gets what arrived, the failure is
    logged, and the model is no longer counted as busy."""
    router = FakeRouter()
    router.break_stream = True

    async def action(client: httpx.AsyncClient, app: FastAPI) -> int:
        await client.post("/v1/chat/completions", json={"model": CUDA1, "stream": True})
        return scheduler_of(app).loaded[CUDA1].active

    assert serve(router, action) == 0
    assert any(isinstance(event, ProxyFailed) for event in events())


def test_a_client_leaving_mid_stream_ends_the_request(events: Callable[[], list[Event]]) -> None:
    """When the client disconnects, the router request is closed and the admission released.

    Setup: the router's stream is held after nothing has been sent. The client then
    disconnects. Expect the broker to return at once, the router stream to be closed (so
    llama.cpp stops generating), CUDA1 to have no active request, and a ClientDisconnected
    event.
    """
    router = FakeRouter()
    router.chats_flow.clear()

    async def main() -> None:
        app = create_app(SETTINGS, httpx.MockTransport(router.handle))
        async with app.router.lifespan_context(app):
            body = json.dumps({"model": CUDA1, "stream": True}).encode()
            leave = asyncio.Event()
            messages = iter([{"type": "http.request", "body": body, "more_body": False}])

            async def receive() -> dict[str, Any]:
                message = next(messages, None)
                if message is not None:
                    return message
                await leave.wait()
                return {"type": "http.disconnect"}

            async def send(_: Any) -> None:
                pass

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/chat/completions",
                "raw_path": b"/v1/chat/completions",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1),
                "server": ("b", 80),
                "state": {},
            }
            handling = asyncio.create_task(app(scope, receive, send))
            await asyncio.sleep(0.05)
            assert router.open_chats == 1 and scheduler_of(app).loaded[CUDA1].active == 1

            leave.set()
            await asyncio.wait_for(handling, 1)
            assert router.open_chats == 0
            assert scheduler_of(app).loaded[CUDA1].active == 0

    asyncio.run(main())
    assert any(isinstance(event, ClientDisconnected) for event in events())
