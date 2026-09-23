from __future__ import annotations

import asyncio
from typing import cast

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from src.model_broker.application import Settings, create_app

JsonObject = dict[str, object]
ROUTER_DOCUMENT: JsonObject = {
    "openapi": "3.1.0",
    "paths": {
        "/v1/chat/completions": {
            "post": {"operationId": "createChatCompletion", "summary": "Upstream chat"}
        },
        "/v1/embeddings": {
            "post": {"operationId": "createEmbedding", "summary": "Create embeddings"}
        },
        "/models": {"get": {"operationId": "listModels"}},
    },
}


async def request_openapi_and_endpoints(
    app_url: str, app: FastAPI
) -> tuple[JsonObject, Response, Response, Response]:
    """Start the ASGI lifespan and call the application without a network server."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url=app_url) as client:
            return (
                cast(JsonObject, (await client.get("/openapi.json")).json()),
                await client.post("/v1/embeddings", json={"input": "hello"}),
                await client.post("/v1/chat/completions", json={"model": "example"}),
                await client.get("/health"),
            )


async def request_health_and_openapi(app_url: str, app: FastAPI) -> tuple[Response, JsonObject]:
    """Start the ASGI lifespan and request the two explicit diagnostic endpoints."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url=app_url) as client:
            return (
                await client.get("/health"),
                cast(JsonObject, (await client.get("/openapi.json")).json()),
            )


def test_openapi_operations_are_registered_with_a_broker_override() -> None:
    """Expose router operations while ensuring the chat endpoint uses broker-owned behavior.

    The upstream document contains chat completions and two operations which the
    broker does not support yet. The startup step should add placeholders for the
    unsupported operations, preserving their paths and methods in the broker
    OpenAPI document. It must not replace the explicit chat-completions handler.

    This is important because clients can discover the visible API without being
    allowed to bypass future scheduling through a direct upstream-style call.
    """

    def schema_fetcher(_: Settings) -> JsonObject:
        """Provide an upstream document without making a network request."""
        return ROUTER_DOCUMENT

    app = create_app(Settings("http://router.test"), schema_fetcher)
    openapi, generated, overridden, health = asyncio.run(
        request_openapi_and_endpoints("http://broker.test", app)
    )

    paths = cast(JsonObject, openapi["paths"])
    embeddings = cast(JsonObject, paths["/v1/embeddings"])
    embeddings_post = cast(JsonObject, embeddings["post"])
    generated_body = cast(JsonObject, generated.json())
    generated_error = cast(JsonObject, generated_body["error"])
    overridden_body = cast(JsonObject, overridden.json())
    overridden_error = cast(JsonObject, overridden_body["error"])
    health_body = cast(JsonObject, health.json())

    assert "/v1/embeddings" in paths
    assert embeddings_post["operationId"] == "createEmbedding"
    assert "/models" in paths
    assert generated.status_code == 501
    assert generated_error["code"] == "not_implemented"
    assert overridden.status_code == 501
    assert overridden_error["code"] == "broker_scheduling_not_implemented"
    assert health_body["router_openapi"] == {
        "available": True,
        "error": None,
        "generated_operations": 2,
    }


def test_explicit_endpoints_start_when_router_openapi_cannot_be_fetched() -> None:
    """Keep health checks available when the private router is down at startup.

    The design requires the broker's explicit routes to start even if the router
    schema cannot be fetched. This test makes the fetcher fail and checks that
    health explains the missing schema rather than failing the application.

    No generated route is registered in this case, so clients cannot be given a
    misleading route that the broker has not actually learned from the router.
    """

    def unavailable(_: Settings) -> JsonObject:
        """Model a router that is temporarily unreachable during startup."""
        raise RuntimeError("connection refused")

    app = create_app(Settings("http://router.test"), unavailable)
    health, openapi = asyncio.run(request_health_and_openapi("http://broker.test", app))

    health_body = cast(JsonObject, health.json())
    paths = cast(JsonObject, openapi["paths"])
    assert health.status_code == 200
    assert health_body["router_openapi"] == {
        "available": False,
        "error": "connection refused",
        "generated_operations": 0,
    }
    assert "/v1/embeddings" not in paths
