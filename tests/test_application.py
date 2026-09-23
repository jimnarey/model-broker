from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from model_broker.application import (
    RouterSchemaError,
    Settings,
    create_app,
    fetch_router_openapi,
    router_operations,
)

SETTINGS = Settings("http://router.test")
ROUTER_DOCUMENT: dict[str, Any] = {
    "openapi": "3.1.0",
    "paths": {
        "/v1/chat/completions": {
            "post": {"operationId": "createChatCompletion", "summary": "Upstream chat"}
        },
        "/v1/embeddings": {
            "post": {"operationId": "createEmbedding", "summary": "Create embeddings"}
        },
        "/models/{model}": {
            "parameters": [{"name": "model", "in": "path"}],
            "get": {"operationId": "getModel"},
        },
    },
}


async def with_broker_client[T](
    document: object, action: Callable[[httpx.AsyncClient], Awaitable[T]]
) -> T:
    """Start a broker against document, run action through ASGI, then close it cleanly."""

    async def fetch(_: Settings) -> object:
        """Return the supplied router document or raise its configured failure."""
        if isinstance(document, Exception):
            raise document
        return document

    app = create_app(SETTINGS, fetch)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://broker.test") as client:
            return await action(client)


def broker_request[T](document: object, action: Callable[[httpx.AsyncClient], Awaitable[T]]) -> T:
    """Run an ASGI broker request from a synchronous pytest test."""
    return asyncio.run(with_broker_client(document, action))


# Startup with a reachable router


def test_undeclared_router_operations_become_placeholders() -> None:
    """Router operations the broker does not own appear in the broker's OpenAPI document.

    Setup: the router declares chat completions, embeddings, and GET /models/{model}.
    Expect: embeddings and the model lookup keep their operationIds, and health reports two
    generated operations; chat completions is not counted because the broker owns it.
    """

    async def request(client: httpx.AsyncClient) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch the generated schema and health report from the started broker."""
        return (await client.get("/openapi.json")).json(), (await client.get("/health")).json()

    schema, health = broker_request(ROUTER_DOCUMENT, request)
    paths = schema["paths"]

    assert paths["/v1/embeddings"]["post"]["operationId"] == "createEmbedding"
    assert paths["/models/{model}"]["get"]["operationId"] == "getModel"
    assert health["router_openapi"] == {
        "available": True,
        "error": None,
        "generated_operations": 2,
    }


def test_placeholder_returns_openai_style_501() -> None:
    """Calling a placeholder returns 501 in the OpenAI error shape instead of proxying.

    The path parameter case checks that placeholders accept templated paths as well.
    """

    async def request(client: httpx.AsyncClient) -> tuple[httpx.Response, httpx.Response]:
        """Call one generated POST route and one generated path-parameter route."""
        return await client.post("/v1/embeddings", json={}), await client.get("/models/any")

    for response in broker_request(ROUTER_DOCUMENT, request):
        assert response.status_code == 501
        assert response.json()["error"]["type"] == "not_implemented"
        assert response.json()["error"]["code"] == "not_implemented"


def test_router_cannot_replace_the_broker_chat_endpoint() -> None:
    """Chat completions keeps the broker's handler even though the router also declares it.

    Expect the broker-specific error code. If a placeholder had replaced it, a later proxying
    placeholder would let clients bypass scheduling through this path.
    """

    async def request(client: httpx.AsyncClient) -> httpx.Response:
        """Call the broker-owned chat endpoint after generated routes are registered."""
        return await client.post("/v1/chat/completions", json={"model": "example"})

    response = broker_request(ROUTER_DOCUMENT, request)
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "broker_scheduling_not_implemented"


# Startup without a usable router document


def test_broker_starts_when_router_is_unreachable() -> None:
    """Health stays available and explains the failure when the router cannot be reached.

    Expect: /health returns 200 with the fetch error, and no placeholder exists, since the
    broker has not learned any router operations.
    """

    async def request(client: httpx.AsyncClient) -> tuple[httpx.Response, dict[str, Any]]:
        """Fetch broker health and its generated schema after a failed startup fetch."""
        return await client.get("/health"), (await client.get("/openapi.json")).json()

    health, schema = broker_request(RouterSchemaError("connection refused"), request)
    paths = schema["paths"]

    assert health.status_code == 200
    assert health.json()["router_openapi"] == {
        "available": False,
        "error": "connection refused",
        "generated_operations": 0,
    }
    assert "/v1/embeddings" not in paths


def test_malformed_document_registers_nothing() -> None:
    """One malformed path item means no placeholders at all, not the ones before it.

    Setup: a valid embeddings path followed by a path item that is a string.
    Expect: embeddings is absent and health reports 0 operations, so health always matches
    the routes that actually exist.
    """
    document = {"paths": {"/v1/embeddings": {"post": {}}, "/broken": "not an object"}}

    async def request(client: httpx.AsyncClient) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read health and schema after the invalid document is rejected at startup."""
        return (await client.get("/health")).json(), (await client.get("/openapi.json")).json()

    health, schema = broker_request(document, request)
    paths = schema["paths"]

    assert health["router_openapi"]["available"] is False
    assert "/broken" in health["router_openapi"]["error"]
    assert health["router_openapi"]["generated_operations"] == 0
    assert "/v1/embeddings" not in paths


# Document validation


@pytest.mark.parametrize(
    ("document", "error"),
    [
        ([], "no paths object"),
        ({"paths": []}, "no paths object"),
        ({"paths": {"v1/models": {}}}, "must start with a slash"),
        ({"paths": {"/models": []}}, "must be an object"),
        ({"paths": {"/models": {"get": "list"}}}, "GET /models must be an object"),
    ],
    ids=["not-object", "paths-not-object", "relative-path", "path-item", "operation"],
)
def test_router_operations_rejects_malformed_documents(document: object, error: str) -> None:
    """Each structural problem is reported as a RouterSchemaError naming the bad entry."""
    with pytest.raises(RouterSchemaError, match=error):
        router_operations(document)


def test_router_operations_skips_non_operation_keys() -> None:
    """Path-item keys such as ``parameters`` are not HTTP methods and are ignored.

    Methods are lower-cased so that ``GET`` and ``get`` refer to the same operation.
    """
    document = {"paths": {"/models": {"parameters": [], "summary": "x", "GET": {"a": 1}}}}
    assert router_operations(document) == [("/models", "get", {"a": 1})]


# Fetching the router document


def fetch_with(handler: Any) -> object:
    """Fetch one mock router response through the production HTTP client."""
    return asyncio.run(fetch_router_openapi(SETTINGS, httpx.MockTransport(handler)))


def test_fetch_requests_the_router_openapi_path() -> None:
    """The document is fetched from <llama_url>/openapi.json and returned decoded."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the URL and return the test router document."""
        requested.append(str(request.url))
        return httpx.Response(200, json=ROUTER_DOCUMENT)

    assert fetch_with(handler) == ROUTER_DOCUMENT
    assert requested == ["http://router.test/openapi.json"]


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, json={"paths": {}}), httpx.Response(200, text="<html>")],
    ids=["http-error", "not-json"],
)
def test_fetch_failures_raise_router_schema_error(response: httpx.Response) -> None:
    """An error status or a non-JSON body is a RouterSchemaError, which startup tolerates.

    A 500 with a JSON body must not be treated as the document.
    """
    with pytest.raises(RouterSchemaError, match="could not fetch"):
        fetch_with(lambda _: response)


# Settings


def test_settings_defaults_and_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset timeout defaults to 2 seconds; a trailing slash on the URL is removed."""
    monkeypatch.setenv("MODEL_BROKER_LLAMA_URL", "http://llama:8080/")
    monkeypatch.delenv("MODEL_BROKER_OPENAPI_TIMEOUT_SECONDS", raising=False)

    assert Settings.from_environment() == Settings("http://llama:8080", 2.0)


@pytest.mark.parametrize(
    ("timeout", "error"), [("soon", "must be a number"), ("0", "must be positive")]
)
def test_settings_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch, timeout: str, error: str
) -> None:
    """A timeout that is not a positive number stops startup with a clear message."""
    monkeypatch.setenv("MODEL_BROKER_OPENAPI_TIMEOUT_SECONDS", timeout)
    with pytest.raises(ValueError, match=error):
        Settings.from_environment()
