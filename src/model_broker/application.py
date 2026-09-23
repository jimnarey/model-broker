"""FastAPI entry point and safe OpenAPI-derived placeholder routes for model-broker."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast
from urllib.error import URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger(__name__)
OPENAPI_PATH = "/openapi.json"
HTTP_METHODS = frozenset({"delete", "get", "head", "options", "patch", "post", "put", "trace"})
JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration needed for the first API-server phase."""

    llama_url: str
    openapi_timeout_seconds: float = 2.0

    @classmethod
    def from_environment(cls) -> Settings:
        """Read deployment configuration without requiring a settings file."""
        llama_url = os.environ.get("MODEL_BROKER_LLAMA_URL", "http://llama-cpp:8080")
        timeout_text = os.environ.get("MODEL_BROKER_OPENAPI_TIMEOUT_SECONDS", "2")
        try:
            timeout = float(timeout_text)
        except ValueError as error:
            raise ValueError("MODEL_BROKER_OPENAPI_TIMEOUT_SECONDS must be a number") from error
        if timeout <= 0:
            raise ValueError("MODEL_BROKER_OPENAPI_TIMEOUT_SECONDS must be positive")
        return cls(llama_url=llama_url.rstrip("/"), openapi_timeout_seconds=timeout)


RouterSchemaFetcher = Callable[[Settings], JsonObject]


@dataclass(slots=True)
class RouterSchemaStatus:
    """The latest startup attempt to obtain the private router schema."""

    available: bool = False
    error: str | None = None
    generated_operations: int = 0


def fetch_router_openapi(settings: Settings) -> JsonObject:
    """Download and validate the router OpenAPI document with a bounded timeout."""
    request = UrlRequest(
        f"{settings.llama_url}{OPENAPI_PATH}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=settings.openapi_timeout_seconds) as response:
            decoded: object = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, URLError) as error:
        raise RuntimeError(f"could not fetch router OpenAPI document: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError("router OpenAPI document must be a JSON object")
    document = cast(JsonObject, decoded)
    if not isinstance(document.get("paths"), dict):
        raise ValueError("router OpenAPI document has no paths object")
    return document


def openai_error(message: str, code: str) -> dict[str, object]:
    """Return the stable error shape used by broker-owned placeholder endpoints."""
    return {"error": {"message": message, "type": "not_implemented", "code": code}}


def route_operations(app: FastAPI) -> set[tuple[str, str]]:
    """Return the path and HTTP-method pairs already owned by the broker."""
    operations: set[tuple[str, str]] = set()
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if isinstance(path, str) and methods is not None:
            operations.update((path, method.lower()) for method in methods)
    return operations


def placeholder_endpoint(path: str, method: str) -> Callable[[Request], Awaitable[JSONResponse]]:
    """Build one endpoint that advertises an upstream operation without proxying it."""

    async def endpoint(_: Request) -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content=openai_error(
                " ".join(
                    (
                        f"{method.upper()} {path} is declared by the upstream router",
                        "but is not implemented by model-broker.",
                    )
                ),
                "not_implemented",
            ),
        )

    endpoint.__name__ = (
        f"generated_{method}_{path.strip('/').replace('/', '_').replace('{', '').replace('}', '')}"
    )
    return endpoint


def register_router_placeholders(app: FastAPI, document: JsonObject) -> int:
    """Register one 501 route for each upstream operation not explicitly broker-owned."""
    raw_paths = document["paths"]
    if not isinstance(raw_paths, dict):
        raise ValueError("router OpenAPI document has no paths object")
    paths = cast(JsonObject, raw_paths)
    existing = route_operations(app)
    generated = 0
    for path, raw_path_item in paths.items():
        if not path.startswith("/"):
            raise ValueError("router OpenAPI path names must start with a slash")
        if not isinstance(raw_path_item, dict):
            raise ValueError(f"router OpenAPI path item for {path!r} must be an object")
        path_item = cast(JsonObject, raw_path_item)
        for method, raw_operation in path_item.items():
            if method.lower() not in HTTP_METHODS:
                continue
            normalised_method = method.lower()
            if not isinstance(raw_operation, dict):
                raise ValueError(
                    f"router OpenAPI operation for {method.upper()} {path} must be an object"
                )
            if (path, normalised_method) in existing:
                continue
            operation = cast(JsonObject, raw_operation)
            operation_id = operation.get("operationId")
            summary = operation.get("summary")
            description = operation.get("description")
            app.add_api_route(
                path,
                placeholder_endpoint(path, normalised_method),
                methods=[normalised_method.upper()],
                status_code=501,
                operation_id=operation_id if isinstance(operation_id, str) else None,
                summary=summary if isinstance(summary, str) else None,
                description=description if isinstance(description, str) else None,
                responses={
                    501: {
                        "description": "The broker does not implement this upstream operation yet."
                    }
                },
            )
            existing.add((path, normalised_method))
            generated += 1
    app.openapi_schema = None
    return generated


def create_app(
    settings: Settings | None = None,
    schema_fetcher: RouterSchemaFetcher = fetch_router_openapi,
) -> FastAPI:
    """Create the broker API with explicit endpoints and startup-derived placeholders."""
    configured_settings = settings or Settings.from_environment()
    status = RouterSchemaStatus()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            document = await asyncio.to_thread(schema_fetcher, configured_settings)
            status.generated_operations = register_router_placeholders(app, document)
            status.available = True
            status.error = None
            LOGGER.info(
                "registered %s OpenAPI-derived router operations", status.generated_operations
            )
        except (OSError, RuntimeError, ValueError) as error:
            status.available = False
            status.error = str(error)
            LOGGER.warning("router OpenAPI document is unavailable: %s", error)
        yield

    app = FastAPI(
        title="Model broker",
        version="0.1.0",
        description="Broker-owned API surface for scheduled local model inference.",
        lifespan=lifespan,
    )
    app.state.router_schema_status = status

    @app.get("/health", tags=["broker"])
    async def health() -> dict[str, object]:
        """Report broker liveness and whether the router API surface is current."""
        return {
            "status": "ok",
            "router_openapi": {
                "available": status.available,
                "error": status.error,
                "generated_operations": status.generated_operations,
            },
        }

    @app.post(
        "/v1/chat/completions",
        status_code=501,
        tags=["broker"],
        summary="Broker-owned chat-completions override",
        responses={501: {"description": "Scheduling and proxying are not implemented yet."}},
    )
    async def chat_completions(_: Request) -> JSONResponse:
        """Reserve the upstream chat endpoint for future validation, scheduling, and proxying."""
        return JSONResponse(
            status_code=501,
            content=openai_error(
                "Chat completions are broker-owned and require scheduling support that is not "
                "implemented yet.",
                "broker_scheduling_not_implemented",
            ),
        )

    return app


app = create_app()
