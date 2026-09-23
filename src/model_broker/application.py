"""FastAPI entry point and safe OpenAPI-derived placeholder routes for model-broker.

Run with ``uvicorn --factory model_broker.application:create_app``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import cast

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.routing import BaseRoute, Route

LOGGER = logging.getLogger(__name__)
OPENAPI_PATH = "/openapi.json"
HTTP_METHODS = frozenset({"delete", "get", "head", "options", "patch", "post", "put", "trace"})
JsonObject = dict[str, object]
RouterOperation = tuple[str, str, JsonObject]


class RouterSchemaError(ValueError):
    """The router OpenAPI document could not be fetched or is unusable."""


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


@dataclass(slots=True)
class RouterSchemaStatus:
    """The startup attempt to obtain the private router schema."""

    available: bool = False
    error: str | None = None
    generated_operations: int = 0


RouterSchemaFetcher = Callable[[Settings], Awaitable[object]]


async def fetch_router_openapi(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> object:
    """Download the router OpenAPI document with a bounded timeout."""
    try:
        async with httpx.AsyncClient(
            transport=transport, timeout=settings.openapi_timeout_seconds
        ) as client:
            response = await client.get(
                f"{settings.llama_url}{OPENAPI_PATH}", headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            return cast(object, response.json())
    except (httpx.HTTPError, ValueError) as error:
        raise RouterSchemaError(f"could not fetch router OpenAPI document: {error}") from error


def router_operations(document: object) -> list[RouterOperation]:
    """Validate the whole document and return each (path, method, operation) it declares.

    Validation finishes before anything is registered, so a malformed entry cannot leave the
    broker with only some of the router's operations.
    """
    paths = cast(JsonObject, document).get("paths") if isinstance(document, dict) else None
    if not isinstance(paths, dict):
        raise RouterSchemaError("router OpenAPI document has no paths object")
    operations: list[RouterOperation] = []
    for path, path_item in cast(JsonObject, paths).items():
        if not path.startswith("/"):
            raise RouterSchemaError(f"router OpenAPI path {path!r} must start with a slash")
        if not isinstance(path_item, dict):
            raise RouterSchemaError(f"router OpenAPI path item for {path!r} must be an object")
        # Path items also hold non-operation keys such as "parameters" and "summary".
        for method, operation in cast(JsonObject, path_item).items():
            if method.lower() not in HTTP_METHODS:
                continue
            if not isinstance(operation, dict):
                raise RouterSchemaError(
                    f"router OpenAPI operation for {method.upper()} {path} must be an object"
                )
            operations.append((path, method.lower(), cast(JsonObject, operation)))
    return operations


def not_implemented(message: str, code: str) -> JSONResponse:
    """Return a 501 response in the OpenAI error shape."""
    error = {"message": message, "type": "not_implemented", "code": code}
    return JSONResponse(status_code=501, content={"error": error})


def placeholder_endpoint(path: str, method: str) -> Callable[[], Awaitable[JSONResponse]]:
    """Build an endpoint that advertises an upstream operation without proxying it."""
    message = (
        f"{method.upper()} {path} is declared by the upstream router "
        "but is not implemented by model-broker."
    )

    async def endpoint() -> JSONResponse:
        return not_implemented(message, "not_implemented")

    return endpoint


def optional_text(operation: JsonObject, key: str) -> str | None:
    value = operation.get(key)
    return value if isinstance(value, str) else None


def served_operations(routes: Iterable[BaseRoute]) -> set[tuple[str, str]]:
    return {
        (route.path, method.lower())
        for route in routes
        if isinstance(route, Route)
        for method in route.methods or ()
    }


def register_placeholders(app: FastAPI, operations: list[RouterOperation]) -> int:
    """Register a 501 route for each operation the broker does not already own."""
    # FastAPI keeps an included router as one opaque entry in app.routes, so the broker's
    # own routes are read from the router itself; app.routes adds /openapi.json and /docs.
    owned = served_operations(app.routes) | served_operations(router.routes)
    new = [(path, method, op) for path, method, op in operations if (path, method) not in owned]
    for path, method, operation in new:
        app.add_api_route(
            path,
            placeholder_endpoint(path, method),
            methods=[method.upper()],
            status_code=501,
            operation_id=optional_text(operation, "operationId"),
            summary=optional_text(operation, "summary"),
            description=optional_text(operation, "description"),
            responses={501: {"description": "The broker does not implement this operation yet."}},
        )
    app.openapi_schema = None
    return len(new)


router = APIRouter(tags=["broker"])


@router.get("/health")
async def health(request: Request) -> JsonObject:
    """Report broker liveness and whether the router API surface is current."""
    status = cast(RouterSchemaStatus, request.app.state.router_schema_status)
    return {"status": "ok", "router_openapi": asdict(status)}


@router.post(
    "/v1/chat/completions",
    status_code=501,
    summary="Broker-owned chat-completions override",
    responses={501: {"description": "Scheduling and proxying are not implemented yet."}},
)
async def chat_completions() -> JSONResponse:
    """Reserve the upstream chat endpoint for future validation, scheduling, and proxying."""
    return not_implemented(
        "Chat completions are broker-owned and require scheduling support that is not "
        "implemented yet.",
        "broker_scheduling_not_implemented",
    )


def create_app(
    settings: Settings | None = None,
    fetch_schema: RouterSchemaFetcher = fetch_router_openapi,
) -> FastAPI:
    """Create the broker API with explicit endpoints and startup-derived placeholders."""
    configured = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        status = cast(RouterSchemaStatus, app.state.router_schema_status)
        try:
            operations = router_operations(await fetch_schema(configured))
        except RouterSchemaError as error:
            status.error = str(error)
            LOGGER.warning("router OpenAPI document is unavailable: %s", error)
        else:
            status.generated_operations = register_placeholders(app, operations)
            status.available = True
            LOGGER.info("registered %s router placeholder operations", status.generated_operations)
        yield

    app = FastAPI(
        title="Model broker",
        version="0.1.0",
        description="Broker-owned API surface for scheduled local model inference.",
        lifespan=lifespan,
    )
    app.state.router_schema_status = RouterSchemaStatus()
    app.include_router(router)
    return app
