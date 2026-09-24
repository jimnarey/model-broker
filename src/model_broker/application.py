"""FastAPI entry point for model-broker.

Run with ``uvicorn --factory model_broker.application:create_app``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from model_broker.catalogue import Variant, load_catalogue
from model_broker.llama import LlamaAdapter
from model_broker.scheduler import RouterView, inspect_router

LOGGER = logging.getLogger(__name__)
# Bounds each router control and observation request; inference has its own timeout.
ROUTER_TIMEOUT_SECONDS = 10.0
JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class Settings:
    """Deployment configuration read from the environment."""

    llama_url: str
    preset_path: Path

    @classmethod
    def from_environment(cls) -> Settings:
        llama_url = os.environ.get("MODEL_BROKER_LLAMA_URL", "http://llama-cpp:8080")
        preset_path = os.environ.get("MODEL_BROKER_PRESET_PATH")
        if not preset_path:
            raise ValueError("MODEL_BROKER_PRESET_PATH must name the mounted llama.cpp preset")
        return cls(llama_url=llama_url.rstrip("/"), preset_path=Path(preset_path))


def not_implemented(message: str, code: str) -> JSONResponse:
    """Return a 501 response in the OpenAI error shape."""
    error = {"message": message, "type": "not_implemented", "code": code}
    return JSONResponse(status_code=501, content={"error": error})


router = APIRouter(tags=["broker"])


@router.get("/health")
async def health(request: Request) -> JsonObject:
    """Report the catalogue, the models the router holds, and why the broker cannot schedule."""
    catalogue = cast(dict[str, Variant], request.app.state.catalogue)
    view = cast(RouterView, request.app.state.router_view)
    return {
        "status": "degraded" if view.problems else "ok",
        "models": len(catalogue),
        "loaded": sorted(view.loaded),
        "router": {"build": view.build, "problems": view.problems},
    }


@router.post(
    "/v1/chat/completions",
    status_code=501,
    summary="Broker-owned chat completions",
    responses={501: {"description": "Scheduling and proxying are not implemented yet."}},
)
async def chat_completions() -> JSONResponse:
    """Reserve the chat endpoint for validation, scheduling, and proxying."""
    return not_implemented(
        "Chat completions are broker-owned and require scheduling support that is not "
        "implemented yet.",
        "broker_scheduling_not_implemented",
    )


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Create the broker API. Paths the broker does not serve return 404.

    Startup fails on an invalid preset. An unreachable or misconfigured router does not stop
    startup; it is reported by /health and blocks scheduling.
    """
    configured = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        catalogue = load_catalogue(configured.preset_path)
        async with httpx.AsyncClient(
            base_url=configured.llama_url, timeout=ROUTER_TIMEOUT_SECONDS, transport=transport
        ) as client:
            adapter = LlamaAdapter(client)
            view = await inspect_router(adapter, catalogue, clock())
            for problem in view.problems:
                LOGGER.warning("broker cannot schedule: %s", problem)
            LOGGER.info(
                "catalogue has %s models; router holds %s", len(catalogue), sorted(view.loaded)
            )
            app.state.catalogue = catalogue
            app.state.adapter = adapter
            app.state.router_view = view
            yield

    app = FastAPI(
        title="Model broker",
        version="0.1.0",
        description="Broker-owned API surface for scheduled local model inference.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
