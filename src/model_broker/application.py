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
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from model_broker.auth import ApiKeys, client_for, load_api_keys
from model_broker.catalogue import load_catalogue
from model_broker.llama import LlamaAdapter
from model_broker.scheduler import Scheduler, inspect_router

LOGGER = logging.getLogger(__name__)
# Bounds each router control and observation request; inference has its own timeout.
ROUTER_TIMEOUT_SECONDS = 10.0
KEYS_FILE = "MODEL_BROKER_API_KEYS_FILE"
JsonObject = dict[str, object]


def seconds_setting(name: str, default: float, positive: bool = False) -> float:
    """Read a non-negative (or, if ``positive``, a positive) number of seconds."""
    text = os.environ.get(name)
    try:
        value = float(text) if text else default
    except ValueError as error:
        raise ValueError(f"{name} must be a number of seconds, not {text!r}") from error
    # "not >=" rather than "<" so that NaN is rejected too.
    if not value >= 0:
        raise ValueError(f"{name} must not be negative")
    if positive and value == 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Deployment configuration read from the environment."""

    llama_url: str
    preset_path: Path
    # How long a model stays protected from unloading after it was last used.
    idle_protect_seconds: float = 300.0
    # How long a request waits for busy conflicting models before a 503.
    max_wait_seconds: float = 30.0
    # How long a load or unload may take before it counts as failed.
    load_timeout_seconds: float = 600.0
    # Secrets file of "name key" lines; None turns authentication off.
    api_keys_file: Path | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        llama_url = os.environ.get("MODEL_BROKER_LLAMA_URL", "http://llama-cpp:8080")
        preset_path = os.environ.get("MODEL_BROKER_PRESET_PATH")
        if not preset_path:
            raise ValueError("MODEL_BROKER_PRESET_PATH must name the mounted llama.cpp preset")
        return cls(
            llama_url=llama_url.rstrip("/"),
            preset_path=Path(preset_path),
            idle_protect_seconds=seconds_setting("MODEL_BROKER_IDLE_PROTECT_SECONDS", 300.0),
            max_wait_seconds=seconds_setting("MODEL_BROKER_MAX_WAIT_SECONDS", 30.0),
            load_timeout_seconds=seconds_setting(
                "MODEL_BROKER_LOAD_TIMEOUT_SECONDS", 600.0, positive=True
            ),
            api_keys_file=Path(keys_file) if (keys_file := os.environ.get(KEYS_FILE)) else None,
        )


def not_implemented(message: str, code: str) -> JSONResponse:
    """Return a 501 response in the OpenAI error shape."""
    error = {"message": message, "type": "not_implemented", "code": code}
    return JSONResponse(status_code=501, content={"error": error})


class Unauthenticated(Exception):
    """The request did not present a valid API key."""


def unauthenticated(_: Request, __: Exception) -> JSONResponse:
    """Answer as OpenAI does for a missing or wrong API key."""
    error = {
        "message": "Missing or invalid API key. Send it as 'Authorization: Bearer <key>'.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
    headers = {"WWW-Authenticate": "Bearer"}
    return JSONResponse(status_code=401, content={"error": error}, headers=headers)


async def authenticate(request: Request) -> None:
    """Record the calling client's name, requiring a valid key when keys are configured.

    With authentication off every request is accepted, whatever key it sends, and the
    client is recorded as None.
    """
    keys = cast(ApiKeys | None, request.app.state.api_keys)
    client = None
    if keys is not None:
        client = client_for(keys, request.headers.get("authorization"))
        if client is None:
            raise Unauthenticated
    request.state.client = client


router = APIRouter(tags=["broker"])
# Every /v1/ endpoint goes here, so none can be added without authentication.
api = APIRouter(prefix="/v1", tags=["broker"], dependencies=[Depends(authenticate)])


@router.get("/health")
async def health(request: Request) -> JsonObject:
    """Report the catalogue, the models the router holds, and why the broker cannot schedule."""
    scheduler = cast(Scheduler, request.app.state.scheduler)
    view = scheduler.view
    return {
        "status": "degraded" if view.problems else "ok",
        "models": len(scheduler.catalogue),
        "loaded": sorted(view.loaded),
        "router": {"build": view.build, "problems": view.problems},
        "authentication": "off" if request.app.state.api_keys is None else "on",
    }


@api.post(
    "/chat/completions",
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

    Startup fails on an invalid preset or API keys file. An unreachable or misconfigured
    router does not stop startup; it is reported by /health and blocks scheduling.
    """
    configured = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        catalogue = load_catalogue(configured.preset_path)
        if configured.api_keys_file is None:
            app.state.api_keys = None
            LOGGER.warning("authentication is off: set %s to require API keys", KEYS_FILE)
        else:
            app.state.api_keys = load_api_keys(configured.api_keys_file)
            LOGGER.info("authentication is on for %s clients", len(app.state.api_keys))
        async with httpx.AsyncClient(
            base_url=configured.llama_url, timeout=ROUTER_TIMEOUT_SECONDS, transport=transport
        ) as client:
            adapter = LlamaAdapter(client, state_timeout_seconds=configured.load_timeout_seconds)
            view = await inspect_router(adapter, catalogue, clock())
            for problem in view.problems:
                LOGGER.warning("broker cannot schedule: %s", problem)
            LOGGER.info(
                "catalogue has %s models; router holds %s", len(catalogue), sorted(view.loaded)
            )
            app.state.scheduler = Scheduler(
                adapter,
                catalogue,
                view,
                protect_seconds=configured.idle_protect_seconds,
                max_wait_seconds=configured.max_wait_seconds,
                clock=clock,
            )
            await app.state.scheduler.start()
            yield

    app = FastAPI(
        title="Model broker",
        version="0.1.0",
        description="Broker-owned API surface for scheduled local model inference.",
        lifespan=lifespan,
    )
    app.add_exception_handler(Unauthenticated, unauthenticated)
    app.include_router(router)
    app.include_router(api)
    return app
