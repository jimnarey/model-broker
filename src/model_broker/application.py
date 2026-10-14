"""FastAPI entry point for model-broker.

Run with ``uvicorn --factory model_broker.application:create_app``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from model_broker.auth import ApiKeys, client_for, load_api_keys
from model_broker.backend import Backend
from model_broker.catalogue import (
    Variant,
    load_catalogue,
    load_service_catalogue,
    merge_catalogues,
)
from model_broker.events import (
    LEVELS,
    REQUEST,
    AuthenticationOff,
    AuthenticationOn,
    BrokerStarted,
    RouterProblemsFound,
    configure_logging,
    emit,
)
from model_broker.instrumentation import RequestEvents, router_event_hooks
from model_broker.llama import LlamaAdapter
from model_broker.proxy import ProxiedChat
from model_broker.scheduler import (
    LoadFailed,
    ResourceBusy,
    Scheduler,
    SchedulerError,
    UnknownModel,
    inspect_router,
)
from model_broker.servicecontrol import ServiceControl, ServiceController
from model_broker.strata import STRATA_SERVICE, strata_service

# Bounds each router control and observation request; inference has its own timeout.
ROUTER_TIMEOUT_SECONDS = 10.0
# One request to the service-control helper, which may recreate a container or wait for one
# to stop. The helper limits itself to 180 s by default, so this is longer.
SERVICE_CONTROL_TIMEOUT_SECONDS = 300.0
KEYS_FILE = "MODEL_BROKER_API_KEYS_FILE"
SERVICE_PRESET = "MODEL_BROKER_SERVICE_PRESET_PATH"
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
    # DEBUG adds every admission, wait, observation, router call and broker request.
    log_level: str = "INFO"
    # Services that change model by restarting their container (Strata) are optional: they
    # are used only when their preset is named. The host's service-control helper restarts
    # them, over its Unix socket.
    service_preset_path: Path | None = None
    service_control_socket: Path = Path("/run/model-broker-service-control/control.sock")
    # A service can take far longer to start than a llama.cpp model: Strata's first start
    # builds its engine (15-40 minutes) and every start reads tens of GB into RAM.
    service_load_timeout_seconds: float = 3600.0
    # Strata's own API, which serves the model, and its API key if it has one (a bearer token).
    strata_url: str = "http://strata:8080"
    strata_api_key: str | None = None

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
            log_level=log_level_setting(),
            service_preset_path=Path(preset)
            if (preset := os.environ.get(SERVICE_PRESET))
            else None,
            service_control_socket=Path(
                os.environ.get(
                    "MODEL_BROKER_SERVICE_CONTROL_SOCKET",
                    "/run/model-broker-service-control/control.sock",
                )
            ),
            service_load_timeout_seconds=seconds_setting(
                "MODEL_BROKER_SERVICE_LOAD_TIMEOUT_SECONDS", 3600.0, positive=True
            ),
            strata_url=os.environ.get("MODEL_BROKER_STRATA_URL", "http://strata:8080").rstrip("/"),
            strata_api_key=os.environ.get("MODEL_BROKER_STRATA_API_KEY") or None,
        )


def log_level_setting() -> str:
    level = os.environ.get("MODEL_BROKER_LOG_LEVEL", "INFO").upper()
    if level not in LEVELS:
        raise ValueError(f"MODEL_BROKER_LOG_LEVEL must be one of {', '.join(LEVELS)}")
    return level


def openai_error(
    status: int,
    message: str,
    code: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return an error in OpenAI's shape, which OpenAI client libraries understand."""
    error = {"message": message, "type": error_type, "param": param, "code": code}
    return JSONResponse(status_code=status, content={"error": error}, headers=headers)


def model_not_found(model: str) -> JSONResponse:
    return openai_error(
        404, f"The model {model!r} does not exist.", "model_not_found", param="model"
    )


def refusal(error: SchedulerError, model: str) -> JSONResponse:
    """The response for a request the scheduler would not admit."""
    match error:
        case UnknownModel():
            return model_not_found(model)
        case ResourceBusy():
            return openai_error(
                503,
                f"The resources for {model!r} are busy: {error}",
                "resource_busy",
                "server_error",
                headers={"Retry-After": str(error.retry_after)},
            )
        case LoadFailed():
            return openai_error(502, str(error), "model_load_failed", "server_error")
        case _:
            return openai_error(503, str(error), "router_unavailable", "server_error")


class Unauthenticated(Exception):
    """The request did not present a valid API key."""


def unauthenticated(_: Request, __: Exception) -> JSONResponse:
    """Answer as OpenAI does for a missing or wrong API key."""
    return openai_error(
        401,
        "Missing or invalid API key. Send it as 'Authorization: Bearer <key>'.",
        "invalid_api_key",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def authenticate(request: Request) -> None:
    """Require a valid key when keys are configured, and record the client's name.

    With authentication off every request is accepted, whatever key it sends, and no client
    is recorded. The name goes on the request context, so every event the request causes
    carries it.
    """
    keys = cast(ApiKeys | None, request.app.state.api_keys)
    if keys is None:
        return
    client = client_for(keys, request.headers.get("authorization"))
    if client is None:
        raise Unauthenticated
    context = REQUEST.get()
    if context is not None:
        context.client = client


router = APIRouter(tags=["broker"])
# Every /v1/ endpoint goes here, so none can be added without authentication.
api = APIRouter(prefix="/v1", tags=["broker"], dependencies=[Depends(authenticate)])


@router.get("/health")
async def health(request: Request) -> JsonObject:
    """Report the catalogue, the models the router holds, and why the broker cannot schedule."""
    scheduler = cast(Scheduler, request.app.state.scheduler)
    view = scheduler.view
    report: JsonObject = {
        "status": "degraded" if view.problems else "ok",
        "models": len(scheduler.catalogue),
        "loaded": sorted(view.loaded),
        "router": {"build": view.build, "problems": view.problems},
        "authentication": "off" if request.app.state.api_keys is None else "on",
    }
    services = sorted(set(scheduler.adapters) - {"llama"})
    if services:
        report["services"] = services
    return report


def model_object(variant: Variant, created: int) -> JsonObject:
    """A catalogue variant in OpenAI's model shape."""
    return {"id": variant.id, "object": "model", "created": created, "owned_by": "model-broker"}


@api.get("/models")
async def list_models(request: Request) -> JsonObject:
    """Every model in the catalogue, loaded or not."""
    scheduler = cast(Scheduler, request.app.state.scheduler)
    created = cast(int, request.app.state.started)
    data = [model_object(variant, created) for variant in scheduler.catalogue.values()]
    return {"object": "list", "data": data}


@api.get("/models/{model}", response_model=None)
async def get_model(model: str, request: Request) -> JsonObject | JSONResponse:
    scheduler = cast(Scheduler, request.app.state.scheduler)
    variant = scheduler.catalogue.get(model)
    if variant is None:
        return model_not_found(model)
    return model_object(variant, cast(int, request.app.state.started))


@api.post("/chat/completions", response_model=None)
async def chat_completions(request: Request) -> ProxiedChat | JSONResponse:
    """Schedule the requested model, then proxy the request and the router's response.

    The body is forwarded unchanged. The admission is taken before responding, so that a
    refusal gets its own status; it is released when the proxied response ends, however
    it ends.
    """
    try:
        payload: object = json.loads(await request.body())
    except ValueError:
        return openai_error(400, "The request body must be JSON.", "invalid_json")
    model = cast(JsonObject, payload).get("model") if isinstance(payload, dict) else None
    if not isinstance(model, str) or not model:
        return openai_error(400, "The request must name a model.", "missing_model", param="model")
    scheduler = cast(Scheduler, request.app.state.scheduler)
    resources = AsyncExitStack()
    try:
        variant = await resources.enter_async_context(scheduler.admit(model))
    except SchedulerError as error:
        return refusal(error, model)
    return ProxiedChat(scheduler.backend_for(variant), cast(JsonObject, payload), model, resources)


async def open_service_api(
    clients: AsyncExitStack,
    open_client: Callable[..., httpx.AsyncClient],
    name: str,
    settings: Settings,
) -> LlamaAdapter:
    """Connect to the API of a restart-managed service, which serves its model and says
    whether it is ready and busy."""
    if name != STRATA_SERVICE:
        raise ValueError(f"the preset names service {name!r}, which the broker has no client for")
    key = settings.strata_api_key
    client = await clients.enter_async_context(
        open_client(
            settings.strata_url,
            name,
            ROUTER_TIMEOUT_SECONDS,
            {"Authorization": f"Bearer {key}"} if key else None,
        )
    )
    return LlamaAdapter(client, name="Strata")


def service_backend(
    name: str,
    catalogue: Mapping[str, Variant],
    api: LlamaAdapter,
    settings: Settings,
    control: ServiceController | None,
) -> Backend:
    """The backend for the preset's models of one restart-managed service.

    ``control`` is the client of the host's service-control helper, which tests replace."""
    models = {model_id: item.env for model_id, item in catalogue.items() if item.backend == name}
    control = control or ServiceControl(
        settings.service_control_socket, SERVICE_CONTROL_TIMEOUT_SECONDS
    )
    return strata_service(control, models, api, settings.service_load_timeout_seconds)


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
    service_control: ServiceController | None = None,
) -> FastAPI:
    """Create the broker API. Paths the broker does not serve return 404.

    Startup fails on an invalid preset or API keys file. An unreachable or misconfigured
    router does not stop startup; it is reported by /health and blocks scheduling.
    """
    configured = settings or Settings.from_environment()
    configure_logging(configured.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        catalogue = load_catalogue(configured.preset_path)
        if configured.service_preset_path is not None:
            services = load_service_catalogue(configured.service_preset_path)
            catalogue = merge_catalogues(catalogue, services)
        if configured.api_keys_file is None:
            app.state.api_keys = None
            emit(AuthenticationOff())
        else:
            app.state.api_keys = load_api_keys(configured.api_keys_file)
            emit(AuthenticationOn(len(app.state.api_keys)))
        async with AsyncExitStack() as clients:

            def open_client(
                url: str, backend: str, timeout: float, headers: dict[str, str] | None = None
            ) -> httpx.AsyncClient:
                return httpx.AsyncClient(
                    base_url=url,
                    timeout=timeout,
                    headers=headers,
                    transport=transport,
                    event_hooks=router_event_hooks(backend),
                )

            llama = await clients.enter_async_context(
                open_client(configured.llama_url, "llama", ROUTER_TIMEOUT_SECONDS)
            )
            adapters: dict[str, Backend] = {
                "llama": LlamaAdapter(llama, state_timeout_seconds=configured.load_timeout_seconds)
            }
            for name in sorted({item.backend for item in catalogue.values()} - {"llama"}):
                api = await open_service_api(clients, open_client, name, configured)
                adapters[name] = service_backend(name, catalogue, api, configured, service_control)
            view = await inspect_router(adapters, catalogue, clock())
            if view.problems:
                emit(RouterProblemsFound(view.problems))
            app.state.scheduler = Scheduler(
                adapters,
                catalogue,
                view,
                protect_seconds=configured.idle_protect_seconds,
                max_wait_seconds=configured.max_wait_seconds,
                clock=clock,
            )
            await app.state.scheduler.start()
            app.state.started = int(time.time())
            emit(BrokerStarted(len(catalogue), sorted(view.loaded), view.build))
            yield

    app = FastAPI(
        title="Model broker",
        version="0.1.0",
        description="Broker-owned API surface for scheduled local model inference.",
        lifespan=lifespan,
    )
    app.add_exception_handler(Unauthenticated, unauthenticated)
    app.add_middleware(RequestEvents)
    app.include_router(router)
    app.include_router(api)
    return app
