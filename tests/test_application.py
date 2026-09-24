from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from model_broker.application import Settings, create_app
from model_broker.catalogue import CatalogueError

PRESET = Path(__file__).parent / "fixtures" / "models-preset.ini"
SETTINGS = Settings("http://router.test", PRESET)
CUDA0 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda0"
Handler = Callable[[httpx.Request], httpx.Response]
GOOD_PROPS = {"role": "router", "max_instances": 3, "models_autoload": False, "build_info": "b1"}


def router(props: dict[str, Any] = GOOD_PROPS, loaded: tuple[str, ...] = ()) -> Handler:
    """A mock router with the given /props and these models loaded."""
    models = {"data": [{"id": model, "status": {"value": "loaded"}} for model in loaded]}
    responses = {"/props": props, "/models": models}
    return lambda request: httpx.Response(200, json=responses[request.url.path])


def broker_request[T](
    action: Callable[[httpx.AsyncClient], Awaitable[T]],
    handler: Handler | None = None,
    settings: Settings = SETTINGS,
) -> T:
    """Start a broker whose router is ``handler``, run action through ASGI, then stop it."""

    async def main() -> T:
        app = create_app(settings, httpx.MockTransport(handler or router()), clock=lambda: 1.0)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://broker.test"
            ) as client:
                return await action(client)

    return asyncio.run(main())


def health(handler: Handler) -> dict[str, Any]:
    return broker_request(lambda client: client.get("/health"), handler).json()


# Startup


def test_health_reports_catalogue_and_loaded_models() -> None:
    """A correctly configured router gives status ok, with the models it already holds.

    Setup: the five-model fixture preset, and a router with models-max 3, autoload off and
    CUDA0 loaded. Expect status ok, 5 models, CUDA0 adopted, and no problems.
    """
    assert health(router(loaded=(CUDA0,))) == {
        "status": "ok",
        "models": 5,
        "loaded": [CUDA0],
        "router": {"build": "b1", "problems": []},
    }


def test_misconfigured_router_degrades_health() -> None:
    """A router that would choose models to unload itself is reported and blocks scheduling.

    Setup: models-max 1 and autoload on, as the production container ran on 2026-09-24.
    Expect status degraded with both problems, while the broker itself still serves.
    """
    props = GOOD_PROPS | {"max_instances": 1, "models_autoload": True}
    report = health(router(props))

    assert report["status"] == "degraded"
    assert [problem.split(";")[0] for problem in report["router"]["problems"]] == [
        "router models-max is 1, but 3 models can be loaded together",
        "router loads models automatically",
    ]


def test_unreachable_router_does_not_stop_startup() -> None:
    """If the router is down, the broker still starts, and /health says it cannot schedule."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    report = health(refuse)

    assert report["status"] == "degraded"
    assert report["router"]["problems"] == [
        "router is unavailable: router GET /props failed: connection refused"
    ]


def test_invalid_preset_stops_startup(tmp_path: Path) -> None:
    """The broker does not serve with a preset it cannot validate."""
    preset = tmp_path / "models-preset.ini"
    preset.write_text("[no-version--cpu]\nmodel = /m.gguf\n")

    with pytest.raises(CatalogueError, match="must start with version = 1"):
        broker_request(lambda client: client.get("/health"), settings=Settings("x", preset))


# Endpoints


def test_chat_completions_is_not_implemented_yet() -> None:
    """Chat returns the broker's own 501 in the OpenAI error shape until scheduling exists."""
    response = broker_request(
        lambda client: client.post("/v1/chat/completions", json={"model": "example"})
    )

    assert response.status_code == 501
    assert response.json()["error"]["code"] == "broker_scheduling_not_implemented"


@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/v1/embeddings"), ("GET", "/slots"), ("POST", "/models/load")],
    ids=["other-openai-endpoint", "router-slots", "router-model-control"],
)
def test_unsupported_paths_return_404(method: str, path: str) -> None:
    """Paths the broker does not serve are 404, including router management endpoints.

    The router's own control API must never be reachable through the broker.
    """
    response = broker_request(lambda client: client.request(method, path))
    assert response.status_code == 404


# Settings


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The router URL defaults to the Compose service name, without a trailing slash."""
    monkeypatch.delenv("MODEL_BROKER_LLAMA_URL", raising=False)
    monkeypatch.setenv("MODEL_BROKER_PRESET_PATH", "/config/models-preset.ini")
    assert Settings.from_environment() == Settings(
        "http://llama-cpp:8080", Path("/config/models-preset.ini")
    )

    monkeypatch.setenv("MODEL_BROKER_LLAMA_URL", "http://llama:8080/")
    assert Settings.from_environment().llama_url == "http://llama:8080"


def test_settings_require_the_preset_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_BROKER_PRESET_PATH", raising=False)
    with pytest.raises(ValueError, match="MODEL_BROKER_PRESET_PATH must name"):
        Settings.from_environment()
