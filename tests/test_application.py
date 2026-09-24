from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from model_broker.application import Settings, create_app
from model_broker.auth import ApiKeysError
from model_broker.catalogue import CatalogueError
from model_broker.events import (
    AuthenticationOff,
    Event,
    HealthChecked,
    RequestCompleted,
    RequestContext,
    RequestReceived,
    RouterCalled,
)

PRESET = Path(__file__).parent / "fixtures" / "models-preset.ini"
SETTINGS = Settings("http://router.test", PRESET)
CUDA0 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda0"
Handler = Callable[[httpx.Request], httpx.Response]
GOOD_PROPS = {"role": "router", "max_instances": 3, "models_autoload": False, "build_info": "b1"}


def router(props: dict[str, Any] = GOOD_PROPS, loaded: tuple[str, ...] = ()) -> Handler:
    """A mock router with the given /props and these models loaded, all idle."""
    models = {"data": [{"id": model, "status": {"value": "loaded"}} for model in loaded]}
    idle = (
        "llamacpp:requests_processing 0\nllamacpp:requests_deferred 0\nllamacpp:n_decode_total 0\n"
    )
    responses = {
        "/props": httpx.Response(200, json=props),
        "/models": httpx.Response(200, json=models),
        "/slots": httpx.Response(200, json=[{"id": 0, "is_processing": False}]),
        "/metrics": httpx.Response(200, text=idle),
    }
    return lambda request: responses[request.url.path]


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
        "authentication": "off",
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


# Authentication

KEY = "test-key-for-vscode"
CHAT = {"model": "example"}


def with_keys(tmp_path: Path) -> Settings:
    """Settings with authentication on, for one client ``vscode`` holding ``KEY``."""
    keys_file = tmp_path / "api-keys"
    keys_file.write_text(f"vscode {KEY}\n")
    return Settings("http://router.test", PRESET, api_keys_file=keys_file)


def chat(settings: Settings, headers: dict[str, str]) -> httpx.Response:
    return broker_request(
        lambda client: client.post("/v1/chat/completions", json=CHAT, headers=headers),
        settings=settings,
    )


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer sk-no-key-required"}],
    ids=["no-key", "placeholder-key"],
)
def test_without_keys_file_every_request_is_accepted(headers: dict[str, str]) -> None:
    """With authentication off, a request reaches the endpoint with or without a key.

    Many OpenAI clients always send some key, so a placeholder must not be refused. The
    endpoint's own 501 shows that the request got past authentication.
    """
    assert chat(SETTINGS, headers).status_code == 501


def test_authentication_off_is_warned_about(events: Callable[[], list[Event]]) -> None:
    """Running without keys is allowed, but startup warns about it."""
    broker_request(lambda client: client.get("/health"))
    assert AuthenticationOff() in events()


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer test-key-for-vscodX"}, {"Authorization": f"Basic {KEY}"}],
    ids=["no-key", "wrong-key", "wrong-scheme"],
)
def test_with_keys_file_a_request_without_a_valid_key_is_refused(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    """With authentication on, a missing or wrong key gets OpenAI's 401 before the endpoint.

    OpenAI clients recognise ``invalid_api_key``; ``WWW-Authenticate`` tells other HTTP
    clients which scheme to use.
    """
    response = chat(with_keys(tmp_path), headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_with_keys_file_a_valid_key_reaches_the_endpoint(tmp_path: Path) -> None:
    response = chat(with_keys(tmp_path), {"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 501


def test_health_needs_no_key(tmp_path: Path) -> None:
    """/health stays open so that the container health check needs no secret."""
    response = broker_request(lambda client: client.get("/health"), settings=with_keys(tmp_path))

    assert response.status_code == 200
    assert response.json()["authentication"] == "on"


def test_invalid_keys_file_stops_startup(tmp_path: Path) -> None:
    keys_file = tmp_path / "api-keys"
    keys_file.write_text("vscode short\n")
    settings = Settings("http://router.test", PRESET, api_keys_file=keys_file)

    with pytest.raises(ApiKeysError, match="shorter than 16"):
        broker_request(lambda client: client.get("/health"), settings=settings)


# Logging


def in_request(caplog: pytest.LogCaptureFixture) -> list[tuple[Event, RequestContext]]:
    """Each event emitted while handling a broker request, with that request's context."""
    found = ((getattr(r, "event", None), getattr(r, "request", None)) for r in caplog.records)
    return [
        (event, context)
        for event, context in found
        if isinstance(event, Event) and isinstance(context, RequestContext)
    ]


def test_each_request_is_logged_with_its_id_and_client(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A broker request produces a received and a completed event, sharing one request ID.

    With authentication on, the completed event carries the client's name. The name is
    known only after authentication, so this shows that later events see it.
    """
    caplog.set_level(logging.DEBUG, logger="model_broker")
    chat(replace(with_keys(tmp_path), log_level="DEBUG"), {"Authorization": f"Bearer {KEY}"})
    (received, first), (completed, last) = [
        (event, context)
        for event, context in in_request(caplog)
        if isinstance(event, RequestReceived | RequestCompleted)
    ]

    assert received == RequestReceived("POST", "/v1/chat/completions")
    assert isinstance(completed, RequestCompleted) and completed.status == 501
    assert first.request_id == last.request_id and last.client == "vscode"


def test_health_checks_are_logged_quietly(events: Callable[[], list[Event]]) -> None:
    """/health is a HealthChecked event at DEBUG, not an INFO RequestCompleted, so that a
    health check every few seconds does not fill the log."""
    broker_request(
        lambda client: client.get("/health"), settings=replace(SETTINGS, log_level="DEBUG")
    )

    kinds = {type(event) for event in events()}
    assert HealthChecked in kinds and RequestCompleted not in kinds


def test_router_calls_are_logged(events: Callable[[], list[Event]]) -> None:
    """Every call to the router is a RouterCalled event: startup reads /props and /models."""
    broker_request(
        lambda client: client.get("/health"), settings=replace(SETTINGS, log_level="DEBUG")
    )

    calls = [(e.method, e.path, e.status) for e in events() if isinstance(e, RouterCalled)]
    assert calls[:2] == [("GET", "/props", 200), ("GET", "/models", 200)]


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


def test_settings_read_scheduling_times(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protection window and maximum wait default to 300 s and 30 s; 0 is allowed."""
    monkeypatch.setenv("MODEL_BROKER_PRESET_PATH", "/config/models-preset.ini")
    settings = Settings.from_environment()
    assert (settings.idle_protect_seconds, settings.max_wait_seconds) == (300.0, 30.0)

    monkeypatch.setenv("MODEL_BROKER_IDLE_PROTECT_SECONDS", "90")
    monkeypatch.setenv("MODEL_BROKER_MAX_WAIT_SECONDS", "0")
    settings = Settings.from_environment()
    assert (settings.idle_protect_seconds, settings.max_wait_seconds) == (90.0, 0.0)


@pytest.mark.parametrize(
    ("value", "error"),
    [("soon", "must be a number of seconds"), ("-1", "must not be negative"), ("nan", "negative")],
    ids=["not-a-number", "negative", "nan"],
)
def test_settings_reject_invalid_times(
    monkeypatch: pytest.MonkeyPatch, value: str, error: str
) -> None:
    monkeypatch.setenv("MODEL_BROKER_PRESET_PATH", "/config/models-preset.ini")
    monkeypatch.setenv("MODEL_BROKER_MAX_WAIT_SECONDS", value)
    with pytest.raises(ValueError, match=error):
        Settings.from_environment()


def test_settings_read_the_api_keys_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keys file is optional: unset means authentication is off."""
    monkeypatch.setenv("MODEL_BROKER_PRESET_PATH", "/config/models-preset.ini")
    monkeypatch.delenv("MODEL_BROKER_API_KEYS_FILE", raising=False)
    assert Settings.from_environment().api_keys_file is None

    monkeypatch.setenv("MODEL_BROKER_API_KEYS_FILE", "/run/secrets/broker-api-keys")
    assert Settings.from_environment().api_keys_file == Path("/run/secrets/broker-api-keys")


def test_settings_require_a_positive_load_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load timeout of 0 would fail every load at once, so it is refused."""
    monkeypatch.setenv("MODEL_BROKER_PRESET_PATH", "/config/models-preset.ini")
    assert Settings.from_environment().load_timeout_seconds == 600.0
    monkeypatch.setenv("MODEL_BROKER_LOAD_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        Settings.from_environment()


@pytest.mark.parametrize(("value", "level"), [(None, "INFO"), ("debug", "DEBUG")])
def test_settings_read_the_log_level(
    monkeypatch: pytest.MonkeyPatch, value: str | None, level: str
) -> None:
    monkeypatch.setenv("MODEL_BROKER_PRESET_PATH", "/config/models-preset.ini")
    if value is None:
        monkeypatch.delenv("MODEL_BROKER_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("MODEL_BROKER_LOG_LEVEL", value)
    assert Settings.from_environment().log_level == level


def test_settings_reject_an_unknown_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_BROKER_PRESET_PATH", "/config/models-preset.ini")
    monkeypatch.setenv("MODEL_BROKER_LOG_LEVEL", "chatty")
    with pytest.raises(ValueError, match="must be one of DEBUG, INFO, WARNING, ERROR"):
        Settings.from_environment()
