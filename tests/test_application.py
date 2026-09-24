from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest

from model_broker.application import Settings, create_app

SETTINGS = Settings("http://router.test")


def broker_request[T](action: Callable[[httpx.AsyncClient], Awaitable[T]]) -> T:
    """Run action against a started broker through ASGI, from a synchronous pytest test."""

    async def main() -> T:
        app = create_app(SETTINGS)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://broker.test"
            ) as client:
                return await action(client)

    return asyncio.run(main())


def test_health_reports_ok() -> None:
    response = broker_request(lambda client: client.get("/health"))
    assert response.json() == {"status": "ok"}


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


def test_settings_default_router_url_and_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    """The router URL defaults to the Compose service name; a trailing slash is removed."""
    monkeypatch.delenv("MODEL_BROKER_LLAMA_URL", raising=False)
    assert Settings.from_environment() == Settings("http://llama-cpp:8080")

    monkeypatch.setenv("MODEL_BROKER_LLAMA_URL", "http://llama:8080/")
    assert Settings.from_environment() == Settings("http://llama:8080")
