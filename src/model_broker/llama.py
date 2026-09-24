"""Small HTTP boundary for one already-running llama.cpp router."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import cast

import httpx

JsonObject = dict[str, object]


class LlamaError(RuntimeError):
    """The attached llama.cpp router could not complete a broker operation."""


@dataclass(frozen=True, slots=True)
class LlamaAdapter:
    """Use the documented router API without knowing how the router was deployed.

    The adapter owns no scheduling policy and never asks llama.cpp to autoload a model. It is a
    narrow boundary for observing model state, issuing explicit control requests, and forwarding
    an already-admitted chat request.
    """

    url: str
    request_timeout_seconds: float = 10.0
    state_timeout_seconds: float = 60.0
    poll_interval_seconds: float = 0.1
    transport: httpx.AsyncBaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject unusable timeouts before an adapter starts handling broker work."""
        if not self.url.rstrip("/"):
            raise ValueError("llama router URL must not be empty")
        for name, value in (
            ("request_timeout_seconds", self.request_timeout_seconds),
            ("state_timeout_seconds", self.state_timeout_seconds),
            ("poll_interval_seconds", self.poll_interval_seconds),
        ):
            if value < 0 or (name != "poll_interval_seconds" and value == 0):
                raise ValueError(f"{name} must be positive")

    def endpoint(self, path: str) -> str:
        """Join one router-relative path to the configured URL without deployment assumptions."""
        return f"{self.url.rstrip('/')}{path}"

    async def model_states(self) -> dict[str, str]:
        """Return each router model ID and its observed status value."""
        document = await self.json_request("GET", "/models")
        if not isinstance(document, dict):
            raise LlamaError("router GET /models response must be a JSON object")
        response = cast(JsonObject, document)
        data = response.get("data")
        if not isinstance(data, list):
            raise LlamaError("router GET /models response has no data list")
        entries = cast(list[object], data)
        states: dict[str, str] = {}
        for item in entries:
            if not isinstance(item, dict):
                raise LlamaError("router GET /models response has a non-object model entry")
            entry = cast(JsonObject, item)
            model_id = entry.get("id")
            status = entry.get("status")
            status_object = cast(JsonObject, status) if isinstance(status, dict) else None
            value = status_object.get("value") if status_object is not None else None
            if not isinstance(model_id, str) or not isinstance(value, str):
                raise LlamaError("router GET /models response has an invalid model status")
            if model_id in states:
                raise LlamaError(f"router GET /models response repeats model {model_id!r}")
            states[model_id] = value
        return states

    async def load(self, model_id: str) -> None:
        """Explicitly load a model, then wait until the router reports it loaded."""
        await self.control("load", model_id)
        await self.wait_for_state(model_id, "loaded")

    async def unload(self, model_id: str) -> None:
        """Explicitly unload a model, then wait until the router reports it unloaded."""
        await self.control("unload", model_id)
        await self.wait_for_state(model_id, "unloaded")

    async def control(self, action: str, model_id: str) -> None:
        """Send one explicit model-management action without triggering an inference request."""
        if action not in {"load", "unload"}:
            raise ValueError(f"unsupported llama model action {action!r}")
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.request_timeout_seconds
            ) as client:
                response = await client.post(
                    self.endpoint(f"/models/{action}"), json={"model": model_id}
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise LlamaError(f"router POST /models/{action} failed: {error}") from error

    async def wait_for_state(self, model_id: str, expected: str) -> None:
        """Poll router state until one model reaches expected, or report a bounded failure."""
        deadline = time.monotonic() + self.state_timeout_seconds
        while True:
            state = (await self.model_states()).get(model_id)
            if state is None:
                raise LlamaError(f"router does not list model {model_id!r}")
            if state == expected:
                return
            if time.monotonic() >= deadline:
                raise LlamaError(
                    f"router model {model_id!r} did not become {expected!r}; "
                    f"last state was {state!r}"
                )
            await asyncio.sleep(self.poll_interval_seconds)

    @asynccontextmanager
    async def chat_completions(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[httpx.Response]:
        """Yield one successful chat response while its HTTP client remains open for streaming."""
        try:
            async with (
                httpx.AsyncClient(
                    transport=self.transport, timeout=self.request_timeout_seconds
                ) as client,
                client.stream(
                    "POST",
                    self.endpoint("/v1/chat/completions"),
                    json=dict(payload),
                    headers={"Accept": "application/json, text/event-stream"},
                ) as response,
            ):
                response.raise_for_status()
                yield response
        except httpx.HTTPError as error:
            raise LlamaError(f"router POST /v1/chat/completions failed: {error}") from error

    async def json_request(self, method: str, path: str) -> object:
        """Request and decode one router JSON document with one consistent error boundary."""
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.request_timeout_seconds
            ) as client:
                response = await client.request(method, self.endpoint(path))
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LlamaError(f"router {method} {path} failed: {error}") from error
