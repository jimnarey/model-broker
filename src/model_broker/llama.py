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
# Prompt processing on a long context can take minutes before the first byte, and a stream
# can pause between tokens, so inference has no read limit; connecting still has one.
INFERENCE_TIMEOUT = httpx.Timeout(10.0, read=None)


class LlamaError(RuntimeError):
    """The attached llama.cpp router could not complete a broker operation."""


@dataclass(frozen=True, slots=True)
class ModelState:
    """One model's router status.

    ``value`` is ``unloaded``, ``loading``, ``loaded`` or ``sleeping``. A failed load is
    reported as ``unloaded`` with ``failed`` set and the worker's exit code.
    """

    value: str
    failed: bool = False
    exit_code: int | None = None


def parse_model_states(document: object) -> dict[str, ModelState]:
    """Validate a router ``GET /models`` document and return each model's state."""
    data = cast(JsonObject, document).get("data") if isinstance(document, dict) else None
    if not isinstance(data, list):
        raise LlamaError("router GET /models response has no data list")
    states: dict[str, ModelState] = {}
    for item in cast(list[object], data):
        entry = cast(JsonObject, item) if isinstance(item, dict) else {}
        status = entry.get("status")
        status = cast(JsonObject, status) if isinstance(status, dict) else {}
        model_id, value = entry.get("id"), status.get("value")
        if not isinstance(model_id, str) or not isinstance(value, str):
            raise LlamaError("router GET /models response has an invalid model entry")
        if model_id in states:
            raise LlamaError(f"router GET /models response repeats model {model_id!r}")
        failed, exit_code = status.get("failed", False), status.get("exit_code")
        # type() rather than isinstance(), because JSON true is also a Python int.
        if type(failed) is not bool or (exit_code is not None and type(exit_code) is not int):
            raise LlamaError(
                f"router GET /models response has an invalid failure status for {model_id!r}"
            )
        states[model_id] = ModelState(value=value, failed=failed, exit_code=exit_code)
    return states


@dataclass(frozen=True, slots=True)
class LlamaAdapter:
    """Use the documented router API without knowing how the router was deployed.

    ``client`` is long-lived, owned by the caller, and has the router URL as its base URL;
    its timeout bounds the control requests. The adapter owns no scheduling policy and never
    relies on llama.cpp's autoload.
    """

    client: httpx.AsyncClient
    state_timeout_seconds: float = 60.0
    poll_interval_seconds: float = 0.1
    inference_timeout: httpx.Timeout = field(default_factory=lambda: INFERENCE_TIMEOUT)

    def __post_init__(self) -> None:
        # "not >" rather than "<=" so that NaN is rejected too.
        if not self.state_timeout_seconds > 0:
            raise ValueError("state_timeout_seconds must be positive")
        if not self.poll_interval_seconds >= 0:
            raise ValueError("poll_interval_seconds must not be negative")

    async def model_states(self) -> dict[str, ModelState]:
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            document: object = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise LlamaError(f"router GET /models failed: {error}") from error
        return parse_model_states(document)

    async def model_state(self, model_id: str) -> ModelState:
        state = (await self.model_states()).get(model_id)
        if state is None:
            raise LlamaError(f"router does not list model {model_id!r}")
        return state

    async def post_model(self, action: str, model_id: str) -> httpx.Response:
        try:
            return await self.client.post(f"/models/{action}", json={"model": model_id})
        except httpx.HTTPError as error:
            raise LlamaError(f"router POST /models/{action} failed: {error}") from error

    async def load(self, model_id: str) -> None:
        """Explicitly load a model and wait until it is loaded, failing fast if its load fails."""
        response = await self.post_model("load", model_id)
        if response.is_error:
            raise LlamaError(
                f"router refused to load {model_id!r}: {response.status_code} {response.text}"
            )
        await self.wait_for_state(model_id, "loaded")

    async def unload(self, model_id: str) -> None:
        """Explicitly unload a model and wait until it is unloaded.

        The router answers 400 when the model is not running. If it is indeed unloaded (for
        example its worker crashed), the unload has already happened, so that is success.
        """
        response = await self.post_model("unload", model_id)
        if response.is_error:
            if (await self.model_state(model_id)).value == "unloaded":
                return
            raise LlamaError(
                f"router refused to unload {model_id!r}: {response.status_code} {response.text}"
            )
        await self.wait_for_state(model_id, "unloaded")

    async def wait_for_state(self, model_id: str, expected: str) -> None:
        """Poll until the model reaches ``expected``; fail on a failed load or after a timeout."""
        deadline = time.monotonic() + self.state_timeout_seconds
        while True:
            state = await self.model_state(model_id)
            if state.failed:
                raise LlamaError(
                    f"router model {model_id!r} failed to load (exit code {state.exit_code})"
                )
            if state.value == expected:
                return
            if time.monotonic() >= deadline:
                raise LlamaError(
                    f"router model {model_id!r} did not become {expected!r} within "
                    f"{self.state_timeout_seconds:g}s; last state was {state.value!r}"
                )
            await asyncio.sleep(self.poll_interval_seconds)

    @asynccontextmanager
    async def chat_completions(
        self, payload: Mapping[str, object]
    ) -> AsyncGenerator[httpx.Response]:
        """Forward an admitted chat request and yield the router's response, open for streaming.

        The response is yielded whatever its status: a router error such as an over-long
        prompt belongs to the client and is passed back unchanged. Only failing to reach the
        router, or to read its stream, is a LlamaError.
        """
        try:
            async with self.client.stream(
                "POST",
                "/v1/chat/completions",
                json=dict(payload),
                headers={"Accept": "application/json, text/event-stream"},
                timeout=self.inference_timeout,
            ) as response:
                yield response
        except httpx.HTTPError as error:
            raise LlamaError(f"router POST /v1/chat/completions failed: {error}") from error
