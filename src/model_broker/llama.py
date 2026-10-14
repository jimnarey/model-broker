"""Small HTTP boundary for one already-running llama.cpp router, or a server with its API."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import cast

import httpx

from model_broker.backend import BackendError, ModelActivity, ModelState, RouterProps

JsonObject = dict[str, object]
# Prompt processing on a long context can take minutes before the first byte, and a stream
# can pause between tokens, so inference has no read limit; connecting still has one.
INFERENCE_TIMEOUT = httpx.Timeout(10.0, read=None)


class LlamaError(BackendError):
    """The llama.cpp router could not complete a broker operation."""


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


def parse_router_props(document: object, role: str = "router") -> RouterProps:
    """Validate a ``GET /props`` document from a server whose ``role`` is as given."""
    props = cast(JsonObject, document) if isinstance(document, dict) else {}
    max_instances, autoload, build = (
        props.get("max_instances"),
        props.get("models_autoload"),
        props.get("build_info"),
    )
    if props.get("role") != role:
        raise LlamaError(f"GET /props did not come from a {role} (its role is not {role!r})")
    # type() rather than isinstance(), because JSON true is also a Python int.
    if type(max_instances) is not int or type(autoload) is not bool:
        raise LlamaError("router GET /props response has invalid max_instances or autoload")
    return RouterProps(max_instances, autoload, build if isinstance(build, str) else None)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse unlabelled ``llamacpp:<name> <value>`` samples from a metrics page."""
    samples: dict[str, float] = {}
    for line in text.splitlines():
        name, _, value = line.strip().rpartition(" ")
        if name and not line.startswith("#"):
            try:
                samples[name.removeprefix("llamacpp:")] = float(value)
            except ValueError as error:
                raise LlamaError(f"router metrics line is not a sample: {line!r}") from error
    return samples


def parse_model_activity(slots: object, metrics: str) -> ModelActivity:
    """Combine a model's ``/slots`` document and ``/metrics`` page into its activity."""
    entries = cast(list[object], slots) if isinstance(slots, list) else None
    if not entries or any(not isinstance(slot, dict) for slot in entries):
        raise LlamaError("router GET /slots response is not a list of slots")
    processing = [cast(JsonObject, slot).get("is_processing") for slot in entries]
    if any(type(flag) is not bool for flag in processing):
        raise LlamaError("router GET /slots response has an invalid is_processing value")
    samples = parse_prometheus(metrics)
    missing = {"requests_processing", "requests_deferred", "n_decode_total"} - samples.keys()
    if missing:
        raise LlamaError(f"router metrics lack {', '.join(sorted(missing))}; is --metrics set?")
    return ModelActivity(
        processing=any(processing) or samples["requests_processing"] > 0,
        deferred=int(samples["requests_deferred"]),
        decode_calls=int(samples["n_decode_total"]),
    )


@dataclass(frozen=True, slots=True)
class LlamaAdapter:
    """Use the documented router API without knowing how the router was deployed.

    ``client`` is long-lived, owned by the caller, and has the router URL as its base URL;
    its timeout bounds the control requests. The adapter owns no scheduling policy and never
    relies on llama.cpp's autoload.

    The model-management API (``/props``, ``/models``, ``/models/load``, ``/models/unload``)
    is also served by Strata's supervisor, so this adapter drives it too, with ``role`` set
    to what that server's ``/props`` reports. ``name`` is how errors refer to the server.
    """

    client: httpx.AsyncClient
    state_timeout_seconds: float = 60.0
    poll_interval_seconds: float = 0.1
    inference_timeout: httpx.Timeout = field(default_factory=lambda: INFERENCE_TIMEOUT)
    name: str = "router"
    role: str = "router"

    def __post_init__(self) -> None:
        # "not >" rather than "<=" so that NaN is rejected too.
        if not self.state_timeout_seconds > 0:
            raise ValueError("state_timeout_seconds must be positive")
        if not self.poll_interval_seconds >= 0:
            raise ValueError("poll_interval_seconds must not be negative")

    async def get(self, path: str, **params: str) -> httpx.Response:
        """GET a router path, turning transport failures and error statuses into LlamaError."""
        try:
            response = await self.client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise LlamaError(f"{self.name} GET {path} failed: {error}") from error
        return response

    async def get_json(self, path: str, **params: str) -> object:
        response = await self.get(path, **params)
        try:
            return cast(object, response.json())
        except ValueError as error:
            raise LlamaError(f"{self.name} GET {path} failed: response is not JSON") from error

    async def props(self) -> RouterProps:
        return parse_router_props(await self.get_json("/props"), self.role)

    async def activity(self, model_id: str) -> ModelActivity:
        """Read a loaded model's slots and metrics. The router needs ``--metrics``."""
        slots = await self.get_json("/slots", model=model_id)
        metrics = await self.get("/metrics", model=model_id)
        return parse_model_activity(slots, metrics.text)

    async def model_states(self) -> dict[str, ModelState]:
        return parse_model_states(await self.get_json("/models"))

    async def model_state(self, model_id: str) -> ModelState:
        state = (await self.model_states()).get(model_id)
        if state is None:
            raise LlamaError(f"{self.name} does not list model {model_id!r}")
        return state

    async def post_model(self, action: str, model_id: str) -> httpx.Response:
        try:
            return await self.client.post(f"/models/{action}", json={"model": model_id})
        except httpx.HTTPError as error:
            raise LlamaError(f"{self.name} POST /models/{action} failed: {error}") from error

    async def load(self, model_id: str) -> None:
        """Explicitly load a model and wait until it is loaded, failing fast if its load fails."""
        response = await self.post_model("load", model_id)
        if response.is_error:
            raise LlamaError(
                f"{self.name} refused to load {model_id!r}: {response.status_code} {response.text}"
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
                f"{self.name} refused to unload {model_id!r}: "
                f"{response.status_code} {response.text}"
            )
        await self.wait_for_state(model_id, "unloaded")

    async def wait_for_state(self, model_id: str, expected: str) -> None:
        """Poll until the model reaches ``expected``; fail on a failed load or after a timeout."""
        deadline = time.monotonic() + self.state_timeout_seconds
        while True:
            state = await self.model_state(model_id)
            if state.failed:
                raise LlamaError(
                    f"{self.name} model {model_id!r} failed to load (exit code {state.exit_code})"
                )
            if state.value == expected:
                return
            if time.monotonic() >= deadline:
                raise LlamaError(
                    f"{self.name} model {model_id!r} did not become {expected!r} within "
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
            raise LlamaError(f"{self.name} POST /v1/chat/completions failed: {error}") from error
