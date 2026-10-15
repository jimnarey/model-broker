"""A service that changes model by restarting its container.

Some servers, Strata among them, choose their model when they start and cannot change it
while running. ``ManagedService`` gives such a service the ``Backend`` interface: loading a
model starts the service's container with that model's settings, and unloading stops it. The
host's service-control helper does the starting and stopping (see ``servicecontrol``).

The service holds one model at a time, so a switch is an unload and then a load. Which model
is loaded is read from the settings the container was created with, and whether it is ready
from the service's own API.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

import httpx

from model_broker.backend import (
    BackendError,
    ModelActivity,
    ModelState,
    RouterProps,
)
from model_broker.llama import LlamaAdapter
from model_broker.servicecontrol import ServiceController, ServiceStatus

ActivityReader = Callable[[LlamaAdapter], Awaitable[ModelActivity]]
# Docker states in which a container that should be starting has instead stopped.
FAILED_STATES = frozenset({"exited", "dead", "missing"})


async def api_is_healthy(api: LlamaAdapter) -> bool:
    """Whether the service's ``/health`` answers; a server that is not up yet does not."""
    try:
        await api.get("/health")
    except BackendError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class ManagedService:
    """Load and unload a service's models by starting and stopping its container.

    ``models`` maps each public model ID to the settings its container is started with. The
    settings of the running container identify which model is loaded, so they must all be
    given for every model, even empty ones. ``api`` is a ``LlamaAdapter`` on the service's own
    server, used for readiness, activity and chat. ``name`` describes the service in errors.
    """

    name: str
    service: str
    control: ServiceController
    models: Mapping[str, Mapping[str, str]]
    api: LlamaAdapter
    read_activity: ActivityReader
    state_timeout_seconds: float = 3600.0
    poll_interval_seconds: float = 2.0
    is_ready: Callable[[LlamaAdapter], Awaitable[bool]] = api_is_healthy

    @property
    def unrecognised(self) -> str:
        """The ID reported for a running container whose settings match no model."""
        return f"{self.service}:unrecognised"

    def identify(self, status: ServiceStatus) -> str | None:
        """The model whose settings the container has, if any."""
        for model_id, settings in self.models.items():
            if all(status.settings.get(name) == value for name, value in settings.items()):
                return model_id
        return None

    async def props(self) -> RouterProps:
        """One model at a time, never loaded by itself. Asking the helper also proves it is up."""
        await self.control.status(self.service)
        return RouterProps(max_instances=1, models_autoload=False, build=None)

    async def read_states(self) -> tuple[dict[str, ModelState], ServiceStatus]:
        status = await self.control.status(self.service)
        states = {model_id: ModelState("unloaded") for model_id in self.models}
        if status.occupying:
            running = self.identify(status) or self.unrecognised
            ready = status.state == "running" and await self.is_ready(self.api)
            states[running] = ModelState("loaded" if ready else "loading")
        return states, status

    async def model_states(self) -> dict[str, ModelState]:
        return (await self.read_states())[0]

    async def model_state(self, model_id: str) -> ModelState:
        state = (await self.model_states()).get(model_id)
        if state is None:
            raise BackendError(f"{self.name} does not list model {model_id!r}")
        return state

    async def load(self, model_id: str) -> None:
        """Start the container with the model's settings and wait until it is ready.

        Refused while a different model is running: replacing the container would kill any
        request it is serving, and the scheduler only loads after unloading."""
        settings = self.models.get(model_id)
        if settings is None:
            raise BackendError(f"{self.name} does not list model {model_id!r}")
        status = await self.control.status(self.service)
        if status.occupying and self.identify(status) != model_id:
            raise BackendError(
                f"{self.name} is running {self.identify(status) or 'an unrecognised model'!r}; "
                f"refusing to replace it with {model_id!r}"
            )
        if not status.occupying:
            await self.control.start(self.service, settings)
        await self.wait_for_state(model_id, "loaded")

    async def unload(self, model_id: str) -> None:
        """Stop the container, if it is running this model, and wait until it has stopped."""
        status = await self.control.status(self.service)
        if status.occupying:
            running = self.identify(status)
            if running != model_id:
                raise BackendError(
                    f"{self.name} is running {running or 'an unrecognised model'!r}, "
                    f"not {model_id!r}; refusing to stop it"
                )
            await self.control.stop(self.service)
        await self.wait_for_state(model_id, "unloaded")

    async def wait_for_state(self, model_id: str, expected: str) -> None:
        """Poll until the model reaches ``expected``.

        Waiting for ``loaded`` fails as soon as the container has stopped, rather than at the
        timeout. Waiting for ``unloaded`` is not failed by a stopped container's exit code:
        Docker reports 137 for any container it had to kill, which is how a stop looks."""
        deadline = time.monotonic() + self.state_timeout_seconds
        while True:
            states, status = await self.read_states()
            state = states.get(model_id)
            if state is None:
                raise BackendError(f"{self.name} does not list model {model_id!r}")
            if state.value == expected:
                return
            if expected == "loaded" and status.state in FAILED_STATES:
                raise BackendError(self.failure(model_id, status))
            if expected == "loaded" and status.occupying and self.identify(status) != model_id:
                raise BackendError(self.mismatch(model_id, status))
            if time.monotonic() >= deadline:
                raise BackendError(
                    f"{self.name} model {model_id!r} did not become {expected!r} within "
                    f"{self.state_timeout_seconds:g}s; last state was {state.value!r}"
                )
            await asyncio.sleep(self.poll_interval_seconds)

    def mismatch(self, model_id: str, status: ServiceStatus) -> str:
        """Why a container that should be running this model is not recognised as it."""
        differing = [
            f"{name} (wanted {value!r}, found {status.settings.get(name)!r})"
            for name, value in self.models[model_id].items()
            if status.settings.get(name) != value
        ]
        return (
            f"{self.name} is running, but its container does not have model {model_id!r}'s "
            f"settings: {', '.join(differing)}. Is each variable passed through in the "
            "Compose file?"
        )

    def failure(self, model_id: str, status: ServiceStatus) -> str:
        how = (
            "was killed for want of memory" if status.oom_killed else f"exited ({status.exit_code})"
        )
        if status.state == "missing":
            how = "has no container"
        return f"{self.name} model {model_id!r} failed to start: the service {how}"

    async def activity(self, model_id: str) -> ModelActivity:
        return await self.read_activity(self.api)

    def chat_completions(
        self, payload: Mapping[str, object]
    ) -> AbstractAsyncContextManager[httpx.Response]:
        return self.api.chat_completions(payload)
