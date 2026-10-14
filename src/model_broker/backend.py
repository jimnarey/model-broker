"""What the scheduler needs from a model server, whichever one it is."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol

import httpx


class BackendError(RuntimeError):
    """A model server could not complete a broker operation."""


@dataclass(frozen=True, slots=True)
class ModelState:
    """One model's router status.

    ``value`` is ``unloaded``, ``loading``, ``loaded`` or ``sleeping``. A failed load is
    reported as ``unloaded`` with ``failed`` set and the worker's exit code.
    """

    value: str
    failed: bool = False
    exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class RouterProps:
    """The router settings the broker depends on, from ``GET /props``."""

    max_instances: int
    models_autoload: bool
    build: str | None


@dataclass(frozen=True, slots=True)
class ModelActivity:
    """What a loaded model's server reports it is doing.

    ``decode_calls`` counts llama_decode() calls. It rises while the model processes a prompt
    or generates, whereas the token counters only change when a request finishes, so a
    change between two observations shows the model worked in between.
    """

    processing: bool
    deferred: int
    decode_calls: int


class Backend(Protocol):
    """A server that loads and unloads models on request, and serves them.

    ``LlamaAdapter`` is one (a llama.cpp router); ``ManagedService`` is another (a service
    whose model changes by restarting its container). Every method raises ``BackendError``
    when the server cannot be reached or answers unexpectedly.
    """

    @property
    def name(self) -> str:
        """How problems with this server are described."""
        ...

    async def props(self) -> RouterProps: ...

    async def model_states(self) -> dict[str, ModelState]: ...

    async def model_state(self, model_id: str) -> ModelState: ...

    async def activity(self, model_id: str) -> ModelActivity: ...

    async def load(self, model_id: str) -> None:
        """Load the model and wait until it is loaded."""
        ...

    async def unload(self, model_id: str) -> None:
        """Unload the model and wait until it is unloaded."""
        ...

    async def wait_for_state(self, model_id: str, expected: str) -> None: ...

    def chat_completions(
        self, payload: Mapping[str, object]
    ) -> AbstractAsyncContextManager[httpx.Response]:
        """Forward an admitted chat request, yielding the server's response for streaming."""
        ...
