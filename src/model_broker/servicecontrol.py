"""Client for the host's service-control helper, which starts and stops Compose services.

The broker has no Docker access. For a service that cannot change model while running, it
asks the helper (model-broker-service-control.py) over a Unix socket to start the service's
container with new settings, or to stop it.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from model_broker.backend import BackendError
from model_broker.events import ServiceControlCalled, emit

PROTOCOL_VERSION = 1
JsonObject = dict[str, object]


class ServiceControlError(BackendError):
    """The helper could not be reached, refused a request, or could not carry it out."""


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """A service's container, as the helper reports it.

    ``state`` is Docker's: ``missing`` (no container), ``created``, ``running``,
    ``restarting``, ``paused``, ``exited`` or ``dead``. ``settings`` are the helper-allowed
    environment variables the container was created with, whether or not it is running;
    ``exit_code`` is set once it has stopped.
    """

    state: str
    exit_code: int | None
    oom_killed: bool
    settings: Mapping[str, str]

    @property
    def occupying(self) -> bool:
        """Whether the container holds, or is about to hold, the service's resources."""
        return self.state in ("running", "restarting", "paused")


class ServiceController(Protocol):
    """What a managed service needs of the helper; the real client and test fakes provide it."""

    async def status(self, service: str) -> ServiceStatus: ...

    async def start(self, service: str, settings: Mapping[str, str]) -> ServiceStatus: ...

    async def stop(self, service: str) -> ServiceStatus: ...


def parse_status(document: object) -> ServiceStatus:
    result = cast(JsonObject, document) if isinstance(document, dict) else {}
    state, exit_code, oom = result.get("state"), result.get("exit_code"), result.get("oom_killed")
    settings = result.get("settings")
    # type() rather than isinstance(), because JSON true is also a Python int.
    valid_settings = isinstance(settings, dict) and all(
        isinstance(k, str) and isinstance(v, str)
        for k, v in cast(dict[object, object], settings).items()
    )
    if (
        not isinstance(state, str)
        or (exit_code is not None and type(exit_code) is not int)
        or type(oom) is not bool
        or not valid_settings
    ):
        raise ServiceControlError("service-control status response is malformed")
    return ServiceStatus(state, exit_code, oom, cast(dict[str, str], settings))


@dataclass(frozen=True, slots=True)
class ServiceControl:
    """A client of the helper's socket. ``timeout_seconds`` bounds one request, including the
    time the helper takes to start or stop a container, so it exceeds the helper's own."""

    socket_path: Path
    timeout_seconds: float = 300.0

    async def request(
        self, operation: str, service: str, settings: Mapping[str, str] | None = None
    ) -> ServiceStatus:
        started = time.monotonic()
        error: str | None = None
        try:
            return await self.exchange(operation, service, settings)
        except ServiceControlError as failure:
            error = str(failure)
            raise
        finally:
            emit(
                ServiceControlCalled(
                    operation, service, round(time.monotonic() - started, 3), error
                )
            )

    async def exchange(
        self, operation: str, service: str, settings: Mapping[str, str] | None
    ) -> ServiceStatus:
        message: JsonObject = {
            "version": PROTOCOL_VERSION,
            "id": secrets.token_hex(8),
            "request": operation,
            "service": service,
        }
        if operation == "start":
            message["settings"] = dict(settings or {})
        try:
            async with asyncio.timeout(self.timeout_seconds):
                reader, writer = await asyncio.open_unix_connection(str(self.socket_path))
                try:
                    writer.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
                    await writer.drain()
                    line = await reader.readline()
                finally:
                    writer.close()
        except TimeoutError as error:
            raise ServiceControlError(
                f"service control {operation} {service} timed out after {self.timeout_seconds:g}s"
            ) from error
        except OSError as error:
            raise ServiceControlError(f"service control {self.socket_path}: {error}") from error
        return self.read_response(line, message["id"], operation, service)

    @staticmethod
    def read_response(
        line: bytes, request_id: object, operation: str, service: str
    ) -> ServiceStatus:
        try:
            reply = cast(JsonObject, json.loads(line))
        except ValueError as error:
            raise ServiceControlError("service control sent no valid response") from error
        if reply.get("ok") is not True:
            raise ServiceControlError(
                f"service control refused {operation} {service}: {reply.get('error')}"
            )
        if reply.get("id") != request_id or reply.get("version") != PROTOCOL_VERSION:
            raise ServiceControlError("service control response does not match the request")
        return parse_status(reply.get("result"))

    async def status(self, service: str) -> ServiceStatus:
        return await self.request("status", service)

    async def start(self, service: str, settings: Mapping[str, str]) -> ServiceStatus:
        return await self.request("start", service, settings)

    async def stop(self, service: str) -> ServiceStatus:
        return await self.request("stop", service)
