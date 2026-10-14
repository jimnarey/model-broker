"""The broker's log: typed events, written as one JSON object per line.

Code reports what happened by emitting an event, a small dataclass of facts, rather than
writing a message. Each event class sets its own level, and its name in snake case is the
event name in the log, so the wording and levels of the whole log live in this module.

Events emitted while a broker request is being handled carry that request's ID and client
name, from ``RequestContext``. Background work started by a request, such as a model load,
inherits them, so it can be traced back to the request that caused it.

Events never carry prompts, completions, or API keys: no event has a field for them.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

LOGGER = logging.getLogger("model_broker")
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(slots=True)
class RequestContext:
    """The broker request being handled: its ID, and the client once authenticated."""

    request_id: str
    client: str | None = None


REQUEST: ContextVar[RequestContext | None] = ContextVar("model_broker_request", default=None)


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for log events. Subclasses are frozen dataclasses of the event's facts."""

    level: ClassVar[int] = logging.INFO

    @classmethod
    def name(cls) -> str:
        return snake_case(cls.__name__)


def snake_case(name: str) -> str:
    """``ModelLoaded`` becomes ``model_loaded``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def emit(event: Event) -> None:
    if LOGGER.isEnabledFor(event.level):
        LOGGER.log(event.level, event.name(), extra={"event": event, "request": REQUEST.get()})


class JsonFormatter(logging.Formatter):
    """Format a record as one JSON object: time, level, event, request, then the facts.

    Records that are not events (from libraries, for example) keep their message and add
    the logger's name.
    """

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        event = getattr(record, "event", None)
        request = getattr(record, "request", None)
        if isinstance(request, RequestContext):
            line |= dataclasses.asdict(request)
        if isinstance(event, Event):
            line |= dataclasses.asdict(event)
        else:
            line["logger"] = record.name
        if record.exc_info:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def configure_logging(level: str) -> None:
    """Write the broker's events to stderr as JSON lines, at ``level`` and above.

    Replaces any handler this function installed before, so it can be called again.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.set_name("model_broker_json")
    for existing in [h for h in LOGGER.handlers if h.get_name() == handler.get_name()]:
        LOGGER.removeHandler(existing)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def seconds_since(started: float) -> float:
    return round(time.monotonic() - started, 3)


# Startup and configuration


@dataclass(frozen=True, slots=True)
class BrokerStarted(Event):
    """The broker validated its preset, inspected the router, and is serving."""

    models: int
    loaded: list[str]
    router_build: str | None


@dataclass(frozen=True, slots=True)
class AuthenticationOff(Event):
    """No API keys file is configured, so every request is accepted."""

    level = logging.WARNING


@dataclass(frozen=True, slots=True)
class AuthenticationOn(Event):
    clients: int


@dataclass(frozen=True, slots=True)
class ModelAdopted(Event):
    """The router already held this model when the broker started or checked it again."""

    model: str
    state: str


@dataclass(frozen=True, slots=True)
class RouterProblemsFound(Event):
    """The broker will not schedule until these router problems are resolved."""

    level = logging.WARNING
    problems: list[str]


@dataclass(frozen=True, slots=True)
class RouterProblemsCleared(Event):
    """A fresh check of the router found no problems; scheduling resumes."""

    router_build: str | None


# Scheduling


@dataclass(frozen=True, slots=True)
class RequestAdmitted(Event):
    level = logging.DEBUG
    model: str
    waited_seconds: float


@dataclass(frozen=True, slots=True)
class RequestWaiting(Event):
    """A request cannot be placed yet. Emitted when the reason first appears or changes."""

    level = logging.DEBUG
    model: str
    reason: str


@dataclass(frozen=True, slots=True)
class RequestRefused(Event):
    """The scheduler refused a request: ``cause`` is the kind of refusal."""

    model: str
    cause: str
    detail: str
    waited_seconds: float


@dataclass(frozen=True, slots=True)
class RequestFinished(Event):
    """An admitted request's response ended, however it ended."""

    level = logging.DEBUG
    model: str
    active_seconds: float


@dataclass(frozen=True, slots=True)
class SwapStarted(Event):
    """The scheduler decided to unload ``unloading`` and then load ``model``."""

    model: str
    unloading: list[str]


@dataclass(frozen=True, slots=True)
class ModelUnloaded(Event):
    model: str
    seconds: float


@dataclass(frozen=True, slots=True)
class ModelUnloadFailed(Event):
    """The model keeps its resources; the swap that needed it did not happen."""

    level = logging.ERROR
    model: str
    error: str


@dataclass(frozen=True, slots=True)
class ModelLoaded(Event):
    model: str
    seconds: float


@dataclass(frozen=True, slots=True)
class ModelLoadFailed(Event):
    """``released`` says whether the router confirmed the model unloaded, freeing its
    resources; if not, it keeps them until the router is checked again."""

    level = logging.ERROR
    model: str
    error: str
    released: bool


@dataclass(frozen=True, slots=True)
class ModelObserved(Event):
    """What llama.cpp reported about a model the scheduler considered unloading."""

    level = logging.DEBUG
    model: str
    processing: bool
    deferred: int
    decode_calls: int
    recently_active: bool


@dataclass(frozen=True, slots=True)
class ModelUnobservable(Event):
    """llama.cpp could not report on a model, so it counts as busy."""

    level = logging.DEBUG
    model: str
    error: str


@dataclass(frozen=True, slots=True)
class BaselineRecorded(Event):
    """The decode count against which later unseen work is detected; None if unreadable."""

    level = logging.DEBUG
    model: str
    decode_calls: int | None


# Proxying


@dataclass(frozen=True, slots=True)
class ClientDisconnected(Event):
    """The client left before the router's response finished; the router request was
    closed, which stops llama.cpp working on it."""

    model: str
    seconds: float


@dataclass(frozen=True, slots=True)
class ProxyFailed(Event):
    """The router could not be reached, or its response broke off part-way."""

    level = logging.ERROR
    model: str
    error: str


@dataclass(frozen=True, slots=True)
class ProxyFinished(Event):
    """The router's whole response reached the client."""

    level = logging.DEBUG
    model: str
    status: int
    bytes: int
    seconds: float


# HTTP


@dataclass(frozen=True, slots=True)
class RequestReceived(Event):
    level = logging.DEBUG
    method: str
    path: str


@dataclass(frozen=True, slots=True)
class RequestCompleted(Event):
    method: str
    path: str
    status: int
    seconds: float


@dataclass(frozen=True, slots=True)
class HealthChecked(Event):
    """A /health request, logged at DEBUG so that frequent health checks stay quiet."""

    level = logging.DEBUG
    status: int
    seconds: float


@dataclass(frozen=True, slots=True)
class RequestCancelled(Event):
    """The client went away before the response finished."""

    method: str
    path: str
    seconds: float


@dataclass(frozen=True, slots=True)
class RequestFailed(Event):
    level = logging.ERROR
    method: str
    path: str
    error: str
    seconds: float


@dataclass(frozen=True, slots=True)
class ServiceControlCalled(Event):
    """One request from the broker to the host's service-control helper, which starts and
    stops a service's container. ``error`` is set when the request failed."""

    level = logging.DEBUG
    operation: str
    service: str
    seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RouterCalled(Event):
    """One HTTP call from the broker to a model server (``backend`` is ``llama`` or
    ``strata``); for a stream, ``seconds`` is the time to the response headers."""

    level = logging.DEBUG
    method: str
    path: str
    model: str | None
    status: int
    seconds: float
    backend: str = "llama"
