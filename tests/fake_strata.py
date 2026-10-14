"""FakeStrata: an in-memory Strata container and API for tests.

``FakeStrata`` stands in for both halves of a managed service. As the service-control
helper's client (``status``, ``start`` and ``stop``) it models a container: its Docker state
and the environment it was created with. As Strata's own server (``handle``, behind
httpx.MockTransport) it answers ``/health`` once the container is running and ``ready``, plus
``/v1/status`` and chat. ``Hub`` serves a FakeRouter and a FakeStrata from one transport and
logs the load and unload calls of both, in order.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import httpx

from model_broker.catalogue import load_service_catalogue
from model_broker.servicecontrol import ServiceControlError, ServiceStatus

from .fake_router import FakeRouter

STRATA_CATALOGUE = load_service_catalogue(Path(__file__).parent / "fixtures" / "strata-preset.ini")
PLAIN, UNCENSORED, SWIFT = STRATA_CATALOGUE
STRATA_URL = "http://strata.test:8080"
ENV_TO_MODEL = {tuple(sorted(v.env.items())): model_id for model_id, v in STRATA_CATALOGUE.items()}


class FakeStrata(FakeRouter):
    """A Strata container and its API.

    It subclasses FakeRouter only for the chat endpoint and its recording. ``state`` is the
    container's Docker state; ``settings`` the environment it was created with. While the
    container is running, ``/health`` answers only once ``ready`` is set. ``busy`` and
    ``requests`` are what ``/v1/status`` reports. A start that finds ``exits_with`` set leaves
    the container exited with that code, and ``fail`` holds ("start" | "stop") operations
    the helper refuses.
    """

    def __init__(self, running: str | None = None) -> None:
        super().__init__()
        self.state = "missing"
        self.exit_code: int | None = None
        self.settings: dict[str, str] = {}
        self.ready = True
        self.exits_with: int | None = None
        self.helper_down = False
        self.busy = 0
        self.requests = 0
        self.api_headers: list[httpx.Headers] = []
        self.log: list[tuple[str, str, str]] | None = None
        self.control_calls: list[tuple[str, str]] = []
        if running is not None:
            self.state, self.settings = "running", dict(STRATA_CATALOGUE[running].env)

    def record(self, action: str) -> None:
        model = ENV_TO_MODEL.get(tuple(sorted(self.settings.items())), "?")
        self.control_calls.append((action, model))
        if self.log is not None:
            self.log.append(("strata", action, model))

    def snapshot(self) -> ServiceStatus:
        gone = self.state in ("exited", "dead", "created")
        return ServiceStatus(
            self.state, self.exit_code if gone else None, False, dict(self.settings)
        )

    # The service-control helper's client

    async def status(self, service: str) -> ServiceStatus:
        if self.helper_down:
            raise ServiceControlError("service control /run/x.sock: connection refused")
        return self.snapshot()

    async def start(self, service: str, settings: Mapping[str, str]) -> ServiceStatus:
        if self.helper_down or ("start" in self.fail_helper):
            raise ServiceControlError("service control refused start strata: stub refusal")
        self.settings = dict(settings)
        self.record("start")
        if self.exits_with is None:
            self.state, self.exit_code = "running", None
        else:
            self.state, self.exit_code = "exited", self.exits_with
        return self.snapshot()

    async def stop(self, service: str) -> ServiceStatus:
        if self.helper_down or ("stop" in self.fail_helper):
            raise ServiceControlError("service control refused stop strata: stub refusal")
        self.record("stop")
        if self.state != "missing":
            # Docker reports a container it had to kill as exit code 137.
            self.state, self.exit_code = "exited", 137
        return self.snapshot()

    fail_helper: frozenset[str] = frozenset()

    # Strata's own server

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.api_headers.append(request.headers)
        if request.url.path == "/health":
            up = self.state == "running" and self.ready
            return httpx.Response(200 if up else 503, json={})
        if request.url.path == "/v1/status":
            if not self.observable:
                return httpx.Response(500, text="unavailable")
            activity = {"in_flight": self.busy, "requests": self.requests}
            return httpx.Response(200, json={"service": "strata", "activity": activity})
        if request.url.path == "/v1/chat/completions":
            return self.chat(request)
        return httpx.Response(404)


class Hub:
    """Send requests for strata.test to a FakeStrata and all others to a FakeRouter.

    ``calls`` lists the load and unload requests to either, in order, as (server, action,
    model), the server being ``llama`` or ``strata``.
    """

    def __init__(self, router: FakeRouter, strata: FakeStrata) -> None:
        self.router, self.strata = router, strata
        self.calls: list[tuple[str, str, str]] = []
        strata.log = self.calls

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "strata.test":
            return self.strata.handle(request)
        before = len(self.router.calls)
        response = self.router.handle(request)
        if len(self.router.calls) > before:
            action, model = self.router.calls[-1]
            self.calls.append(("llama", action, model))
        return response
