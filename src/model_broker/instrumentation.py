"""Automatic events for HTTP traffic: every broker request, and every call to the router."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import cast

import httpx
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from model_broker.events import (
    REQUEST,
    HealthChecked,
    RequestCancelled,
    RequestCompleted,
    RequestContext,
    RequestFailed,
    RequestReceived,
    RouterCalled,
    emit,
    seconds_since,
)

HEALTH_PATH = "/health"
STARTED = "model_broker_started"


class RequestEvents:
    """ASGI middleware giving each broker request an ID, and reporting how it ended.

    It sets ``REQUEST`` for the request, so every event emitted while handling it, and by
    background work it starts, carries the request ID and, once authenticated, the client.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method, path = cast(str, scope["method"]), cast(str, scope["path"])
        token = REQUEST.set(RequestContext(uuid.uuid4().hex[:8]))
        started = time.monotonic()
        status = 500

        async def send_and_note_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = cast(int, message["status"])
            await send(message)

        emit(RequestReceived(method, path))
        try:
            await self.app(scope, receive, send_and_note_status)
        except asyncio.CancelledError:
            emit(RequestCancelled(method, path, seconds_since(started)))
            raise
        except Exception as error:
            emit(RequestFailed(method, path, repr(error), seconds_since(started)))
            raise
        else:
            if path == HEALTH_PATH:
                emit(HealthChecked(status, seconds_since(started)))
            else:
                emit(RequestCompleted(method, path, status, seconds_since(started)))
        finally:
            REQUEST.reset(token)


def requested_model(request: httpx.Request) -> str | None:
    """The model a router call is about: its ``model`` query parameter or JSON field.

    Only the model name is taken from the body; nothing else in it is logged.
    """
    if "model" in request.url.params:
        return request.url.params["model"]
    try:
        body: object = json.loads(request.content) if request.content else None
    except (ValueError, httpx.RequestNotRead):
        return None
    model = cast(dict[str, object], body).get("model") if isinstance(body, dict) else None
    return model if isinstance(model, str) else None


async def note_start(request: httpx.Request) -> None:
    request.extensions[STARTED] = time.monotonic()


async def report_call(response: httpx.Response) -> None:
    request = response.request
    started = cast(float, request.extensions.get(STARTED, time.monotonic()))
    emit(
        RouterCalled(
            request.method,
            request.url.path,
            requested_model(request),
            response.status_code,
            seconds_since(started),
        )
    )


ROUTER_EVENT_HOOKS = {"request": [note_start], "response": [report_call]}
"""httpx event hooks for the router client: one ``RouterCalled`` event per call."""
