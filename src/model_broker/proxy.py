"""Stream an admitted chat request to the router and the router's response to the client."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from contextlib import AsyncExitStack

import anyio
import httpx
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from model_broker.events import ClientDisconnected, ProxyFailed, ProxyFinished, emit
from model_broker.llama import LlamaAdapter, LlamaError

# Headers not relayed: those describing the router's connection rather than the response,
# and content-encoding, because the body is relayed decoded.
HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "content-length",
        "content-encoding",
        "date",
        "server",
    }
)


def openai_error_body(message: str, error_type: str, code: str) -> bytes:
    error = {"message": message, "type": error_type, "param": None, "code": code}
    return json.dumps({"error": error}).encode()


class ProxiedChat(Response):
    """ASGI response for one admitted chat request.

    It sends the request to the router, then relays the router's status, headers and body
    unchanged as they arrive, so streams stay streams and router errors reach the client.
    Meanwhile it watches for the client leaving, and if it does, stops at once and closes
    the router request. However it ends, it closes ``resources`` (the router request and the
    scheduler admission), shielded from cancellation, so the admission is always released.

    It subclasses Response only so that FastAPI passes it through; ``__call__`` does all the
    sending.
    """

    def __init__(
        self,
        adapter: LlamaAdapter,
        payload: Mapping[str, object],
        model: str,
        resources: AsyncExitStack,
    ) -> None:
        super().__init__()
        # Set just before the last part of the body is sent. The server then reports the
        # connection as finished, which is not the client leaving early.
        self.finishing = False
        self.adapter = adapter
        self.payload = payload
        self.model = model
        self.resources = resources

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = time.monotonic()
        disconnected = False
        try:
            async with anyio.create_task_group() as group:

                async def watch_for_disconnect() -> None:
                    nonlocal disconnected
                    while (await receive())["type"] != "http.disconnect":
                        pass
                    if not self.finishing:
                        disconnected = True
                        group.cancel_scope.cancel()

                group.start_soon(watch_for_disconnect)
                await self.relay(send, started)
                group.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                await self.resources.aclose()
            if disconnected:
                emit(ClientDisconnected(self.model, round(time.monotonic() - started, 3)))

    async def relay(self, send: Send, started: float) -> None:
        try:
            upstream = await self.resources.enter_async_context(
                self.adapter.chat_completions(self.payload)
            )
        except LlamaError as error:
            emit(ProxyFailed(self.model, str(error)))
            body = openai_error_body(str(error), "server_error", "router_unavailable")
            self.finishing = True
            await send_whole(send, 502, [(b"content-type", b"application/json")], body)
            return
        headers = [
            (name.encode(), value.encode())
            for name, value in upstream.headers.items()
            if name.lower() not in HOP_HEADERS
        ]
        await send(
            {"type": "http.response.start", "status": upstream.status_code, "headers": headers}
        )
        sent = 0
        try:
            async for chunk in upstream.aiter_bytes():
                sent += len(chunk)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        except httpx.HTTPError as error:
            # The status has been sent, so the client can only see the body end early.
            emit(ProxyFailed(self.model, str(error)))
        else:
            seconds = round(time.monotonic() - started, 3)
            emit(ProxyFinished(self.model, upstream.status_code, sent, seconds))
        self.finishing = True
        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def send_whole(
    send: Send, status: int, headers: list[tuple[bytes, bytes]], body: bytes
) -> None:
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})
