"""FakeRouter: an in-memory llama.cpp router for tests, behind httpx.MockTransport.

It serves the real preset fixture's models through the endpoints the broker uses.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from model_broker.catalogue import load_catalogue

# Real sections from the unified preset.
CATALOGUE = load_catalogue(Path(__file__).parent / "fixtures" / "models-preset.ini")
CUDA0 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda0"
CUDA1 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda1"
MOE_CUDA1 = "GLM-4.5-Air-UD-Q4_K_XL--cuda1"
DUAL = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0--cuda0-cuda1"
GOOD_PROPS = {"role": "router", "max_instances": 3, "models_autoload": False, "build_info": "b1"}


CHAT_REPLY = {"object": "chat.completion", "choices": [{"message": {"content": "hello"}}]}
STREAM_CHUNKS = [b'data: {"delta": "hel"}\n\n', b'data: {"delta": "lo"}\n\n', b"data: [DONE]\n\n"]


class FakeRouter:
    """An in-memory llama.cpp router with the endpoints the broker uses.

    Loads and unloads finish at once unless ``loads_finish`` or ``unloads_finish`` is
    cleared, in which case the model reports ``loading``, or stays ``loaded``, until it is
    set. ``fail`` holds (action, model) pairs whose control request is refused with a 500.
    """

    def __init__(self, loaded: tuple[str, ...] = (), loading: tuple[str, ...] = ()) -> None:
        self.props = dict(GOOD_PROPS)
        self.states = {model: "loaded" if model in loaded else "unloaded" for model in CATALOGUE}
        self.states.update(dict.fromkeys(loading, "loading"))
        self.processing: set[str] = set()
        self.deferred: dict[str, int] = {}
        self.decode_calls: dict[str, int] = {}
        self.observable = True
        self.fail: set[tuple[str, str]] = set()
        self.calls: list[tuple[str, str]] = []
        self.loads_finish = asyncio.Event()
        self.loads_finish.set()
        self.unloads_finish = asyncio.Event()
        self.unloads_finish.set()
        self.unloading: set[str] = set()
        self.chat_payloads: list[dict[str, Any]] = []
        self.chat_error: tuple[int, dict[str, Any]] | None = None
        self.chats_flow = asyncio.Event()
        self.chats_flow.set()
        self.break_stream = False
        self.open_chats = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/props":
            return httpx.Response(200, json=self.props)
        if path == "/models":
            if self.loads_finish.is_set():
                self.states = {m: "loaded" if s == "loading" else s for m, s in self.states.items()}
            if self.unloads_finish.is_set():
                self.states.update(dict.fromkeys(self.unloading, "unloaded"))
                self.unloading.clear()
            data = [{"id": m, "status": {"value": s}} for m, s in self.states.items()]
            return httpx.Response(200, json={"data": data})
        if path in ("/slots", "/metrics"):
            return self.observe(path, request.url.params["model"])
        if path == "/v1/chat/completions":
            return self.chat(request)
        action, model = path.removeprefix("/models/"), json.loads(request.content)["model"]
        self.calls.append((action, model))
        if (action, model) in self.fail:
            return httpx.Response(500, text="refused")
        if action == "load":
            self.states[model] = "loading"
        else:
            self.unloading.add(model)
        return httpx.Response(200, json={"success": True})

    def observe(self, path: str, model: str) -> httpx.Response:
        if not self.observable:
            return httpx.Response(500, text="unavailable")
        if path == "/slots":
            return httpx.Response(200, json=[{"id": 0, "is_processing": model in self.processing}])
        metrics = (
            f"llamacpp:requests_processing {int(model in self.processing)}\n"
            f"llamacpp:requests_deferred {self.deferred.get(model, 0)}\n"
            f"llamacpp:n_decode_total {self.decode_calls.get(model, 0)}\n"
        )
        return httpx.Response(200, text=metrics)

    # Chat completions

    def chat(self, request: httpx.Request) -> httpx.Response:
        """Answer a chat request with JSON, or with an SSE stream if it asks to stream.

        A stream sends ``STREAM_CHUNKS``, pausing before each while ``chats_flow`` is
        cleared, so a test can hold it open. ``open_chats`` counts streams not yet closed,
        which shows whether the broker closed the router request.
        """
        payload = json.loads(request.content)
        self.chat_payloads.append(payload)
        if self.chat_error is not None:
            return httpx.Response(self.chat_error[0], json=self.chat_error[1])
        if not payload.get("stream"):
            return httpx.Response(200, json=CHAT_REPLY)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=self.stream()
        )

    async def stream(self) -> AsyncIterator[bytes]:
        self.open_chats += 1
        try:
            for chunk in STREAM_CHUNKS:
                await self.chats_flow.wait()
                if self.break_stream:
                    raise httpx.ReadError("router connection lost")
                yield chunk
        finally:
            self.open_chats -= 1
