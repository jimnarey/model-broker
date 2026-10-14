"""Tests for scheduling across llama.cpp and Strata, and the application with Strata attached.

The real Scheduler and adapters run against a FakeRouter and a FakeStrata behind one hub
transport. Models on both servers compete for the same GPU and RAM, so a request for one
server's model can mean unloading the other's.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from model_broker.application import Settings, create_app
from model_broker.events import Event, RouterCalled
from model_broker.llama import LlamaAdapter
from model_broker.scheduler import (
    LoadFailed,
    ResourceBusy,
    RouterUnfit,
    Scheduler,
    inspect_router,
)
from model_broker.strata import strata_service

from .fake_router import CATALOGUE, CHAT_REPLY, CUDA0, CUDA1, FakeRouter
from .fake_strata import PLAIN, STRATA_CATALOGUE, STRATA_URL, SWIFT, UNCENSORED, FakeStrata, Hub

CATALOGUE_BOTH = CATALOGUE | STRATA_CATALOGUE
FIXTURES = Path(__file__).parent / "fixtures"


def run(
    hub: Hub,
    body: Callable[[Scheduler], Awaitable[object]],
    protect: float = 0.0,
    max_wait: float = 0.0,
) -> None:
    """Start a scheduler over both fakes, adopting what they have loaded, and run ``body``."""

    async def main() -> None:
        transport = httpx.MockTransport(hub.handle)
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://router.test") as llama,
            httpx.AsyncClient(transport=transport, base_url=STRATA_URL) as api,
        ):
            strata = strata_service(
                hub.strata,
                {model_id: dict(v.env) for model_id, v in STRATA_CATALOGUE.items()},
                LlamaAdapter(api, name="Strata"),
                state_timeout_seconds=60,
            )
            adapters = {
                "llama": LlamaAdapter(llama, 60, poll_interval_seconds=0.005),
                "strata": replace(strata, poll_interval_seconds=0.005),
            }
            view = await inspect_router(adapters, CATALOGUE_BOTH, time.monotonic())
            scheduler = Scheduler(
                adapters,
                CATALOGUE_BOTH,
                view,
                protect_seconds=protect,
                max_wait_seconds=max_wait,
                poll_seconds=0.01,
            )
            await scheduler.start()
            await body(scheduler)

    asyncio.run(main())


def fakes(llama: tuple[str, ...] = (), strata: str | None = None) -> Hub:
    return Hub(FakeRouter(loaded=llama), FakeStrata(running=strata))


async def admitted(scheduler: Scheduler, model: str) -> str:
    async with scheduler.admit(model) as variant:
        return variant.id


def test_strata_request_unloads_an_idle_llama_model_on_its_gpu_first() -> None:
    """Ask for a Strata model while an idle llama.cpp model holds GPU1.

    Expect the router to unload the llama.cpp model before Strata's container is started
    (calls in that order, each to its own server), and only the Strata model loaded."""
    hub = fakes(llama=(CUDA1,))

    async def body(scheduler: Scheduler) -> None:
        assert await admitted(scheduler, PLAIN) == PLAIN
        assert set(scheduler.loaded) == {PLAIN}

    run(hub, body)
    assert hub.calls == [("llama", "unload", CUDA1), ("strata", "start", PLAIN)]


def test_llama_request_unloads_an_idle_strata_model_on_its_gpu_first() -> None:
    """Ask for a llama.cpp GPU1 model while an idle Strata model is running.

    Expect Strata's container to be stopped before the router loads the llama.cpp model."""
    hub = fakes(strata=PLAIN)

    async def body(scheduler: Scheduler) -> None:
        assert await admitted(scheduler, CUDA1) == CUDA1
        assert set(scheduler.loaded) == {CUDA1}

    run(hub, body)
    assert hub.calls == [("strata", "stop", PLAIN), ("llama", "load", CUDA1)]


def test_strata_loads_beside_a_llama_model_on_the_other_gpu() -> None:
    """Ask for a Strata model (GPU1) while a GPU0 llama.cpp model is loaded.

    Expect no unload and both loaded: they share no exclusive resource. The long protection
    window shows it does not matter without a conflict."""
    hub = fakes(llama=(CUDA0,))

    async def body(scheduler: Scheduler) -> None:
        await admitted(scheduler, PLAIN)
        assert set(scheduler.loaded) == {CUDA0, PLAIN}

    run(hub, body, protect=60)
    assert hub.calls == [("strata", "start", PLAIN)]


def test_switching_strata_models_stops_then_starts_the_container() -> None:
    """Ask for the uncensored model while the plain one (same weights) is loaded and idle.

    Expect the plain model unloaded and the uncensored one loaded, both through the
    helper, in that order: Strata cannot hold two, so a switch is stop then start."""
    hub = fakes(strata=PLAIN)

    async def body(scheduler: Scheduler) -> None:
        await admitted(scheduler, UNCENSORED)
        assert set(scheduler.loaded) == {UNCENSORED}

    run(hub, body)
    assert hub.calls == [("strata", "stop", PLAIN), ("strata", "start", UNCENSORED)]


def test_a_loaded_strata_model_is_admitted_without_calls() -> None:
    """Start with a Strata model already running and request it. Expect admission and no
    start or stop: the broker adopts what the helper reports at startup."""
    hub = fakes(strata=SWIFT)

    async def body(scheduler: Scheduler) -> None:
        assert await admitted(scheduler, SWIFT) == SWIFT

    run(hub, body, protect=60)
    assert hub.calls == []


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        pytest.param(lambda strata: setattr(strata, "busy", 1), "queued in Strata", id="working"),
        pytest.param(
            lambda strata: setattr(strata, "observable", False),
            "cannot see whether",
            id="unobservable",
        ),
    ],
)
def test_busy_strata_model_blocks_a_conflicting_llama_request(
    setup: Callable[[FakeStrata], Any], reason: str
) -> None:
    """Request a llama.cpp GPU1 model while Strata is running a request, or cannot be asked.

    Setup: Strata's /v1/status reports one request in flight, or fails. Expect a refusal
    naming Strata (not llama.cpp) and nothing unloaded: a failed observation counts as busy."""
    hub = fakes(strata=PLAIN)
    setup(hub.strata)

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(ResourceBusy, match=reason):
            await admitted(scheduler, CUDA1)
        assert set(scheduler.loaded) == {PLAIN}

    run(hub, body)
    assert hub.calls == []


def test_strata_request_count_advancing_counts_as_recent_use() -> None:
    """Work that reached Strata directly restarts the protection window when it is noticed.

    Setup: a 0.3 s window that has passed. Strata is busy at the first request (refused, and
    the count 10 recorded). It is then idle but the count is 50, so it worked in between and
    the request is refused as recently used. After another 0.35 s unchanged, it proceeds."""
    hub = fakes(strata=PLAIN)

    async def body(scheduler: Scheduler) -> None:
        await asyncio.sleep(0.35)
        hub.strata.busy, hub.strata.requests = 1, 10
        with pytest.raises(ResourceBusy, match="working"):
            await admitted(scheduler, CUDA1)
        hub.strata.busy, hub.strata.requests = 0, 50
        with pytest.raises(ResourceBusy, match="used recently"):
            await admitted(scheduler, CUDA1)
        await asyncio.sleep(0.35)
        assert await admitted(scheduler, CUDA1) == CUDA1

    run(hub, body, protect=0.3)


def test_failed_strata_start_is_an_error_after_the_llama_model_has_gone() -> None:
    """The helper refuses to start the Strata model that a llama.cpp unload made room for.

    Expect LoadFailed carrying the helper's refusal, the llama.cpp model unloaded, and nothing
    left loaded: the container was never started, so no resources are held."""
    hub = fakes(llama=(CUDA1,))
    hub.strata.fail_helper = frozenset({"start"})

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(LoadFailed, match="refused start strata"):
            await admitted(scheduler, PLAIN)
        assert scheduler.loaded == {}

    run(hub, body)
    assert hub.calls == [("llama", "unload", CUDA1)]


def test_a_strata_container_that_exits_while_starting_fails_the_load() -> None:
    """Start a Strata model whose container exits with code 1 (say its model files are gone).

    Expect LoadFailed naming the exit code, and nothing left loaded, promptly: the broker
    sees the container has stopped and does not wait out the load timeout."""
    hub = fakes()
    hub.strata.exits_with = 1

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(LoadFailed, match=r"exited \(1\)"):
            await admitted(scheduler, PLAIN)
        assert scheduler.loaded == {}

    run(hub, body)


def test_both_servers_holding_conflicting_models_is_a_problem() -> None:
    """Start with a llama.cpp GPU1 model and a Strata model both loaded.

    Each server alone is consistent; only the broker sees that they claim GPU1 together.
    Expect a problem naming both, and requests refused (the broker never allows this state)."""
    hub = fakes(llama=(CUDA1,), strata=PLAIN)

    async def body(scheduler: Scheduler) -> None:
        assert any(
            "different servers" in p and CUDA1 in p and PLAIN in p for p in scheduler.view.problems
        )
        with pytest.raises(RouterUnfit):
            await admitted(scheduler, CUDA0)

    run(hub, body)


def test_unreachable_strata_stops_all_scheduling_until_it_is_back() -> None:
    """Take the service-control helper down at startup and ask for a llama.cpp model on the
    other GPU.

    Expect a problem naming Strata and a refusal, even though CUDA0 does not conflict with
    Strata's GPU1: with the helper unreachable the broker cannot tell what Strata holds. Once
    it is back, the next request re-inspects and is admitted."""
    hub = fakes()
    hub.strata.helper_down = True

    async def body(scheduler: Scheduler) -> None:
        assert scheduler.view.problems[0].startswith("Strata is unavailable")
        with pytest.raises(RouterUnfit, match="Strata is unavailable"):
            await admitted(scheduler, CUDA0)
        hub.strata.helper_down = False
        assert await admitted(scheduler, CUDA0) == CUDA0

    run(hub, body)


def test_a_strata_model_found_loading_at_startup_is_waited_for() -> None:
    """Start with Strata's container running but not yet answering, then request the model.

    Expect the request to be admitted once Strata answers, with no start call of the broker's
    own: the container is adopted, not restarted."""
    hub = fakes(strata=PLAIN)
    hub.strata.ready = False

    async def body(scheduler: Scheduler) -> None:
        request = asyncio.create_task(admitted(scheduler, PLAIN))
        await asyncio.sleep(0.05)
        assert not request.done()
        hub.strata.ready = True
        assert await request == PLAIN

    run(hub, body)
    assert hub.calls == []


# The application with Strata attached

PRESET = FIXTURES / "models-preset.ini"


def settings(**overrides: Any) -> Settings:
    return Settings(
        "http://router.test",
        PRESET,
        idle_protect_seconds=0,
        max_wait_seconds=0,
        service_preset_path=FIXTURES / "strata-preset.ini",
        strata_url=STRATA_URL,
        **overrides,
    )


def with_broker[T](
    hub: Hub, action: Callable[[httpx.AsyncClient], Awaitable[T]], config: Settings | None = None
) -> T:
    """Start a broker over both fakes, run ``action`` against it through ASGI, then stop it."""

    async def main() -> T:
        app = create_app(
            config or settings(), httpx.MockTransport(hub.handle), service_control=hub.strata
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://broker.test"
            ) as client:
                return await action(client)

    return asyncio.run(main())


def test_models_and_health_cover_both_catalogues() -> None:
    """GET /v1/models and /health with Strata configured.

    Expect every llama.cpp and every Strata model listed (5 + 3), health ok, and health
    listing the managed service."""
    hub = fakes()

    async def action(client: httpx.AsyncClient) -> tuple[list[str], dict[str, Any]]:
        models = (await client.get("/v1/models")).json()["data"]
        return [m["id"] for m in models], (await client.get("/health")).json()

    ids, health = with_broker(hub, action)
    assert (sorted(ids), health["status"], health["services"]) == (
        sorted(CATALOGUE_BOTH),
        "ok",
        ["strata"],
    )


def test_chat_for_a_strata_model_swaps_models_and_is_served_by_strata() -> None:
    """POST a chat request for a Strata model while a llama.cpp model holds GPU1.

    Expect 200 with Strata's reply, the payload delivered unchanged to Strata's server and
    not to the router, and the llama.cpp model unloaded before the Strata container was
    started with the model's settings."""
    hub = fakes(llama=(CUDA1,))
    payload = {"model": PLAIN, "messages": [{"role": "user", "content": "hi"}]}

    async def action(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post("/v1/chat/completions", json=payload)

    response = with_broker(hub, action)
    assert (response.status_code, response.json()) == (200, CHAT_REPLY)
    assert (hub.strata.chat_payloads, hub.router.chat_payloads) == ([payload], [])
    assert hub.calls == [("llama", "unload", CUDA1), ("strata", "start", PLAIN)]
    assert hub.strata.settings == dict(STRATA_CATALOGUE[PLAIN].env)


def test_chat_for_a_llama_model_still_goes_to_the_router() -> None:
    """POST a chat request for a llama.cpp model with Strata attached.

    Expect the router to receive it and Strata's server to receive nothing."""
    hub = fakes(llama=(CUDA0,))
    payload = {"model": CUDA0, "messages": []}

    async def action(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post("/v1/chat/completions", json=payload)

    assert with_broker(hub, action).status_code == 200
    assert (hub.router.chat_payloads, hub.strata.chat_payloads) == ([payload], [])


def test_strata_api_key_is_sent_with_every_request_to_stratas_server() -> None:
    """Configure a Strata API key and chat with a model Strata is already running.

    Expect the bearer key on every request the broker makes to Strata's server (the readiness
    check at startup, and the chat), and none to the router."""
    hub = fakes(strata=PLAIN)

    async def action(client: httpx.AsyncClient) -> httpx.Response:
        return await client.post("/v1/chat/completions", json={"model": PLAIN})

    with_broker(hub, action, settings(strata_api_key="secret"))
    assert len(hub.strata.api_headers) >= 2
    assert {h["authorization"] for h in hub.strata.api_headers} == {"Bearer secret"}


def test_router_calls_are_logged_with_their_server(events: Callable[[], list[Event]]) -> None:
    """Start the broker with Strata running at DEBUG and read the RouterCalled events.

    Expect startup's calls tagged by server: llama.cpp's /props and /models, and Strata's
    /health (readiness) and /v1/status (the activity baseline) for the model it is running.
    Both kinds of server can answer the same paths, so the log needs the tag."""
    hub = fakes(strata=PLAIN)
    with_broker(hub, lambda client: client.get("/health"), settings(log_level="DEBUG"))
    calls = sorted(
        (e.backend, e.path) for e in events() if isinstance(e, RouterCalled) and e.method == "GET"
    )
    assert calls == [
        ("llama", "/models"),
        ("llama", "/props"),
        ("strata", "/health"),
        ("strata", "/v1/status"),
    ]


def test_service_preset_that_reuses_a_llama_model_id_stops_startup(tmp_path: Path) -> None:
    """Start the broker with a service preset that defines a llama.cpp model's ID.

    Expect startup to fail naming the clash, as for any invalid preset."""
    clash = tmp_path / "services.ini"
    clash.write_text(f"version = 1\n[{CUDA1}]\nservice = strata\ndevice = CUDA1\n")
    config = replace(settings(), service_preset_path=clash)
    with pytest.raises(Exception, match="more than one preset"):
        with_broker(fakes(), lambda client: client.get("/health"), config)


def test_a_preset_naming_an_unknown_service_stops_startup(tmp_path: Path) -> None:
    """Start the broker with a section for a service it has no client for.

    Expect startup to fail naming the service: its models would otherwise be listed and could
    never be served."""
    unknown = tmp_path / "services.ini"
    unknown.write_text("version = 1\n[x--cuda1]\nservice = comfyui\ndevice = CUDA1\n")
    config = replace(settings(), service_preset_path=unknown)
    with pytest.raises(ValueError, match="comfyui"):
        with_broker(fakes(), lambda client: client.get("/health"), config)
