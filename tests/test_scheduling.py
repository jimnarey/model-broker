"""Tests for Scheduler admission: the busy rules, waiting, ordering, and failures.

Each test runs a real Scheduler and LlamaAdapter against FakeRouter, an in-memory router
behind httpx.MockTransport. Times are real but short: protection windows and waits are
fractions of a second.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from model_broker.catalogue import load_catalogue
from model_broker.events import (
    Event,
    ModelLoaded,
    ModelLoadFailed,
    ModelObserved,
    ModelUnloaded,
    RequestAdmitted,
    RequestFinished,
    RequestRefused,
    RequestWaiting,
    RouterProblemsFound,
    SwapStarted,
)
from model_broker.llama import LlamaAdapter
from model_broker.scheduler import (
    BUSY_RETRY_AFTER_SECONDS,
    LoadFailed,
    ResourceBusy,
    RouterUnfit,
    Scheduler,
    SchedulerError,
    UnknownModel,
    inspect_router,
)

# Real sections from the unified preset.
CATALOGUE = load_catalogue(Path(__file__).parent / "fixtures" / "models-preset.ini")
CUDA0 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda0"
CUDA1 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda1"
MOE_CUDA1 = "GLM-4.5-Air-UD-Q4_K_XL--cuda1"
DUAL = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0--cuda0-cuda1"
GOOD_PROPS = {"role": "router", "max_instances": 3, "models_autoload": False, "build_info": "b1"}


class FakeRouter:
    """An in-memory llama.cpp router with the endpoints the scheduler uses.

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


def run(
    router: FakeRouter,
    body: Callable[[Scheduler], Awaitable[object]],
    protect: float = 0.0,
    max_wait: float = 0.0,
    load_timeout: float = 60.0,
) -> None:
    """Start a scheduler against ``router``, adopting its loaded models, and run ``body``."""

    async def main() -> None:
        transport = httpx.MockTransport(router.handle)
        async with httpx.AsyncClient(transport=transport, base_url="http://router.test") as client:
            adapter = LlamaAdapter(client, load_timeout, poll_interval_seconds=0.005)
            view = await inspect_router(adapter, CATALOGUE, time.monotonic())
            scheduler = Scheduler(
                adapter,
                CATALOGUE,
                view,
                protect_seconds=protect,
                max_wait_seconds=max_wait,
                poll_seconds=0.01,
            )
            await scheduler.start()
            await body(scheduler)

    asyncio.run(main())


async def hold(scheduler: Scheduler, model: str, until: asyncio.Event) -> None:
    """Keep one request admitted to ``model`` until ``until`` is set, like a long stream."""
    async with scheduler.admit(model):
        await until.wait()


async def admitted(scheduler: Scheduler, model: str) -> str:
    async with scheduler.admit(model) as variant:
        return variant.id


# Admission without conflict


def test_loaded_model_is_admitted_without_router_calls() -> None:
    """A request for a loaded model is admitted at once and counted active while it runs."""
    router = FakeRouter(loaded=(CUDA0,))

    async def body(scheduler: Scheduler) -> None:
        async with scheduler.admit(CUDA0) as variant:
            assert variant.id == CUDA0
            assert scheduler.loaded[CUDA0].active == 1
        assert scheduler.loaded[CUDA0].active == 0

    run(router, body, protect=60)
    assert router.calls == []


def test_model_on_another_gpu_loads_alongside() -> None:
    """A GPU1 model loads next to a loaded GPU0 model; nothing is unloaded.

    The protection window is long, to show that it does not matter without a conflict.
    """
    router = FakeRouter(loaded=(CUDA0,))

    async def body(scheduler: Scheduler) -> None:
        assert await admitted(scheduler, CUDA1) == CUDA1
        assert set(scheduler.loaded) == {CUDA0, CUDA1}

    run(router, body, protect=60)
    assert router.calls == [("load", CUDA1)]


def test_idle_conflicting_model_is_unloaded_first() -> None:
    """An idle GPU1 model is unloaded, and only then is the requested GPU1 model loaded."""
    router = FakeRouter(loaded=(CUDA1,))

    async def body(scheduler: Scheduler) -> None:
        await admitted(scheduler, MOE_CUDA1)
        assert set(scheduler.loaded) == {MOE_CUDA1}

    run(router, body)
    assert router.calls == [("unload", CUDA1), ("load", MOE_CUDA1)]


# The busy rules


@pytest.mark.parametrize(
    ("setup", "protect", "reason"),
    [
        (lambda router: router.processing.add(CUDA1), 0.0, "working or has requests queued"),
        (lambda router: router.deferred.update({CUDA1: 2}), 0.0, "requests queued in llama"),
        (lambda router: None, 60.0, "was used recently"),
        (lambda router: setattr(router, "observable", False), 0.0, "cannot see whether"),
    ],
    ids=["llama-processing", "llama-queued", "recently-used", "unobservable"],
)
def test_busy_conflicting_model_refuses_with_no_wait(
    setup: Callable[[FakeRouter], Any], protect: float, reason: str
) -> None:
    """With a maximum wait of 0, a busy conflicting model means an immediate refusal.

    CUDA1 is loaded and the request is for MOE_CUDA1, which also needs GPU1. Each case makes
    CUDA1 busy in one way the broker did not cause itself: a processing slot or queued
    request in llama.cpp (as from a client calling the router directly), use within the
    protection window (it was adopted at startup, which counts as use), or a failed
    observation, which must count as busy. Nothing is unloaded.
    """
    router = FakeRouter(loaded=(CUDA1,))
    setup(router)

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(ResourceBusy, match=reason):
            await admitted(scheduler, MOE_CUDA1)
        assert set(scheduler.loaded) == {CUDA1}

    run(router, body, protect=protect)
    assert router.calls == []


def test_request_waits_for_an_active_request_then_loads() -> None:
    """A conflicting request waits while the loaded model serves a request, then proceeds.

    Setup: one request holds CUDA1 (like an open stream); another asks for MOE_CUDA1 with a
    2 s maximum wait. Expect it to still be waiting, with no router calls, while CUDA1 is in
    use, and to unload CUDA1 and load MOE_CUDA1 as soon as that request ends.
    """
    router = FakeRouter(loaded=(CUDA1,))

    async def body(scheduler: Scheduler) -> None:
        stream_ends = asyncio.Event()
        stream = asyncio.create_task(hold(scheduler, CUDA1, stream_ends))
        await asyncio.sleep(0.02)
        request = asyncio.create_task(admitted(scheduler, MOE_CUDA1))
        await asyncio.sleep(0.1)
        assert not request.done() and router.calls == []

        stream_ends.set()
        assert await request == MOE_CUDA1
        await stream

    run(router, body, max_wait=2)
    assert router.calls == [("unload", CUDA1), ("load", MOE_CUDA1)]


def test_retry_after_is_when_the_protection_window_ends() -> None:
    """A refusal caused only by recent use says when to retry: the window's end.

    With a 60 s window starting at startup, Retry-After is 60 s, rounded up. A refusal for
    a model that is working has no known end, so it uses the default.
    """
    router = FakeRouter(loaded=(CUDA1,))

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(ResourceBusy) as refused:
            await admitted(scheduler, MOE_CUDA1)
        assert refused.value.retry_after == 60

        router.processing.add(CUDA1)
        with pytest.raises(ResourceBusy) as refused:
            await admitted(scheduler, MOE_CUDA1)
        assert refused.value.retry_after == BUSY_RETRY_AFTER_SECONDS

    run(router, body, protect=60)


def test_request_is_admitted_when_the_protection_window_ends() -> None:
    """A request waiting only for the protection window proceeds once it ends.

    The window is 0.2 s from startup and the maximum wait is 2 s, so the request should
    wait about 0.2 s and then load, not wait for the whole 2 s.
    """
    router = FakeRouter(loaded=(CUDA1,))

    async def body(scheduler: Scheduler) -> None:
        started = time.monotonic()
        await admitted(scheduler, MOE_CUDA1)
        assert 0.15 < time.monotonic() - started < 1.0

    run(router, body, protect=0.2, max_wait=2)
    assert router.calls == [("unload", CUDA1), ("load", MOE_CUDA1)]


def test_decode_count_advancing_counts_as_recent_use() -> None:
    """Work that bypassed the broker restarts the protection window when it is noticed.

    Setup: a 0.3 s window from startup, which has passed before the first request.
    1. CUDA1 is processing, so a request for MOE_CUDA1 is refused. That observation
       records the decode count, 10.
    2. CUDA1 is now idle, but its decode count is 50: it worked since the last look, so
       the window restarts and the request is refused as recently used.
    3. After another 0.35 s with the count unchanged, the request proceeds.
    """
    router = FakeRouter(loaded=(CUDA1,))

    async def body(scheduler: Scheduler) -> None:
        await asyncio.sleep(0.35)
        router.processing.add(CUDA1)
        router.decode_calls[CUDA1] = 10
        with pytest.raises(ResourceBusy, match="working"):
            await admitted(scheduler, MOE_CUDA1)

        router.processing.discard(CUDA1)
        router.decode_calls[CUDA1] = 50
        with pytest.raises(ResourceBusy, match="used recently"):
            await admitted(scheduler, MOE_CUDA1)

        await asyncio.sleep(0.35)
        assert await admitted(scheduler, MOE_CUDA1) == MOE_CUDA1

    run(router, body, protect=0.3)


def test_direct_work_before_the_first_check_is_noticed() -> None:
    """Work that bypassed the broker between startup and its first check still counts.

    Setup: CUDA1 is adopted with decode count 10, recorded at startup. The 0.2 s window
    passes; then a client calls the router directly, raising the count to 50, and finishes
    before the broker looks. Expect the request for MOE_CUDA1 to be refused as recently
    used. Without the startup baseline, the first check would only record 50 and see
    nothing.
    """
    router = FakeRouter(loaded=(CUDA1,))
    router.decode_calls[CUDA1] = 10

    async def body(scheduler: Scheduler) -> None:
        await asyncio.sleep(0.25)
        router.decode_calls[CUDA1] = 50
        with pytest.raises(ResourceBusy, match="used recently"):
            await admitted(scheduler, MOE_CUDA1)

    run(router, body, protect=0.2)


def test_an_unknown_decode_count_counts_as_recent_use() -> None:
    """If the baseline could not be read, the first check assumes unseen work.

    Setup: the router cannot be observed at startup, so CUDA1 has no baseline. After the
    window has passed and observation works again, the first check must not conclude that
    CUDA1 is idle.
    """
    router = FakeRouter(loaded=(CUDA1,))
    router.observable = False

    async def body(scheduler: Scheduler) -> None:
        await asyncio.sleep(0.25)
        router.observable = True
        with pytest.raises(ResourceBusy, match="used recently"):
            await admitted(scheduler, MOE_CUDA1)

    run(router, body, protect=0.2)


# Loading, queueing and order


def test_a_model_found_loading_at_startup_is_waited_for() -> None:
    """A model the router is loading when the broker starts is not ready, but holds GPU1.

    Expect a request for it to wait for the load, and a request for MOE_CUDA1, which also
    needs GPU1, to be refused meanwhile rather than unload a model still loading.
    """
    router = FakeRouter(loading=(CUDA1,))
    router.loads_finish.clear()

    async def body(scheduler: Scheduler) -> None:
        waiting = asyncio.create_task(admitted(scheduler, CUDA1))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        with pytest.raises(ResourceBusy, match="loading"):
            await admitted(scheduler, MOE_CUDA1)

        router.loads_finish.set()
        assert await waiting == CUDA1

    run(router, body)
    assert router.calls == []


def test_requests_for_the_same_model_share_one_load() -> None:
    """Concurrent requests for an unloaded model cause one load, and all are admitted."""
    router = FakeRouter()
    router.loads_finish.clear()

    async def body(scheduler: Scheduler) -> None:
        requests = [asyncio.create_task(admitted(scheduler, CUDA1)) for _ in range(3)]
        await asyncio.sleep(0.05)
        router.loads_finish.set()
        assert await asyncio.gather(*requests) == [CUDA1] * 3

    run(router, body)
    assert router.calls == [("load", CUDA1)]


def test_requests_waiting_for_a_loading_model_protect_it() -> None:
    """Requests queued in the broker for a loading model make it busy.

    Setup: CUDA1 is loading with two requests waiting for it. A request for MOE_CUDA1, which
    also needs GPU1, is refused instead of unloading CUDA1 before its users are served.
    """
    router = FakeRouter()
    router.loads_finish.clear()

    async def body(scheduler: Scheduler) -> None:
        waiting = [asyncio.create_task(admitted(scheduler, CUDA1)) for _ in range(2)]
        await asyncio.sleep(0.05)
        assert scheduler.loaded[CUDA1].queued == 2
        with pytest.raises(ResourceBusy, match="requests waiting"):
            await admitted(scheduler, MOE_CUDA1)

        router.loads_finish.set()
        await asyncio.gather(*waiting)

    run(router, body)


def test_a_long_load_does_not_hold_up_other_resources() -> None:
    """While one model loads, a model that needs none of its resources loads alongside.

    Setup: CUDA1 is loading and stays so. A request for CUDA0 must neither wait for that
    load nor be refused: only models competing for the same resources wait for each other.
    """
    router = FakeRouter()
    router.loads_finish.clear()

    async def body(scheduler: Scheduler) -> None:
        loading = asyncio.create_task(admitted(scheduler, CUDA1))
        await asyncio.sleep(0.05)
        other = asyncio.create_task(admitted(scheduler, CUDA0))
        await asyncio.sleep(0.05)
        assert router.calls == [("load", CUDA1), ("load", CUDA0)]

        router.loads_finish.set()
        assert await asyncio.gather(loading, other) == [CUDA1, CUDA0]

    run(router, body)


def test_a_model_being_unloaded_keeps_its_resources() -> None:
    """A model counts as holding its GPUs until the router confirms it has unloaded.

    Setup: the dual-GPU model is loaded and idle. A request for CUDA1 starts unloading it,
    and the unload is slow. Meanwhile a request for CUDA0 arrives. GPU0 is not needed by
    CUDA1, but the dual model is still on it, so CUDA0 must wait; loading it at once could
    run the card out of memory. Once the unload finishes, both load.
    """
    router = FakeRouter(loaded=(DUAL,))
    router.unloads_finish.clear()

    async def body(scheduler: Scheduler) -> None:
        gpu1 = asyncio.create_task(admitted(scheduler, CUDA1))
        await asyncio.sleep(0.05)
        gpu0 = asyncio.create_task(admitted(scheduler, CUDA0))
        await asyncio.sleep(0.1)
        assert router.calls == [("unload", DUAL)] and not gpu0.done()
        assert scheduler.loaded[DUAL].leaving

        router.unloads_finish.set()
        assert await asyncio.gather(gpu1, gpu0) == [CUDA1, CUDA0]

    run(router, body, max_wait=2)
    assert router.calls[1:] == [("load", CUDA1), ("load", CUDA0)]


def test_conflicting_requests_are_placed_in_arrival_order() -> None:
    """A later request waits behind an earlier one for the same resources, even if it could
    load at once.

    Setup: CUDA0 is in use. Request B, for the dual-GPU model, waits for it. Request C, for
    CUDA1, arrives next: GPU1 is free, but B needs GPU1 too and came first. Expect C to wait,
    B to load when CUDA0's request ends, and C to load after B's request ends.
    """
    router = FakeRouter(loaded=(CUDA0,))

    async def body(scheduler: Scheduler) -> None:
        first_ends, dual_ends = asyncio.Event(), asyncio.Event()
        first = asyncio.create_task(hold(scheduler, CUDA0, first_ends))
        await asyncio.sleep(0.02)
        dual = asyncio.create_task(hold(scheduler, DUAL, dual_ends))
        await asyncio.sleep(0.02)
        later = asyncio.create_task(admitted(scheduler, CUDA1))
        await asyncio.sleep(0.1)
        assert not later.done() and router.calls == []

        first_ends.set()
        await asyncio.sleep(0.1)
        assert router.calls == [("unload", CUDA0), ("load", DUAL)] and not later.done()

        dual_ends.set()
        assert await later == CUDA1
        await asyncio.gather(first, dual)

    run(router, body, max_wait=2)
    assert router.calls[2:] == [("unload", DUAL), ("load", CUDA1)]


def test_cancelled_request_no_longer_holds_its_place() -> None:
    """A waiting request that is cancelled (its client left) stops blocking later ones.

    Setup as in the arrival-order test, but B is cancelled before C arrives, so C loads
    CUDA1 at once next to the busy CUDA0.
    """
    router = FakeRouter(loaded=(CUDA0,))

    async def body(scheduler: Scheduler) -> None:
        first_ends = asyncio.Event()
        first = asyncio.create_task(hold(scheduler, CUDA0, first_ends))
        await asyncio.sleep(0.02)
        dual = asyncio.create_task(admitted(scheduler, DUAL))
        await asyncio.sleep(0.02)
        dual.cancel()
        await asyncio.sleep(0)

        assert await admitted(scheduler, CUDA1) == CUDA1
        first_ends.set()
        await first

    run(router, body, max_wait=2)
    assert router.calls == [("load", CUDA1)]


# Failures


def test_failed_load_is_an_error_and_leaves_nothing_loaded() -> None:
    router = FakeRouter()
    router.fail.add(("load", CUDA1))

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(LoadFailed, match="refused to load"):
            await admitted(scheduler, CUDA1)
        assert scheduler.loaded == {}

    run(router, body)


def test_failed_load_fails_every_request_waiting_for_it() -> None:
    """Requests that joined a load which then fails are released with the error, not left
    waiting."""
    router = FakeRouter()
    router.fail.add(("load", CUDA1))

    async def body(scheduler: Scheduler) -> None:
        requests = [asyncio.create_task(admitted(scheduler, CUDA1)) for _ in range(2)]
        results = await asyncio.gather(*requests, return_exceptions=True)
        assert all(isinstance(result, LoadFailed) for result in results)

    run(router, body)
    assert router.calls == [("load", CUDA1)]


def test_a_timed_out_load_keeps_its_resources() -> None:
    """A load that times out may still be running, so its model keeps its GPU.

    Setup: CUDA0 is serving a request, and loading CUDA1 outlasts the 0.1 s load timeout.
    Expect:
    - the request for CUDA1 fails, but CUDA1 stays in the loaded set, and the router is
      marked unfit because the broker cannot confirm where CUDA1 is;
    - the next request makes the broker check the router again. It still reports CUDA1
      loading, so a request for MOE_CUDA1, which also needs GPU1, is refused rather than
      loaded onto a card that CUDA1 may be filling;
    - the check keeps CUDA0's record, so its running request is still counted.
    """
    router = FakeRouter(loaded=(CUDA0,))
    router.loads_finish.clear()

    async def body(scheduler: Scheduler) -> None:
        first_ends = asyncio.Event()
        first = asyncio.create_task(hold(scheduler, CUDA0, first_ends))
        await asyncio.sleep(0.02)
        serving = scheduler.loaded[CUDA0]

        with pytest.raises(LoadFailed, match="did not become 'loaded'"):
            await admitted(scheduler, CUDA1)
        assert CUDA1 in scheduler.loaded
        assert "not confirmed it is unloaded" in scheduler.view.problems[0]

        with pytest.raises(ResourceBusy, match="loading"):
            await admitted(scheduler, MOE_CUDA1)
        assert scheduler.view.problems == []
        assert scheduler.loaded[CUDA0] is serving and serving.active == 1

        first_ends.set()
        await first

    run(router, body, load_timeout=0.1)
    assert ("load", MOE_CUDA1) not in router.calls


def test_failed_unload_keeps_the_model_and_its_resources() -> None:
    """If a conflicting model will not unload, it still counts as loaded and holding GPU1.

    The requested model is not loaded, so the broker never believes two GPU1 models share
    the card.
    """
    router = FakeRouter(loaded=(CUDA1,))
    router.fail.add(("unload", CUDA1))

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(LoadFailed, match="refused to unload"):
            await admitted(scheduler, MOE_CUDA1)
        assert set(scheduler.loaded) == {CUDA1}

    run(router, body)
    assert router.calls == [("unload", CUDA1)]


def test_unfit_router_is_refused_until_it_is_fixed() -> None:
    """Router problems refuse every request, and are checked again on the next request.

    Setup: models-max 1 at startup. Expect a refusal naming it. After the router is
    restarted with models-max 3, the next request re-inspects it and is admitted without a
    broker restart.
    """
    router = FakeRouter()
    router.props["max_instances"] = 1

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(RouterUnfit, match="models-max is 1"):
            await admitted(scheduler, CUDA1)
        router.props["max_instances"] = 3
        assert await admitted(scheduler, CUDA1) == CUDA1

    run(router, body)


def test_unknown_model_is_refused() -> None:
    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(UnknownModel):
            await admitted(scheduler, "no-such-model--cuda0")

    run(FakeRouter(), body)


# Events


def test_a_swap_is_logged_step_by_step(events: Callable[[], list[Event]]) -> None:
    """Replacing a model logs the decision, the unload, the load, then the request.

    Setup: idle CUDA1 is loaded; a request asks for MOE_CUDA1. Expect, in order: the swap
    naming what it unloads, CUDA1 unloaded, MOE_CUDA1 loaded, the request admitted, and
    the request finished.
    """
    run(FakeRouter(loaded=(CUDA1,)), lambda scheduler: admitted(scheduler, MOE_CUDA1))

    steps = [
        (type(event), getattr(event, "model", None))
        for event in events()
        if isinstance(
            event, SwapStarted | ModelUnloaded | ModelLoaded | RequestAdmitted | RequestFinished
        )
    ]
    assert steps == [
        (SwapStarted, MOE_CUDA1),
        (ModelUnloaded, CUDA1),
        (ModelLoaded, MOE_CUDA1),
        (RequestAdmitted, MOE_CUDA1),
        (RequestFinished, MOE_CUDA1),
    ]
    assert SwapStarted(MOE_CUDA1, [CUDA1]) in events()


@pytest.mark.parametrize(
    ("model", "cause"),
    [(MOE_CUDA1, "resource_busy"), ("no-such-model--cuda0", "unknown_model")],
    ids=["busy", "unknown"],
)
def test_a_refusal_is_logged_with_its_cause(
    events: Callable[[], list[Event]], model: str, cause: str
) -> None:
    """The cause is the refusal's kind; the detail is the scheduler's reason."""

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(SchedulerError):
            await admitted(scheduler, model)

    run(FakeRouter(loaded=(CUDA1,)), body, protect=60)
    refusals = [event for event in events() if isinstance(event, RequestRefused)]
    assert [(r.model, r.cause) for r in refusals] == [(model, cause)]


def test_waiting_is_logged_once_per_reason(events: Callable[[], list[Event]]) -> None:
    """A waiting request re-checks every poll, but logs its reason only when it changes.

    Setup: CUDA1 serves a request throughout; a request for MOE_CUDA1 waits 0.2 s, polling
    every 0.01 s, then is refused. Expect one RequestWaiting, not one per poll.
    """
    router = FakeRouter(loaded=(CUDA1,))

    async def body(scheduler: Scheduler) -> None:
        stream_ends = asyncio.Event()
        stream = asyncio.create_task(hold(scheduler, CUDA1, stream_ends))
        await asyncio.sleep(0.02)
        with pytest.raises(ResourceBusy):
            await admitted(scheduler, MOE_CUDA1)
        stream_ends.set()
        await stream

    run(router, body, max_wait=0.2)
    waiting = [event for event in events() if isinstance(event, RequestWaiting)]
    assert waiting == [
        RequestWaiting(MOE_CUDA1, f"{CUDA1!r} is loading, working, or has requests waiting")
    ]


def test_observations_are_logged(events: Callable[[], list[Event]]) -> None:
    """What llama.cpp reported about a model the scheduler considered unloading is logged,
    including whether it counted as recent activity."""
    router = FakeRouter(loaded=(CUDA1,))
    router.deferred[CUDA1] = 2

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(ResourceBusy):
            await admitted(scheduler, MOE_CUDA1)

    run(router, body)
    assert ModelObserved(CUDA1, False, 2, 0, False) in events()


@pytest.mark.parametrize(
    ("hold_load", "released"),
    [(False, True), (True, False)],
    ids=["refused-load", "timed-out-load"],
)
def test_a_failed_load_is_logged_with_whether_it_was_released(
    events: Callable[[], list[Event]], hold_load: bool, released: bool
) -> None:
    """A refused load is known to hold nothing; a timed-out one may still be loading, so it
    is not released, and the router problem that follows is logged too."""
    router = FakeRouter()
    if hold_load:
        router.loads_finish.clear()
    else:
        router.fail.add(("load", CUDA1))

    async def body(scheduler: Scheduler) -> None:
        with pytest.raises(LoadFailed):
            await admitted(scheduler, CUDA1)

    run(router, body, load_timeout=0.1)
    failures = [event for event in events() if isinstance(event, ModelLoadFailed)]
    assert [(f.model, f.released) for f in failures] == [(CUDA1, released)]
    assert any(isinstance(event, RouterProblemsFound) for event in events()) is not released
