"""Decide when a requested model can be loaded, and load it.

A request for a loaded model is admitted at once. Otherwise the loaded models whose claims
conflict with the requested one must all be idle (no work, no queue, and no recent
activity) before the scheduler unloads them and loads the requested model. See design.md,
Scheduling and request lifecycle.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

from model_broker.catalogue import Variant, claims_conflict, max_loaded_models
from model_broker.events import (
    BaselineRecorded,
    ModelAdopted,
    ModelLoaded,
    ModelLoadFailed,
    ModelObserved,
    ModelUnloaded,
    ModelUnloadFailed,
    ModelUnobservable,
    RequestAdmitted,
    RequestFinished,
    RequestRefused,
    RequestWaiting,
    RouterProblemsCleared,
    RouterProblemsFound,
    SwapStarted,
    emit,
    snake_case,
)
from model_broker.llama import LlamaAdapter, LlamaError, ModelState, RouterProps

# The router statuses the broker understands. Any status but "unloaded", including one the
# broker does not know, is taken to hold the model's resources.
KNOWN_STATES = frozenset({"unloaded", "loading", "loaded", "sleeping"})
# How often a waiting request re-checks models it has no end time for, such as busy ones.
POLL_SECONDS = 1.0
# Retry-After for a refusal with no known end time.
BUSY_RETRY_AFTER_SECONDS = 5


class SchedulerError(Exception):
    """A request cannot be admitted."""


class UnknownModel(SchedulerError):
    """The requested model is not in the catalogue."""


class RouterUnfit(SchedulerError):
    """The router's settings or state make scheduling unsafe."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class ResourceBusy(SchedulerError):
    """A conflicting model stayed busy for the whole maximum wait."""

    def __init__(self, reason: str, retry_after: int) -> None:
        super().__init__(reason)
        self.retry_after = retry_after


class LoadFailed(SchedulerError):
    """The router failed to unload a conflicting model or to load the requested one."""


def ready_event(ready: bool) -> asyncio.Event:
    event = asyncio.Event()
    if ready:
        event.set()
    return event


@dataclass(slots=True, eq=False)
class LoadedModel:
    """A model the router holds or is loading, and the broker's view of its use.

    ``ready`` is set once the model is loaded, or once its load has failed, in which case
    ``error`` says why. ``leaving`` marks a model being unloaded: it keeps its resources
    until the router confirms it has gone, but admits no new requests. ``decode_calls`` is
    the router's decode count at the last observation, used to notice work that did not
    pass through the broker.
    """

    variant: Variant
    last_activity: float
    active: int = 0
    queued: int = 0
    decode_calls: int | None = None
    ready: asyncio.Event = field(default_factory=lambda: ready_event(True))
    error: str | None = None
    leaving: bool = False


@dataclass(slots=True)
class RouterView:
    """What the broker learned from the router: its loaded models, and any problems.

    ``problems`` lists every reason the broker must not schedule; it is empty when the
    router is fit to use.
    """

    loaded: dict[str, LoadedModel]
    problems: list[str]
    build: str | None


@dataclass(frozen=True, slots=True)
class Blocked:
    """Why a request cannot be placed yet, and when that ends if known."""

    reason: str
    until: float | None = None


@dataclass(eq=False, slots=True)
class Ticket:
    """One request waiting to have its model loaded; tickets are kept in arrival order."""

    variant: Variant


def router_problems(props: RouterProps, catalogue: Mapping[str, Variant]) -> list[str]:
    """Check the router settings the design depends on.

    ``models-max`` must never force the router to choose a model to unload, so it must allow
    as many models as can be loaded without conflict. llama.cpp uses 0 for unlimited.
    """
    problems: list[str] = []
    needed = max_loaded_models(catalogue)
    if 0 < props.max_instances < needed:
        problems.append(
            f"router models-max is {props.max_instances}, but {needed} models can be loaded "
            f"together; set --models-max {needed} or higher"
        )
    if props.models_autoload:
        problems.append("router loads models automatically; start it with --no-models-autoload")
    return problems


def reconcile(
    catalogue: Mapping[str, Variant], states: Mapping[str, ModelState], now: float
) -> tuple[dict[str, LoadedModel], list[str]]:
    """Adopt the models the router holds, recording problems that make scheduling unsafe.

    Every model not ``unloaded`` is adopted, so that it keeps its resources; one still
    ``loading`` is not ready. Adopted models start with last activity ``now``, so the idle
    protection window applies after a broker restart. A status the broker does not know, a
    loaded model outside the catalogue (its claims are unknown), and two conflicting loaded
    models (which the broker never allows) are problems.
    """
    loaded: dict[str, LoadedModel] = {}
    problems: list[str] = []
    for model_id, state in states.items():
        if state.value == "unloaded":
            continue
        if state.value not in KNOWN_STATES:
            problems.append(
                f"router reports {model_id!r} in unknown state {state.value!r}; it is taken "
                "to hold its resources"
            )
        if state.value == "sleeping":
            problems.append(f"router reports {model_id!r} sleeping; idle sleeping must be off")
        variant = catalogue.get(model_id)
        if variant is None:
            problems.append(f"router has {model_id!r} {state.value}, but the preset lacks it")
        else:
            ready = ready_event(state.value != "loading")
            loaded[model_id] = LoadedModel(variant, last_activity=now, ready=ready)
    for first, second in itertools.combinations(loaded.values(), 2):
        if claims_conflict(first.variant, second.variant):
            problems.append(
                f"router has conflicting models loaded: {first.variant.id!r} and "
                f"{second.variant.id!r}"
            )
    return loaded, problems


async def inspect_router(
    adapter: LlamaAdapter, catalogue: Mapping[str, Variant], now: float
) -> RouterView:
    """Check the router's settings and adopt its loaded models."""
    try:
        props = await adapter.props()
        states = await adapter.model_states()
    except LlamaError as error:
        return RouterView(loaded={}, problems=[f"router is unavailable: {error}"], build=None)
    loaded, problems = reconcile(catalogue, states, now)
    return RouterView(loaded, router_problems(props, catalogue) + problems, props.build)


class Scheduler:
    """Admit requests into loaded models, loading and unloading models when it is safe.

    ``lock`` covers only the decision to unload and load. The unload and load themselves run
    in a background task, so a long load does not hold up decisions about other resources.
    While they run, the models involved stay in ``loaded``, so they keep their resources and
    count as busy for any other decision.

    Call ``start()`` once the event loop is running and before admitting requests.
    """

    def __init__(
        self,
        adapter: LlamaAdapter,
        catalogue: Mapping[str, Variant],
        view: RouterView,
        *,
        protect_seconds: float,
        max_wait_seconds: float,
        poll_seconds: float = POLL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.adapter = adapter
        self.catalogue = catalogue
        self.view = view
        self.protect_seconds = protect_seconds
        self.max_wait_seconds = max_wait_seconds
        self.poll_seconds = poll_seconds
        self.clock = clock
        self.lock = asyncio.Lock()
        self.changed = asyncio.Condition()
        self.waiting: list[Ticket] = []
        self.tasks: set[asyncio.Task[None]] = set()

    @property
    def loaded(self) -> dict[str, LoadedModel]:
        return self.view.loaded

    async def start(self) -> None:
        """Begin tracking the models adopted from the router."""
        await self.track(list(self.loaded.values()))

    def seconds_since(self, started: float) -> float:
        return round(self.clock() - started, 3)

    async def track(self, models: list[LoadedModel]) -> None:
        """Watch newly adopted models: wait for any still loading, and record a baseline
        decode count for the others."""
        for model in models:
            state = "loaded" if model.ready.is_set() else "loading"
            emit(ModelAdopted(model.variant.id, state))
            if model.ready.is_set():
                await self.record_baseline(model)
            else:
                self.background(self.finish_load(model))

    def background(self, work: Coroutine[None, None, None]) -> None:
        task = asyncio.create_task(work)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    @asynccontextmanager
    async def admit(self, model_id: str) -> AsyncGenerator[Variant]:
        """Hold an admission to ``model_id`` for the duration of one proxied request."""
        started = self.clock()
        try:
            model = await self.acquire(model_id)
        except SchedulerError as error:
            cause = snake_case(type(error).__name__)
            emit(RequestRefused(model_id, cause, str(error), self.seconds_since(started)))
            raise
        emit(RequestAdmitted(model_id, self.seconds_since(started)))
        admitted = self.clock()
        try:
            yield model.variant
        finally:
            model.active -= 1
            model.last_activity = self.clock()
            emit(RequestFinished(model_id, self.seconds_since(admitted)))
            await self.notify()

    async def acquire(self, model_id: str) -> LoadedModel:
        variant = self.catalogue.get(model_id)
        if variant is None:
            raise UnknownModel(model_id)
        await self.check_router()
        deadline = self.clock() + self.max_wait_seconds
        ticket = Ticket(variant)
        self.waiting.append(ticket)
        reported = ""
        try:
            while True:
                model = self.loaded.get(model_id)
                if model is not None and not model.leaving:
                    self.remove(ticket)
                    return await self.join(model)
                placed = await self.place(ticket)
                if isinstance(placed, LoadedModel):
                    return await self.join(placed)
                now = self.clock()
                if placed.until is not None and placed.until <= now:
                    continue  # the obstacle has already gone; look again at once
                if placed.reason != reported:
                    emit(RequestWaiting(model_id, placed.reason))
                    reported = placed.reason
                if now >= deadline:
                    retry = placed.until - now if placed.until else BUSY_RETRY_AFTER_SECONDS
                    raise ResourceBusy(placed.reason, max(1, math.ceil(retry)))
                recheck = placed.until if placed.until else now + self.poll_seconds
                await self.wait_for_change(min(recheck, deadline) - now)
        finally:
            if ticket in self.waiting:
                self.remove(ticket)
                await self.notify()

    async def check_router(self) -> None:
        """Refuse while the router is unfit, inspecting it again in case it has been fixed."""
        if not self.view.problems:
            return
        async with self.lock:
            if self.view.problems:
                await self.reinspect()
        if self.view.problems:
            raise RouterUnfit(self.view.problems)

    async def reinspect(self) -> None:
        """Check the router again, merging what it reports with the models already tracked.

        Problems can arise while other models are serving requests, so the existing record
        of each model the router still holds is kept, with its active and queued counts. A
        record is replaced only if its load failed. A model the router no longer holds is
        dropped, unless a request or an unload or load is still in progress on it. If the
        router cannot be reached, nothing is known, so every tracked model is kept.
        """
        try:
            props = await self.adapter.props()
            states = await self.adapter.model_states()
        except LlamaError as error:
            self.view.problems = [f"router is unavailable: {error}"]
            emit(RouterProblemsFound(self.view.problems))
            return
        fresh, problems = reconcile(self.catalogue, states, self.clock())
        merged: dict[str, LoadedModel] = {}
        for model_id, adopted in fresh.items():
            current = self.loaded.get(model_id)
            merged[model_id] = current if current and current.error is None else adopted
        for model_id, current in self.loaded.items():
            in_progress = current.active or current.leaving or not current.ready.is_set()
            if model_id not in merged and in_progress:
                merged[model_id] = current
        new = [
            model for model_id, model in merged.items() if self.loaded.get(model_id) is not model
        ]
        self.view = RouterView(
            merged, router_problems(props, self.catalogue) + problems, props.build
        )
        if self.view.problems:
            emit(RouterProblemsFound(self.view.problems))
        else:
            emit(RouterProblemsCleared(props.build))
        await self.track(new)

    async def join(self, model: LoadedModel) -> LoadedModel:
        """Admit a request into a model, first waiting for it if it is still loading."""
        if not model.ready.is_set():
            model.queued += 1
            try:
                await model.ready.wait()
            finally:
                model.queued -= 1
        if model.error is not None:
            raise LoadFailed(model.error)
        model.active += 1
        return model

    async def place(self, ticket: Ticket) -> LoadedModel | Blocked:
        """Start loading the ticket's model if every conflicting loaded model is idle.

        Returns the model, still loading, or why it cannot be placed yet. An earlier waiting
        request for a conflicting model goes first, so requests are placed in arrival order.
        """
        variant = ticket.variant
        earlier = self.waiting[: self.waiting.index(ticket)]
        if any(
            other.variant.id != variant.id and claims_conflict(other.variant, variant)
            for other in earlier
        ):
            return Blocked("an earlier request is waiting for the same resources")
        async with self.lock:
            existing = self.loaded.get(variant.id)
            if existing is not None:
                if existing.leaving:
                    return Blocked(f"{variant.id!r} is being unloaded")
                return Blocked("loaded by another request", until=self.clock())
            conflicts = [m for m in self.loaded.values() if claims_conflict(m.variant, variant)]
            blocked = await self.busy(conflicts)
            if blocked is not None:
                return blocked
            # Observing llama.cpp awaited, so a request may have joined a conflicting model
            # since; check again with no await between this check and the decision.
            if any(model.active or model.queued or model.leaving for model in conflicts):
                return Blocked("a conflicting model was just used", until=self.clock())
            self.remove(ticket)
            for victim in conflicts:
                victim.leaving = True
            model = LoadedModel(variant, self.clock(), ready=ready_event(False))
            self.loaded[variant.id] = model
            emit(SwapStarted(variant.id, [victim.variant.id for victim in conflicts]))
        self.background(self.swap(conflicts, model))
        return model

    async def swap(self, victims: list[LoadedModel], model: LoadedModel) -> None:
        """Unload the victims, then load ``model``, recording any failure on ``model``.

        Each victim keeps its resources until the router confirms it has unloaded. If an
        unload fails, that victim and any not yet unloaded stay loaded, and ``model`` is not
        loaded; the router was never asked to load it, so it holds nothing. ``model`` is
        marked ready either way, releasing its waiters.
        """
        victim = None
        try:
            for victim in victims:
                started = self.clock()
                await self.adapter.unload(victim.variant.id)
                del self.loaded[victim.variant.id]
                emit(ModelUnloaded(victim.variant.id, self.seconds_since(started)))
        except LlamaError as error:
            if victim is not None:
                emit(ModelUnloadFailed(victim.variant.id, str(error)))
            for victim in victims:
                victim.leaving = False
            del self.loaded[model.variant.id]
            model.error = str(error)
            model.ready.set()
            await self.notify()
            return
        started = self.clock()
        try:
            await self.adapter.load(model.variant.id)
        except LlamaError as error:
            await self.load_failed(model, error)
        else:
            emit(ModelLoaded(model.variant.id, self.seconds_since(started)))
            await self.record_baseline(model)
        finally:
            model.last_activity = self.clock()
            model.ready.set()
            await self.notify()

    async def finish_load(self, model: LoadedModel) -> None:
        """Wait for an adopted model that the router is still loading."""
        started = self.clock()
        try:
            await self.adapter.wait_for_state(model.variant.id, "loaded")
        except LlamaError as error:
            await self.load_failed(model, error)
        else:
            emit(ModelLoaded(model.variant.id, self.seconds_since(started)))
            await self.record_baseline(model)
        finally:
            model.ready.set()
            await self.notify()

    async def load_failed(self, model: LoadedModel, error: LlamaError) -> None:
        """Record a failed load, releasing the model's resources only if that is certain.

        A timeout or a lost connection does not show that the worker has gone: it may still
        be loading onto its GPUs. So the model is dropped only when the router confirms it is
        unloaded (or not listed). Otherwise it keeps its resources, and the router is marked
        unfit until a fresh inspection shows the model's real state.
        """
        model_id = model.variant.id
        model.error = str(error)
        try:
            state = (await self.adapter.model_states()).get(model_id)
        except LlamaError:
            unloaded = False
        else:
            unloaded = state is None or state.value == "unloaded"
        emit(ModelLoadFailed(model_id, str(error), released=unloaded))
        if unloaded:
            if self.loaded.get(model_id) is model:
                del self.loaded[model_id]
        else:
            self.view.problems.append(
                f"loading {model_id!r} failed ({error}), and the router has not confirmed it "
                "is unloaded, so it keeps its resources until the router is checked again"
            )
            emit(RouterProblemsFound(self.view.problems))

    async def record_baseline(self, model: LoadedModel) -> None:
        """Record the model's decode count, so later work bypassing the broker shows up.

        If it cannot be read, the count stays unknown and the next observation counts as
        recent activity.
        """
        with suppress(LlamaError):
            model.decode_calls = (await self.adapter.activity(model.variant.id)).decode_calls
        emit(BaselineRecorded(model.variant.id, model.decode_calls))

    async def busy(self, conflicts: list[LoadedModel]) -> Blocked | None:
        """Return why a conflicting model cannot be unloaded yet, or None if all are idle."""
        until = self.clock()
        reason = ""
        for model in conflicts:
            name = model.variant.id
            if model.leaving:
                return Blocked(f"{name!r} is being unloaded")
            if model.active or model.queued or not model.ready.is_set():
                return Blocked(f"{name!r} is loading, working, or has requests waiting")
            try:
                activity = await self.adapter.activity(name)
            except LlamaError as error:
                emit(ModelUnobservable(name, str(error)))
                return Blocked(f"cannot see whether {name!r} is idle: {error}")
            # Without a baseline, unseen work cannot be ruled out, so assume there was some.
            worked = model.decode_calls is None or activity.decode_calls != model.decode_calls
            if worked:
                model.last_activity = self.clock()
            model.decode_calls = activity.decode_calls
            emit(
                ModelObserved(
                    name, activity.processing, activity.deferred, activity.decode_calls, worked
                )
            )
            if activity.processing or activity.deferred:
                return Blocked(f"{name!r} is working or has requests queued in llama.cpp")
            protected_until = model.last_activity + self.protect_seconds
            if protected_until > until:
                until, reason = protected_until, f"{name!r} was used recently"
        return Blocked(reason, until) if reason else None

    def remove(self, ticket: Ticket) -> None:
        with suppress(ValueError):
            self.waiting.remove(ticket)

    async def notify(self) -> None:
        async with self.changed:
            self.changed.notify_all()

    async def wait_for_change(self, timeout: float) -> None:
        """Wait until something the scheduler depends on changes, or ``timeout`` elapses."""
        async with self.changed:
            with suppress(TimeoutError):
                await asyncio.wait_for(self.changed.wait(), max(0.0, timeout))
