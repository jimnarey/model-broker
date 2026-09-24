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
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

from model_broker.catalogue import Variant, claims_conflict, max_loaded_models
from model_broker.llama import LlamaAdapter, LlamaError, ModelState, RouterProps

# Statuses in which a model holds, or is about to hold, its resources.
OCCUPYING_STATES = frozenset({"loaded", "loading", "sleeping"})
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
    """Adopt the models the router already holds, recording problems that make it unsafe.

    Adopted models start with last activity ``now``, so the idle protection window applies
    after a broker restart. A loaded model outside the catalogue has unknown claims, and two
    conflicting loaded models break the broker's invariant, so both are problems.
    """
    loaded: dict[str, LoadedModel] = {}
    problems: list[str] = []
    for model_id, state in states.items():
        if state.value not in OCCUPYING_STATES:
            continue
        if state.value == "sleeping":
            problems.append(f"router reports {model_id!r} sleeping; idle sleeping must be off")
        variant = catalogue.get(model_id)
        if variant is None:
            problems.append(f"router has {model_id!r} {state.value}, but the preset lacks it")
        else:
            loaded[model_id] = LoadedModel(variant, last_activity=now)
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
        self.swaps: set[asyncio.Task[None]] = set()

    @property
    def loaded(self) -> dict[str, LoadedModel]:
        return self.view.loaded

    @asynccontextmanager
    async def admit(self, model_id: str) -> AsyncGenerator[Variant]:
        """Hold an admission to ``model_id`` for the duration of one proxied request."""
        model = await self.acquire(model_id)
        try:
            yield model.variant
        finally:
            model.active -= 1
            model.last_activity = self.clock()
            await self.notify()

    async def acquire(self, model_id: str) -> LoadedModel:
        variant = self.catalogue.get(model_id)
        if variant is None:
            raise UnknownModel(model_id)
        await self.check_router()
        deadline = self.clock() + self.max_wait_seconds
        ticket = Ticket(variant)
        self.waiting.append(ticket)
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
        """Refuse while the router is unfit, inspecting it again in case it has been fixed.

        Nothing is admitted while there are problems, so replacing the adopted models from a
        fresh inspection cannot lose track of a running request.
        """
        if not self.view.problems:
            return
        async with self.lock:
            if self.view.problems:
                self.view = await inspect_router(self.adapter, self.catalogue, self.clock())
        if self.view.problems:
            raise RouterUnfit(self.view.problems)

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
        swap = asyncio.create_task(self.swap(conflicts, model))
        self.swaps.add(swap)
        swap.add_done_callback(self.swaps.discard)
        return model

    async def swap(self, victims: list[LoadedModel], model: LoadedModel) -> None:
        """Unload the victims, then load ``model``, recording any failure on ``model``.

        Each victim keeps its resources until the router confirms it has unloaded. A victim
        that fails to unload, and any not yet unloaded, stay loaded; the requested model is
        then not loaded. ``model`` is marked ready either way, releasing its waiters.
        """
        try:
            for victim in victims:
                await self.adapter.unload(victim.variant.id)
                del self.loaded[victim.variant.id]
            await self.adapter.load(model.variant.id)
        except LlamaError as error:
            for victim in victims:
                victim.leaving = False
            del self.loaded[model.variant.id]
            model.error = str(error)
        finally:
            model.last_activity = self.clock()
            model.ready.set()
            await self.notify()

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
                return Blocked(f"cannot see whether {name!r} is idle: {error}")
            if model.decode_calls is not None and activity.decode_calls != model.decode_calls:
                model.last_activity = self.clock()
            model.decode_calls = activity.decode_calls
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
