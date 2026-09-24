"""The broker's record of loaded models, and the router checks that gate scheduling."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass

from model_broker.catalogue import Variant, claims_conflict, max_loaded_models
from model_broker.llama import LlamaAdapter, LlamaError, ModelState, RouterProps

# Statuses in which a model holds, or is about to hold, its resources.
OCCUPYING_STATES = frozenset({"loaded", "loading", "sleeping"})


@dataclass(slots=True)
class LoadedModel:
    """A model the router has loaded or is loading, and the broker's view of its use."""

    variant: Variant
    last_activity: float
    active: int = 0
    queued: int = 0


@dataclass(slots=True)
class RouterView:
    """What the broker learned from the router at startup.

    ``problems`` lists every reason the broker must not schedule; it is empty when the
    router is fit to use.
    """

    loaded: dict[str, LoadedModel]
    problems: list[str]
    build: str | None


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
