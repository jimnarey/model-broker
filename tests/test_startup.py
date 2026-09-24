from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from model_broker.catalogue import load_catalogue
from model_broker.llama import LlamaAdapter, ModelState, RouterProps
from model_broker.scheduler import inspect_router, reconcile, router_problems

# Real sections from the unified preset: GPU0, GPU1, MoE offload on GPU1, dual GPU, and CPU.
CATALOGUE = load_catalogue(Path(__file__).parent / "fixtures" / "models-preset.ini")
CUDA0 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda0"
CUDA1 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda1"
MOE_CUDA1 = "GLM-4.5-Air-UD-Q4_K_XL--cuda1"
CPU = "Qwen3.5-9B-Q4_K_M--cpu"
NOW = 1000.0


def props(max_instances: int = 3, autoload: bool = False) -> RouterProps:
    return RouterProps(max_instances, autoload, "b10920-eafe15a5e")


# Router settings


@pytest.mark.parametrize(
    ("router", "problems"),
    [
        (props(max_instances=3), []),
        (props(max_instances=4), []),
        (props(max_instances=0), []),
        (props(max_instances=1), ["router models-max is 1, but 3 models can be loaded together"]),
        (props(autoload=True), ["router loads models automatically"]),
    ],
    ids=["exact", "higher", "unlimited", "too-low", "autoload"],
)
def test_router_problems(router: RouterProps, problems: list[str]) -> None:
    """models-max must allow every model that can be loaded at once, and autoload must be off.

    The fixture's exclusive resources are GPU0, GPU1 and RAM:weights, so at most 3 models can
    be loaded together; with a lower limit the router, not the broker, would choose which
    model to drop. llama.cpp uses 0 for unlimited. ``too-low`` is the setting the production
    container ran with on 2026-09-24.
    """
    found = router_problems(router, CATALOGUE)

    assert len(found) == len(problems)
    assert all(found[n].startswith(problem) for n, problem in enumerate(problems))


# Reconciliation


def test_reconcile_adopts_loaded_and_loading_models() -> None:
    """Models the router holds or is loading are adopted; unloaded ones are ignored.

    A loading model is adopted but not ready, so requests for it wait for the load.

    Adopted models get last activity ``NOW``, so a restarted broker does not unload a model
    that someone was using a moment ago. A failed load is ``unloaded`` and so holds nothing.
    """
    states = {
        CUDA0: ModelState("loaded"),
        CUDA1: ModelState("loading"),
        CPU: ModelState("unloaded"),
        MOE_CUDA1: ModelState("unloaded", failed=True, exit_code=1),
    }

    loaded, problems = reconcile(CATALOGUE, states, NOW)

    assert problems == []
    assert sorted(loaded) == [CUDA0, CUDA1]
    assert loaded[CUDA0].ready.is_set() and not loaded[CUDA1].ready.is_set()
    assert all(model.last_activity == NOW for model in loaded.values())
    assert all(model.active == model.queued == 0 for model in loaded.values())


@pytest.mark.parametrize(
    ("states", "adopted", "problem"),
    [
        ({"not-in-preset--cuda0": ModelState("loaded")}, [], "but the preset lacks it"),
        ({CUDA0: ModelState("sleeping")}, [CUDA0], "sleeping; idle sleeping must be off"),
        ({CUDA0: ModelState("unloading")}, [CUDA0], "unknown state 'unloading'"),
        (
            {CUDA1: ModelState("loaded"), MOE_CUDA1: ModelState("loaded")},
            [CUDA1, MOE_CUDA1],
            "conflicting models loaded",
        ),
    ],
    ids=["unknown-model", "sleeping", "unknown-status", "conflicting-models"],
)
def test_reconcile_reports_unsafe_router_state(
    states: dict[str, ModelState], adopted: list[str], problem: str
) -> None:
    """Router state the broker cannot account for is a problem that blocks scheduling.

    ``unknown-model`` has no claims the broker knows. ``sleeping`` is adopted, because it
    may still hold memory, but idle sleeping should be off. ``unknown-status`` (a status a
    newer router might add) is adopted too: treating it as unloaded could put a second
    model on a GPU that is still occupied. ``conflicting-models`` has two
    GPU1 models loaded, which the broker never allows, for example after router autoload.
    """
    loaded, problems = reconcile(CATALOGUE, states, NOW)

    assert set(loaded) == set(adopted)
    assert len(problems) == 1 and problem in problems[0]


# Inspecting the router


def inspect(handler: httpx.MockTransport) -> tuple[list[str], list[str], str | None]:
    """Run inspect_router against a mock router; return loaded IDs, problems and build."""

    async def main() -> tuple[list[str], list[str], str | None]:
        async with httpx.AsyncClient(transport=handler, base_url="http://router.test") as client:
            view = await inspect_router(LlamaAdapter(client), CATALOGUE, NOW)
        return sorted(view.loaded), view.problems, view.build

    return asyncio.run(main())


def test_inspect_router_combines_settings_and_loaded_models() -> None:
    """Both the settings check and reconciliation feed the result.

    Setup: models-max 1 with CUDA0 loaded. Expect CUDA0 adopted, the models-max problem, and
    the router build recorded for /health.
    """
    router_props = {
        "role": "router",
        "max_instances": 1,
        "models_autoload": False,
        "build_info": "b1",
    }
    models = {"data": [{"id": CUDA0, "status": {"value": "loaded"}}]}
    responses = {
        "/props": httpx.Response(200, json=router_props),
        "/models": httpx.Response(200, json=models),
    }

    loaded, problems, build = inspect(httpx.MockTransport(lambda r: responses[r.url.path]))

    assert (loaded, build) == ([CUDA0], "b1")
    assert len(problems) == 1 and "models-max is 1" in problems[0]


def test_inspect_router_reports_an_unreachable_router() -> None:
    """A router that cannot be reached is one problem, and nothing is adopted."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    loaded, problems, build = inspect(httpx.MockTransport(refuse))

    assert (loaded, build) == ([], None)
    assert problems == ["router is unavailable: router GET /props failed: connection refused"]
