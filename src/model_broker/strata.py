"""Strata, the first service managed by restarting its container.

Strata serves one model per process, fixed when it starts, so ``ManagedService`` does the
loading and unloading. This module adds what is particular to Strata: how to tell whether
it is busy, from its own ``GET /v1/status``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from model_broker.backend import BackendError, ModelActivity
from model_broker.llama import JsonObject, LlamaAdapter
from model_broker.managed import ManagedService
from model_broker.servicecontrol import ServiceController

STRATA_SERVICE = "strata"


def parse_strata_activity(document: object) -> ModelActivity:
    """Read what Strata's ``GET /v1/status`` says about its one model's work.

    ``in_flight`` counts the running request and those waiting behind it. ``requests``
    counts finished requests plus the running one, so it moves when a request starts or
    ends. It plays the part of llama.cpp's decode counter: a change between two
    observations shows the model worked in between.
    """
    activity = cast(JsonObject, document).get("activity") if isinstance(document, dict) else None
    activity = cast(JsonObject, activity) if isinstance(activity, dict) else {}
    in_flight, requests = activity.get("in_flight"), activity.get("requests")
    # type() rather than isinstance(), because JSON true is also a Python int.
    if type(in_flight) is not int or type(requests) is not int:
        raise BackendError("Strata GET /v1/status response has no activity in_flight and requests")
    return ModelActivity(processing=in_flight > 0, deferred=0, decode_calls=requests)


async def read_strata_activity(api: LlamaAdapter) -> ModelActivity:
    return parse_strata_activity(await api.get_json("/v1/status"))


def strata_service(
    control: ServiceController,
    models: Mapping[str, Mapping[str, str]],
    api: LlamaAdapter,
    state_timeout_seconds: float,
) -> ManagedService:
    """Strata as a managed service. Strata ignores the model name in a request and answers
    with whatever it is running, so the scheduler must admit a request only while the model
    it names is the loaded one, which it does."""
    return ManagedService(
        name="Strata",
        service=STRATA_SERVICE,
        control=control,
        models=models,
        api=api,
        read_activity=read_strata_activity,
        state_timeout_seconds=state_timeout_seconds,
    )
