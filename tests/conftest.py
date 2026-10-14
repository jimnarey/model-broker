from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from model_broker.events import Event

HOST_FACTS_SCRIPT = Path(__file__).parents[1] / "model-broker-host-facts.py"
SERVICE_CONTROL_SCRIPT = Path(__file__).parents[1] / "model-broker-service-control.py"


@pytest.fixture(scope="session")
def host_facts() -> Any:
    """Load the standalone host-facts script, whose file name is not importable, as a module."""
    spec = importlib.util.spec_from_file_location("model_broker_host_facts", HOST_FACTS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def service_control() -> Any:
    """Load the standalone service-control script, which is not importable by name, as a module."""
    spec = importlib.util.spec_from_file_location(
        "model_broker_service_control", SERVICE_CONTROL_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def events(caplog: pytest.LogCaptureFixture) -> Callable[[], list[Event]]:
    """Collect the events the broker emits, as Event objects, from DEBUG up.

    Returns a function listing the events emitted so far. Tests that start the application
    must also set log_level="DEBUG" to see DEBUG events, since startup sets the level.
    """
    caplog.set_level(logging.DEBUG, logger="model_broker")

    def emitted() -> list[Event]:
        found = (getattr(record, "event", None) for record in caplog.records)
        return [event for event in found if isinstance(event, Event)]

    return emitted
