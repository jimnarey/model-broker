from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

HOST_FACTS_SCRIPT = Path(__file__).parents[1] / "model-broker-host-facts.py"


@pytest.fixture(scope="session")
def host_facts() -> Any:
    """Load the standalone host-facts script, whose file name is not importable, as a module."""
    spec = importlib.util.spec_from_file_location("model_broker_host_facts", HOST_FACTS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
