"""Acceptance tests: the broker, over HTTP, against a real llama.cpp router on the GPUs.

They are opt-in, and need:

- ``MODEL_BROKER_RUN_ACCEPTANCE=1``;
- the router image, built with ``tests/acceptance/build-router.sh``;
- ``MODEL_BROKER_ACCEPTANCE_PRESET``: the deployment's unified preset;
- ``MODEL_BROKER_ACCEPTANCE_MODELS``: the host directory holding the preset's models
  (mounted at ``/models`` in the router); and
- both GPUs idle, since the tests load models onto them.

The router runs in its own container, on localhost, with the broker's settings
(models-max 3, no autoload). Each test starts from a router with nothing loaded and a
fresh broker process, then checks the router's own view of what is loaded and working.
The tests choose the smallest suitable models from the preset, so they need no model names
from the deployment.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from model_broker.catalogue import Variant, load_catalogue

PROJECT = Path(__file__).parents[2]
LLAMA_ENV = Path(__file__).parent / "llama.env"
TIMEOUT = httpx.Timeout(300.0)
# Models the tests may use: fully on their GPUs (or CPU), with no extra files to mount.
EXTRA_FILE_KEYS = ("chat-template-file", "spec-draft-model", "mmproj")
LONG_ANSWER = "Count from 1 to 400, one number per line, with nothing else."


# Environment


@dataclass(frozen=True)
class Environment:
    image: str
    preset: Path
    models: Path


def pinned_commit() -> str:
    for line in LLAMA_ENV.read_text().splitlines():
        if line.startswith("LLAMA_COMMIT="):
            return line.partition("=")[2].strip()
    raise AssertionError(f"{LLAMA_ENV} has no LLAMA_COMMIT")


def gpu_memory_used_mib() -> list[int]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [int(line) for line in output.split()]


@pytest.fixture(scope="session")
def environment() -> Environment:
    if os.environ.get("MODEL_BROKER_RUN_ACCEPTANCE") != "1":
        pytest.skip("set MODEL_BROKER_RUN_ACCEPTANCE=1 to run the acceptance tests")
    preset = os.environ.get("MODEL_BROKER_ACCEPTANCE_PRESET")
    models = os.environ.get("MODEL_BROKER_ACCEPTANCE_MODELS")
    if not preset or not models:
        pytest.skip("set MODEL_BROKER_ACCEPTANCE_PRESET and MODEL_BROKER_ACCEPTANCE_MODELS")
    if shutil.which("docker") is None or shutil.which("nvidia-smi") is None:
        pytest.skip("Docker and nvidia-smi are required")
    image = f"model-broker-acceptance-llama:{pinned_commit()[:12]}"
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        pytest.skip(f"{image} is not built; run tests/acceptance/build-router.sh")
    busy = [mib for mib in gpu_memory_used_mib() if mib > 1024]
    if busy:
        pytest.skip(f"GPUs are in use ({busy} MiB); unload their models first")
    return Environment(image, Path(preset), Path(models))


# Models


@dataclass(frozen=True)
class Models:
    """The models the tests use: the smallest suitable ones for each placement."""

    gpu0: str
    gpu1: str
    other_gpu1: str
    dual: str
    cpu: str


def smallest(catalogue: dict[str, Variant], models: Path, suffix: str, count: int) -> list[str]:
    def size(variant: Variant) -> int:
        path = models / variant.model_path.removeprefix("/models/")
        if "-00001-of-" in path.name:
            stem = path.name.split("-00001-of-")[0]
            return sum(part.stat().st_size for part in path.parent.glob(f"{stem}-0*"))
        return path.stat().st_size

    candidates = [
        variant
        for variant in catalogue.values()
        if variant.id.endswith(suffix)
        and variant.resource_class in ("gpu_resident", "cpu_weights")
        and not any(key in variant.settings for key in EXTRA_FILE_KEYS)
        and (models / variant.model_path.removeprefix("/models/")).exists()
    ]
    chosen = [variant.id for variant in sorted(candidates, key=size)[:count]]
    assert len(chosen) == count, f"the preset has too few small {suffix} models"
    return chosen


@pytest.fixture(scope="session")
def models(environment: Environment) -> Models:
    catalogue = load_catalogue(environment.preset)
    gpu1, other_gpu1 = smallest(catalogue, environment.models, "--cuda1", 2)
    return Models(
        gpu0=smallest(catalogue, environment.models, "--cuda0", 1)[0],
        gpu1=gpu1,
        other_gpu1=other_gpu1,
        dual=smallest(catalogue, environment.models, "--cuda0-cuda1", 1)[0],
        cpu=smallest(catalogue, environment.models, "--cpu", 1)[0],
    )


# The router


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="session")
def router(environment: Environment) -> Iterator[str]:
    """A router container with the broker's settings; yields its URL."""
    port = free_port()
    started = subprocess.run(
        [
            "docker", "run", "--detach", "--rm", "--gpus", "all",
            "--publish", f"127.0.0.1:{port}:8080",
            "--env", "NCCL_CUMEM_ENABLE=0",
            "--volume", f"{environment.models}:/models:ro",
            "--volume", f"{environment.preset}:/models-preset.ini:ro",
            environment.image,
            "--models-preset", "/models-preset.ini",
            "--models-max", "3",
            "--no-models-autoload",
            "--metrics",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--parallel", "1",
            "--flash-attn", "on",
            "--no-webui",
        ],
        capture_output=True,
        text=True,
        check=True,
    )  # fmt: skip
    container = started.stdout.strip()
    url = f"http://127.0.0.1:{port}"
    try:
        wait_until(lambda: httpx.get(f"{url}/models").status_code == 200, 60)
        yield url
    finally:
        subprocess.run(["docker", "rm", "--force", container], capture_output=True)


def wait_until(condition: Callable[[], bool], seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if condition():
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise AssertionError(f"condition not met within {seconds} s")


def router_models(router: str, state: str = "loaded") -> set[str]:
    data = httpx.get(f"{router}/models").json()["data"]
    return {model["id"] for model in data if model["status"]["value"] == state}


def router_processing(router: str, model: str) -> bool:
    slots = httpx.get(f"{router}/slots", params={"model": model}).json()
    return any(slot["is_processing"] for slot in slots)


@pytest.fixture(autouse=True)
def empty_router(router: str) -> None:
    """Unload everything, directly through the router, so each test starts clean."""
    for model in router_models(router) | router_models(router, "loading"):
        httpx.post(f"{router}/models/unload", json={"model": model})
    wait_until(lambda: not router_models(router) and not router_models(router, "loading"), 60)


# The broker


@dataclass
class Broker:
    url: str
    process: subprocess.Popen[bytes]
    log: Path

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(10)


@pytest.fixture
def start_broker(
    environment: Environment, router: str, tmp_path: Path
) -> Iterator[Callable[..., Broker]]:
    """Start broker processes against the router; each is stopped after the test.

    Keyword arguments are broker settings, e.g. ``idle_protect_seconds=0``. Each broker
    logs at DEBUG to a file in the test's temporary directory.
    """
    brokers: list[Broker] = []

    def start(idle_protect_seconds: float = 0, max_wait_seconds: float = 0) -> Broker:
        port = free_port()
        log = tmp_path / f"broker-{len(brokers)}.log"
        env = os.environ | {
            "MODEL_BROKER_LLAMA_URL": router,
            "MODEL_BROKER_PRESET_PATH": str(environment.preset),
            "MODEL_BROKER_IDLE_PROTECT_SECONDS": str(idle_protect_seconds),
            "MODEL_BROKER_MAX_WAIT_SECONDS": str(max_wait_seconds),
            "MODEL_BROKER_LOAD_TIMEOUT_SECONDS": "300",
            "MODEL_BROKER_LOG_LEVEL": "DEBUG",
        }
        env.pop("MODEL_BROKER_API_KEYS_FILE", None)
        process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "--app-dir", str(PROJECT / "src"),
                "--factory", "model_broker.application:create_app",
                "--host", "127.0.0.1", "--port", str(port), "--no-access-log",
            ],
            env=env,
            stdout=log.open("wb"),
            stderr=subprocess.STDOUT,
        )  # fmt: skip
        broker = Broker(f"http://127.0.0.1:{port}", process, log)
        brokers.append(broker)
        wait_until(lambda: httpx.get(f"{broker.url}/health").status_code == 200, 30)
        health = httpx.get(f"{broker.url}/health").json()
        assert health["status"] == "ok", health["router"]["problems"]
        return broker

    yield start
    for broker in brokers:
        if broker.process.poll() is None:
            broker.stop()


def chat_body(
    model: str, prompt: str = "Say hello.", max_tokens: int = 8, **extra: Any
) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    return {"model": model, "messages": messages, "max_tokens": max_tokens, **extra}


async def chat(broker: Broker, model: str) -> httpx.Response:
    async with httpx.AsyncClient(base_url=broker.url, timeout=TIMEOUT) as client:
        return await client.post("/v1/chat/completions", json=chat_body(model))


def run(test: Callable[[], Any]) -> None:
    asyncio.run(test())


async def stream_lines(response: httpx.Response) -> AsyncIterator[str]:
    async for line in response.aiter_lines():
        if line:
            yield line


# The tests


def test_gpu0_and_gpu1_models_stay_loaded_together(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """Models on different GPUs do not conflict: loading the second keeps the first.

    This is what the router's own LRU cannot do: with models-max 1 it would unload the
    GPU0 model for no reason.
    """
    broker = start_broker(idle_protect_seconds=60)

    async def test() -> None:
        assert (await chat(broker, models.gpu0)).status_code == 200
        assert (await chat(broker, models.gpu1)).status_code == 200

    run(test)
    assert router_models(router) == {models.gpu0, models.gpu1}


def test_only_the_conflicting_model_is_unloaded(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """A second GPU1 model replaces the first GPU1 model, and the GPU0 model stays."""
    broker = start_broker()

    async def test() -> None:
        for model in (models.gpu0, models.gpu1, models.other_gpu1):
            assert (await chat(broker, model)).status_code == 200

    run(test)
    assert router_models(router) == {models.gpu0, models.other_gpu1}


def test_a_gpu_model_and_a_cpu_model_run_together(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """A GPU-resident model only records its RAM use, so a CPU model loads beside two GPU
    models: three models at once, the most the router's models-max 3 allows."""
    broker = start_broker()

    async def test() -> None:
        for model in (models.gpu0, models.gpu1, models.cpu):
            assert (await chat(broker, model)).status_code == 200

    run(test)
    assert router_models(router) == {models.gpu0, models.gpu1, models.cpu}


def test_a_dual_gpu_model_replaces_both_single_gpu_models(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """The dual-GPU model needs both cards, so both single-GPU models are unloaded first;
    a single-GPU model then replaces it in turn."""
    broker = start_broker()

    async def test() -> None:
        for model in (models.gpu0, models.gpu1, models.dual):
            assert (await chat(broker, model)).status_code == 200
        assert router_models(router) == {models.dual}
        assert (await chat(broker, models.gpu0)).status_code == 200

    run(test)
    assert router_models(router) == {models.gpu0}


def test_an_active_stream_is_never_interrupted(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """A request needing a streaming model's GPU waits for the stream to end.

    Setup: a long answer streams from the GPU1 model; once it has started, a request for the
    other GPU1 model is sent, with up to 60 s to wait. Expect the stream to finish with
    ``[DONE]``, and the second request to succeed only after that.
    """
    broker = start_broker(max_wait_seconds=60)

    async def test() -> None:
        async with httpx.AsyncClient(base_url=broker.url, timeout=TIMEOUT) as client:
            body = chat_body(models.gpu1, LONG_ANSWER, max_tokens=400, stream=True)
            async with client.stream("POST", "/v1/chat/completions", json=body) as stream:
                lines = stream_lines(stream)
                await anext(lines)
                competing = asyncio.create_task(chat(broker, models.other_gpu1))
                received = [line async for line in lines]
                stream_ended = time.monotonic()
            response = await competing
            answered = time.monotonic()

        assert received[-1] == "data: [DONE]"
        assert response.status_code == 200
        assert answered > stream_ended

    run(test)
    assert router_models(router) == {models.other_gpu1}


def test_recent_use_refuses_then_admits(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """Within the protection window a conflicting request is refused with Retry-After; after
    it, the same request succeeds."""
    broker = start_broker(idle_protect_seconds=5)

    async def test() -> None:
        assert (await chat(broker, models.gpu1)).status_code == 200
        refused = await chat(broker, models.other_gpu1)
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "resource_busy"
        retry_after = int(refused.headers["retry-after"])
        assert 1 <= retry_after <= 5

        await asyncio.sleep(retry_after + 0.5)
        assert (await chat(broker, models.other_gpu1)).status_code == 200

    run(test)
    assert router_models(router) == {models.other_gpu1}


def test_work_sent_directly_to_the_router_is_respected(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """While the GPU1 model serves a request that bypassed the broker, the broker will not
    unload it: it sees the work through llama.cpp's slots and metrics."""
    broker = start_broker()

    async def test() -> None:
        assert (await chat(broker, models.gpu1)).status_code == 200
        async with httpx.AsyncClient(base_url=router, timeout=TIMEOUT) as direct:
            body = chat_body(models.gpu1, LONG_ANSWER, max_tokens=400, stream=True)
            async with direct.stream("POST", "/v1/chat/completions", json=body) as stream:
                lines = stream_lines(stream)
                await anext(lines)
                refused = await chat(broker, models.other_gpu1)
                assert refused.status_code == 503
                assert "working" in refused.json()["error"]["message"]
                _ = [line async for line in lines]

    run(test)
    assert router_models(router) == {models.gpu1}


def test_a_client_leaving_stops_generation_and_frees_the_model(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """When a streaming client disconnects, llama.cpp stops generating for it, and the model
    becomes displaceable at once.

    Setup: a long answer streams from the GPU1 model; after its first lines, the client
    closes the connection. Expect the router to stop processing within a few seconds, and a
    request for the other GPU1 model (with no wait allowed) then to succeed.
    """
    broker = start_broker()

    async def test() -> None:
        async with httpx.AsyncClient(base_url=broker.url, timeout=TIMEOUT) as client:
            body = chat_body(models.gpu1, LONG_ANSWER, max_tokens=2000, stream=True)
            async with client.stream("POST", "/v1/chat/completions", json=body) as stream:
                lines = stream_lines(stream)
                for _ in range(3):
                    await anext(lines)
                assert router_processing(router, models.gpu1)
        await asyncio.to_thread(wait_until, lambda: not router_processing(router, models.gpu1), 5)
        assert (await chat(broker, models.other_gpu1)).status_code == 200

    run(test)
    assert router_models(router) == {models.other_gpu1}


def test_a_broker_restart_keeps_the_loaded_models(
    start_broker: Callable[..., Broker], router: str, models: Models
) -> None:
    """A restarted broker adopts what the router holds instead of discarding it, and
    serves those models without loading them again."""
    first = start_broker()

    async def load() -> None:
        for model in (models.gpu0, models.gpu1):
            assert (await chat(first, model)).status_code == 200

    run(load)
    first.stop()

    second = start_broker()
    assert httpx.get(f"{second.url}/health").json()["loaded"] == sorted([models.gpu0, models.gpu1])

    async def use() -> None:
        assert (await chat(second, models.gpu0)).status_code == 200

    run(use)
    assert router_models(router) == {models.gpu0, models.gpu1}
    assert '"swap_started"' not in second.log.read_text()
