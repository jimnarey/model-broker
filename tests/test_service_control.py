"""Tests for the service-control helper, with a stub ``docker`` in place of Docker.

The stub keeps one container's state in a JSON file beside it, records every call with the
environment it received, and can be told to fail, hang or stop slowly.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

STUB = r"""#!{python}
import json, os, sys, time
from pathlib import Path

path = Path(__file__).with_name("state.json")
state = json.loads(path.read_text())
args = sys.argv[1:]
base = {{"PATH", "HOME", "DOCKER_CONFIG"}}
state["calls"].append({{"argv": args, "env": dict(os.environ), "cwd": os.getcwd()}})
code = 0
if state.get("fail"):
    print(state["fail"], file=sys.stderr)
    code = 1
elif state.get("hang"):
    path.write_text(json.dumps(state))
    time.sleep(60)
elif args[0] == "ps":
    ids = ["c0ffee"] * (2 if state.get("duplicate") else int(bool(state["container"])))
    print("\n".join(ids))
elif args[0] == "inspect":
    box = state["container"]
    print(json.dumps([{{
        "State": {{"Status": box["status"], "ExitCode": box["exit"], "OOMKilled": False,
                  "Health": {{"Status": "healthy"}}}},
        "Config": {{"Env": ["PATH=/bin"] + [f"{{k}}={{v}}" for k, v in box["env"].items()]}},
    }}]))
elif args[0] == "compose":
    verb = next(a for a in args[1:] if a in ("up", "stop"))
    if verb == "up":
        env = {{k: v for k, v in os.environ.items() if k not in base}}
        state["container"] = {{"status": "running", "exit": 0, "env": env}}
    else:
        time.sleep(state.get("stop_delay", 0))
        if state["container"]:
            state["container"].update(status="exited", exit=137)
path.write_text(json.dumps(state))
sys.exit(code)
"""

CONFIG = """\
[control]
docker = {docker}
compose-file = {compose}
project-directory = {directory}
project-name = demo
start-timeout-seconds = {start}
stop-timeout-seconds = {stop}
[strata]
stop-grace-seconds = 7
allowed-env = STRATA_MODEL STRATA_VISION
[comfyui]
compose-service = comfy
"""


class World:
    """A stub docker, a configuration that uses it, and access to what the stub saw."""

    def __init__(self, directory: Path, service_control: Any, start: float, stop: float) -> None:
        self.module = service_control
        self.docker = directory / "docker"
        self.docker.write_text(STUB.format(python=sys.executable))
        self.docker.chmod(0o755)
        self.state_file = directory / "state.json"
        self.configure(container=None)
        compose = directory / "compose.yml"
        compose.write_text("services: {}\n")
        self.config_file = directory / "service-control.conf"
        self.config_file.write_text(
            CONFIG.format(
                docker=self.docker, compose=compose, directory=directory, start=start, stop=stop
            )
        )
        self.config = service_control.load_config(self.config_file)
        self.controller = service_control.Controller(self.config, directory / "state")
        self.strata = self.config.services["strata"]

    def configure(self, **changes: Any) -> None:
        state = {"calls": [], "container": None} | (
            json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        )
        self.state_file.write_text(json.dumps(state | changes))

    def state(self) -> dict[str, Any]:
        return json.loads(self.state_file.read_text())

    def verbs(self) -> list[str]:
        return [
            c["argv"][0]
            if c["argv"][0] != "compose"
            else "compose " + c["argv"][-4 if "up" in c["argv"] else -3]
            for c in self.state()["calls"]
        ]


@pytest.fixture
def world(tmp_path: Path, service_control: Any) -> World:
    return World(tmp_path, service_control, start=10, stop=10)


def request_bytes(operation: str, service: str = "strata", **extra: Any) -> bytes:
    message = {"version": 1, "id": "r1", "request": operation, "service": service} | extra
    return json.dumps(message).encode()


# Configuration


def test_shipped_example_configuration_is_valid(service_control: Any) -> None:
    """Load model-broker-service-control.conf.example without checking the paths exist.

    Expect Strata with its seven allowed variables and a 60 s grace period, and the defaults
    inherited from [control]; a second service needs no more than its section."""
    example = Path(__file__).parents[1] / "model-broker-service-control.conf.example"
    config = service_control.load_config(example)
    strata = config.services["strata"]
    assert (len(strata.allowed_env), strata.stop_grace_seconds, strata.project_name) == (
        7,
        60,
        "server_containers",
    )


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        pytest.param("", "[control]", id="no-control-section"),
        pytest.param("[control]\n", "no services", id="no-services"),
        pytest.param("[control]\nproject-name = p\n[a]\n", "absolute path", id="no-compose-file"),
        pytest.param(
            "[control]\ncompose-file = /c\nproject-directory = /d\n[a]\n",
            "project-name",
            id="no-project-name",
        ),
        pytest.param(
            "[control]\ncompose-file = /c\nproject-directory = /d\nproject-name = p\n"
            "[a]\nallowed-env = lower\n",
            "invalid name",
            id="bad-env-name",
        ),
        pytest.param(
            "[control]\ncompose-file = /c\nproject-directory = /d\nproject-name = p\n[Bad Name]\n",
            "valid Compose service name",
            id="bad-service-name",
        ),
        pytest.param(
            "[control]\ncompose-file = /c\nproject-directory = /d\nproject-name = p\n"
            "start-timeout-seconds = 0\n[a]\n",
            "must be positive",
            id="zero-timeout",
        ),
    ],
)
def test_bad_configuration_is_rejected(
    tmp_path: Path, service_control: Any, text: str, fragment: str
) -> None:
    """Load a configuration with one defect each, as the id says. Expect ConfigError naming it:
    a helper with root-equivalent reach must refuse to start on a configuration it half-reads."""
    path = tmp_path / "bad.conf"
    path.write_text(text)
    with pytest.raises(service_control.ConfigError, match=fragment):
        service_control.load_config(path)


def test_missing_paths_stop_startup(world: World) -> None:
    """Check files() on a configuration whose compose file has been deleted. Expect ConfigError
    naming it, so a typo surfaces at startup rather than on the broker's first load."""
    world.strata.compose_file.unlink()
    with pytest.raises(world.module.ConfigError, match=r"compose\.yml"):
        world.module.check_files(world.config)


# Requests


def test_start_request_is_parsed(world: World) -> None:
    """Parse a start request with two allowed settings. Expect the service, the operation and the
    settings, unchanged."""
    request = world.module.parse_request(
        request_bytes("start", settings={"STRATA_MODEL": "IQ2_XS", "STRATA_VISION": ""}),
        world.config,
    )
    assert (request.operation, request.service.name, dict(request.settings)) == (
        "start",
        "strata",
        {"STRATA_MODEL": "IQ2_XS", "STRATA_VISION": ""},
    )


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        pytest.param(b"not json", "UTF-8 JSON", id="not-json"),
        pytest.param(b"[]", "must contain only", id="not-an-object"),
        pytest.param(request_bytes("status", extra=1), "must contain only", id="extra-key"),
        pytest.param(
            request_bytes("status", settings={}), "must contain only", id="settings-on-status"
        ),
        pytest.param(request_bytes("start"), "must contain only", id="start-without-settings"),
        pytest.param(request_bytes("restart"), "not permitted", id="unknown-operation"),
        pytest.param(
            request_bytes("status", service="nope"), "not controlled", id="unknown-service"
        ),
        pytest.param(
            request_bytes("status", version=True), "unsupported protocol", id="boolean-version"
        ),
        pytest.param(
            request_bytes("status", version=2), "unsupported protocol", id="wrong-version"
        ),
        pytest.param(request_bytes("status", id=""), "non-empty string", id="empty-id"),
        pytest.param(
            request_bytes("start", settings={"PATH": "/x"}), "not allowed", id="unlisted-variable"
        ),
        pytest.param(
            request_bytes("start", settings={"STRATA_MODEL": "a b"}), "not permitted", id="space"
        ),
        pytest.param(
            request_bytes("start", settings={"STRATA_MODEL": "$(id)"}), "not permitted", id="dollar"
        ),
        pytest.param(
            request_bytes("start", settings={"STRATA_MODEL": "a\nb"}), "not permitted", id="newline"
        ),
        pytest.param(
            request_bytes("start", settings={"STRATA_MODEL": "x" * 513}),
            "not permitted",
            id="too-long",
        ),
        pytest.param(
            request_bytes("start", settings={"STRATA_MODEL": 5}), "not permitted", id="not-a-string"
        ),
        pytest.param(request_bytes("start", settings=[]), "must be an object", id="settings-list"),
    ],
)
def test_bad_requests_are_refused(world: World, raw: bytes, fragment: str) -> None:
    """Parse a request with one defect each, as the id says. Expect RequestError naming it, and
    in particular for the unlisted variable and the unsafe values: only listed names with plain
    characters may reach Compose's interpolation."""
    with pytest.raises(world.module.RequestError, match=fragment):
        world.module.parse_request(raw, world.config)


@pytest.mark.parametrize(
    ("credentials", "groups", "allowed"),
    [
        pytest.param((10, 0, 0), "", True, id="root"),
        pytest.param((10, 1000, 42), "", True, id="primary-group"),
        pytest.param((10, 1000, 1000), "Groups:\t4 42 100\n", True, id="supplementary-group"),
        pytest.param((10, 1000, 1000), "Groups:\t4 100\n", False, id="not-a-member"),
        pytest.param((10, 1000, 1000), "", False, id="status-unreadable"),
    ],
)
def test_peer_authorisation(
    service_control: Any, credentials: tuple[int, int, int], groups: str, allowed: bool
) -> None:
    """Check a peer (pid, uid, gid) against allowed group 42, with /proc/pid/status as given.

    Root and members of group 42 (primary or supplementary) are admitted; anyone else, or a
    peer whose groups cannot be read, is refused."""
    packed = struct.pack("3i", *credentials)
    assert service_control.peer_is_authorised(packed, 42, lambda _: groups) is allowed


# Docker, through the stub


def test_status_of_a_service_without_a_container_is_missing(world: World) -> None:
    """Ask for the status when Docker lists no container. Expect state missing and no settings,
    with Docker asked for this Compose project's service only (not one-off runs)."""
    status = world.controller.status(world.strata)
    argv = world.state()["calls"][0]["argv"]
    assert (status["state"], status["settings"]) == ("missing", {})
    assert [a for a in argv if a.startswith("label=")] == [
        "label=com.docker.compose.project=demo",
        "label=com.docker.compose.service=strata",
        "label=com.docker.compose.oneoff=False",
    ]


def test_status_reports_only_the_allowed_settings_and_exit_code_when_gone(world: World) -> None:
    """Give the stub a stopped container whose environment holds an allowed setting, an unlisted
    one (SECRET) and PATH. Expect only the allowed one back, and the exit code (137) since the
    container is gone; a running container reports none."""
    box = {"status": "exited", "exit": 137, "env": {"STRATA_MODEL": "m", "SECRET": "s"}}
    world.configure(container=box)
    gone = world.controller.status(world.strata)
    world.configure(container=box | {"status": "running", "exit": 0})
    running = world.controller.status(world.strata)
    assert (gone["settings"], gone["exit_code"], running["exit_code"]) == (
        {"STRATA_MODEL": "m"},
        137,
        None,
    )


def test_start_runs_compose_up_with_fixed_arguments_and_only_the_settings(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start with one setting while the helper's own environment holds a secret.

    Expect compose called with this service's file, directory and project, `up --detach
    --no-deps --no-build --pull never strata`, run in the project directory, and an
    environment of PATH, HOME, DOCKER_CONFIG plus the setting: nothing inherited, so the
    secret cannot reach Compose. The result is the container's new status."""
    monkeypatch.setenv("SECRET", "leaked")
    result = world.controller.start(world.strata, {"STRATA_MODEL": "m"})
    call = next(c for c in world.state()["calls"] if "up" in c["argv"])
    assert call["argv"] == [
        "compose",
        "--file", str(world.strata.compose_file),
        "--project-directory", str(world.strata.project_directory),
        "--project-name", "demo",
        "up", "--detach", "--no-deps", "--no-build", "--pull", "never", "strata",
    ]  # fmt: skip
    # The stub is a Python script, and Python adds LC_CTYPE to its own environment.
    environment = set(call["env"]) - {"LC_CTYPE"}
    assert (environment, call["env"]["STRATA_MODEL"], call["cwd"]) == (
        {"PATH", "HOME", "DOCKER_CONFIG", "STRATA_MODEL"},
        "m",
        str(world.strata.project_directory),
    )
    assert (result["state"], result["settings"]) == ("running", {"STRATA_MODEL": "m"})


def test_stop_runs_compose_stop_with_the_services_grace_period(world: World) -> None:
    """Start then stop. Expect `stop --timeout 7 strata` (the configured grace) and a status of
    exited with exit code 137, which is how Docker reports a container it had to kill: the
    caller must not read that as a crash."""
    world.controller.start(world.strata, {})
    result = world.controller.stop(world.strata)
    call = next(c for c in world.state()["calls"] if "stop" in c["argv"])
    assert (call["argv"][-3:], result["state"], result["exit_code"]) == (
        ["--timeout", "7", "strata"],
        "exited",
        137,
    )


def test_compose_service_name_is_used_when_configured(world: World) -> None:
    """Stop `comfyui`, whose Compose service is named `comfy`. Expect the compose argument to be
    `comfy`, not the section name."""
    world.controller.stop(world.config.services["comfyui"])
    argv = next(c["argv"] for c in world.state()["calls"] if "stop" in c["argv"])
    assert argv[-1] == "comfy"


def test_a_docker_failure_is_reported_with_its_message(world: World) -> None:
    """Make docker exit 1 with a message. Expect OperationError carrying the exit code and the
    message, so the broker can log why a model did not start."""
    world.configure(fail="no such image: strata:latest")
    with pytest.raises(world.module.OperationError, match=r"failed \(1\): no such image"):
        world.controller.start(world.strata, {})


def test_a_hung_docker_command_times_out(tmp_path: Path, service_control: Any) -> None:
    """Make docker hang with a 0.3 s start timeout. Expect OperationError saying it timed out,
    not a hang: a stuck Compose must not hold the helper."""
    world = World(tmp_path, service_control, start=0.3, stop=10)
    world.configure(hang=True)
    with pytest.raises(service_control.OperationError, match="timed out"):
        world.controller.start(world.strata, {})


def test_two_containers_for_one_service_is_an_error(world: World) -> None:
    """Make Docker list two containers for the service. Expect OperationError: the helper
    cannot say which one the broker means."""
    world.configure(duplicate=True, container={"status": "running", "exit": 0, "env": {}})
    with pytest.raises(world.module.OperationError, match="2 containers"):
        world.controller.status(world.strata)


# Over the socket


@pytest.fixture
def server(world: World) -> Iterator[tuple[World, Path]]:
    """A ControlServer on a temporary socket, admitting this process's own group."""
    path = Path(f"/tmp/mbsc-{os.getpid()}-{time.monotonic_ns()}.sock")
    instance = world.module.ControlServer(str(path), world.controller, os.getgid())
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield world, path
    finally:
        instance.shutdown()
        instance.server_close()
        path.unlink(missing_ok=True)


def send(path: Path, raw: bytes, timeout: float = 10) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(raw + b"\n")
        return json.loads(client.makefile("rb").readline())


def test_status_round_trip_over_the_socket(server: tuple[World, Path]) -> None:
    """Send a status request over the socket and read the reply. Expect ok, the request's id and
    version echoed, and the container's status as the result."""
    world, path = server
    world.configure(container={"status": "running", "exit": 0, "env": {"STRATA_MODEL": "m"}})
    reply = send(path, request_bytes("status"))
    assert (reply["ok"], reply["id"], reply["version"], reply["result"]["settings"]) == (
        True,
        "r1",
        1,
        {"STRATA_MODEL": "m"},
    )


def test_a_refused_request_gets_an_error_and_no_docker_call(server: tuple[World, Path]) -> None:
    """Send a start with an unlisted variable. Expect ok false with the reason, and the stub
    never run: refusal happens before any command is built."""
    world, path = server
    reply = send(path, request_bytes("start", settings={"PATH": "/evil"}))
    assert (reply["ok"], "not allowed" in reply["error"], world.state()["calls"]) == (
        False,
        True,
        [],
    )


def test_a_peer_outside_the_group_is_refused(world: World) -> None:
    """Serve with an allowed group that is not ours (and we are not root). Expect ok false
    'not authorised' and no docker call. Skipped when the tests run as root."""
    if os.getuid() == 0:
        pytest.skip("root is always authorised")
    path = Path(f"/tmp/mbsc-deny-{os.getpid()}.sock")
    instance = world.module.ControlServer(str(path), world.controller, os.getgid() + 12345)
    threading.Thread(target=instance.serve_forever, daemon=True).start()
    try:
        reply = send(path, request_bytes("status"))
    finally:
        instance.shutdown()
        instance.server_close()
        path.unlink(missing_ok=True)
    assert (reply["ok"], "not authorised" in reply["error"], world.state()["calls"]) == (
        False,
        True,
        [],
    )


def test_an_oversized_request_is_refused(server: tuple[World, Path]) -> None:
    """Send 20 000 bytes with no newline. Expect ok false 'exceeds maximum size'."""
    _, path = server
    reply = send(path, b"x" * 20_000)
    assert (reply["ok"], "maximum size" in reply["error"]) == (False, True)


def test_a_slow_stop_does_not_block_status_requests(server: tuple[World, Path]) -> None:
    """Start a stop that takes 1 s, then ask for status 0.2 s in.

    Expect the status reply well before the stop finishes (under 0.8 s), because requests are
    served on separate threads; the broker polls status while a model is unloading."""
    world, path = server
    world.configure(container={"status": "running", "exit": 0, "env": {}}, stop_delay=1.0)
    stopper = threading.Thread(target=send, args=(path, request_bytes("stop")))
    stopper.start()
    time.sleep(0.2)
    started = time.monotonic()
    reply = send(path, request_bytes("status"))
    waited = time.monotonic() - started
    stopper.join()
    assert (reply["ok"], waited < 0.8) == (True, True)


# The socket


def test_prepare_socket_gives_the_group_the_directory(tmp_path: Path, service_control: Any) -> None:
    """Prepare a socket path in a new directory, naming our own group.

    Expect the group's ID back and the directory made group-readable (0750): the broker
    container mounts the directory, so a restart that replaces the socket file does not leave
    it holding a stale one."""
    group = os.getgid()
    import grp

    gid = service_control.prepare_socket(tmp_path / "run" / "c.sock", grp.getgrgid(group).gr_name)
    assert (gid, oct((tmp_path / "run").stat().st_mode & 0o777)) == (group, "0o750")


def test_prepare_socket_removes_a_stale_socket_but_not_other_files(
    tmp_path: Path, service_control: Any
) -> None:
    """Prepare over a stale socket (removed) and over a regular file (refused)."""
    import grp

    name = grp.getgrgid(os.getgid()).gr_name
    stale = tmp_path / "s.sock"
    with socket.socket(socket.AF_UNIX) as old:
        old.bind(str(stale))
    service_control.prepare_socket(stale, name)
    stale.write_text("not a socket")
    with pytest.raises(ValueError, match="non-socket"):
        service_control.prepare_socket(stale, name)


# The broker's client against this helper


def test_the_brokers_client_drives_the_helper(server: tuple[World, Path]) -> None:
    """Start, check and stop a service through model_broker.servicecontrol.ServiceControl,
    talking to the real helper server (with the stub docker) over its socket.

    Expect the start's settings to reach Docker, the statuses to parse, and the exit code 137
    that Docker gives a stopped container to come back as such. This is the one test that has
    both halves of the protocol in it, so they cannot drift apart unnoticed."""
    import asyncio

    from model_broker.servicecontrol import ServiceControl

    world, path = server

    async def drive() -> list[Any]:
        control = ServiceControl(path, timeout_seconds=10)
        started = await control.start("strata", {"STRATA_MODEL": "m", "STRATA_VISION": ""})
        checked = await control.status("strata")
        stopped = await control.stop("strata")
        return [started, checked, stopped]

    started, checked, stopped = asyncio.run(drive())
    assert (started.state, dict(started.settings), started.occupying) == (
        "running",
        {"STRATA_MODEL": "m", "STRATA_VISION": ""},
        True,
    )
    assert checked.settings == started.settings
    assert (stopped.state, stopped.exit_code, stopped.occupying) == ("exited", 137, False)
    up = next(c for c in world.state()["calls"] if "up" in c["argv"])
    assert (up["env"]["STRATA_MODEL"], up["env"]["STRATA_VISION"]) == ("m", "")


def test_a_refusal_from_the_helper_reaches_the_broker_as_an_error(
    server: tuple[World, Path],
) -> None:
    """Ask the helper, through the broker's client, to set a variable it does not allow.

    Expect ServiceControlError carrying the helper's reason, and no Docker call."""
    import asyncio

    from model_broker.servicecontrol import ServiceControl, ServiceControlError

    world, path = server

    async def drive() -> None:
        await ServiceControl(path, timeout_seconds=10).start("strata", {"PATH": "/evil"})

    with pytest.raises(ServiceControlError, match="not allowed for strata"):
        asyncio.run(drive())
    assert world.state()["calls"] == []
