"""Opt-in container test for installation, systemd activation, and migration.

Build the image first: ``docker build --file tests/systemd/Dockerfile --tag
model-broker-systemd-test:local tests``, then set MODEL_BROKER_RUN_SYSTEMD_INTEGRATION=1.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

IMAGE = "model-broker-systemd-test:local"
PROJECT = Path(__file__).parents[1].resolve()
SOCKET = "/run/model-broker-host-facts/facts.sock"
CTL = "/usr/local/libexec/model-broker-host-facts/model-broker-host-factsctl.py"
LEGACY_UNIT = """\
[Unit]
Description=legacy Docker supervisor
[Service]
ExecStart=/bin/sleep infinity
[Install]
WantedBy=multi-user.target
"""

Exec = Callable[..., subprocess.CompletedProcess[str]]


def run(*arguments: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, input=input_text, capture_output=True, text=True, check=False)


@pytest.fixture
def container() -> Iterator[Exec]:
    """A disposable, network-isolated systemd container with the project mounted read-only.

    Yields a function that runs one command inside it.
    """
    if os.environ.get("MODEL_BROKER_RUN_SYSTEMD_INTEGRATION") != "1":
        pytest.skip("set MODEL_BROKER_RUN_SYSTEMD_INTEGRATION=1 to run Docker tests")
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")
    if run("docker", "image", "inspect", IMAGE).returncode != 0:
        pytest.skip(f"Docker image {IMAGE} is not built; see tests/systemd/Dockerfile")
    # --privileged and host cgroups let systemd run as PID 1 in the container.
    started = run(
        "docker", "run", "--detach", "--privileged", "--cgroupns=host", "--network", "none",
        "--tmpfs", "/run", "--tmpfs", "/run/lock",
        "--volume", f"{PROJECT}:/opt/model-broker:ro",
        "--name", f"model-broker-host-facts-test-{uuid.uuid4().hex}",
        IMAGE,
    )  # fmt: skip
    if started.returncode != 0:
        pytest.skip(f"Docker cannot start a systemd container: {started.stderr.strip()}")
    container_id = started.stdout.strip()

    def execute(*arguments: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return run(
            "docker", "exec", "--interactive", container_id, *arguments, input_text=input_text
        )

    try:
        # "degraded" (exit 1) is normal in a container: some units cannot start there.
        if execute("systemctl", "is-system-running", "--wait").returncode not in (0, 1):
            pytest.skip("systemd did not start in the container")
        yield execute
    finally:
        run("docker", "rm", "--force", container_id)


def test_installer_replaces_legacy_supervisor_and_serves_facts(container: Exec) -> None:
    """The installer disables the old supervisor, starts the helper, and the helper answers.

    Setup: a stand-in ``llama-supervisor.service`` is installed, enabled, and running, as on a
    host from before the redesign. Then install-model-broker-host-facts.sh runs as root.
    Expect:
    - the new service is active;
    - the legacy service is both stopped and disabled, so it cannot come back at boot (the
      migration must remove the old Docker-capable control path, not run beside it);
    - the socket is root:model-broker-host-facts-client with mode 660, which is the access
      boundary the broker container is granted through group_add;
    - the installed control client gets a successful inventory response.
    """
    legacy = container(
        "tee", "/etc/systemd/system/llama-supervisor.service", input_text=LEGACY_UNIT
    )
    assert legacy.returncode == 0
    assert container("systemctl", "daemon-reload").returncode == 0
    assert container("systemctl", "enable", "--now", "llama-supervisor.service").returncode == 0
    assert container("systemctl", "is-active", "llama-supervisor.service").returncode == 0

    installed = container("bash", "/opt/model-broker/install-model-broker-host-facts.sh")
    assert installed.returncode == 0, installed.stderr

    assert container("systemctl", "is-active", "model-broker-host-facts.service").returncode == 0
    assert container("systemctl", "is-active", "llama-supervisor.service").returncode != 0
    assert container("systemctl", "is-enabled", "llama-supervisor.service").returncode != 0
    ownership = container("stat", "--format=%U:%G:%a", SOCKET).stdout.strip()
    assert ownership == "root:model-broker-host-facts-client:660"
    response = container("python3", CTL, "inventory")
    assert response.returncode == 0, response.stderr
    assert json.loads(response.stdout)["ok"] is True
