"""Opt-in container test for a clean host-facts installation under systemd.

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
DIRECTORY = "/run/model-broker-host-facts"
SOCKET = f"{DIRECTORY}/facts.sock"
CTL = "/usr/local/libexec/model-broker-host-facts/model-broker-host-factsctl.py"

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


def test_clean_install_starts_the_service_and_serves_authorised_clients(container: Exec) -> None:
    """A clean install starts the helper and admits a client in its dedicated group.

    Setup: an otherwise empty systemd container runs the installer as root. The test then
    creates an ordinary user in the socket's client group and bind-mounts the runtime
    *directory* into ``/mnt``. The mount models deployment: the broker container receives the
    directory, not the socket file inside it, which the helper replaces whenever it restarts.

    Expect: systemd reports the service active, the directory and socket have the documented
    root, group, and permissions (0750 and 0660), and the ordinary group member can request both
    supported operations through the mount, which needs the group to be able to enter the
    directory. This checks the installed service's useful behaviour without creating or testing
    any legacy supervisor or migration path.
    """
    installed = container("bash", "/opt/model-broker/install-model-broker-host-facts.sh")
    assert installed.returncode == 0, installed.stderr

    assert container("systemctl", "is-active", "model-broker-host-facts.service").returncode == 0
    ownership = container("stat", "--format=%U:%G:%a", DIRECTORY, SOCKET).stdout.split()
    assert ownership == [
        "root:model-broker-host-facts-client:750",
        "root:model-broker-host-facts-client:660",
    ]
    assert (
        container(
            "useradd",
            "--create-home",
            "--groups",
            "model-broker-host-facts-client",
            "broker-client",
        ).returncode
        == 0
    )
    assert container("mkdir", "-p", "/mnt/facts").returncode == 0
    mounted = container("mount", "--bind", DIRECTORY, "/mnt/facts")
    assert mounted.returncode == 0, mounted.stderr

    for operation in ("inventory", "utilisation"):
        response = ask_facts(container, operation)
        assert response.returncode == 0, response.stderr
        assert json.loads(response.stdout)["ok"] is True


def ask_facts(container: Exec, operation: str) -> subprocess.CompletedProcess[str]:
    """Send a host-facts request as the unprivileged group member, through the mount."""
    return container(
        "runuser", "--user", "broker-client", "--", "python3", CTL,
        "--socket", "/mnt/facts/facts.sock", operation,
    )  # fmt: skip


def test_host_facts_restart_keeps_the_directory_the_broker_mounts(container: Exec) -> None:
    """Restart the installed facts helper and check the broker's mount would still work.

    Setup: install, mount the runtime directory as the broker container does, restart the unit,
    then request the inventory through the old mount as the group member.

    Expect the directory to keep its inode (systemd deletes a RuntimeDirectory on stop unless
    told to preserve it) and the request through the old mount to succeed. Mounting the socket
    file instead would fail here even with the directory kept, as the helper replaces the file."""
    assert container("bash", "/opt/model-broker/install-model-broker-host-facts.sh").returncode == 0
    container(
        "useradd", "--create-home", "--groups", "model-broker-host-facts-client", "broker-client"
    )
    container("mkdir", "-p", "/mnt/facts")
    assert container("mount", "--bind", DIRECTORY, "/mnt/facts").returncode == 0
    before = container("stat", "--format=%i", DIRECTORY).stdout

    restarted = container("systemctl", "restart", "model-broker-host-facts.service")
    assert restarted.returncode == 0, restarted.stderr
    after = container("stat", "--format=%i", DIRECTORY).stdout

    response = ask_facts(container, "inventory")
    assert (before == after, response.returncode) == (True, 0), response.stderr


SERVICE_CONTROL_SOCKET = "/run/model-broker-service-control/control.sock"
SERVICE_CONTROL_CTL = (
    "/usr/local/libexec/model-broker-service-control/model-broker-service-controlctl.py"
)
SERVICE_CONTROL_CONFIG = """\
[control]
docker = /usr/bin/docker
compose-file = /srv/compose.yml
project-directory = /srv
project-name = demo
[strata]
allowed-env = STRATA_MODEL
"""


def test_service_control_install_serves_authorised_clients_only(container: Exec) -> None:
    """A clean install of the service-control helper starts it and admits only its client group.

    Setup: the systemd container gets a configuration naming one service, a Compose file, and a
    stand-in `docker` that lists no containers (so no Docker daemon is needed). The installer
    runs as root and keeps that configuration. A group member and an outsider then ask for the
    service's status, with the socket's *directory* bind-mounted into ``/mnt`` as the broker
    container would receive it.

    Expect: the unit active; the directory and socket owned by root and the client group with
    0750 and 0660; the group member told the service is missing; the outsider refused; and an
    unlisted service refused. The helper must also survive its own hardening: ProtectSystem and
    the rest allow it to run the (stand-in) docker and answer."""
    prepared = container(
        "bash",
        "-c",
        "mkdir -p /etc/model-broker /srv && touch /srv/compose.yml"
        " && printf '#!/bin/sh\\nexit 0\\n' > /usr/bin/docker && chmod 755 /usr/bin/docker"
        f" && cat > /etc/model-broker/service-control.conf <<'EOF'\n{SERVICE_CONTROL_CONFIG}EOF",
    )
    assert prepared.returncode == 0, prepared.stderr
    installed = container("bash", "/opt/model-broker/install-model-broker-service-control.sh")
    assert installed.returncode == 0, installed.stderr

    assert (
        container("systemctl", "is-active", "model-broker-service-control.service").returncode == 0
    )
    owners = container(
        "stat",
        "--format=%U:%G:%a",
        "/run/model-broker-service-control",
        SERVICE_CONTROL_SOCKET,
    ).stdout.split()
    assert owners == [
        "root:model-broker-control-client:750",
        "root:model-broker-control-client:660",
    ]

    container("useradd", "--create-home", "--groups", "model-broker-control-client", "member")
    container("useradd", "--create-home", "outsider")
    container("mkdir", "-p", "/mnt/control")
    mounted = container("mount", "--bind", "/run/model-broker-service-control", "/mnt/control")
    assert mounted.returncode == 0, mounted.stderr

    def ask(user: str, service: str) -> subprocess.CompletedProcess[str]:
        return container(
            "runuser", "--user", user, "--", "python3", SERVICE_CONTROL_CTL,
            "--socket", "/mnt/control/control.sock", "status", service,
        )  # fmt: skip

    member = ask("member", "strata")
    assert member.returncode == 0, member.stderr
    assert json.loads(member.stdout)["result"]["state"] == "missing"
    assert "not controlled" in ask("member", "other").stdout
    assert ask("outsider", "strata").returncode != 0


def test_service_control_restart_keeps_the_directory_the_broker_mounts(container: Exec) -> None:
    """Restart the installed helper and check the broker's mount would still work.

    Setup: install as in the test above, bind-mount the runtime directory (as the broker
    container does), restart the unit, then ask for a status through the mount.

    Expect the directory to keep its inode across the restart (systemd deletes a
    RuntimeDirectory on stop unless told to preserve it), and the request through the old mount
    to succeed. With a new directory the mount would show the deleted one, empty, and the
    broker could not reach the helper again without being restarted itself."""
    container(
        "bash",
        "-c",
        "mkdir -p /etc/model-broker /srv && touch /srv/compose.yml"
        " && printf '#!/bin/sh\\nexit 0\\n' > /usr/bin/docker && chmod 755 /usr/bin/docker"
        f" && cat > /etc/model-broker/service-control.conf <<'EOF'\n{SERVICE_CONTROL_CONFIG}EOF",
    )
    assert (
        container("bash", "/opt/model-broker/install-model-broker-service-control.sh").returncode
        == 0
    )
    container("mkdir", "-p", "/mnt/control")
    assert (
        container("mount", "--bind", "/run/model-broker-service-control", "/mnt/control").returncode
        == 0
    )
    before = container("stat", "--format=%i", "/run/model-broker-service-control").stdout

    restarted = container("systemctl", "restart", "model-broker-service-control.service")
    assert restarted.returncode == 0, restarted.stderr
    after = container("stat", "--format=%i", "/run/model-broker-service-control").stdout

    status = container(
        "python3", SERVICE_CONTROL_CTL, "--socket", "/mnt/control/control.sock", "status", "strata"
    )
    assert (before == after, status.returncode) == (True, 0), status.stderr
