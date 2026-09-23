from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path


class SystemdIntegrationTests(unittest.TestCase):
    """Opt-in container tests for installation, systemd activation, and migration."""

    image = "model-broker-systemd-test:local"
    container_id: str

    @classmethod
    def setUpClass(cls) -> None:
        """Require an explicitly enabled Docker environment with a local systemd image."""
        if os.environ.get("MODEL_BROKER_RUN_SYSTEMD_INTEGRATION") != "1":
            raise unittest.SkipTest(
                "set MODEL_BROKER_RUN_SYSTEMD_INTEGRATION=1 to run Docker tests"
            )
        if shutil.which("docker") is None:
            raise unittest.SkipTest("Docker is not installed")
        available = subprocess.run(
            ["docker", "image", "inspect", cls.image], capture_output=True, text=True, check=False
        )
        if available.returncode != 0:
            raise unittest.SkipTest(
                f"Docker image {cls.image} is not available locally; build it with "
                "tests/systemd/Dockerfile"
            )

    def setUp(self) -> None:
        """Start one privileged systemd container without Docker Compose or network access."""
        name = f"model-broker-host-facts-test-{uuid.uuid4().hex}"
        project = Path(__file__).parents[1].resolve()
        started = subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--privileged",
                "--cgroupns=host",
                "--network",
                "none",
                "--tmpfs",
                "/run",
                "--tmpfs",
                "/run/lock",
                "--volume",
                f"{project}:/opt/model-broker:ro",
                "--name",
                name,
                self.image,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if started.returncode != 0:
            self.skipTest(f"Docker cannot start a systemd test container: {started.stderr.strip()}")
        self.container_id = started.stdout.strip()
        self.addCleanup(self._remove_container)
        ready = self._exec("systemctl", "is-system-running", "--wait")
        if ready.returncode not in (0, 1):
            self.skipTest(f"systemd did not start: {ready.stderr.strip()}")

    def _remove_container(self) -> None:
        """Remove the disposable systemd container even after a failed assertion."""
        subprocess.run(
            ["docker", "rm", "--force", self.container_id], capture_output=True, check=False
        )

    def _exec(
        self, *arguments: str, input_text: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run one command in the disposable container and retain its diagnostics."""
        return subprocess.run(
            ["docker", "exec", "--interactive", self.container_id, *arguments],
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_installer_migrates_legacy_service_and_serves_facts(self) -> None:
        """Verify the real installation and safe replacement of the old host service.

        The test first creates and starts a small stand-in for the legacy
        Docker-capable supervisor. It then runs the new installer in a
        disposable systemd container. After installation, the new helper must be
        running while the legacy service is both stopped and disabled.

        The test also checks that the socket has the promised root ownership,
        dedicated group, and restrictive permissions, then makes an inventory
        request through the installed control client. Finally it confirms that
        the container has no Docker socket. Together, these checks show that the
        migration removes the old control path rather than merely adding a new
        service beside it.
        """
        legacy = (
            "[Unit]\n"
            "Description=legacy Docker supervisor\n"
            "[Service]\n"
            "ExecStart=/bin/sleep infinity\n"
            "[Install]\n"
            "WantedBy=multi-user.target\n"
        )
        self.assertEqual(
            self._exec(
                "tee", "/etc/systemd/system/llama-supervisor.service", input_text=legacy
            ).returncode,
            0,
        )
        self.assertEqual(self._exec("systemctl", "daemon-reload").returncode, 0)
        self.assertEqual(
            self._exec("systemctl", "enable", "--now", "llama-supervisor.service").returncode, 0
        )
        self.assertEqual(
            self._exec("systemctl", "is-active", "--quiet", "llama-supervisor.service").returncode,
            0,
        )

        installed = self._exec("bash", "/opt/model-broker/install-model-broker-host-facts.sh")
        self.assertEqual(installed.returncode, 0, installed.stderr)
        self.assertEqual(
            self._exec(
                "systemctl", "is-active", "--quiet", "model-broker-host-facts.service"
            ).returncode,
            0,
        )
        self.assertNotEqual(
            self._exec("systemctl", "is-active", "--quiet", "llama-supervisor.service").returncode,
            0,
        )
        self.assertNotEqual(
            self._exec("systemctl", "is-enabled", "llama-supervisor.service").returncode,
            0,
        )

        socket_path = "/run/model-broker-host-facts/facts.sock"
        socket_mode = self._exec("stat", "--format=%U:%G:%a", socket_path)
        self.assertEqual(socket_mode.stdout.strip(), "root:model-broker-host-facts-client:660")
        response = self._exec(
            "python3",
            "/usr/local/libexec/model-broker-host-facts/model-broker-host-factsctl.py",
            "inventory",
        )
        self.assertEqual(response.returncode, 0, response.stderr)
        self.assertTrue(json.loads(response.stdout)["ok"])
        self.assertEqual(self._exec("test", "!", "-e", "/var/run/docker.sock").returncode, 0)
