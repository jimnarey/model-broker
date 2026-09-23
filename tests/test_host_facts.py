from __future__ import annotations

import importlib.util
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any


class HostFactsTests(unittest.TestCase):
    """Unit and local-socket integration tests for the host-facts protocol."""

    host_facts: Any

    @classmethod
    def setUpClass(cls) -> None:
        """Load the standalone helper script as a module for direct testing."""
        script = Path(__file__).parents[1] / "model-broker-host-facts.py"
        spec = importlib.util.spec_from_file_location("model_broker_host_facts", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.host_facts = module

    def _start_server(self) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
        """Start a local server using predictable facts and the current primary group."""
        if not hasattr(socket, "SO_PEERCRED"):
            self.skipTest("Linux SO_PEERCRED is required for this integration test")
        directory = tempfile.TemporaryDirectory[str]()
        path = Path(directory.name) / "facts.sock"

        class FakeFacts:
            def inventory(self) -> dict[str, Any]:
                """Return a stable result so this test exercises protocol wiring only."""
                return {"kind": "inventory"}

            def utilisation(self) -> dict[str, Any]:
                """Return a stable result so this test exercises protocol wiring only."""
                return {"kind": "utilisation"}

        try:
            server = self.host_facts.FactsServer(str(path), FakeFacts(), os.getgid())
        except PermissionError as error:
            directory.cleanup()
            self.skipTest(f"Unix-domain sockets are unavailable in this environment: {error}")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.addCleanup(directory.cleanup)
        return path, directory

    @staticmethod
    def _request(path: Path, message: bytes) -> dict[str, Any]:
        """Send exactly one newline-delimited protocol request and decode its response."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(path))
            client.sendall(message + b"\n")
            response = client.makefile("rb").readline()
        return json.loads(response)

    def test_protocol_accepts_only_the_fixed_requests(self) -> None:
        """Verify that the request parser accepts only the two published operations.

        The host helper is deliberately an observation service. A client may ask
        for inventory or current utilisation, but it must not be able to turn a
        request into a Docker command, a shell command, or a future accidental
        extension of the protocol. This test checks one valid request, an
        unsupported operation, and an extra command-shaped field.

        Keeping this boundary strict matters because the process is installed
        with host privileges. Rejecting unexpected input before any work is
        performed prevents the socket from becoming a general host-control API.
        """
        request_id, request = self.host_facts.parse_request(
            b'{"version":1,"id":"request-1","request":"inventory"}'
        )

        self.assertEqual(request_id, "request-1")
        self.assertEqual(request, "inventory")
        with self.assertRaisesRegex(self.host_facts.RequestError, "not permitted"):
            self.host_facts.parse_request(b'{"version":1,"id":"request-2","request":"docker"}')
        with self.assertRaisesRegex(self.host_facts.RequestError, "only version"):
            self.host_facts.parse_request(
                b'{"version":1,"id":"request-3","request":"inventory","command":"id"}'
            )

    def test_utilisation_marks_the_first_cpu_sample_unready(self) -> None:
        """Verify that CPU use is reported only after there is a real comparison.

        CPU use is calculated from the difference between two readings. The
        first call therefore has no earlier reading to compare with and must say
        that the sample is not ready, rather than inventing a percentage. The
        second call uses fixed counters to check the calculation and the memory
        fields returned alongside it.

        This matters to the broker and to monitoring: an unknown first reading
        is honest and safe, while a made-up value could make an idle or busy host
        look like the opposite.
        """
        counters = iter(((100, 30), (150, 45)))
        original_counters = self.host_facts.cpu_counters
        original_meminfo = self.host_facts.meminfo
        self.host_facts.cpu_counters = lambda: next(counters)
        self.host_facts.meminfo = lambda: {"MemTotal": 1_000, "MemAvailable": 400}

        class Nvidia:
            def utilisation(self) -> dict[str, Any]:
                """Avoid loading real NVIDIA libraries in this deterministic unit test."""
                return {"available": False, "reason": "not installed", "devices": []}

        try:
            facts = self.host_facts.Facts.__new__(self.host_facts.Facts)
            facts.nvidia = Nvidia()
            facts.previous_cpu = None
            first = facts.utilisation()
            second = facts.utilisation()
        finally:
            self.host_facts.cpu_counters = original_counters
            self.host_facts.meminfo = original_meminfo

        self.assertEqual(first["cpu"], {"percent": None, "sample_ready": False})
        self.assertEqual(second["cpu"], {"percent": 70.0, "sample_ready": True})
        self.assertEqual(
            second["memory"],
            {"total_bytes": 1_000, "available_bytes": 400, "used_bytes": 600},
        )

    def test_unix_socket_serves_only_the_documented_operations(self) -> None:
        """Verify the complete local socket path, not just the request parser.

        The test starts the actual server on a temporary Unix socket, sends an
        inventory request, and checks that the response is successful, retains
        the request ID, and contains the facts supplied by the test. It then
        sends an unsupported operation through that same socket and checks that
        the service rejects it.

        This catches wiring mistakes that a parser-only test would miss, such as
        serving the wrong handler, dropping the request ID, or accepting a
        forbidden operation after a client has connected.
        """
        path, _ = self._start_server()
        inventory = self._request(
            path, b'{"version":1,"id":"inventory-check","request":"inventory"}'
        )
        rejected = self._request(path, b'{"version":1,"id":"nope","request":"compose"}')

        self.assertEqual(inventory["id"], "inventory-check")
        self.assertEqual(inventory["result"], {"kind": "inventory"})
        self.assertTrue(inventory["ok"])
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["error"], "request is not permitted")

    def test_peer_authorisation_allows_only_root_or_the_broker_group(self) -> None:
        """Verify which local processes may read facts from the privileged helper.

        The helper accepts root and members of the dedicated broker group. A
        process can have that group as either its main group or an additional
        group, so the test covers both arrangements. It also covers an unreadable
        status file and a peer whose groups do not include the broker group.

        These checks matter because socket file permissions alone are not the
        whole story for a mounted socket. The service must make the same careful
        decision for every connecting process and deny access when it cannot
        establish that the process is allowed.
        """
        authorised_group = 4321
        peer = self.host_facts.peer_is_authorised

        def supplementary(_: Path) -> str:
            """Model a peer whose supplementary groups include the broker group."""
            return "Name:\tbroker\nGroups:\t10 4321 5000\n"

        self.assertTrue(peer(struct.pack("3i", 7, 1000, authorised_group), authorised_group))
        self.assertTrue(peer(struct.pack("3i", 7, 0, 2000), authorised_group))
        self.assertTrue(peer(struct.pack("3i", 7, 1000, 2000), authorised_group, supplementary))

        def no_status(_: Path) -> None:
            """Model an unreadable peer status file."""
            return None

        def other_groups(_: Path) -> str:
            """Model a peer that is not a member of the broker group."""
            return "Groups:\t8\n"

        self.assertFalse(peer(struct.pack("3i", 7, 1000, 2000), authorised_group, no_status))
        self.assertFalse(peer(struct.pack("3i", 7, 1000, 2000), authorised_group, other_groups))

    def test_procfs_and_sysfs_inventory_preserves_observed_values(self) -> None:
        """Verify how the helper reads host files without needing real host hardware.

        The test builds a small temporary copy of the files normally provided by
        the operating system: memory information, a NUMA node, and a GPU PCI
        device. It checks byte conversion, CPU placement, the kernel card name,
        and the observed PCIe speed and width.

        It deliberately does not create the maximum PCIe values. The expected
        result is therefore ``None`` for those fields, which proves that missing
        facts stay unknown instead of being guessed. That distinction is needed
        before these values can be trusted in broker reports or later policy.
        """
        original_meminfo = self.host_facts.PROC_MEMINFO
        original_node = self.host_facts.SYS_NODE
        original_pci = self.host_facts.SYS_PCI
        with tempfile.TemporaryDirectory[str]() as directory:
            root = Path(directory)
            proc_meminfo = root / "proc" / "meminfo"
            proc_meminfo.parent.mkdir()
            proc_meminfo.write_text(
                "MemTotal:       2048 kB\nMemAvailable:   512 kB\n", encoding="utf-8"
            )
            node = root / "sys" / "node" / "node0"
            node.mkdir(parents=True)
            (node / "cpulist").write_text("0-3\n", encoding="utf-8")
            (node / "meminfo").write_text("Node 0 MemTotal: 1024 kB\n", encoding="utf-8")
            device = root / "sys" / "pci" / "0000:65:00.0"
            (device / "drm" / "card2").mkdir(parents=True)
            (device / "numa_node").write_text("0\n", encoding="utf-8")
            (device / "current_link_speed").write_text("16.0 GT/s PCIe\n", encoding="utf-8")
            (device / "current_link_width").write_text("16\n", encoding="utf-8")

            self.host_facts.PROC_MEMINFO = proc_meminfo
            self.host_facts.SYS_NODE = root / "sys" / "node"
            self.host_facts.SYS_PCI = root / "sys" / "pci"
            try:
                self.assertEqual(
                    self.host_facts.meminfo(),
                    {"MemTotal": 2_097_152, "MemAvailable": 524_288},
                )
                self.assertEqual(
                    self.host_facts.numa_nodes(),
                    [{"node": 0, "cpus": "0-3", "memory_total_bytes": 1_048_576}],
                )
                pci = self.host_facts.pci_facts("0000:65:00.0")
            finally:
                self.host_facts.PROC_MEMINFO = original_meminfo
                self.host_facts.SYS_NODE = original_node
                self.host_facts.SYS_PCI = original_pci

        self.assertEqual(pci["kernel_devices"], ["card2"])
        self.assertEqual(pci["pcie"]["current_link_speed"], "16.0 GT/s PCIe")
        self.assertEqual(pci["pcie"]["current_link_width"], 16)
        self.assertIsNone(pci["pcie"]["max_link_speed"])
        self.assertIsNone(pci["pcie"]["max_link_width"])
