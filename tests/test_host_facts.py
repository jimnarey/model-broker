from __future__ import annotations

import grp
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

NVIDIA_DISABLED = "NVIDIA disabled for tests"


def request_line(request: str = "inventory", request_id: str = "request-1") -> bytes:
    return json.dumps({"version": 1, "id": request_id, "request": request}).encode()


@pytest.fixture
def short_directory() -> Iterator[Path]:
    """A directory with a short path: Unix socket paths are limited to 108 bytes."""
    directory = Path(tempfile.mkdtemp(prefix="hf-"))
    yield directory
    shutil.rmtree(directory)


# Request parsing and peer authorisation


def test_parse_request_returns_id_and_operation(host_facts: Any) -> None:
    """A well-formed request yields its id and operation unchanged."""
    assert host_facts.parse_request(request_line("utilisation", "abc")) == ("abc", "utilisation")


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (b"\xff", "UTF-8 JSON"),
        (b"not json", "UTF-8 JSON"),
        (b'["inventory"]', "only version, id, and request"),
        (b'{"version":1,"id":"a"}', "only version, id, and request"),
        (b'{"version":1,"id":"a","request":"inventory","command":"id"}', "only version"),
        (b'{"version":2,"id":"a","request":"inventory"}', "unsupported protocol version"),
        (b'{"version":true,"id":"a","request":"inventory"}', "unsupported protocol version"),
        (b'{"version":1.0,"id":"a","request":"inventory"}', "unsupported protocol version"),
        (b'{"version":1,"id":"","request":"inventory"}', "id must be a non-empty string"),
        (b'{"version":1,"id":7,"request":"inventory"}', "id must be a non-empty string"),
        (b'{"version":1,"id":"a","request":"docker"}', "request is not permitted"),
    ],
    ids=[
        "not-utf8",
        "not-json",
        "not-an-object",
        "missing-field",
        "extra-field",
        "wrong-version",
        "boolean-version",
        "float-version",
        "empty-id",
        "numeric-id",
        "unknown-operation",
    ],
)
def test_parse_request_rejects_malformed_requests(host_facts: Any, raw: bytes, error: str) -> None:
    """Each malformed request is rejected with a message naming what is wrong.

    The helper runs as root, so anything other than the exact three-field envelope with a
    known operation is refused before any work is done. ``true`` and ``1.0`` are listed
    because they compare equal to 1 in Python, so a plain ``==`` check would accept them.
    """
    with pytest.raises(host_facts.RequestError, match=error):
        host_facts.parse_request(raw)


BROKER_GID = 4321


@pytest.mark.parametrize(
    ("uid", "gid", "status", "allowed"),
    [
        (0, 2000, None, True),
        (1000, BROKER_GID, None, True),
        (1000, 2000, "Name:\tbroker\nGroups:\t10 4321 5000\n", True),
        (1000, 2000, "Groups:\t8 43210\n", False),
        (1000, 2000, "Name:\tbroker\n", False),
        (1000, 2000, None, False),
    ],
    ids=[
        "root",
        "primary-group",
        "supplementary-group",
        "other-groups",
        "no-groups-line",
        "status-unreadable",
    ],
)
def test_peer_authorisation(
    host_facts: Any, uid: int, gid: int, status: str | None, allowed: bool
) -> None:
    """Root and members of the broker group (primary or supplementary) may connect.

    Setup: SO_PEERCRED credentials for pid 7 with the given uid and gid, and a stand-in for
    /proc/7/status. Expect: only the first three are allowed. ``43210`` checks that group IDs
    are matched whole, not as substrings, and an unreadable status file denies access.
    """
    credentials = struct.pack("3i", 7, uid, gid)
    assert host_facts.peer_is_authorised(credentials, BROKER_GID, lambda _: status) is allowed


def test_peer_authorisation_rejects_truncated_credentials(host_facts: Any) -> None:
    """Credentials that are not three ints cannot be trusted, so the peer is denied."""
    assert host_facts.peer_is_authorised(b"\x00" * 4, BROKER_GID) is False


# procfs parsing and CPU arithmetic


def test_parse_meminfo_converts_kb_lines_to_bytes(host_facts: Any) -> None:
    """kB values become bytes, node-prefixed keys are normalised, and unitless lines are skipped.

    ``HugePages_Total`` is a page count, not kB, so multiplying it by 1024 would be wrong.
    """
    text = (
        "MemTotal:       2048 kB\n"
        "Node 0 MemAvailable:   512 kB\n"
        "HugePages_Total:       4\n"
        "garbage line\n"
    )
    assert host_facts.parse_meminfo(text) == {"MemTotal": 2_097_152, "MemAvailable": 524_288}


def test_parse_cpu_counters_excludes_guest_time(host_facts: Any) -> None:
    """Total is user..steal and idle is idle + iowait, from the aggregate ``cpu`` line.

    Setup: user=10 nice=20 system=30 idle=40 iowait=5 irq=1 softirq=2 steal=3 guest=100
    guest_nice=200, with a per-CPU line before it that must be ignored.
    Expect: total = 10+20+30+40+5+1+2+3 = 111 and idle = 45. The guest fields are already
    included in user and nice, so adding them would double-count.
    """
    text = "cpu0 1 1 1 1 1 1 1 1 1 1\ncpu  10 20 30 40 5 1 2 3 100 200\nintr 0\n"
    assert host_facts.parse_cpu_counters(text) == (111, 45)


def test_parse_cpu_counters_without_aggregate_line(host_facts: Any) -> None:
    """Unreadable or truncated /proc/stat reports no sample rather than a wrong one."""
    assert host_facts.parse_cpu_counters(None) is None
    assert host_facts.parse_cpu_counters("cpu  1 2 3 4\n") is None


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        ((100, 30), (150, 45), 70.0),
        (None, (150, 45), None),
        ((100, 30), None, None),
        ((100, 30), (100, 30), None),
    ],
    ids=["busy-share", "first-sample", "unreadable-sample", "no-time-elapsed"],
)
def test_cpu_percent(
    host_facts: Any,
    previous: tuple[int, int] | None,
    current: tuple[int, int] | None,
    expected: float | None,
) -> None:
    """CPU use is the non-idle share of the change in counters between two samples.

    ``busy-share``: total rises by 50 and idle by 15, so busy = (50 - 15) / 50 = 70%.
    Without two samples, or with no elapsed time, the result is None rather than 0, because
    reporting 0% would describe an unknown host as idle.
    """
    assert host_facts.cpu_percent(previous, current) == expected


# sysfs: NUMA and PCIe


def test_numa_nodes_reads_cpus_and_memory_in_numeric_order(host_facts: Any, tmp_path: Path) -> None:
    """Each node reports its CPU list and MemTotal in bytes, ordered node2 before node10.

    A plain string sort would put node10 first.
    """
    for number, cpus in ((10, "8-15"), (2, "0-7")):
        node = tmp_path / f"node{number}"
        node.mkdir()
        (node / "cpulist").write_text(f"{cpus}\n")
        (node / "meminfo").write_text(f"Node {number} MemTotal: 1024 kB\n")
    (tmp_path / "online").write_text("2,10\n")

    assert host_facts.numa_nodes(tmp_path) == [
        {"node": 2, "cpus": "0-7", "memory_total_bytes": 1_048_576},
        {"node": 10, "cpus": "8-15", "memory_total_bytes": 1_048_576},
    ]


def test_pci_address_matches_sysfs_and_cuda_format(host_facts: Any) -> None:
    """NVML's numeric fields format as ``0000:2b:00.0``: 4-digit domain, lowercase hex.

    NVML's own bus_id string is ``00000000:2B:00.0``, which names no sysfs directory and
    matches no CUDA address; using it left every PCI field null.
    """
    assert host_facts.pci_address(0, 0x2B, 0) == "0000:2b:00.0"


def write_port(path: Path, max_speed: str, max_width: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "max_link_speed").write_text(f"{max_speed}\n")
    (path / "max_link_width").write_text(f"{max_width}\n")


def chipset_gpu(root: Path, bridge_speed: str = "8.0 GT/s PCIe") -> Path:
    """Build the sysfs tree of this server's GPU0 and return the /sys/bus/pci/devices dir.

    Root port (Gen4 x8) -> chipset switch (Gen3 x4) -> downstream port (Gen3 x4) -> GPU
    (Gen5 x8). The GPU is idle, so its link has trained down to 2.5 GT/s x4.
    """
    root_port = root / "devices" / "pci0000:00" / "0000:00:01.2"
    switch = root_port / "0000:02:00.2"
    downstream = switch / "0000:03:00.0"
    gpu = downstream / "0000:04:00.0"
    write_port(root_port, "16.0 GT/s PCIe", 8)
    write_port(switch, bridge_speed, 4)
    write_port(downstream, "8.0 GT/s PCIe", 4)
    write_port(gpu, "32.0 GT/s PCIe", 8)
    (gpu / "current_link_speed").write_text("2.5 GT/s PCIe\n")
    (gpu / "current_link_width").write_text("4\n")
    (gpu / "numa_node").write_text("-1\n")
    (gpu / "drm" / "card1").mkdir(parents=True)
    (gpu / "drm" / "renderD128").mkdir()
    devices = root / "bus" / "pci" / "devices"
    devices.mkdir(parents=True)
    (devices / "0000:04:00.0").symlink_to(gpu)
    return devices


def test_pci_facts_report_the_path_bottleneck_not_the_card_maximum(
    host_facts: Any, tmp_path: Path
) -> None:
    """The slot limit is the lowest max_link_* along the upstream ports, not the card's own.

    Setup: a copy of this server's GPU0 path (see ``chipset_gpu``). The card supports Gen5
    x8, but the chipset ports allow only 8 GT/s x4.
    Expect: device maximum 32.0 x8, path maximum 8.0 x4, and the three upstream ports in
    order. Reporting the device maximum alone would make both of this server's GPUs look
    identical when one is limited to a quarter of the other's bandwidth. Also, numa_node -1
    (the kernel's "unknown") becomes None, and only DRM ``card`` nodes are listed.
    """
    facts = host_facts.pci_facts("0000:04:00.0", chipset_gpu(tmp_path))

    assert facts == {
        "pci_bus_id": "0000:04:00.0",
        "kernel_devices": ["card1"],
        "numa_node": None,
        "pcie": {
            "current_link_speed_gts": 2.5,
            "current_link_width": 4,
            "device_max_link_speed_gts": 32.0,
            "device_max_link_width": 8,
            "path_max_link_speed_gts": 8.0,
            "path_max_link_width": 4,
            "upstream_ports": ["0000:03:00.0", "0000:02:00.2", "0000:00:01.2"],
        },
    }


def test_pci_facts_path_limit_is_unknown_when_any_port_is_unknown(
    host_facts: Any, tmp_path: Path
) -> None:
    """One port reporting ``Unknown`` speed makes the path speed None, not the min of the rest.

    Ignoring that port would report 8 GT/s from the others, which overstates the limit if the
    unknown port is slower. The widths are all known, so the path width is still 4.
    """
    devices = chipset_gpu(tmp_path, bridge_speed="Unknown")
    pcie = host_facts.pci_facts("0000:04:00.0", devices)["pcie"]

    assert pcie["path_max_link_speed_gts"] is None
    assert pcie["path_max_link_width"] == 4


def test_pci_facts_for_missing_device(host_facts: Any, tmp_path: Path) -> None:
    """A device NVML reports but sysfs lacks has every fact unknown rather than invented."""
    facts = host_facts.pci_facts("0000:99:00.0", tmp_path)

    assert facts == {
        "pci_bus_id": "0000:99:00.0",
        "kernel_devices": [],
        "numa_node": None,
        "pcie": {
            "current_link_speed_gts": None,
            "current_link_width": None,
            "device_max_link_speed_gts": None,
            "device_max_link_width": None,
            "path_max_link_speed_gts": None,
            "path_max_link_width": None,
            "upstream_ports": [],
        },
    }


def test_pci_facts_without_address(host_facts: Any) -> None:
    """When NVML cannot report a PCI address the device has no PCI facts at all."""
    assert host_facts.pci_facts(None) == {
        "pci_bus_id": None,
        "kernel_devices": [],
        "numa_node": None,
        "pcie": None,
    }


# NVIDIA


def test_gpu_facts_when_nvml_is_unavailable(host_facts: Any) -> None:
    """Without NVML both requests say NVIDIA is unavailable and why, with no devices."""
    expected = {"available": False, "reason": NVIDIA_DISABLED, "devices": []}
    assert host_facts.gpu_inventory(NVIDIA_DISABLED) == expected
    assert host_facts.gpu_utilisation(NVIDIA_DISABLED) == expected


def test_open_nvidia_reports_missing_library(
    host_facts: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If libnvidia-ml cannot be found, the reason is returned instead of raising."""
    monkeypatch.setattr(host_facts.ctypes.util, "find_library", lambda _: None)
    assert host_facts.open_nvidia() == "NVML is unavailable: libnvidia-ml was not found"


def test_nvidia_inventory_on_real_hardware(host_facts: Any) -> None:
    """On a host with NVIDIA GPUs, every GPU maps to a sysfs device and a CUDA ordinal.

    Skipped where NVML is unavailable. On the broker host this checks what the unit tests
    cannot: that the address built from NVML names a real sysfs directory, that CUDA reports
    the same address, and that the path limits can be read. Formatting the address from
    NVML's 8-digit-domain bus_id, as an earlier version did, fails the first assertion.
    """
    nvidia = host_facts.open_nvidia()
    if isinstance(nvidia, str):
        pytest.skip(nvidia)
    devices = host_facts.gpu_inventory(nvidia)["devices"]

    assert devices
    for device in devices:
        assert (host_facts.SYS_PCI / device["pci_bus_id"]).is_dir()
        assert device["cuda_device"] is not None
        assert device["kernel_devices"]
        assert device["pcie"]["path_max_link_speed_gts"] is not None
        assert device["memory_total_bytes"]


def test_nvidia_utilisation_on_real_hardware(host_facts: Any) -> None:
    """On a host with NVIDIA GPUs, utilisation covers the same GPUs as inventory.

    The broker joins the two replies by UUID, so the UUID sets must match, and each reading
    must be a real value: None would mean an NVML call failed.
    """
    nvidia = host_facts.open_nvidia()
    if isinstance(nvidia, str):
        pytest.skip(nvidia)
    inventory = host_facts.gpu_inventory(nvidia)["devices"]
    readings = host_facts.gpu_utilisation(nvidia)["devices"]

    assert {gpu["uuid"] for gpu in readings} == {gpu["uuid"] for gpu in inventory}
    for reading in readings:
        assert 0 <= reading["gpu_percent"] <= 100
        assert 0 < reading["memory_used_bytes"] <= reading["memory_total_bytes"]


# Socket setup


def test_prepare_socket_refuses_to_replace_a_regular_file(host_facts: Any, tmp_path: Path) -> None:
    """A non-socket at the socket path is left alone and startup fails.

    The helper runs as root, so unlinking whatever is at a configured path could delete an
    arbitrary file.
    """
    path = tmp_path / "facts.sock"
    path.write_text("keep me")

    with pytest.raises(ValueError, match="refusing to replace non-socket path"):
        host_facts.prepare_socket(path, grp.getgrgid(os.getgid()).gr_name)
    assert path.read_text() == "keep me"


def test_prepare_socket_removes_a_stale_socket(host_facts: Any, short_directory: Path) -> None:
    """A socket left by a previous run is removed and the group's ID returned."""
    path = short_directory / "facts.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
        stale.bind(str(path))

    group_id = host_facts.prepare_socket(path, grp.getgrgid(os.getgid()).gr_name)

    assert group_id == os.getgid()
    assert not path.exists()


def test_prepare_socket_rejects_unknown_group(host_facts: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="socket group does not exist"):
        host_facts.prepare_socket(tmp_path / "facts.sock", "no-such-group-model-broker")


# Unix socket server


def start_server(host_facts: Any, directory: Path, allowed_gid: int) -> tuple[Path, Any]:
    if not hasattr(socket, "SO_PEERCRED"):
        pytest.skip("Linux SO_PEERCRED is required")
    path = directory / "facts.sock"
    server = host_facts.FactsServer(str(path), NVIDIA_DISABLED, allowed_gid)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return path, server


@pytest.fixture
def server(host_facts: Any, short_directory: Path) -> Iterator[tuple[Path, Any]]:
    """A running server that admits the test process through its primary group."""
    path, server = start_server(host_facts, short_directory, os.getgid())
    yield path, server
    server.shutdown()
    server.server_close()


def send(path: Path, message: bytes | None) -> dict[str, Any]:
    """Send one newline-terminated message (or nothing) and decode the one-line response."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(str(path))
        if message is not None:
            client.sendall(message + b"\n")
        return json.loads(client.makefile("rb").readline())


def test_server_answers_inventory_and_utilisation(server: tuple[Path, Any]) -> None:
    """Both operations succeed over the socket and echo the request id and version.

    Setup: a real server on a temporary socket with NVIDIA disabled, so results come from
    this machine's /proc and /sys. Expect: the NVIDIA section carries the disabled reason,
    and the first utilisation reply marks the CPU sample as not ready while the second has a
    percentage, which shows the server keeps the previous sample between requests. The pause
    lets the kernel's CPU counters (10 ms ticks) advance between the two samples.
    """
    path, _ = server
    inventory = send(path, request_line("inventory", "inv-1"))
    first = send(path, request_line("utilisation", "util-1"))
    time.sleep(0.05)
    second = send(path, request_line("utilisation", "util-2"))

    assert inventory["ok"] and inventory["id"] == "inv-1" and inventory["version"] == 1
    assert inventory["result"]["nvidia"]["reason"] == NVIDIA_DISABLED
    assert first["result"]["cpu"]["sample_ready"] is False
    assert second["id"] == "util-2"
    assert second["result"]["cpu"]["sample_ready"] is True


def test_server_rejects_unknown_operation(server: tuple[Path, Any]) -> None:
    """A rejected request produces an error reply, not a dropped connection."""
    path, _ = server
    assert send(path, request_line("compose")) == {"ok": False, "error": "request is not permitted"}


def test_server_rejects_oversized_request(host_facts: Any, server: tuple[Path, Any]) -> None:
    """A line longer than MAX_MESSAGE_BYTES is refused before it is parsed."""
    path, _ = server
    response = send(path, b"a" * (host_facts.MAX_MESSAGE_BYTES + 1))
    assert response == {"ok": False, "error": "request exceeds maximum size"}


def test_idle_client_times_out_without_blocking_the_server(
    host_facts: Any, server: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client that connects and sends nothing is dropped after the timeout.

    The server handles one connection at a time, so without the timeout this client would
    block every other client indefinitely. Expect: the idle client gets "request timed out",
    and the next client is served normally.
    """
    monkeypatch.setattr(host_facts.FactsHandler, "timeout", 0.2)
    path, _ = server

    assert send(path, None) == {"ok": False, "error": "request timed out"}
    assert send(path, request_line())["ok"] is True


def test_unexpected_failure_still_gets_a_reply(
    server: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If collecting facts raises, the client gets a generic error instead of silence."""
    path, facts_server = server

    def fail(_: str) -> None:
        raise RuntimeError("sysfs exploded")

    monkeypatch.setattr(facts_server, "collect", fail)
    assert send(path, request_line()) == {"ok": False, "error": "internal error"}


def test_server_rejects_peer_outside_the_broker_group(
    host_facts: Any, short_directory: Path
) -> None:
    """A peer that is neither root nor in the allowed group is refused per connection.

    Setup: the server allows a group ID that the test process does not have. Expect: the
    request is refused even though the socket file itself is reachable, which is the case
    this check exists for (for example, a socket bind-mounted into a container).
    """
    if os.geteuid() == 0:
        pytest.skip("root is always authorised")
    other_gid = max([os.getgid(), *os.getgroups()]) + 1000
    path, facts_server = start_server(host_facts, short_directory, other_gid)
    try:
        response = send(path, request_line())
    finally:
        facts_server.shutdown()
        facts_server.server_close()

    assert response == {"ok": False, "error": "connecting peer is not authorised"}
