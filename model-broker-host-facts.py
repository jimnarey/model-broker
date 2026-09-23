#!/usr/bin/env python3
"""Read-only privileged host facts service for model-broker.

This program intentionally has no Docker, Compose, systemd, model, or
scheduling integration.  It serves a small versioned JSON protocol over a
Unix socket.  The only supported requests are ``inventory`` and
``utilisation``.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import grp
import json
import os
import platform
import socket
import socketserver
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024
REQUESTS = frozenset({"inventory", "utilisation"})
PROC_STAT = Path("/proc/stat")
PROC_MEMINFO = Path("/proc/meminfo")
SYS_PCI = Path("/sys/bus/pci/devices")
SYS_NODE = Path("/sys/devices/system/node")


class RequestError(ValueError):
    """A client request that cannot be served."""


class NvmlPciInfo(ctypes.Structure):
    """The stable nvmlPciInfo_t layout used by nvmlDeviceGetPciInfo_v3."""

    _fields_ = [
        ("bus_id_legacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pci_device_id", ctypes.c_uint),
        ("pci_subsystem_id", ctypes.c_uint),
        ("bus_id", ctypes.c_char * 32),
    ]


class NvmlMemory(ctypes.Structure):
    """The nvmlMemory_t layout."""

    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlUtilisation(ctypes.Structure):
    """The nvmlUtilization_t layout."""

    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_int(path: Path) -> int | None:
    value = read_text(path)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    content = read_text(PROC_MEMINFO)
    if content is None:
        return values
    for line in content.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        number = rest.strip().split(maxsplit=1)
        if number and number[0].isdigit():
            values[key] = int(number[0]) * 1024
    return values


def cpu_counters() -> tuple[int, int] | None:
    content = read_text(PROC_STAT)
    if content is None:
        return None
    for line in content.splitlines():
        if line.startswith("cpu "):
            fields = [int(value) for value in line.split()[1:]]
            if len(fields) < 4:
                return None
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
            # guest and guest_nice are already included in user and nice.
            return sum(fields[:8]), idle
    return None


def numa_nodes() -> list[dict[str, Any]]:
    if not SYS_NODE.is_dir():
        return []
    nodes: list[dict[str, Any]] = []
    for node in sorted(SYS_NODE.glob("node[0-9]*")):
        node_memory = read_text(node / "meminfo")
        total: int | None = None
        if node_memory is not None:
            for line in node_memory.splitlines():
                if "MemTotal:" in line:
                    number = line.split("MemTotal:", maxsplit=1)[1].strip().split()[0]
                    total = int(number) * 1024
                    break
        nodes.append(
            {
                "node": int(node.name[4:]),
                "cpus": read_text(node / "cpulist"),
                "memory_total_bytes": total,
            }
        )
    return nodes


def pci_facts(address: str) -> dict[str, Any]:
    device = SYS_PCI / address
    cards = sorted(path.name for path in (device / "drm").glob("card[0-9]*"))
    return {
        "pci_bus_id": address,
        "kernel_devices": cards,
        "numa_node": read_int(device / "numa_node"),
        "pcie": {
            "current_link_speed": read_text(device / "current_link_speed"),
            "current_link_width": read_int(device / "current_link_width"),
            "max_link_speed": read_text(device / "max_link_speed"),
            "max_link_width": read_int(device / "max_link_width"),
        },
    }


class NvidiaManagement:
    """Minimal optional NVML and CUDA-driver bindings, with no subprocesses."""

    def __init__(self) -> None:
        self.nvml: Any | None = None
        self.cuda: Any | None = None
        self.error: str | None = None
        self._open()

    def _open(self) -> None:
        nvml_name = ctypes.util.find_library("nvidia-ml")
        cuda_name = ctypes.util.find_library("cuda")
        if not nvml_name:
            self.error = "NVML library is unavailable"
            return
        try:
            self.nvml = ctypes.CDLL(nvml_name)
            self.nvml.nvmlInit_v2.restype = ctypes.c_int
            self.nvml.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
            self.nvml.nvmlDeviceGetCount_v2.restype = ctypes.c_int
            self.nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
            self.nvml.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            self.nvml.nvmlDeviceGetName.restype = ctypes.c_int
            self.nvml.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            self.nvml.nvmlDeviceGetUUID.restype = ctypes.c_int
            self.nvml.nvmlDeviceGetPciInfo_v3.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(NvmlPciInfo),
            ]
            self.nvml.nvmlDeviceGetPciInfo_v3.restype = ctypes.c_int
            self.nvml.nvmlDeviceGetMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(NvmlMemory),
            ]
            self.nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
            self.nvml.nvmlDeviceGetUtilizationRates.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(NvmlUtilisation),
            ]
            self.nvml.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
            if self.nvml.nvmlInit_v2() != 0:
                self.error = "NVML initialisation failed"
                self.nvml = None
                return
            if cuda_name:
                self.cuda = ctypes.CDLL(cuda_name)
                self.cuda.cuInit.argtypes = [ctypes.c_uint]
                self.cuda.cuInit.restype = ctypes.c_int
                self.cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
                self.cuda.cuDeviceGetCount.restype = ctypes.c_int
                self.cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
                self.cuda.cuDeviceGet.restype = ctypes.c_int
                self.cuda.cuDeviceGetPCIBusId.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                self.cuda.cuDeviceGetPCIBusId.restype = ctypes.c_int
        except (AttributeError, OSError) as error:
            self.error = f"NVIDIA management library could not be opened: {error}"
            self.nvml = None

    @staticmethod
    def _ok(code: int) -> bool:
        return code == 0

    def _handles(self) -> list[ctypes.c_void_p]:
        if self.nvml is None:
            return []
        count = ctypes.c_uint()
        if not self._ok(self.nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))):
            return []
        handles: list[ctypes.c_void_p] = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if self._ok(self.nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle))):
                handles.append(handle)
        return handles

    def _cuda_addresses(self) -> dict[str, int]:
        if self.cuda is None or not self._ok(self.cuda.cuInit(0)):
            return {}
        count = ctypes.c_int()
        if not self._ok(self.cuda.cuDeviceGetCount(ctypes.byref(count))):
            return {}
        addresses: dict[str, int] = {}
        for ordinal in range(count.value):
            device = ctypes.c_int()
            buffer = ctypes.create_string_buffer(32)
            if self._ok(self.cuda.cuDeviceGet(ctypes.byref(device), ordinal)) and self._ok(
                self.cuda.cuDeviceGetPCIBusId(buffer, len(buffer), device)
            ):
                addresses[buffer.value.decode("ascii").lower()] = ordinal
        return addresses

    def _device(self, handle: ctypes.c_void_p, cuda_addresses: dict[str, int]) -> dict[str, Any]:
        assert self.nvml is not None
        name = ctypes.create_string_buffer(96)
        uuid = ctypes.create_string_buffer(96)
        pci = NvmlPciInfo()
        memory_info = NvmlMemory()
        address: str | None = None
        if self._ok(self.nvml.nvmlDeviceGetPciInfo_v3(handle, ctypes.byref(pci))):
            address = pci.bus_id.decode("ascii").lower() or None
        result: dict[str, Any] = (
            pci_facts(address)
            if address
            else {
                "pci_bus_id": None,
                "kernel_devices": [],
                "numa_node": None,
                "pcie": {},
            }
        )
        self.nvml.nvmlDeviceGetName(handle, name, len(name))
        self.nvml.nvmlDeviceGetUUID(handle, uuid, len(uuid))
        self.nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory_info))
        result.update(
            {
                "cuda_device": (
                    f"CUDA{cuda_addresses[address]}" if address in cuda_addresses else None
                ),
                "name": name.value.decode("utf-8", errors="replace") or None,
                "uuid": uuid.value.decode("ascii", errors="replace") or None,
                "memory_total_bytes": int(memory_info.total),
            }
        )
        return result

    def inventory(self) -> dict[str, Any]:
        if self.nvml is None:
            return {"available": False, "reason": self.error, "devices": []}
        cuda_addresses = self._cuda_addresses()
        return {
            "available": True,
            "reason": None if cuda_addresses else "CUDA ordinal mapping is unavailable",
            "devices": [self._device(handle, cuda_addresses) for handle in self._handles()],
        }

    def utilisation(self) -> dict[str, Any]:
        if self.nvml is None:
            return {"available": False, "reason": self.error, "devices": []}
        devices: list[dict[str, Any]] = []
        for handle in self._handles():
            memory = NvmlMemory()
            utilisation = NvmlUtilisation()
            self.nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
            self.nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(utilisation))
            uuid = ctypes.create_string_buffer(96)
            self.nvml.nvmlDeviceGetUUID(handle, uuid, len(uuid))
            devices.append(
                {
                    "uuid": uuid.value.decode("ascii", errors="replace") or None,
                    "gpu_percent": int(utilisation.gpu),
                    "memory_percent": int(utilisation.memory),
                    "memory_used_bytes": int(memory.used),
                    "memory_total_bytes": int(memory.total),
                }
            )
        return {"available": True, "reason": None, "devices": devices}


class Facts:
    def __init__(self) -> None:
        self.nvidia = NvidiaManagement()
        self.previous_cpu: tuple[float, int, int] | None = None

    @staticmethod
    def timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def inventory(self) -> dict[str, Any]:
        memory = meminfo()
        return {
            "collected_at": self.timestamp(),
            "cpu": {"architecture": platform.machine(), "logical_cpus": os.cpu_count()},
            "memory": {"total_bytes": memory.get("MemTotal")},
            "numa_nodes": numa_nodes(),
            "nvidia": self.nvidia.inventory(),
        }

    def utilisation(self) -> dict[str, Any]:
        now = time.monotonic()
        counters = cpu_counters()
        cpu_percent: float | None = None
        if counters is not None and self.previous_cpu is not None:
            _, old_total, old_idle = self.previous_cpu
            total = counters[0] - old_total
            idle = counters[1] - old_idle
            if total > 0:
                cpu_percent = round(100 * (total - idle) / total, 2)
        if counters is not None:
            self.previous_cpu = (now, *counters)
        memory = meminfo()
        total_memory = memory.get("MemTotal")
        available = memory.get("MemAvailable")
        return {
            "collected_at": self.timestamp(),
            "cpu": {"percent": cpu_percent, "sample_ready": cpu_percent is not None},
            "memory": {
                "total_bytes": total_memory,
                "available_bytes": available,
                "used_bytes": total_memory - available
                if total_memory is not None and available is not None
                else None,
            },
            "nvidia": self.nvidia.utilisation(),
        }


def parse_request(raw: bytes) -> tuple[str, str]:
    try:
        decoded: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestError("request must be UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise RequestError("request must contain only version, id, and request")
    message = cast(dict[str, object], decoded)
    if set(message) != {"version", "id", "request"}:
        raise RequestError("request must contain only version, id, and request")
    if message.get("version") != PROTOCOL_VERSION:
        raise RequestError("unsupported protocol version")
    request_id = message.get("id")
    operation = message.get("request")
    if not isinstance(request_id, str) or not request_id:
        raise RequestError("id must be a non-empty string")
    if not isinstance(operation, str) or operation not in REQUESTS:
        raise RequestError("request is not permitted")
    return request_id, operation


def peer_is_authorised(
    credentials: bytes, allowed_gid: int, status_reader: Callable[[Path], str | None] = read_text
) -> bool:
    """Check Linux Unix-socket peer credentials against the broker group."""
    try:
        pid, uid, gid = struct.unpack("3i", credentials)
    except struct.error:
        return False
    if uid == 0 or gid == allowed_gid:
        return True
    groups = status_reader(Path("/proc") / str(pid) / "status")
    if groups is None:
        return False
    for line in groups.splitlines():
        if line.startswith("Groups:"):
            try:
                return allowed_gid in {int(value) for value in line.split()[1:]}
            except ValueError:
                return False
    return False


class FactsServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(self, address: str, facts: Facts, allowed_gid: int) -> None:
        self.facts = facts
        self.allowed_gid = allowed_gid
        super().__init__(address, FactsHandler)


class FactsHandler(socketserver.StreamRequestHandler):
    def _peer_is_allowed(self) -> bool:
        credentials = self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        server = cast(FactsServer, self.server)
        return peer_is_authorised(credentials, server.allowed_gid)

    def handle(self) -> None:
        if not self._peer_is_allowed():
            self._respond({"ok": False, "error": "connecting peer is not authorised"})
            return
        raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
        if len(raw) > MAX_MESSAGE_BYTES:
            self._respond({"ok": False, "error": "request exceeds maximum size"})
            return
        try:
            request_id, operation = parse_request(raw)
            server = cast(FactsServer, self.server)
            handlers: dict[str, Callable[[], dict[str, Any]]] = {
                "inventory": server.facts.inventory,
                "utilisation": server.facts.utilisation,
            }
            self._respond(
                {
                    "ok": True,
                    "version": PROTOCOL_VERSION,
                    "id": request_id,
                    "result": handlers[operation](),
                }
            )
        except RequestError as error:
            self._respond({"ok": False, "error": str(error)})

    def _respond(self, response: dict[str, Any]) -> None:
        payload = json.dumps(response, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        self.wfile.write(payload)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True, help="Unix socket path to create")
    parser.add_argument(
        "--socket-group", required=True, help="group permitted to connect to the socket"
    )
    return parser.parse_args()


def prepare_socket(path: Path, group: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not path.is_socket():
            raise ValueError(f"refusing to replace non-socket path: {path}")
        path.unlink()
    try:
        return grp.getgrnam(group).gr_gid
    except KeyError as error:
        raise ValueError(f"socket group does not exist: {group}") from error


def main() -> int:
    arguments = parse_arguments()
    try:
        group_id = prepare_socket(arguments.socket, arguments.socket_group)
        server = FactsServer(str(arguments.socket), Facts(), group_id)
        if arguments.socket.stat().st_gid != group_id:
            os.chown(arguments.socket, -1, group_id)
        os.chmod(arguments.socket, 0o660)
    except (OSError, ValueError) as error:
        print(f"model-broker-host-facts.py: {error}", file=sys.stderr)
        return 2
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if arguments.socket.is_socket():
            arguments.socket.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
