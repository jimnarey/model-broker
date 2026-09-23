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
import logging
import os
import platform
import socket
import socketserver
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

LOGGER = logging.getLogger("model-broker-host-facts")
PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024
CLIENT_TIMEOUT_SECONDS = 5.0
REQUESTS = frozenset({"inventory", "utilisation"})
PEERCRED_FORMAT = "3i"
NVML_STRING_BYTES = 96
PROC_STAT = Path("/proc/stat")
PROC_MEMINFO = Path("/proc/meminfo")
SYS_PCI = Path("/sys/bus/pci/devices")
SYS_NODE = Path("/sys/devices/system/node")

JsonObject = dict[str, Any]
CpuSample = tuple[int, int]


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


# Argument types of every C function used; all of them return an int status (0 is success).
NVML_SIGNATURES: dict[str, list[Any]] = {
    "nvmlInit_v2": [],
    "nvmlDeviceGetCount_v2": [ctypes.POINTER(ctypes.c_uint)],
    "nvmlDeviceGetHandleByIndex_v2": [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
    "nvmlDeviceGetName": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
    "nvmlDeviceGetUUID": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
    "nvmlDeviceGetPciInfo_v3": [ctypes.c_void_p, ctypes.POINTER(NvmlPciInfo)],
    "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p, ctypes.POINTER(NvmlMemory)],
    "nvmlDeviceGetUtilizationRates": [ctypes.c_void_p, ctypes.POINTER(NvmlUtilisation)],
}
CUDA_SIGNATURES: dict[str, list[Any]] = {
    "cuInit": [ctypes.c_uint],
    "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
}


# procfs and sysfs


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_int(path: Path) -> int | None:
    try:
        return int(read_text(path) or "")
    except ValueError:
        return None


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_meminfo(text: str | None) -> dict[str, int]:
    """Parse the kB lines of /proc/meminfo or a NUMA node's meminfo into bytes.

    Node files prefix each key with ``Node N``, so the key is the last word before the colon.
    """
    values: dict[str, int] = {}
    for line in (text or "").splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if key.strip() and fields[1:] == ["kB"] and fields[0].isdigit():
            values[key.split()[-1]] = int(fields[0]) * 1024
    return values


def parse_cpu_counters(text: str | None) -> CpuSample | None:
    """Return (total, idle) jiffies from the aggregate ``cpu`` line of /proc/stat."""
    for line in (text or "").splitlines():
        if line.startswith("cpu "):
            fields = [int(value) for value in line.split()[1:]]
            if len(fields) < 5:
                return None
            # Fields are user nice system idle iowait irq softirq steal guest guest_nice.
            # guest and guest_nice are already counted in user and nice, so stop at steal.
            return sum(fields[:8]), fields[3] + fields[4]
    return None


def cpu_percent(previous: CpuSample | None, current: CpuSample | None) -> float | None:
    """Return the busy share of CPU time between two samples, or None without two samples."""
    if previous is None or current is None:
        return None
    total = current[0] - previous[0]
    idle = current[1] - previous[1]
    return round(100 * (total - idle) / total, 2) if total > 0 else None


def numa_nodes(root: Path = SYS_NODE) -> list[JsonObject]:
    nodes = sorted(root.glob("node[0-9]*"), key=lambda node: int(node.name[4:]))
    return [
        {
            "node": int(node.name[4:]),
            "cpus": read_text(node / "cpulist"),
            "memory_total_bytes": parse_meminfo(read_text(node / "meminfo")).get("MemTotal"),
        }
        for node in nodes
    ]


def pci_address(domain: int, bus: int, device: int) -> str:
    """Format a GPU's PCI address as sysfs and CUDA do: 4-digit domain, lowercase hex.

    NVML's own ``bus_id`` string uses an 8-digit uppercase domain, which matches neither.
    """
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"


def link_speed_gts(text: str | None) -> float | None:
    """Parse a sysfs link speed such as ``8.0 GT/s PCIe``; ``Unknown`` becomes None."""
    try:
        return float((text or "").split()[0])
    except (IndexError, ValueError):
        return None


def link_chain(device: Path) -> list[Path]:
    """Return the device followed by each upstream PCIe port on its path to the CPU."""
    chain: list[Path] = []
    path = device.resolve()
    while (path / "max_link_speed").is_file():
        chain.append(path)
        path = path.parent
    return chain


def lowest[T: (int, float)](values: list[T | None]) -> T | None:
    """Return the smallest value, or None when there are none or any is unknown."""
    if not values or None in values:
        return None
    return min(cast(list[T], values))


def pcie_facts(device: Path) -> JsonObject:
    """Report the device's own link and the bottleneck on its path to the CPU.

    ``max_link_*`` on the device is what the card supports, not what its slot allows. A link
    runs at the lower of its two ends, so the lowest maximum along the chain of upstream ports
    is the most the device can reach. ``current_link_*`` drops while the GPU is idle.
    """
    chain = link_chain(device)
    speeds = [link_speed_gts(read_text(port / "max_link_speed")) for port in chain]
    widths = [read_int(port / "max_link_width") for port in chain]
    return {
        "current_link_speed_gts": link_speed_gts(read_text(device / "current_link_speed")),
        "current_link_width": read_int(device / "current_link_width"),
        "device_max_link_speed_gts": speeds[0] if chain else None,
        "device_max_link_width": widths[0] if chain else None,
        "path_max_link_speed_gts": lowest(speeds),
        "path_max_link_width": lowest(widths),
        "upstream_ports": [port.name for port in chain[1:]],
    }


def pci_facts(address: str | None, sys_pci: Path = SYS_PCI) -> JsonObject:
    if address is None:
        return {"pci_bus_id": None, "kernel_devices": [], "numa_node": None, "pcie": None}
    device = sys_pci / address
    numa_node = read_int(device / "numa_node")
    return {
        "pci_bus_id": address,
        "kernel_devices": sorted(path.name for path in (device / "drm").glob("card[0-9]*")),
        # The kernel reports -1 when the device has no known NUMA affinity.
        "numa_node": numa_node if numa_node is not None and numa_node >= 0 else None,
        "pcie": pcie_facts(device),
    }


# NVIDIA management library (NVML) and CUDA driver, loaded without subprocesses


@dataclass(frozen=True, slots=True)
class Nvidia:
    nvml: Any
    cuda: Any | None


def load_library(name: str, signatures: dict[str, list[Any]]) -> Any:
    path = ctypes.util.find_library(name)
    if path is None:
        raise OSError(f"lib{name} was not found")
    library = ctypes.CDLL(path)
    for function_name, argtypes in signatures.items():
        function = getattr(library, function_name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int
    return library


def open_nvidia() -> Nvidia | str:
    """Load NVML (required) and the CUDA driver (optional), or return why NVML is unusable."""
    try:
        nvml = load_library("nvidia-ml", NVML_SIGNATURES)
    except (AttributeError, OSError) as error:
        return f"NVML is unavailable: {error}"
    if nvml.nvmlInit_v2() != 0:
        return "NVML initialisation failed"
    try:
        cuda = load_library("cuda", CUDA_SIGNATURES)
    except (AttributeError, OSError):
        cuda = None
    if cuda is not None and cuda.cuInit(0) != 0:
        cuda = None
    return Nvidia(nvml, cuda)


def nvml_handles(nvml: Any) -> list[ctypes.c_void_p]:
    count = ctypes.c_uint()
    if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
        return []
    handles: list[ctypes.c_void_p] = []
    for index in range(count.value):
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) == 0:
            handles.append(handle)
    return handles


def nvml_query[T: ctypes.Structure](
    function: Callable[..., int], handle: ctypes.c_void_p, result: T
) -> T | None:
    """Fill a result struct, returning None rather than a zeroed struct when the call fails."""
    return result if function(handle, ctypes.byref(result)) == 0 else None


def nvml_string(function: Callable[..., int], handle: ctypes.c_void_p) -> str | None:
    buffer = ctypes.create_string_buffer(NVML_STRING_BYTES)
    if function(handle, buffer, len(buffer)) != 0:
        return None
    return buffer.value.decode("utf-8", errors="replace") or None


def cuda_ordinals(cuda: Any | None) -> dict[str, int]:
    """Map each sysfs-style PCI address to its CUDA ordinal in the default device order."""
    count = ctypes.c_int()
    if cuda is None or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return {}
    ordinals: dict[str, int] = {}
    for ordinal in range(count.value):
        device = ctypes.c_int()
        buffer = ctypes.create_string_buffer(32)
        if (
            cuda.cuDeviceGet(ctypes.byref(device), ordinal) == 0
            and cuda.cuDeviceGetPCIBusId(buffer, len(buffer), device) == 0
        ):
            ordinals[buffer.value.decode("ascii").lower()] = ordinal
    return ordinals


def gpu_inventory(nvidia: Nvidia | str) -> JsonObject:
    if isinstance(nvidia, str):
        return {"available": False, "reason": nvidia, "devices": []}
    nvml = nvidia.nvml
    ordinals = cuda_ordinals(nvidia.cuda)
    devices: list[JsonObject] = []
    for handle in nvml_handles(nvml):
        pci = nvml_query(nvml.nvmlDeviceGetPciInfo_v3, handle, NvmlPciInfo())
        address = pci_address(pci.domain, pci.bus, pci.device) if pci else None
        memory = nvml_query(nvml.nvmlDeviceGetMemoryInfo, handle, NvmlMemory())
        ordinal = ordinals.get(address) if address else None
        devices.append(
            {
                **pci_facts(address),
                "cuda_device": None if ordinal is None else f"CUDA{ordinal}",
                "name": nvml_string(nvml.nvmlDeviceGetName, handle),
                "uuid": nvml_string(nvml.nvmlDeviceGetUUID, handle),
                "memory_total_bytes": memory.total if memory else None,
            }
        )
    reason = None if ordinals else "CUDA ordinal mapping is unavailable"
    return {"available": True, "reason": reason, "devices": devices}


def gpu_utilisation(nvidia: Nvidia | str) -> JsonObject:
    if isinstance(nvidia, str):
        return {"available": False, "reason": nvidia, "devices": []}
    nvml = nvidia.nvml
    devices: list[JsonObject] = []
    for handle in nvml_handles(nvml):
        memory = nvml_query(nvml.nvmlDeviceGetMemoryInfo, handle, NvmlMemory())
        rates = nvml_query(nvml.nvmlDeviceGetUtilizationRates, handle, NvmlUtilisation())
        devices.append(
            {
                "uuid": nvml_string(nvml.nvmlDeviceGetUUID, handle),
                "gpu_percent": rates.gpu if rates else None,
                "memory_percent": rates.memory if rates else None,
                "memory_used_bytes": memory.used if memory else None,
                "memory_total_bytes": memory.total if memory else None,
            }
        )
    return {"available": True, "reason": None, "devices": devices}


# Requests


def inventory(nvidia: Nvidia | str) -> JsonObject:
    return {
        "collected_at": timestamp(),
        "cpu": {"architecture": platform.machine(), "logical_cpus": os.cpu_count()},
        "memory": {"total_bytes": parse_meminfo(read_text(PROC_MEMINFO)).get("MemTotal")},
        "numa_nodes": numa_nodes(),
        "nvidia": gpu_inventory(nvidia),
    }


def utilisation(
    nvidia: Nvidia | str, previous_cpu: CpuSample | None
) -> tuple[JsonObject, CpuSample | None]:
    """Collect utilisation, returning the CPU sample to compare against on the next call."""
    counters = parse_cpu_counters(read_text(PROC_STAT))
    percent = cpu_percent(previous_cpu, counters)
    memory = parse_meminfo(read_text(PROC_MEMINFO))
    total, available = memory.get("MemTotal"), memory.get("MemAvailable")
    result = {
        "collected_at": timestamp(),
        "cpu": {"percent": percent, "sample_ready": percent is not None},
        "memory": {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": None if total is None or available is None else total - available,
        },
        "nvidia": gpu_utilisation(nvidia),
    }
    return result, counters


def parse_request(raw: bytes) -> tuple[str, str]:
    try:
        decoded: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestError("request must be UTF-8 JSON") from error
    message = cast(dict[str, object], decoded) if isinstance(decoded, dict) else {}
    if set(message) != {"version", "id", "request"}:
        raise RequestError("request must contain only version, id, and request")
    version, request_id, operation = message["version"], message["id"], message["request"]
    # type() rather than ==, because JSON true and 1.0 both compare equal to 1.
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise RequestError("unsupported protocol version")
    if not isinstance(request_id, str) or not request_id:
        raise RequestError("id must be a non-empty string")
    if not isinstance(operation, str) or operation not in REQUESTS:
        raise RequestError("request is not permitted")
    return request_id, operation


def peer_is_authorised(
    credentials: bytes, allowed_gid: int, read_status: Callable[[Path], str | None] = read_text
) -> bool:
    """Allow root, or a peer whose primary or supplementary groups include allowed_gid."""
    try:
        pid, uid, gid = struct.unpack(PEERCRED_FORMAT, credentials)
    except struct.error:
        return False
    if uid == 0 or gid == allowed_gid:
        return True
    for line in (read_status(Path(f"/proc/{pid}/status")) or "").splitlines():
        if line.startswith("Groups:"):
            return str(allowed_gid) in line.split()[1:]
    return False


# Unix socket server


class FactsServer(socketserver.UnixStreamServer):
    """Single-threaded server; it also keeps the CPU sample between utilisation requests."""

    def __init__(self, address: str, nvidia: Nvidia | str, allowed_gid: int) -> None:
        self.nvidia = nvidia
        self.allowed_gid = allowed_gid
        self.previous_cpu: CpuSample | None = None
        super().__init__(address, FactsHandler)

    def collect(self, operation: str) -> JsonObject:
        if operation == "inventory":
            return inventory(self.nvidia)
        result, self.previous_cpu = utilisation(self.nvidia, self.previous_cpu)
        return result


class FactsHandler(socketserver.StreamRequestHandler):
    # Applied to each connection so that an idle client cannot block the server.
    timeout = CLIENT_TIMEOUT_SECONDS

    def handle(self) -> None:
        server = cast(FactsServer, self.server)
        credentials = self.request.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize(PEERCRED_FORMAT)
        )
        try:
            if not peer_is_authorised(credentials, server.allowed_gid):
                raise RequestError("connecting peer is not authorised")
            raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
            if len(raw) > MAX_MESSAGE_BYTES:
                raise RequestError("request exceeds maximum size")
            request_id, operation = parse_request(raw)
            response: JsonObject = {
                "ok": True,
                "version": PROTOCOL_VERSION,
                "id": request_id,
                "result": server.collect(operation),
            }
        except RequestError as error:
            response = {"ok": False, "error": str(error)}
        except TimeoutError:
            response = {"ok": False, "error": "request timed out"}
        except Exception:
            LOGGER.exception("failed to serve request")
            response = {"ok": False, "error": "internal error"}
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
    """Remove a stale socket at path and return the group's ID; refuse to replace anything else."""
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_arguments()
    nvidia = open_nvidia()
    if isinstance(nvidia, str):
        LOGGER.warning("%s; NVIDIA facts will be reported as unavailable", nvidia)
    try:
        group_id = prepare_socket(arguments.socket, arguments.socket_group)
        server = FactsServer(str(arguments.socket), nvidia, group_id)
        os.chown(arguments.socket, -1, group_id)
        os.chmod(arguments.socket, 0o660)
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
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
