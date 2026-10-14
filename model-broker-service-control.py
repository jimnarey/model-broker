#!/usr/bin/env python3
"""Privileged service-control helper for model-broker.

Some services cannot load or unload a model while running, so the broker changes model by
stopping or starting their container. This helper does that on the broker's behalf, because
the broker itself has no Docker access. It is not a general Docker API:

* it serves a small versioned JSON protocol over a Unix socket, with three requests:
  ``status``, ``start`` and ``stop``, each naming a service;
* it controls only the Docker Compose services listed in its root-owned configuration, in
  the one Compose project each belongs to, and never builds, pulls or removes anything;
* the only thing a ``start`` request can set is the environment variables the configuration
  allows for that service, with values restricted to a safe character set. Compose
  interpolates them into the service, so the broker can choose a model but cannot
  choose an image, a mount, a command or a device.

No request text becomes part of a command line other than the service's configured Compose
name, and Compose runs with a scrubbed environment and a time limit.
"""

from __future__ import annotations

import argparse
import configparser
import grp
import json
import logging
import os
import re
import socket
import socketserver
import struct
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

LOGGER = logging.getLogger("model-broker-service-control")
PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024
CLIENT_TIMEOUT_SECONDS = 10.0
OPERATIONS = frozenset({"status", "start", "stop"})
PEERCRED_FORMAT = "3i"
SERVICE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
# Values reach Compose's interpolation, so no space, quote, dollar sign or other shell or
# Compose syntax is allowed. An empty value is allowed: it overrides a value from .env.
ENV_VALUE = re.compile(r"^[A-Za-z0-9_./:,@%+=-]*$")
MAX_VALUE_LENGTH = 512
OUTPUT_TAIL_CHARACTERS = 2000
CLEAN_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PROJECT_LABEL = "com.docker.compose.project"
SERVICE_LABEL = "com.docker.compose.service"
ONEOFF_LABEL = "com.docker.compose.oneoff"
# Docker's container states in which the service's processes are, or are about to be, gone.
GONE_STATES = frozenset({"exited", "dead", "created"})

JsonObject = dict[str, Any]


class ConfigError(ValueError):
    """The configuration file cannot be used."""


class RequestError(ValueError):
    """A client request that cannot be served."""


class OperationError(RuntimeError):
    """A valid request that Docker could not carry out."""


@dataclass(frozen=True, slots=True)
class Service:
    """A Compose service the helper may control, and the settings a start may change."""

    name: str
    compose_service: str
    compose_file: Path
    project_directory: Path
    project_name: str
    allowed_env: frozenset[str]
    stop_grace_seconds: int


@dataclass(frozen=True, slots=True)
class Config:
    docker: Path
    services: Mapping[str, Service]
    start_timeout_seconds: float
    stop_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    operation: str
    service: Service
    settings: Mapping[str, str]


# Configuration


def positive_number(section: configparser.SectionProxy, key: str, default: float) -> float:
    try:
        value = float(section.get(key, str(default)))
    except ValueError as error:
        raise ConfigError(f"[{section.name}] {key} must be a number") from error
    if not value > 0:
        raise ConfigError(f"[{section.name}] {key} must be positive")
    return value


def absolute_path(section: configparser.SectionProxy, key: str, fallback: str | None) -> Path:
    text = section.get(key, fallback)
    if not text or not Path(text).is_absolute():
        raise ConfigError(f"[{section.name}] {key} must be an absolute path")
    return Path(text)


def parse_service(
    section: configparser.SectionProxy, defaults: configparser.SectionProxy
) -> Service:
    name = cast(str, section.name)
    if not SERVICE_NAME.fullmatch(name):
        raise ConfigError(f"service name {name!r} is not a valid Compose service name")
    allowed = frozenset(section.get("allowed-env", "").split())
    for variable in allowed:
        if not ENV_NAME.fullmatch(variable):
            raise ConfigError(f"[{name}] allowed-env has an invalid name {variable!r}")
    project = section.get("project-name", defaults.get("project-name", ""))
    if not project:
        raise ConfigError(f"[{name}] needs a project-name")
    grace = positive_number(section, "stop-grace-seconds", 30)
    return Service(
        name=name,
        compose_service=section.get("compose-service", name),
        compose_file=absolute_path(section, "compose-file", defaults.get("compose-file")),
        project_directory=absolute_path(
            section, "project-directory", defaults.get("project-directory")
        ),
        project_name=project,
        allowed_env=allowed,
        stop_grace_seconds=int(grace),
    )


def load_config(path: Path) -> Config:
    """Read the configuration: a [control] section of defaults, then one section per service."""
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    if not parser.has_section("control"):
        raise ConfigError("the configuration needs a [control] section")
    control = parser["control"]
    services = {
        name: parse_service(parser[name], control)
        for name in parser.sections()
        if name != "control"
    }
    if not services:
        raise ConfigError("the configuration lists no services")
    docker = absolute_path(control, "docker", "/usr/bin/docker")
    return Config(
        docker=docker,
        services=services,
        start_timeout_seconds=positive_number(control, "start-timeout-seconds", 180),
        stop_timeout_seconds=positive_number(control, "stop-timeout-seconds", 180),
    )


def check_files(config: Config) -> None:
    """Fail at startup, not on the first request, if a path the configuration names is missing."""
    missing = [config.docker, *(s.compose_file for s in config.services.values())]
    missing += [s.project_directory for s in config.services.values()]
    absent = [str(path) for path in missing if not path.exists()]
    if absent:
        raise ConfigError(f"configured paths do not exist: {', '.join(sorted(set(absent)))}")


# Requests


def parse_settings(service: Service, raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise RequestError("settings must be an object")
    settings: dict[str, str] = {}
    for name, value in cast(dict[object, object], raw).items():
        if not isinstance(name, str) or name not in service.allowed_env:
            raise RequestError(f"setting {name!r} is not allowed for {service.name}")
        if (
            not isinstance(value, str)
            or len(value) > MAX_VALUE_LENGTH
            or not ENV_VALUE.fullmatch(value)
        ):
            raise RequestError(f"setting {name} has a value that is not permitted")
        settings[name] = value
    return settings


def parse_request(raw: bytes, config: Config) -> Request:
    try:
        decoded: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestError("request must be UTF-8 JSON") from error
    message = cast(dict[str, object], decoded) if isinstance(decoded, dict) else {}
    operation = message.get("request")
    expected = {"version", "id", "request", "service"} | (
        {"settings"} if operation == "start" else set()
    )
    if set(message) != expected:
        raise RequestError(f"{operation or 'request'} must contain only {sorted(expected)}")
    version, request_id, name = message["version"], message["id"], message["service"]
    # type() rather than ==, because JSON true and 1.0 both compare equal to 1.
    if type(version) is not int or version != PROTOCOL_VERSION:
        raise RequestError("unsupported protocol version")
    if not isinstance(request_id, str) or not request_id:
        raise RequestError("id must be a non-empty string")
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise RequestError("request is not permitted")
    if not isinstance(name, str) or name not in config.services:
        raise RequestError("service is not controlled by this helper")
    service = config.services[name]
    settings = parse_settings(service, message["settings"]) if operation == "start" else {}
    return Request(request_id, operation, service, settings)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


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


# Docker


class Controller:
    """Run Docker and Compose for the configured services, one mutation at a time per service."""

    def __init__(self, config: Config, state_directory: Path) -> None:
        self.config = config
        self.state_directory = state_directory
        self.locks = {name: threading.Lock() for name in config.services}

    def environment(self, settings: Mapping[str, str] | None = None) -> dict[str, str]:
        """A scrubbed environment: nothing inherited, so no variable can steer Docker or
        Compose except the service's allowed settings. HOME and DOCKER_CONFIG point at the
        helper's own state directory, because the service has no writable home."""
        base = {
            "PATH": CLEAN_PATH,
            "HOME": str(self.state_directory),
            "DOCKER_CONFIG": str(self.state_directory / "docker"),
        }
        return base | dict(settings or {})

    def run(
        self,
        arguments: list[str],
        service: Service,
        timeout: float,
        settings: Mapping[str, str] | None = None,
    ) -> str:
        """Run one Docker command; return its output, or raise OperationError."""
        command = [str(self.config.docker), *arguments]
        try:
            done = subprocess.run(
                command,
                env=self.environment(settings),
                cwd=service.project_directory,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise OperationError(f"docker {arguments[0]} timed out after {timeout:g}s") from error
        except OSError as error:
            raise OperationError(f"cannot run docker: {error}") from error
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip()[-OUTPUT_TAIL_CHARACTERS:]
            raise OperationError(f"docker {arguments[0]} failed ({done.returncode}): {tail}")
        return done.stdout

    def compose(
        self,
        service: Service,
        arguments: list[str],
        timeout: float,
        settings: Mapping[str, str] | None = None,
    ) -> str:
        project = [
            "compose",
            "--file", str(service.compose_file),
            "--project-directory", str(service.project_directory),
            "--project-name", service.project_name,
        ]  # fmt: skip
        return self.run([*project, *arguments], service, timeout, settings)

    def status(self, service: Service) -> JsonObject:
        """Report the service's container: its state, and the allowed settings it runs with."""
        filters = [
            f"label={PROJECT_LABEL}={service.project_name}",
            f"label={SERVICE_LABEL}={service.compose_service}",
            f"label={ONEOFF_LABEL}=False",
        ]
        listing = ["ps", "--all", "--quiet", "--no-trunc"]
        for item in filters:
            listing += ["--filter", item]
        found = self.run(listing, service, self.config.stop_timeout_seconds).split()
        if not found:
            return {"state": "missing", "exit_code": None, "oom_killed": False, "settings": {}}
        if len(found) > 1:
            raise OperationError(f"{service.name} has {len(found)} containers; expected one")
        inspected = self.run(["inspect", found[0]], service, self.config.stop_timeout_seconds)
        return container_status(json.loads(inspected)[0], service)

    def start(self, service: Service, settings: Mapping[str, str]) -> JsonObject:
        """Create or start the service with these settings; Compose recreates it if they changed.

        Nothing is built or pulled: the image must exist already."""
        with self.locks[service.name]:
            arguments = ["up", "--detach", "--no-deps", "--no-build", "--pull", "never"]
            self.compose(
                service,
                [*arguments, service.compose_service],
                self.config.start_timeout_seconds,
                settings,
            )
            return self.status(service)

    def stop(self, service: Service) -> JsonObject:
        """Stop the service's container, keeping it, so it does not restart by itself."""
        with self.locks[service.name]:
            timeout = self.config.stop_timeout_seconds
            self.compose(
                service,
                ["stop", "--timeout", str(service.stop_grace_seconds), service.compose_service],
                timeout,
            )
            return self.status(service)

    def handle(self, request: Request) -> JsonObject:
        if request.operation == "status":
            return self.status(request.service)
        if request.operation == "start":
            return self.start(request.service, request.settings)
        return self.stop(request.service)


def container_status(info: JsonObject, service: Service) -> JsonObject:
    """Reduce ``docker inspect`` output to what the broker needs."""
    state = cast(JsonObject, info.get("State") or {})
    status = str(state.get("Status", "unknown"))
    environment: dict[str, str] = {}
    for entry in cast(list[str], (info.get("Config") or {}).get("Env") or []):
        name, _, value = entry.partition("=")
        if name in service.allowed_env:
            environment[name] = value
    health = cast(JsonObject, state.get("Health") or {})
    return {
        "state": status,
        "exit_code": state.get("ExitCode") if status in GONE_STATES else None,
        "oom_killed": bool(state.get("OOMKilled", False)),
        "health": health.get("Status"),
        "settings": environment,
    }


# Unix socket server


class ControlServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """Threaded, so a slow stop does not hold up status requests; mutations are locked."""

    daemon_threads = True

    def __init__(self, address: str, controller: Controller, allowed_gid: int) -> None:
        self.controller = controller
        self.allowed_gid = allowed_gid
        super().__init__(address, ControlHandler)


class ControlHandler(socketserver.StreamRequestHandler):
    # Applied to each connection so that an idle client cannot hold a thread.
    timeout = CLIENT_TIMEOUT_SECONDS

    def handle(self) -> None:
        server = cast(ControlServer, self.server)
        credentials = self.request.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize(PEERCRED_FORMAT)
        )
        try:
            if not peer_is_authorised(credentials, server.allowed_gid):
                raise RequestError("connecting peer is not authorised")
            raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
            if len(raw) > MAX_MESSAGE_BYTES:
                raise RequestError("request exceeds maximum size")
            request = parse_request(raw, server.controller.config)
            LOGGER.info(
                "%s %s %s",
                request.operation,
                request.service.name,
                sorted(request.settings.items()),
            )
            response: JsonObject = {
                "ok": True,
                "version": PROTOCOL_VERSION,
                "id": request.id,
                "result": server.controller.handle(request),
            }
        except (RequestError, OperationError) as error:
            LOGGER.warning("refused or failed: %s", error)
            response = {"ok": False, "error": str(error)}
        except TimeoutError:
            response = {"ok": False, "error": "request timed out"}
        except Exception:
            LOGGER.exception("failed to serve request")
            response = {"ok": False, "error": "internal error"}
        payload = json.dumps(response, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        try:
            self.wfile.write(payload)
        except OSError:
            LOGGER.warning("client left before the response was sent")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True, help="Unix socket path to create")
    parser.add_argument(
        "--socket-group", required=True, help="group permitted to connect to the socket"
    )
    parser.add_argument("--config", type=Path, required=True, help="services configuration file")
    parser.add_argument(
        "--state-directory",
        type=Path,
        default=Path("/var/lib/model-broker-service-control"),
        help="writable directory for Docker's client configuration",
    )
    return parser.parse_args()


def prepare_socket(path: Path, group: str) -> int:
    """Remove a stale socket at path, give the group its directory, and return the group's ID.

    The broker container mounts the directory, not the socket file: a restart of this helper
    replaces the socket, and a bind-mounted file would keep pointing at the old one.
    Refuses to replace anything but a socket."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not path.is_socket():
            raise ValueError(f"refusing to replace non-socket path: {path}")
        path.unlink()
    try:
        group_id = grp.getgrnam(group).gr_gid
    except KeyError as error:
        raise ValueError(f"socket group does not exist: {group}") from error
    os.chown(path.parent, -1, group_id)
    os.chmod(path.parent, 0o750)
    return group_id


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = parse_arguments()
    try:
        config = load_config(arguments.config)
        check_files(config)
        arguments.state_directory.mkdir(parents=True, exist_ok=True)
        group_id = prepare_socket(arguments.socket, arguments.socket_group)
        server = ControlServer(
            str(arguments.socket), Controller(config, arguments.state_directory), group_id
        )
        os.chown(arguments.socket, -1, group_id)
        os.chmod(arguments.socket, 0o660)
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2
    LOGGER.info("controlling %s", ", ".join(sorted(config.services)))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if arguments.socket.is_socket():
            arguments.socket.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
