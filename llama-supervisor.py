#!/usr/bin/env python3
"""Capability-limited host executor for approved llama.cpp Compose profiles.

The broker owns scheduling, resource decisions, and request leases.  This
process receives an authenticated profile ID plus one bounded Compose action,
resolves that ID from the tracked llama profile env files, and executes the
corresponding fixed command.  It intentionally has no GPU, RAM, model, or
eviction-policy logic.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socketserver
import subprocess
import sys
import time
from typing import Any


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024
MAX_RESULT_BYTES = 32 * 1024
MAX_CLOCK_SKEW_SECONDS = 30
NONCE_LIFETIME_SECONDS = 120
ACTIONS = frozenset({"up", "start", "stop", "restart", "rm", "ps"})


class RequestError(ValueError):
    """A request that is invalid or not authorised."""


def parse_service_name(profile: Path) -> str:
    """Read SERVICE_NAME without evaluating or sourcing a profile env file."""
    for raw_line in profile.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == "SERVICE_NAME":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                return value
    raise RequestError(f"{profile} has no non-empty SERVICE_NAME")


def resolve_profile(project: Path, service: str) -> Path:
    """Resolve exactly one approved llama profile by its literal service name."""
    config = project / "llama-cpp" / "config"
    profiles = sorted(path for path in config.glob("*/*.env") if path.is_file())
    matches = [profile for profile in profiles if parse_service_name(profile) == service]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RequestError(f"unknown approved service {service!r}")
    raise RequestError(f"service {service!r} is declared by multiple profile env files")


def canonical_request(message: dict[str, Any]) -> bytes:
    """Return the exact protected portion of a protocol request."""
    required = ("version", "id", "timestamp", "nonce", "action", "service")
    if any(field not in message for field in required):
        raise RequestError("request is missing a required field")
    protected = {field: message[field] for field in required}
    return json.dumps(protected, separators=(",", ":"), sort_keys=True).encode("utf-8")


class Authorizer:
    def __init__(self, secret: bytes) -> None:
        if not secret:
            raise ValueError("the HMAC secret must not be empty")
        self.secret = secret
        self.used_nonces: dict[str, float] = {}

    def verify(self, message: dict[str, Any]) -> None:
        if message.get("version") != PROTOCOL_VERSION:
            raise RequestError("unsupported protocol version")
        if not isinstance(message.get("id"), str) or not message["id"]:
            raise RequestError("request id must be a non-empty string")
        if not isinstance(message.get("action"), str) or message["action"] not in ACTIONS:
            raise RequestError("action is not permitted")
        if not isinstance(message.get("service"), str) or not message["service"]:
            raise RequestError("service must be a non-empty string")
        if not isinstance(message.get("timestamp"), int):
            raise RequestError("timestamp must be an integer")
        if not isinstance(message.get("nonce"), str) or len(message["nonce"]) < 16:
            raise RequestError("nonce must be a string of at least 16 characters")
        if not isinstance(message.get("mac"), str):
            raise RequestError("request has no HMAC")

        now = time.time()
        if abs(now - message["timestamp"]) > MAX_CLOCK_SKEW_SECONDS:
            raise RequestError("request timestamp is outside the allowed clock skew")
        self._expire_nonces(now)
        if message["nonce"] in self.used_nonces:
            raise RequestError("request nonce has already been used")

        expected = hmac.new(self.secret, canonical_request(message), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, message["mac"]):
            raise RequestError("request HMAC is invalid")
        self.used_nonces[message["nonce"]] = now + NONCE_LIFETIME_SECONDS

    def _expire_nonces(self, now: float) -> None:
        for nonce, expiry in list(self.used_nonces.items()):
            if expiry <= now:
                del self.used_nonces[nonce]


def clipped(value: str) -> str:
    if len(value) <= MAX_RESULT_BYTES:
        return value
    return "[truncated]\n" + value[-MAX_RESULT_BYTES:]


class Supervisor:
    def __init__(self, project: Path) -> None:
        self.project = project.resolve()
        self.compose_file = self.project / "compose.ai.yml"
        self.renderer = self.project / "llama-cpp" / "render-compose.py"
        if not self.compose_file.is_file() or not self.renderer.is_file():
            raise ValueError(f"project does not contain compose.ai.yml and llama-cpp/render-compose.py: {self.project}")

    def execute(self, action: str, service: str) -> dict[str, Any]:
        profile = resolve_profile(self.project, service)
        rendered = self._render(profile)
        command = self._compose_command(action, service, rendered)
        result = subprocess.run(
            command,
            cwd=self.project,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return {
            "service": service,
            "profile": str(profile.relative_to(self.project)),
            "action": action,
            "exit_code": result.returncode,
            "stdout": clipped(result.stdout),
            "stderr": clipped(result.stderr),
        }

    def _render(self, profile: Path) -> Path:
        result = subprocess.run(
            [sys.executable, os.fspath(self.renderer), os.fspath(profile)],
            cwd=self.project,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            detail = clipped(result.stderr or result.stdout).strip()
            raise RuntimeError(f"render-compose.py failed ({result.returncode}): {detail}")
        output = result.stdout.strip()
        rendered = Path(output)
        if not output or not rendered.is_file():
            raise RuntimeError("render-compose.py did not return an existing Compose file")
        return rendered

    def _compose_command(self, action: str, service: str, rendered: Path) -> list[str]:
        command = [
            "docker", "compose",
            "-f", os.fspath(self.compose_file),
            "-f", os.fspath(rendered),
        ]
        actions = {
            "up": ["up", "-d", "--build", "--no-deps", service],
            "start": ["start", service],
            "stop": ["stop", service],
            "restart": ["restart", service],
            "rm": ["rm", "-f", service],
            "ps": ["ps", "--format", "json", service],
        }
        return [*command, *actions[action]]


class ControlServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(self, address: str, supervisor: Supervisor, authorizer: Authorizer) -> None:
        self.supervisor = supervisor
        self.authorizer = authorizer
        super().__init__(address, ControlHandler)


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
        if len(raw) > MAX_MESSAGE_BYTES:
            self._respond({"ok": False, "error": "request exceeds maximum size"})
            return
        try:
            message = json.loads(raw.decode("utf-8"))
            if not isinstance(message, dict):
                raise RequestError("request must be a JSON object")
            self.server.authorizer.verify(message)  # type: ignore[attr-defined]
            result = self.server.supervisor.execute(message["action"], message["service"])  # type: ignore[attr-defined]
            self._respond({"ok": result["exit_code"] == 0, "id": message["id"], "result": result})
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, RequestError, RuntimeError, ValueError) as error:
            self._respond({"ok": False, "error": str(error)})

    def _respond(self, response: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(response, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="repository containing compose.ai.yml")
    parser.add_argument("--socket", type=Path, required=True, help="Unix socket path to create")
    parser.add_argument("--secret", type=Path, required=True, help="read-only HMAC secret file")
    parser.add_argument("--socket-group", required=True, help="group permitted to connect to the socket")
    return parser.parse_args()


def prepare_socket(path: Path, group: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not path.is_socket():
            raise ValueError(f"refusing to replace non-socket path: {path}")
        path.unlink()
    try:
        gid = grp.getgrnam(group).gr_gid
    except KeyError as error:
        raise ValueError(f"socket group does not exist: {group}") from error
    # socketserver binds after this function returns; ownership is corrected in main.
    return gid


def main() -> int:
    arguments = parse_arguments()
    try:
        group_id = prepare_socket(arguments.socket, arguments.socket_group)
        supervisor = Supervisor(arguments.project)
        authorizer = Authorizer(arguments.secret.read_bytes())
        server = ControlServer(os.fspath(arguments.socket), supervisor, authorizer)
        if arguments.socket.stat().st_gid != group_id:
            os.chown(arguments.socket, -1, group_id)
        os.chmod(arguments.socket, 0o660)
    except (OSError, ValueError) as error:
        print(f"llama-supervisor.py: {error}", file=sys.stderr)
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
