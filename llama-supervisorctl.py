#!/usr/bin/env python3
"""Send one authenticated test or administrative request to llama-supervisor."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import socket
import sys
import time


PROTOCOL_VERSION = 1
ACTIONS = ("up", "start", "stop", "restart", "rm", "ps")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("service", help="literal approved SERVICE_NAME")
    parser.add_argument("--socket", type=Path, default=Path("/run/llama-supervisor/control.sock"))
    parser.add_argument("--secret", type=Path, default=Path("/etc/llama-supervisor/broker.hmac"))
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for a Compose operation")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    protected = {
        "version": PROTOCOL_VERSION,
        "id": secrets.token_hex(12),
        "timestamp": int(time.time()),
        "nonce": secrets.token_hex(24),
        "action": arguments.action,
        "service": arguments.service,
    }
    payload = json.dumps(protected, separators=(",", ":"), sort_keys=True).encode("utf-8")
    secret = arguments.secret.read_bytes()
    request = dict(protected, mac=hmac.new(secret, payload, hashlib.sha256).hexdigest())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(arguments.timeout)
        client.connect(str(arguments.socket))
        client.sendall(json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n")
        response = client.makefile("rb").readline()
    if not response:
        print("llama-supervisorctl.py: supervisor closed the connection without a response", file=sys.stderr)
        return 2
    payload = json.loads(response)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
