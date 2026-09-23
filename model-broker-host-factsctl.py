#!/usr/bin/env python3
"""Send one read-only request to model-broker-host-facts."""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import sys
from pathlib import Path

PROTOCOL_VERSION = 1
REQUESTS = ("inventory", "utilisation")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", choices=REQUESTS)
    parser.add_argument(
        "--socket", type=Path, default=Path("/run/model-broker-host-facts/facts.sock")
    )
    parser.add_argument("--timeout", type=float, default=10, help="socket timeout in seconds")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    request = {
        "version": PROTOCOL_VERSION,
        "id": secrets.token_hex(12),
        "request": arguments.request,
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(arguments.timeout)
            client.connect(str(arguments.socket))
            client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
            response = client.makefile("rb").readline()
        payload = json.loads(response)
    except (OSError, ValueError) as error:
        reason = "no valid response" if isinstance(error, ValueError) else error
        print(f"model-broker-host-factsctl.py: {arguments.socket}: {reason}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
