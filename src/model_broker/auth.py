"""Optional API-key authentication for the broker's public API.

Keys come from a secrets file with one ``name key`` pair per line, one key per client. The
broker keeps only SHA-256 digests of the keys, and compares a presented key's digest with
every stored digest, so the comparison takes the same time whichever key, if any, matches.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from pathlib import Path

# Keys are random secrets, not passwords; a short one is almost certainly a mistake.
MIN_KEY_LENGTH = 16

ApiKeys = Mapping[bytes, str]
"""Client names by the SHA-256 digest of their key."""


class ApiKeysError(ValueError):
    """The API keys file cannot be used. Messages never include a key."""


def digest(key: str) -> bytes:
    return hashlib.sha256(key.encode()).digest()


def parse_api_keys(text: str) -> ApiKeys:
    """Parse ``name key`` lines; blank lines and lines starting with ``#`` are ignored."""
    keys: dict[bytes, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 2:
            raise ApiKeysError(f"API keys line {number} must be a client name and a key")
        name, key = fields
        if len(key) < MIN_KEY_LENGTH:
            raise ApiKeysError(
                f"API key for {name!r} on line {number} is shorter than {MIN_KEY_LENGTH} characters"
            )
        if name in keys.values():
            raise ApiKeysError(f"API keys line {number} repeats client {name!r}")
        if digest(key) in keys:
            raise ApiKeysError(f"API key for {name!r} on line {number} is already in use")
        keys[digest(key)] = name
    if not keys:
        raise ApiKeysError("API keys file has no keys; remove the setting to turn keys off")
    return keys


def load_api_keys(path: Path) -> ApiKeys:
    try:
        return parse_api_keys(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ApiKeysError(f"could not read API keys file {path}: {error}") from error


def client_for(keys: ApiKeys, authorization: str | None) -> str | None:
    """Return the client whose key is in an ``Authorization: Bearer`` header, or None."""
    scheme, _, token = (authorization or "").strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    presented = digest(token.strip())
    client = None
    for stored, name in keys.items():
        if hmac.compare_digest(stored, presented):
            client = name
    return client
