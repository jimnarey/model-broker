from __future__ import annotations

from pathlib import Path

import pytest

from model_broker.auth import ApiKeysError, client_for, digest, load_api_keys, parse_api_keys

VSCODE_KEY = "test-key-for-vscode"
HARNESS_KEY = "test-key-for-harness"
KEYS_FILE = f"""\
# Broker clients: one "name key" pair per line.
vscode  {VSCODE_KEY}

deepseek-harness {HARNESS_KEY}
"""


# The keys file


def test_parse_api_keys_maps_key_digests_to_client_names() -> None:
    """Each line gives a client and its key; comments and blank lines are skipped.

    Only the SHA-256 digest of each key is kept, so the raw keys are not held in memory.
    """
    assert parse_api_keys(KEYS_FILE) == {
        digest(VSCODE_KEY): "vscode",
        digest(HARNESS_KEY): "deepseek-harness",
    }


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("vscode\n", "line 1 must be a client name and a key"),
        (f"vscode {VSCODE_KEY} extra\n", "line 1 must be a client name and a key"),
        ("vscode short-key\n", "'vscode' on line 1 is shorter than 16 characters"),
        (f"vscode {VSCODE_KEY}\nvscode {HARNESS_KEY}\n", "line 2 repeats client 'vscode'"),
        (f"vscode {VSCODE_KEY}\nother {VSCODE_KEY}\n", "'other' on line 2 is already in use"),
        ("# no keys yet\n\n", "has no keys; remove the setting"),
    ],
    ids=["no-key", "extra-field", "short-key", "repeated-client", "shared-key", "empty"],
)
def test_parse_api_keys_rejects_unusable_files(text: str, error: str) -> None:
    """A malformed keys file stops startup, with a message that never contains a key.

    A shared key would make two clients indistinguishable in the logs. An empty file would
    turn authentication on while refusing everyone, so it is refused as a probable mistake.
    """
    with pytest.raises(ApiKeysError, match=error) as raised:
        parse_api_keys(text)
    assert VSCODE_KEY not in str(raised.value) and HARNESS_KEY not in str(raised.value)


def test_load_api_keys_reports_an_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(ApiKeysError, match="could not read API keys file"):
        load_api_keys(tmp_path / "missing")


# Checking a request's key


@pytest.mark.parametrize(
    ("authorization", "client"),
    [
        (f"Bearer {VSCODE_KEY}", "vscode"),
        (f"Bearer {HARNESS_KEY}", "deepseek-harness"),
        (f"bearer {VSCODE_KEY}", "vscode"),
        (f"  Bearer   {VSCODE_KEY}  ", "vscode"),
        ("Bearer not-a-configured-key", None),
        (f"Bearer {VSCODE_KEY[:-1]}", None),
        (f"Basic {VSCODE_KEY}", None),
        ("Bearer", None),
        (None, None),
    ],
    ids=[
        "first-client",
        "second-client",
        "lowercase-scheme",
        "extra-spaces",
        "unknown-key",
        "truncated-key",
        "wrong-scheme",
        "no-key",
        "no-header",
    ],
)
def test_client_for_identifies_the_key_holder(
    authorization: str | None, client: str | None
) -> None:
    """A Bearer header with a configured key names its client; anything else names none.

    The scheme is case-insensitive, as HTTP requires. A key differing in one character, or
    sent under another scheme, is not accepted.
    """
    assert client_for(parse_api_keys(KEYS_FILE), authorization) == client
