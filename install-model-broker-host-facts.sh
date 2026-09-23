#!/usr/bin/env bash
# Install the read-only host-facts helper. Run manually as root from this repository.
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ${EUID} -ne 0 ]]; then
    echo "run with sudo: sudo $0" >&2
    exit 2
fi

if ! getent group model-broker-host-facts-client >/dev/null; then
    groupadd --system model-broker-host-facts-client
fi

# This helper replaces the old Docker-capable supervisor. Do not leave that
# service able to start or stop Compose workloads after this migration.
if systemctl cat llama-supervisor.service >/dev/null 2>&1; then
    systemctl disable --now llama-supervisor.service
fi

install -d -m 0755 -o root -g root /usr/local/libexec/model-broker-host-facts
install -m 0755 -o root -g root \
    "$script_directory/model-broker-host-facts.py" \
    /usr/local/libexec/model-broker-host-facts/model-broker-host-facts.py
install -m 0755 -o root -g root \
    "$script_directory/model-broker-host-factsctl.py" \
    /usr/local/libexec/model-broker-host-facts/model-broker-host-factsctl.py
install -m 0644 -o root -g root \
    "$script_directory/model-broker-host-facts.service" /etc/systemd/system/model-broker-host-facts.service

systemctl daemon-reload
systemctl enable --now model-broker-host-facts.service

group_id=$(getent group model-broker-host-facts-client | cut -d: -f3)
echo "Installed. Add this to the broker container with group_add: [$group_id]."
