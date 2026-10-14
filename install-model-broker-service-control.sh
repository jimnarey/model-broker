#!/usr/bin/env bash
# Install the service-control helper. Run manually as root from this repository.
#
# The helper can start and stop the Compose services named in its configuration, so it is
# root-equivalent on this host. It installs a configuration only if none exists; edit
# /etc/model-broker/service-control.conf before starting it.
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ${EUID} -ne 0 ]]; then
    echo "run with sudo: sudo $0" >&2
    exit 2
fi

if ! getent group model-broker-control-client >/dev/null; then
    groupadd --system model-broker-control-client
fi

install -d -m 0755 -o root -g root /usr/local/libexec/model-broker-service-control
install -m 0755 -o root -g root \
    "$script_directory/model-broker-service-control.py" \
    /usr/local/libexec/model-broker-service-control/model-broker-service-control.py
install -m 0755 -o root -g root \
    "$script_directory/model-broker-service-controlctl.py" \
    /usr/local/libexec/model-broker-service-control/model-broker-service-controlctl.py
install -m 0644 -o root -g root \
    "$script_directory/model-broker-service-control.service" \
    /etc/systemd/system/model-broker-service-control.service

install -d -m 0755 -o root -g root /etc/model-broker
if [[ ! -e /etc/model-broker/service-control.conf ]]; then
    install -m 0644 -o root -g root \
        "$script_directory/model-broker-service-control.conf.example" \
        /etc/model-broker/service-control.conf
    echo "Installed an example configuration; edit /etc/model-broker/service-control.conf." >&2
fi

systemctl daemon-reload
systemctl enable --now model-broker-service-control.service

group_id=$(getent group model-broker-control-client | cut -d: -f3)
echo "Installed. Give the broker container group_add: [$group_id] and mount the directory"
echo "/run/model-broker-service-control (not the socket file inside it, which is replaced"
echo "whenever the helper restarts)."
