#!/usr/bin/env bash
# Install the host-side executor. Run manually as root from this repository.
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ${EUID} -ne 0 ]]; then
    echo "run with sudo: sudo $0" >&2
    exit 2
fi

if ! getent group llama-supervisor-client >/dev/null; then
    groupadd --system llama-supervisor-client
fi
if ! id -u llama-supervisor >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin llama-supervisor
fi
usermod -a -G docker,llama-supervisor-client llama-supervisor

install -d -m 0750 -o root -g llama-supervisor-client /etc/llama-supervisor
if [[ ! -e /etc/llama-supervisor/environment ]]; then
    install -m 0640 -o root -g llama-supervisor-client \
        "$script_directory/llama-supervisor.env.example" /etc/llama-supervisor/environment
fi
if [[ ! -s /etc/llama-supervisor/broker.hmac ]]; then
    umask 0027
    python3 -c 'import secrets, sys; sys.stdout.write(secrets.token_hex(32))' \
        >/etc/llama-supervisor/broker.hmac
    chown root:llama-supervisor-client /etc/llama-supervisor/broker.hmac
    chmod 0640 /etc/llama-supervisor/broker.hmac
fi

install -d -m 0755 -o root -g root /usr/local/libexec/llama-supervisor
install -m 0755 -o root -g root \
    "$script_directory/llama-supervisor.py" /usr/local/libexec/llama-supervisor/llama-supervisor.py
install -m 0755 -o root -g root \
    "$script_directory/llama-supervisorctl.py" /usr/local/libexec/llama-supervisor/llama-supervisorctl.py
install -m 0644 -o root -g root \
    "$script_directory/llama-supervisor.service" /etc/systemd/system/llama-supervisor.service

systemctl daemon-reload
systemctl enable --now llama-supervisor.service

group_id=$(getent group llama-supervisor-client | cut -d: -f3)
echo "Installed. Add this to the broker container with group_add: [$group_id]."
