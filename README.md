# Model broker scaffolding

## Llama supervisor

`llama-supervisor.py` is a host-side executor for the existing rendered
llama.cpp Compose profiles. It is deliberately not a scheduler: it has no GPU
or RAM logic, does not inspect NVML, and does not decide whether a service is
safe to evict. The future broker makes those decisions and sends one of the
bounded actions `up`, `start`, `stop`, `restart`, `rm`, or `ps` for one literal
`SERVICE_NAME`.

Each request is JSON over a Unix-domain socket and has a timestamp, unique
nonce, and HMAC-SHA-256. The supervisor only resolves service names found in
`llama-cpp/config/*/*.env`, renders that selected profile with
`render-compose.py`, and runs the corresponding fixed `docker compose`
arguments. It never accepts Compose paths, env files, Docker options, images,
or commands from its client.

The socket and HMAC secret are the control boundary:

- The socket has no TCP listener and is mode `0660` for the dedicated
  `llama-supervisor-client` group.
- Only the broker container should receive the socket and secret mounts.
- A deliberate local administrator/test account can use the control client by
  joining that group (or by using `sudo`); ordinary containers cannot reach a
  Unix socket merely by sharing a Docker network.
- The supervisor itself runs as a separate unprivileged account, but Docker
  access remains host-root-equivalent. The installed executable and unit are
  root-owned, but the supervisor deliberately reads the configured repository's
  renderer and profile files. Treat write access to that repository as
  administrative authority and do not mount it writable into untrusted
  containers.

Install manually on the host after reviewing the paths and service account:

```sh
sudo ./model-broker/install-llama-supervisor.sh
sudo systemctl status llama-supervisor
```

The installer creates `/etc/llama-supervisor/environment` once. Edit it to use
a different repository path before starting the service. It prints the numeric
group ID required by the broker container.

The future broker Compose service needs these mounts and group membership:

```yaml
group_add:
  - "${LLAMA_SUPERVISOR_CLIENT_GID}"
volumes:
  - /run/llama-supervisor/control.sock:/run/llama-supervisor/control.sock
  - /etc/llama-supervisor/broker.hmac:/run/secrets/llama-supervisor.hmac:ro
environment:
  LLAMA_SUPERVISOR_SOCKET: /run/llama-supervisor/control.sock
  LLAMA_SUPERVISOR_SECRET: /run/secrets/llama-supervisor.hmac
```

Use the installed control client for intentional host-side testing. It needs
the secret, so invoke it with `sudo` unless a dedicated test principal has
been granted access:

```sh
sudo /usr/local/libexec/llama-supervisor/llama-supervisorctl.py \
  ps llama-cpp-gpu-0
sudo /usr/local/libexec/llama-supervisor/llama-supervisorctl.py \
  up llama-cpp-gpu-0
sudo /usr/local/libexec/llama-supervisor/llama-supervisorctl.py \
  stop llama-cpp-gpu-0
```

`stop` releases a running llama process and therefore its lifetime GPU lock.
`rm` is intentionally separate. No operation runs project-wide `docker compose
down`.
