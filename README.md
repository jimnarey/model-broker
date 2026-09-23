# Model broker scaffolding

A broker service for managing llama requests requiring different models
with different resource requirements.

## Broker API

`model_broker.application` is the first FastAPI layer of the broker. On startup
it obtains the private router OpenAPI document from `MODEL_BROKER_LLAMA_URL`.
It registers a safe `501 not_implemented` placeholder for each router operation
that the broker does not own yet, so the broker OpenAPI document exposes the
visible router surface without proxying a request around scheduling.

`GET /health` is broker-owned and remains available when the router schema
cannot be fetched. `POST /v1/chat/completions` is the first bespoke override:
it returns a broker-specific `501` until validation, scheduling, loading, and
proxying are implemented. It is never generated from, or passed through to,
the router.

Run locally with the project environment:

```sh
uv run uvicorn model_broker.application:app --host 127.0.0.1 --port 8000
```

The supplied [`.env.example`](.env.example) lists the current environment
settings.

### Container

The [Dockerfile](Dockerfile) has an Ubuntu 24.04 runtime base. It uses the
locked dependency set and lets `uv` provide the Python 3.13 runtime required by
this project. It does not mount a Docker socket, host device, or host-facts
socket.

```sh
docker build --tag model-broker-dv:local .
docker run --rm --publish 8000:8000 \
  --env MODEL_BROKER_LLAMA_URL=http://llama-cpp:8080 \
  model-broker-dev:local
```

The router address must be reachable from the broker container; in a normal
deployment it is the private service DNS name.

## Model-broker host facts

`model-broker-host-facts.py` is the optional, root-owned observation helper
described in [the design](design.md). It is not a supervisor: it does not use
Docker or Compose, start or stop applications, read model configuration, make
scheduling decisions, or expose a model API.

The helper exposes one Unix-domain socket at
`/run/model-broker-host-facts/facts.sock`. Its deliberately small, versioned
JSON protocol accepts exactly these requests:

| Request | Returned data |
| --- | --- |
| `inventory` | CPU/RAM/NUMA inventory plus CUDA-to-kernel-device and PCIe facts when NVIDIA's libraries are available. |
| `utilisation` | Timestamped CPU and host-memory use, plus NVIDIA GPU and memory utilisation when available. |

Every request must have exactly `version`, `id`, and `request` fields. A
successful response includes the same `id`, protocol version, and structured
`result`. The service does not execute supplied commands or return arbitrary
host output. NVIDIA information is obtained directly through NVML and the
CUDA driver API; it never shells out to `nvidia-smi`.

Socket access is the control boundary. The socket is owned by root and the
dedicated `model-broker-host-facts-client` group with mode `0660`; the service
also checks Linux peer credentials (including supplementary groups) for each
connection. Mount only this socket into the broker container. There is no
HMAC key or Docker socket to mount.

Install manually on the host after reviewing the unit:

```sh
sudo ./install-model-broker-host-facts.sh
sudo systemctl status model-broker-host-facts
```

When present, the installer disables and stops the legacy
`llama-supervisor.service`, so it can no longer control Compose workloads. It
does not remove the old unit or its credentials; review and remove those
legacy files separately when appropriate.

The installer prints the numeric group ID to add to the broker container. A
deployment that enables optional host facts should use:

```yaml
group_add:
  - "${MODEL_BROKER_HOST_FACTS_GID}"
volumes:
  - /run/model-broker-host-facts/facts.sock:/run/model-broker-host-facts/facts.sock
environment:
  MODEL_BROKER_HOST_FACTS_SOCKET: /run/model-broker-host-facts/facts.sock
```

For an intentional host-side check (normally through `sudo`):

```sh
sudo /usr/local/libexec/model-broker-host-facts/model-broker-host-factsctl.py inventory
sudo /usr/local/libexec/model-broker-host-facts/model-broker-host-factsctl.py utilisation
```

The first `utilisation` response reports `cpu.sample_ready: false`, because it
establishes the CPU baseline. Later samples return CPU percentage. NVIDIA or
hardware-interface fields can be unavailable; the broker must preserve that
as unknown rather than infer it.

## Tests

The normal suite needs no Docker:

```sh
pytest -q
```

The systemd installation and migration test is deliberately opt-in. It uses a
privileged, network-isolated `docker run` container and never uses Docker
Compose. Build its small local image first, then run the test:

```sh
docker build --file tests/systemd/Dockerfile --tag model-broker-systemd-test:local tests
MODEL_BROKER_RUN_SYSTEMD_INTEGRATION=1 pytest -q tests/test_systemd_integration.py
```

The integration test creates a legacy `llama-supervisor.service`, runs the
installer, verifies that the legacy service is stopped and disabled, verifies
the new socket ownership and mode, and makes an `inventory` request.
