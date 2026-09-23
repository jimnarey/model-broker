# Model broker scaffolding

A broker service for managing llama requests requiring different models
with different resource requirements.

## Broker API

`model_broker.application` is the first FastAPI layer of the broker. On startup
it obtains the private router OpenAPI document from `MODEL_BROKER_LLAMA_URL`.
It registers a safe `501 not_implemented` placeholder for each router operation
that the broker does not own yet, so the broker OpenAPI document exposes the
visible router surface without proxying a request around scheduling.

For now, these generated entries intentionally have limited OpenAPI detail: they preserve
the path, HTTP method, operation identifier, summary, and description, but not router
parameters, request bodies, response schemas, security requirements, tags, or other
metadata. They are a safe list of visible endpoints, not yet a full copy of the router
contract. This is a temporary limitation to come back to before clients rely on the
broker OpenAPI document for generated client code.

`GET /health` is broker-owned and remains available when the router schema
cannot be fetched. `POST /v1/chat/completions` is the first bespoke override:
it returns a broker-specific `501` until validation, scheduling, loading, and
proxying are implemented. It is never generated from, or passed through to,
the router.

Run locally with the project environment:

```sh
uv run uvicorn --app-dir src --factory model_broker.application:create_app --host 127.0.0.1 --port 8000
```

The supplied [`.env.example`](.env.example) lists the current environment
settings.

### Container

The [Dockerfile](Dockerfile) has an Ubuntu 24.04 runtime base. It uses the
locked dependency set and lets `uv` provide the Python 3.13 runtime required by
this project. It does not mount a Docker socket, host device, or host-facts
socket.

```sh
docker build --tag model-broker-dev:local .
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

Each GPU's `pcie` object separates three things:

| Fields | Meaning |
| --- | --- |
| `current_link_speed_gts`, `current_link_width` | The link right now. GPUs train down to 2.5 GT/s at idle, so this is not a capability. |
| `device_max_link_speed_gts`, `device_max_link_width` | What the card supports, regardless of slot. |
| `path_max_link_speed_gts`, `path_max_link_width` | The lowest maximum along the card's upstream PCIe ports: the most the card can reach in this slot. `null` if any port reports an unknown value. |

`upstream_ports` lists those ports from the card towards the CPU; a card behind the
chipset has several and shares the chipset uplink. On the current broker host, CUDA0
(`0000:04:00.0`) is behind the chipset and limited to 8 GT/s x4 (Gen3 x4), while CUDA1
(`0000:2b:00.0`) is on a CPU root port at 16 GT/s x8 (Gen4 x8); both cards report a
device maximum of 32 GT/s x8.

`cuda_device` uses the CUDA driver's default device order on the host. It matches
llama.cpp's `CUDAn` only while the llama container sees all GPUs in the same order: do not
set `CUDA_VISIBLE_DEVICES` or `CUDA_DEVICE_ORDER` there unless the helper's unit sets the
same values with `Environment=`.

The first `utilisation` response reports `cpu.sample_ready: false`, because it
establishes the CPU baseline. Later samples return CPU percentage. NVIDIA or
hardware-interface fields can be unavailable; the broker must preserve that
as unknown rather than infer it.

## Tests

The normal suite needs no Docker:

```sh
uv run pytest -q
```

Tests marked "real hardware" run only where NVML is available, such as the broker host,
and are skipped elsewhere. The host-facts helper runs under the host's system `python3`
(3.14 on the broker host), not the project environment, so check it with that interpreter
too:

```sh
uv run --isolated --no-project --python /usr/bin/python3 --with pytest \
  pytest -q tests/test_host_facts.py
```

The systemd clean-install test is deliberately opt-in. It uses a
privileged, network-isolated `docker run` container and never uses Docker
Compose. Build its small local image first, then run the test:

```sh
docker build --file tests/systemd/Dockerfile --tag model-broker-systemd-test:local tests
MODEL_BROKER_RUN_SYSTEMD_INTEGRATION=1 uv run pytest -q tests/test_systemd_integration.py
```

The integration test image uses Ubuntu 26.04 to match the broker host. It starts with no
legacy services, runs the installer, verifies that systemd starts the helper and gives its
socket the documented ownership and mode, then verifies that a client in the dedicated
group can request both supported operations. It tests no migration path.
