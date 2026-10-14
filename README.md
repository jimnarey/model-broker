# Model broker

An OpenAI-compatible broker in front of one llama.cpp router, and optionally
services such as Strata that change model by restarting. It decides which
models can be loaded together on this host's GPUs and RAM, loads and unloads
them explicitly, and proxies chat requests, so that a request for one model
does not needlessly unload another on a different GPU. See [the design](design.md).

**Status:** the broker serves `/v1/models` and scheduled, proxied
`POST /v1/chat/completions` (JSON and streamed), with optional API keys and JSON
logs. It is proven against a real router by the acceptance tests. It is not yet
deployed: that happens in the server-containers repository, which also puts the
router on a private network. Other OpenAI endpoints return 404.

## Broker API

`model_broker.application` is the broker's FastAPI application. It serves only
the endpoints the broker supports; every other path, including the router's own
management API, returns 404.

| Endpoint | |
| --- | --- |
| `GET /health` | Status, models loaded, router problems, authentication. No key needed. |
| `GET /v1/models` | The preset's models, in OpenAI's shape. |
| `GET /v1/models/{model}` | One model, or a 404 `model_not_found`. |
| `POST /v1/chat/completions` | Scheduled, then proxied to the router. |

### Startup

The broker validates the preset at `MODEL_BROKER_PRESET_PATH`; an invalid
preset stops startup. It then checks the router's `GET /props` and adopts the
models the router already holds. If the router is unreachable, or would choose
models to unload itself (a `--models-max` below the number of models that can be
loaded together, or autoload enabled), the broker still starts, with `/health`
status `degraded` and the reasons, refuses chat requests, and checks the router
again on the next request.

### Scheduling

The scheduler (`model_broker.scheduler`) loads a requested model at once when no
loaded model competes for its resources, so models on different GPUs stay
loaded together. Otherwise the competing models must be idle before it unloads
them: no running or queued requests, in the broker or in llama.cpp, and no use
within `MODEL_BROKER_IDLE_PROTECT_SECONDS`. A request waits up to
`MODEL_BROKER_MAX_WAIT_SECONDS` for that, then gets `503 resource_busy` with
`Retry-After`. A running request is never interrupted.

### Chat completions

An admitted request is forwarded unchanged, and the router's response is
relayed as it arrives, streams included. Router errors reach the client
unchanged. A refusal is an OpenAI-style error: 404 `model_not_found`, 503
`resource_busy`, 503 `router_unavailable`, 502 `model_load_failed`, or 400 for
a body without a model. When a client disconnects, the broker closes the router
request, so llama.cpp stops generating, and releases the model at once.

### Settings

| Variable | Default | |
| --- | --- | --- |
| `MODEL_BROKER_LLAMA_URL` | `http://llama-cpp:8080` | The router. |
| `MODEL_BROKER_PRESET_PATH` | required | The unified preset, the file the router uses. |
| `MODEL_BROKER_IDLE_PROTECT_SECONDS` | `300` | How long after its last use a model cannot be unloaded. |
| `MODEL_BROKER_MAX_WAIT_SECONDS` | `30` | How long a request waits for busy models; 0 refuses at once. |
| `MODEL_BROKER_LOAD_TIMEOUT_SECONDS` | `600` | How long a load or unload may take. |
| `MODEL_BROKER_API_KEYS_FILE` | unset (off) | API keys; see Authentication. |
| `MODEL_BROKER_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `MODEL_BROKER_SERVICE_PRESET_PATH` | unset (none) | Preset of services started by restarting a container (Strata); naming it turns them on. |
| `MODEL_BROKER_SERVICE_CONTROL_SOCKET` | `/run/model-broker-service-control/control.sock` | The service-control helper's socket. |
| `MODEL_BROKER_SERVICE_LOAD_TIMEOUT_SECONDS` | `3600` | How long such a model may take to start. |
| `MODEL_BROKER_STRATA_URL` | `http://strata:8080` | Strata's own API, which serves the model. |
| `MODEL_BROKER_STRATA_API_KEY` | unset | Strata's API key, if it has one. |

[`.env.example`](.env.example) describes each one.

### Services that change model by restarting

Some servers choose their model when they start and cannot change it while running.
Strata is one: it serves a single model per process, so it cannot join the llama.cpp
router. For these the broker changes model by stopping or starting the service's
container, which it cannot do itself because it has no Docker access. A second host
service, [`model-broker-service-control`](#service-control), does it on request.

Naming a service preset with `MODEL_BROKER_SERVICE_PRESET_PATH` adds the service's models
to `/v1/models`. The preset has the llama preset's layout (`version = 1`, one section per
model, the `--cuda<N>` suffix). Each section names the `service`, its `device`, any other
`exclusive` resources (Strata holds `RAM:weights`), and the environment variables its
container is started with as `env.NAME` keys. The broker needs no other knowledge of the
service's image: `strata/config/models-preset.ini` in the server-containers repository is an
example. Loading a model starts the container with that
section's settings; unloading stops it. The model currently loaded is read from the settings
the running container was created with, and it counts as loaded once the service's own API
answers. A model whose container exits while starting fails at once, with its exit code.

Every model of such a service claims its GPUs and listed resources exclusively, so loading one
unloads whatever competes for them, on either server, and the reverse. Strata's own API serves
the chat requests and reports whether it is busy. If a configured service cannot be reached,
meaning the helper is down, the broker stops scheduling everything until it can, because it
cannot tell what that service holds. Leave the setting unset when no such service is in use.

#### Service control

`model-broker-service-control.py` is a small root-owned systemd service, installed with
`sudo ./install-model-broker-service-control.sh`, next to the read-only host-facts helper but
with a different job: it starts and stops Docker Compose services. It is not a general Docker
API. A root-owned configuration (`/etc/model-broker/service-control.conf`, from
[the example](model-broker-service-control.conf.example)) lists the services it may touch, each
with the Compose project it belongs to and the environment variables a start may set. A
request can only name a listed service and set a listed variable to a value of plain
characters; nothing else in it reaches a command line, and Compose runs with a scrubbed
environment, `--no-build` and `--pull never`. Any service in the Compose project can be added
to the configuration, with or without settings.

It serves a versioned JSON protocol (`status`, `start`, `stop`) on a Unix socket owned by the
group `model-broker-control-client`. Give the broker container that group (`group_add`) and
mount the directory `/run/model-broker-service-control`, not the socket file: the helper
replaces the socket when it restarts, and a mounted file would keep pointing at the old one.
`model-broker-service-controlctl.py` sends a request by hand:

```sh
python3 /usr/local/libexec/model-broker-service-control/model-broker-service-controlctl.py \
  status strata
```

### Logging

The broker writes one JSON object per line to stderr: `time`, `level`, `event`,
then the event's facts. Events emitted while handling a request also carry its
`request_id` and, with authentication on, its `client`, including events from a
model load the request caused. Prompts, completions and keys are never logged.

```json
{"time": "2026-09-25T10:14:03.512+00:00", "level": "info", "event": "model_loaded", "request_id": "3f9c1a2b", "client": "vscode", "model": "qwen2.5-coder-7b-instruct-q4_k_m--cuda1", "seconds": 6.84}
```

`MODEL_BROKER_LOG_LEVEL` (default `INFO`) sets the detail:

- `INFO`: startup, authentication, router problems found or cleared, each
  unload and load (started, done, failed), refused requests, and each completed
  request.
- `DEBUG` adds each admission and finished request, each wait and its reason,
  what llama.cpp reported about each model considered for unloading, every call
  to a model server, every request received, and `/health` checks.

Every event is a dataclass in `model_broker/events.py`, which also sets its
level; code emits events rather than writing messages.

### Authentication

API keys are optional. Without `MODEL_BROKER_API_KEYS_FILE`, authentication is
off: every request is accepted, any key a client sends is ignored, startup logs
a warning, and `/health` reports `"authentication": "off"`. To turn it on, give
each client its own key in a secrets file and point the setting at it:

```text
# client-name  key (generate each key with: openssl rand -hex 24)
vscode            <key>
deepseek-harness  <key>
```

Comments must be on their own line.

Every `/v1/` request then needs `Authorization: Bearer <key>`; a missing or
wrong key gets OpenAI's `401 invalid_api_key`. `/health` needs no key. The file
is read at startup, so restart the broker after changing it. Keys cross the
network in plain HTTP until the broker is behind a TLS proxy.

### Running

Locally, with the project environment:

```sh
MODEL_BROKER_PRESET_PATH=/path/to/models-preset.ini \
  uv run uvicorn --app-dir src --factory model_broker.application:create_app \
  --host 127.0.0.1 --port 8000 --no-access-log
```

### Container

The [Dockerfile](Dockerfile) has an Ubuntu 24.04 runtime base. It uses the
locked dependency set and lets `uv` provide the Python 3.13 runtime required by
this project. It does not mount a Docker socket, host device, or host-facts
socket.

```sh
docker build --tag model-broker-dev:local .
docker run --rm --publish 8000:8000 \
  --env MODEL_BROKER_LLAMA_URL=http://llama-cpp:8080 \
  --env MODEL_BROKER_PRESET_PATH=/config/models-preset.ini \
  --volume /path/to/models-preset.ini:/config/models-preset.ini:ro \
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

### Acceptance tests

`tests/acceptance` proves the broker against a real llama.cpp router on the GPUs,
over HTTP: models on different GPUs stay loaded together, only the conflicting
model is unloaded, a dual-GPU model replaces both single-GPU models, an active
stream is never interrupted, recent use is protected, work sent straight to the
router is respected, a client disconnecting stops generation, and a restarted
broker keeps the loaded models.

They run against a router image built by this repository, from the commit
pinned in [`tests/acceptance/llama.env`](tests/acceptance/llama.env); keep it at
the commit the deployment runs. Build it once per commit (it compiles
llama.cpp with CUDA, which takes a while):

```sh
tests/acceptance/build-router.sh
```

The tests start that router in their own container, on localhost, with the
broker's settings, and a broker process per test. They choose the smallest
suitable models from the preset they are given. Both GPUs must be idle, so
unload any models from the deployment's router first.

```sh
MODEL_BROKER_RUN_ACCEPTANCE=1 \
MODEL_BROKER_ACCEPTANCE_PRESET=/path/to/unified/models-preset.ini \
MODEL_BROKER_ACCEPTANCE_MODELS=/path/to/models \
uv run pytest -q tests/acceptance
```
