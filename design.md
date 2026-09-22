# Model broker design

## Scope and decisions made

The first broker is a custom Python service.  It exposes an OpenAI-compatible
LLM API and forwards requests to the existing llama.cpp services.  It does not
replace llama.cpp, its model presets, the different pinned llama.cpp builds, or
the service-per-GPU-topology approach.

The broker owns the *logical* model catalogue, request routing, resource
knowledge, conflict prevention, and every rule for bringing a service up or
down.  A small host-side supervisor is only a capability-limited Docker
Compose executor.  Neither service should accept arbitrary Docker commands
from an API client.

Initially, Ollama and ComfyUI are external GPU users.  The supervisor observes
them and blocks an incompatible llama request; it does not stop, start, or
otherwise manage them.  ComfyUI will later be exposed through a separate API
adapter, rather than being presented as an OpenAI model.

## Existing llama service contract

`llama-cpp/render-compose.py` is the canonical renderer for an approved
profile env file.  It creates one materialised Compose service with the
correct pinned source build, model/config volumes, GPU visibility, shared GPU
lock, and service identity.  The supervisor will use this renderer and Compose
for now; it will not reproduce the service definition through arbitrary
`docker run` calls.

Each GPU profile holds its GPU lock for the lifetime of its llama-server.  An
idle server therefore still occupies its declared logical resource set until
the supervisor stops it.  The lock is a final collision guard; the supervisor
makes the scheduling decision before trying a start.

The normal upstream server is already configured as a llama.cpp router with
one loaded model maximum and model autoloading.  A started profile can serve
one of the models offered by its preset; a different service/profile is still
selected when a different build, GPU layout, or resource set is required.

## Catalogue

The catalogue is held in one broker-owned JSON file for the first release, for
example `/state/catalogue.json`.  It is written atomically (write temporary
file, fsync, then rename) and survives broker restarts through its persistent
volume.  SQLite is deliberately deferred.

The authoritative source of model IDs is each running llama.cpp service's
`GET /v1/models` response, not a parser for model directories or preset files.
The static broker configuration contains only the approved service profiles,
their resources, and their Compose launch details.

### Explicit catalogue refresh

`POST /admin/catalogue/refresh` starts a distinct, serial maintenance job.
The broker first prevents conflicting provisioning actions and waits for any
existing managed request leases to drain.  For each enabled profile, in a
configured order:

1. The broker checks and reserves the profile's resource set, then asks the
   supervisor to start the service.
2. Once the supervisor reports the container ready, the broker queries that
   service's `/v1/models` endpoint on the private backend network.
3. The broker replaces that profile's previous entries with the returned
   entries, computes additions and removals, and writes the JSON catalogue.
4. It asks the supervisor to stop the temporary service, then releases its
   broker-held resource reservation before considering the next conflicting
   profile.

If a service cannot start or be queried, its previous catalogue entries are
retained and marked `stale`; failure is never interpreted as a model removal.
The refresh endpoint returns a job ID and job status is available separately,
so an HTTP client does not have to remain connected for the whole serial pass.

### Passive refresh

Whenever the supervisor starts a profile for a normal model request, it emits
a `profile_ready` event to the broker.  The broker queries `/v1/models` and
reconciles just that profile before sending a request to it.  This catches
configuration or build changes without running a full refresh.

The external API always uses a stable qualified name:

```text
<profile-id>/<backend-model-id>
```

For example, `upstream-16g-gpu0/qwen2.5-coder-14b-instruct-q6_k` and
`upstream-32g/qwen2.5-coder-14b-instruct-q6_k` are intentionally different
models to clients.  A future unqualified alias may point to a configured
preferred profile, but qualified names are the durable API contract.

### Logging

The broker writes structured JSON Lines to a persistent, rotating log file at
`/state/log/model-broker.jsonl`.  It records catalogue job start/finish,
profile starts/stops, model additions/removals, stale-catalogue failures,
resource denials, and request lifecycle summaries.  It must never record API
keys, Authorization headers, prompts, completions, or model input/output.

Representative entries are:

```json
{"event":"catalogue.model_added","profile":"upstream-16g-gpu0","model":"qwen2.5-coder-14b-instruct-q6_k","external_model":"upstream-16g-gpu0/qwen2.5-coder-14b-instruct-q6_k"}
{"event":"catalogue.model_removed","profile":"upstream-16g-gpu0","model":"obsolete-model"}
{"event":"catalogue.profile_query_failed","profile":"schwerz-16g-gpu1","catalogue_retained":true}
```

## Resources and scheduling

A resource set is a fixed, non-negotiable set of logical tokens assigned to a
profile.  It is not an attempt to infer spare VRAM and pack another process
onto a GPU.  This is intentionally conservative.

The initial logical resources are `GPU0`, `GPU1`, and `RAM`.

| Profile family | Initial resource set |
| --- | --- |
| Upstream 16 GB on GPU 0 | `{GPU0}` |
| Upstream 16 GB on GPU 1 | `{GPU1}` |
| Upstream 32 GB / all GPUs | `{GPU0, GPU1}` |
| Upstream CPU | `{RAM}` |
| GenerelSchwerz 16 GB on GPU 1 | `{GPU1, RAM}` |
| GenerelSchwerz 32 GB / all GPUs | `{GPU0, GPU1}` |
| csantiago78 / all GPUs | `{GPU0, GPU1}` |

`RAM` is a deliberate exclusive logical slot in this first policy.  It means
the CPU service and the single-GPU GenerelSchwerz service are considered
incompatible, even if aggregate host RAM monitoring suggests that a particular
combination might fit.  It can be replaced later by a capacity-aware RAM model
only after measured evidence supports it.

GPU IDs remain the host/NVIDIA indices used by the existing env files and
Compose `device_ids`; the broker does not translate them to UUIDs.  The 16 GB
GenerelSchwerz profile is fixed to GPU 1 (and therefore conflicts with any
other profile using GPU 1).  Before enabling scheduling, the broker runs its
GPU observation preflight for `llama-cpp/gpu-pcie-link.py 1` and requires the
configured PCIe-generation expectation for the PCIe-4-connected card.  The
registry records that expectation and, if useful, a minimum lane width.  A
failed preflight prevents the broker selecting the Schwerz profile.

For a request, the broker selects a profile only when all its resource tokens
are free.  It may ask the supervisor to stop a conflicting *broker-managed*
profile only if that profile has no active broker request lease and has
exceeded its idle grace period.  It never preempts an active request.  Foreign
GPU processes, Ollama, and ComfyUI cause a graceful `resource_busy` refusal
rather than an automatic stop.

## Networking and access

The broker is a Compose service.  Llama services remain Compose-launched for
now, but only the host supervisor starts and stops them.  They join an internal
`model-backends` network with the broker; clients do not join that network.
The broker also joins a separate internal client network for the future harness
and the existing private HTTPS-gateway network when LAN HTTPS access is wanted.

After migration, generated llama services have no published host ports.  The
broker reaches a backend by its approved service name on port 8080.  Existing
clients such as Pi and DeepSeek migrate to the broker endpoint so the broker's
lease accounting becomes authoritative.

The broker authenticates external OpenAI API clients with broker API keys.  A
catalogue refresh is an administrator-only action with a distinct admin key.

## Host supervisor boundary

`model-supervisor` is a thin systemd service, not a network server.  It owns
only Docker daemon access and execution of approved Compose actions.  It has
no knowledge of GPUs, RAM, model catalogues, idle time, request activity,
external blockers, or eviction policy.  The broker owns all of those concerns,
including HTTP proxying and every model API query.

The supervisor is deliberately service-agnostic.  Its root-owned manifest maps
an approved profile ID to a renderer input, Compose project/files, and Compose
service name.  This allows a non-llama service to be added later, but it does
not grant an API caller the ability to name an arbitrary file, image, bind
mount, Compose project, service, or command-line option.

The broker communicates over one Unix-domain socket, for example
`/run/model-supervisor/control.sock`.  systemd creates the private runtime
directory and the supervisor binds this socket; it has no TCP listener.  The
socket is owned by `model-supervisor:model-broker-api` with mode `0660`.
Only the broker container receives that one socket as a bind mount and runs
unprivileged with the dedicated numeric `model-broker-api` group.  Docker
network membership does not grant access to a Unix socket, and no other
container receives the mount.

Messages are length-delimited JSON and contain an HMAC-SHA-256 over a
timestamp, nonce, method, and canonical payload.  The HMAC key is root-created
and mounted read-only only into the broker.  The supervisor rejects stale and
replayed nonces.  Unix permissions and the absent network listener are the
primary access control; the HMAC is defence in depth.  The supervisor still
authorizes every operation against its own root-owned allowlist, so a
compromised broker can request only approved lifecycle operations, never an
arbitrary image, command, bind mount, or Compose path.

Initial protocol methods are:

```text
compose(profile_id, action=up|start|stop|restart|rm|ps|logs)
operation_status(operation_id)
```

The supervisor maps an allowed action to a fixed argument vector.  For example,
`up` runs the approved renderer then `docker compose ... up -d --no-build
--no-deps APPROVED_SERVICE`; `stop` runs `docker compose ... stop
APPROVED_SERVICE`.  It returns the process exit status and sanitised standard
output/error unchanged in meaning.  Thus a real Docker/llama failure such as a
GPU lock collision has the same consequence as it would on the command line;
the supervisor does not reinterpret it as resource policy.  Its action log
goes to journald; the broker's user-visible catalogue log remains the
persistent JSON Lines file described above.

The broker retains its own JSON state for resource reservations and request
leases.  On broker restart it begins conservatively: it queries the supervisor
for `ps`, treats running broker-managed profiles as occupied, and reconstructs
its model state before attempting a conflicting start.  The existing llama GPU
lock remains the final host-side collision guard.

GPU observation also belongs outside the supervisor.  The broker may use a
small private `gpu-observer` companion, or be given NVIDIA's `utility` driver
capability itself, to query NVML/nvidia-smi and the PCIe-link helper.  The
utility capability is sufficient for NVML and nvidia-smi and does not require
the CUDA `compute` capability.  That observer supplies facts; only the broker
turns those facts into a scheduling decision.

## Deferred work

- SQLite, multiple broker replicas, and distributed leases
- Dynamic VRAM/RAM packing or automatic management of non-llama GPU users
- Docker-SDK replacement of the established Compose launch path
- OpenAI Responses API and a dedicated ComfyUI workflow/WebSocket adapter
