# Model broker design

## Scope and decisions

The broker is a standalone Python service that exposes an OpenAI-compatible API, with optional API-key authentication. It sits between clients and one already-running upstream llama.cpp router. For each request it decides whether the requested model *variant* can be loaded now, which loaded model must be unloaded to make room, or whether the request must wait or be refused. It then asks the router to load and unload models and proxies the request.

The broker exists because the router cannot make that decision. The router's model limit is a global process count: it does not know that a GPU1 model and a GPU0 model can stay loaded together, or that a request for another GPU1 model should make only the GPU1 worker a candidate for unloading. Its request order and LRU behaviour are therefore not a scheduling policy for this host. The broker owns the set of loaded models and makes every load and unload decision explicitly.

The router runs in Docker as the `llama-cpp` Compose service. Docker starts it normally and keeps it running, and the broker never starts or stops it. The broker itself never calls Docker, Compose or systemd. For services that can only change model by restarting, it asks a small host service to start or stop a named Compose service (see Service control); that is the one place it causes a container to start or stop.

**Network assumption:** the router is on a private network that only the broker can reach, so every request to it passes through the broker. The deployment provides that network.

The deployment in the server-containers repository supplies the unified llama.cpp preset. The broker only needs a readable path to that file; it does not need that repository, Compose files, model directories, or GPU devices. It is deliberately based on upstream llama.cpp. Experimental MoE forks remain available for benchmarking but are not on the broker path.

The broker schedules the llama.cpp router and, optionally, services that change model by restarting their container, of which Strata is the first. Both sit behind one small `Backend` interface, extracted from the working llama.cpp code when Strata arrived: load, unload, status, activity, and chat forwarding. Other significant-compute applications, such as ComfyUI or Ollama, are not under the broker. Their GPU use is outside its accounting and can still cause out-of-memory failures; bringing one in means adding a backend, and for one that must be restarted, a section in the service preset and in the helper's configuration.

## Standalone configuration

The model-broker repository contains its own container image, application configuration, and `.env.example`. It has no build-time dependency on server-containers. A deployment provides its environment through Docker Compose, systemd, or a normal `.env` file.

Only these settings are required to start:

```dotenv
# Address reachable from the broker container or host process.
MODEL_BROKER_LLAMA_URL=http://llama-cpp:8080

# Read-only path inside the broker container or host process.
MODEL_BROKER_PRESET_PATH=/config/models-preset.ini
```

The server-containers Compose service mounts the same unified preset into the broker at the configured path and sets `MODEL_BROKER_LLAMA_URL` to the llama service's Compose name.

Other settings have safe defaults and are documented in `.env.example`: the idle protection window (`MODEL_BROKER_IDLE_PROTECT_SECONDS`), the maximum request wait (`MODEL_BROKER_MAX_WAIT_SECONDS`), the load and unload timeout (`MODEL_BROKER_LOAD_TIMEOUT_SECONDS`), the API keys file (`MODEL_BROKER_API_KEYS_FILE`), and the log level (`MODEL_BROKER_LOG_LEVEL`). The listen address and port are Uvicorn's command-line options. Secrets are supplied through the deployment environment or a secrets mount, never committed.

## Unified llama.cpp contract

The `llama-cpp` container can see both CUDA devices. The router starts with the unified preset, `--no-models-autoload`, and `--metrics`. It does not receive router-wide placement arguments such as `--device`, `--split-mode`, `--main-gpu`, `--tensor-split`, or `--n-gpu-layers`, which would override the placement encoded in each preset section. Router-driven idle sleeping or any other automatic model discard is disabled: the broker, not the router, decides when a loaded model leaves memory.

Each preset section is self-contained: it holds all effective settings for its model rather than relying on router-wide defaults.

The router's `models-max` never needs to choose a victim. Every variant holds at least one exclusive resource (a GPU, or `RAM:weights`; see below), and no two loaded models can hold the same one. So the most models that can be loaded at once is the largest group of variants whose exclusive resources do not overlap: three with two GPUs (a GPU0 model, a GPU1 model, and a CPU model). If `models-max` is at least that number, it never binds, and the broker needs no process-count accounting. At startup the broker reads the router's `GET /props`, which in router mode reports `max_instances` and `models_autoload`, and refuses to schedule while `max_instances` is below that number or autoload is enabled.

At deployment time, verify the relation between `CUDA0`/`CUDA1` and physical cards with `llama-server --list-devices` in the image. The broker uses llama.cpp device identifiers, not a guessed PCI bus order.

## Preset-derived catalogue

`models-preset.ini` is the sole model catalogue. It is mounted read-only into both the router and the broker. The broker does not scan model directories or maintain a separate catalogue.

On startup, the broker parses and validates every preset section and builds immutable variants in memory:

```python
@dataclass(frozen=True)
class ResourceClaim:
    resource: str               # GPU0, GPU1, CPU, or RAM:weights
    exclusive: bool

@dataclass(frozen=True)
class Variant:
    id: str                     # INI section / public model ID
    model_path: str
    settings: Mapping[str, str] # effective llama.cpp settings
    claims: tuple[ResourceClaim, ...]
    resource_class: str         # gpu_resident, moe_offload, or cpu_weights
```

The public model ID is exactly the preset section name, for example `Qwen3.8-Flash-Next-UD-Q3_K_XL--cuda1`. The final suffix is an intentional, stable part of the ID:

| Suffix | Required preset placement | GPU claim |
| --- | --- | --- |
| `--cuda0` | `device = CUDA0`, `n-gpu-layers = all` | GPU0 |
| `--cuda1` | `device = CUDA1`, `n-gpu-layers = all` | GPU1 |
| `--cuda0-cuda1` | `device = CUDA0,CUDA1`, `n-gpu-layers = all` | GPU0 and GPU1 |
| `--cpu` | `device = none`, `n-gpu-layers = 0` | none |

Startup validation rejects a malformed or ambiguous preset rather than scheduling it. The file must start with `version = 1` and no other top-level setting. Every section needs a model and a recognised suffix, and its `device` (and any draft-model device) must name exactly the suffix's GPUs. GPU sections must use `n-gpu-layers = all`, because `auto` or a layer count lets llama.cpp put weights in host RAM that the claim does not cover. Expert offload to RAM is allowed only explicitly, through `n-cpu-moe`. CPU sections must not set GPU-only options.

## Resources and claims

The broker builds its resource set from the parsed preset; there is no hand-maintained GPU list. Each variant claims:

| Resource | Claim |
| --- | --- |
| Each GPU in its suffix | Exclusive. |
| `RAM:weights` | Exclusive for `cpu_weights` and `moe_offload` (`n-cpu-moe > 0`) variants; recorded but not exclusive for `gpu_resident` variants. |
| `CPU` | Recorded, not exclusive. |

Two variants conflict when both hold some resource exclusively. A non-exclusive claim records demand for a later capacity policy and never blocks another variant. This gives the basic policy:

| Loaded | Requested | Conflict |
| --- | --- | --- |
| `--cuda0` | `--cuda1` | No: both stay loaded. |
| `--cuda1` | another `--cuda1` | Yes: GPU1. The GPU0 worker is untouched. |
| `--cuda0` or `--cuda1` | `--cuda0-cuda1` | Yes: dual-GPU needs both cards. |
| GPU-resident `--cuda0` | `--cpu` | No: the GPU model only records RAM use. |
| MoE offload on GPU1 | `--cpu` | Yes: both hold `RAM:weights` exclusively. |

The `RAM:weights` rule is deliberately more conservative than the hardware. A CPU-weights model and an MoE expert-offload model may fit together, but they compete for RAM capacity, memory bandwidth, and CPU time, so they do not run together until measurements justify a capacity policy.

## Router API contract

The router is a private backend. The broker reaches it on the Compose network as `http://llama-cpp:8080`. Once clients have migrated, the router has no published host or LAN port.

The broker uses these router endpoints:

```text
GET  /props                          router settings: max_instances, models_autoload
GET  /models                         each model's status
POST /models/load     {"model": id}
POST /models/unload   {"model": id}
GET  /slots?model=<id>               per-slot is_processing for a loaded model
GET  /metrics?model=<id>             requests_processing, requests_deferred, token counters
POST /v1/chat/completions            proxied inference, including SSE streams
```

After a load or unload the broker polls `GET /models` until the desired state is reported, or until a bounded timeout. Status values are `unloaded`, `loading`, `loaded`, and `sleeping`. The pinned router reports a failed load as `unloaded` with `failed: true` and the worker's exit code, so the broker fails the load as soon as it sees that. Unloading a model that is not running returns 400; the broker treats it as done when `GET /models` confirms the model is unloaded. A chat request for an unloaded model returns 400 "model is not loaded" rather than starting a worker: the broker is the only component that loads models.

## Services that change model by restarting

Strata runs a large MoE model by spreading its experts over GPU, RAM and SSD. It serves exactly one model per process, fixed at start, and its server has no API to change it. (Later versions can unload and reload that same model, which is not enough.) It therefore cannot be driven like the llama.cpp router. The same goes for other servers that choose their model at start, so the broker has a general backend for them, `ManagedService`, and Strata is the first service on it.

Changing model means stopping or starting the service's container, and the broker has no Docker access. A second host service does it, described under Service control below. The broker asks it to start the container with a model's settings, or to stop it. The container is the unit of loading: the model is loaded when the container runs and the service's own API answers, and unloaded when it has stopped.

### The service preset

A service preset has the llama preset's layout (`version = 1`, `;` comments, one self-contained section per model, a `--cuda<N>` or `--cpu` suffix) and is mounted read-only into the broker, which names it with an optional environment variable: with none, no service is configured. A section holds:

- `service`: which service it is. The broker needs a client for it (today only `strata`); an unknown service stops startup.
- `device`: the GPUs, matching the suffix, as for llama.cpp.
- `exclusive`: resources besides the GPUs that no other model may hold while it is loaded. Strata copies its experts into RAM, so it holds `RAM:weights`, which conflicts with llama.cpp's CPU-weight and expert-offload models.
- `env.NAME`: each environment variable the container is started with. Every section lists all of them, empty ones too, because the container is recreated with exactly these and an empty value overrides whatever `.env` sets.

Nothing else is accepted: a section is the broker's whole description of what to start. The uncensored Strata model is the plain model's section with a control-vector path in `env.STRATA_SPEED_PROJECTION`; the two are separate models because they behave differently, though they share weights.

A service's models claim their GPUs and listed resources exclusively and record CPU, so two models of one service conflict, as they must, and a service model conflicts with a llama.cpp model on its GPU or holding `RAM:weights`. A llama.cpp model on another GPU can stay loaded. Public model IDs must be unique across presets.

### Loading, unloading and what is loaded

The settings a container was created with say which model it runs: the model whose `env` all matches. So the broker learns what is loaded at startup by asking the helper for the container's status, with no state of its own. A running container whose settings match no section is reported as an unrecognised model, loaded and holding its resources, which the existing check turns into a problem that stops scheduling.

A model is `loading` while its container runs but the service's API does not answer, `loaded` once it does, and `unloaded` when the container is stopped. Loading starts the container and waits; unloading stops it and waits. Both refuse to act on a different model from the one running (a start would recreate the container and kill the request it is serving), which the scheduler never asks for, so reaching it means the broker's picture is wrong. If the container exits while the broker waits for it to load, the load fails at once with the exit code rather than at the timeout. After a stop Docker reports the exit code of a killed container, 137, so an exit code says nothing about a stop that was requested.

For Strata the broker reads activity from Strata's own `GET /v1/status`: `in_flight` (the running request and those queued) plays the part of llama.cpp's processing and deferred counts, and `requests` (finished plus running) the part of the decode counter, so work that reached Strata without the broker is noticed in the same way. Chat requests go to the same server unchanged. Strata ignores the model name in a request and answers with whatever it runs, which is safe because the scheduler admits a request only while the model it names is the loaded one. An API key set on Strata is sent as a bearer token.

A Strata start is slow: its first start builds the engine (15 to 40 minutes) and every start reads tens of GB into RAM. `MODEL_BROKER_SERVICE_LOAD_TIMEOUT_SECONDS` (default an hour) replaces the load timeout for such services.

If a service cannot be reached, meaning its helper is down, that is a problem that stops all scheduling, not only the service's models. The broker cannot tell what an unreachable service holds, and although a request for a llama.cpp model on the other GPU might be fine, there is no way to know that the GPU is free. Leave the service unconfigured when it is not in use. Startup also reports two servers holding conflicting models, which only the broker can see.

### Service control

`model-broker-service-control.py` is a root-owned systemd service in this repository, beside the host-facts helper but with the opposite character: that one only reads, this one starts and stops containers, so it is root-equivalent on the host and is built to do very little. It is not a general Docker API:

- It has three requests, `status`, `start` and `stop`, in a versioned JSON protocol over a Unix socket, each naming a service. The socket's group, checked against the peer's credentials, decides who may ask.
- It controls only the Docker Compose services in its root-owned configuration, each in the Compose project that file names, and can control any service in the project. It runs `docker compose up --detach --no-deps --no-build --pull never`, or `stop`, so it never builds, pulls, removes or touches other services.
- A start can set only the environment variables the configuration allows for that service, to values of plain characters (empty is allowed). Compose interpolates them into the service, so the broker can choose a model or a context but not an image, mount, command or device. No request text becomes part of a command line other than the configured Compose service name.
- Compose runs in an empty environment apart from `PATH` and a private Docker configuration directory, with a time limit, so nothing the helper inherited steers it.
- Stopping uses `compose stop`, which keeps the container, so `restart: unless-stopped` does not bring it back. Starting an existing container with unchanged settings does nothing; changed settings make Compose recreate it.
- Requests are served on threads, so a slow stop does not hold up the status polls the broker makes while a model unloads. Mutations are serialised per service.

The helper reports a container's Docker state, its exit code once stopped, whether it was killed for want of memory, and the allowed settings it was created with. The broker container mounts the helper's runtime directory and joins its client group; it never receives the Docker socket.

## Public API and server

The broker, rather than llama.cpp, serves the public API. It is an ASGI application implemented with FastAPI and served by Uvicorn.

| Endpoint | Behaviour |
| --- | --- |
| `GET /health` | Status (`ok` or `degraded`), the number of models, the models loaded, the router's build and any problems that block scheduling, and whether authentication is on. Needs no key. |
| `GET /v1/models` | The validated preset variants in OpenAI's `object: "list"` / `object: "model"` shape. |
| `GET /v1/models/{model}` | One variant, or an OpenAI-style not-found error. |
| `POST /v1/chat/completions` | Validates that `model` is a known variant, schedules it, and proxies the request and any SSE stream (see below). |

A chat request is admitted by the scheduler before the broker responds, so a refusal gets its own OpenAI-style error: 404 `model_not_found`, 503 `resource_busy` with `Retry-After`, 503 `router_unavailable`, 502 `model_load_failed`, or 400 for a body without a model. Once admitted, the body is forwarded unchanged and the router's status, headers, and body are relayed as they arrive, so streams stay streams and router errors, such as a prompt too long for the context, reach the client unchanged. The broker watches for the client leaving throughout, including while the router is processing the prompt. If it leaves, the broker closes the router request, which stops llama.cpp generating, and releases the model at once. However a response ends, its admission is released.

Every other path returns 404. Supporting another endpoint is a deliberate change: implement and test a broker handler that extracts and schedules any model it names. Endpoints with no model effect may later become plain pass-through handlers.

## Scheduling and request lifecycle

There is one broker instance, and it is the only authority for loading and unloading models on this host. Decisions to unload and load are serialised by a scheduling lock. The unloads and loads themselves, and all proxying, happen outside it, so a long load of one model does not hold up a model that needs none of its resources.

For each loaded model the broker keeps:

- **active requests**: requests it has proxied whose response or stream has not finished;
- **queued requests**: requests waiting in the broker for this model; and
- **last activity**: when the model last finished a request or was last observed working.

A model stays loaded until a request needs its resources. There is no idle unload.

### Request flow

1. Authenticate the caller, when API keys are configured, and resolve `model` to a variant; an unknown model is a 404.
2. If the variant is loaded, proxy immediately. If slots are busy, llama.cpp queues the request itself. If the variant is loading, wait for that load.
3. Otherwise, find the loaded models whose claims conflict with the variant. The request proceeds only when every one of them is *displaceable*. A loaded model is **busy** while any of these holds:
   - **it is working**: the broker has active requests for it, or llama.cpp reports a processing slot or `requests_processing > 0`;
   - **it has queued work**: requests are waiting in the broker for it, or llama.cpp reports `requests_deferred > 0`; or
   - **it was active recently**: its last activity is within the idle protection window (`MODEL_BROKER_IDLE_PROTECT_SECONDS`), so an interactive user between turns keeps their model.
4. While any conflicting model is busy, the request waits, up to the maximum wait (`MODEL_BROKER_MAX_WAIT_SECONDS`). A maximum wait of 0 refuses immediately. When the wait expires the broker returns `503 resource_busy` with `Retry-After`, based on when the protection window ends where that is the only obstacle. The maximum wait covers only waiting for busy models, not the unload and load that follow.
5. Once every conflicting model is displaceable, the broker decides, under the scheduling lock, to unload them and load the requested variant, then releases the lock. It unloads each one, waiting for `unloaded`, then loads the variant and waits for `loaded`. Until the router confirms an unload, the model still holds its resources, so no other decision can place a model on a GPU that is being freed; a model being loaded counts as busy. It then records the request as active and proxies it.
6. When the response or stream ends, is cancelled, or fails, the broker decrements the active count, updates last activity, and wakes waiting requests.

An in-progress request is never interrupted to make room for another model.

A loaded model's own requests keep being admitted while another model waits for its resources. Under steady traffic for the loaded model, a conflicting request can therefore wait until its maximum wait expires and be refused. That is intended: the current user finishes first. Requests waiting for the same unloaded variant share one load. Requests for different conflicting variants are admitted in arrival order.

### Observing llama.cpp

The broker's own counts cover every request that passes through it. It also asks llama.cpp, because work can reach the router without the broker (for example while clients migrate), and because the router's queue is the authority on what it has accepted. The broker reads `GET /slots?model=` and `GET /metrics?model=` for a conflicting model whenever a scheduling decision depends on it, not on a timer. If the router's decode counter (`n_decode_total`, which rises during prompt processing and generation) has advanced since the previous observation, the model counts as active at the time of that observation: the broker cannot tell when the work ended, so it assumes the latest possible time. The broker records the counter when it adopts or loads a model, so work before the first check is seen too; if it could not record one, the first check counts as activity.

If an observation fails, the model is treated as busy. The broker never unloads a model it cannot see is idle.

## Restart and failure handling

On startup the broker reads the preset, checks `GET /props`, and reads `GET /models`. Every model the router reports in any state but `unloaded` becomes a loaded model with no active requests, holding its resources, and its last activity is the startup time, so the protection window applies after a broker restart. A model still `loading` is waited for before requests are admitted to it. A status the broker does not recognise is also a problem that blocks scheduling, since the broker cannot tell what it means for the model's resources. Already-loaded router workers are not needlessly discarded, although request streams do not survive a broker restart.

If a load fails, the broker returns a backend error to the caller. It releases the variant's resources only when the router confirms the model is `unloaded`, as after a refused request or a worker that exited. A timeout (`MODEL_BROKER_LOAD_TIMEOUT_SECONDS`) or a lost connection does not show that the worker has gone; it may still be loading onto its GPUs. The model then keeps its resources and the router is marked unfit, so that the next request checks the router again and adopts what it reports. That check keeps the broker's record of every model still loaded, with its running and waiting requests. If an unload fails, the model still counts as loaded, holding its claims, until `GET /models` shows it gone. The broker does not ask the router to evict, and it never moves a loaded model to a different GPU or to the CPU.

Changing the preset is a deployment: update the file and restart the router and the broker. There is no administrative reload API.

## Networking, authentication, and logging

The broker is the only externally reachable API service. The router is on a private network shared only with the broker, so the broker is the only way to reach it. The broker receives the router URL and a read-only preset mount, plus normal configuration and secrets. Network layout belongs to the deployment, not this repository.

Authentication is optional, so that existing clients keep working while they are given keys:

- With no keys file configured, authentication is off. The broker accepts every request and ignores any key a client sends (many OpenAI clients always send one). Startup logs a warning and `/health` reports that authentication is off.
- `MODEL_BROKER_API_KEYS_FILE` names a secrets file with one `name key` pair per line: one key per client. When it is set, every `/v1/` request needs `Authorization: Bearer <key>`, compared in constant time; anything else gets an OpenAI-style 401. `/health` stays unauthenticated for health checks.
- Logs record the client's name, never its key, so one client can be identified or revoked without affecting the others.

A bearer key crossing the LAN in plain HTTP can be read by anyone on the network. The deployment will later put the broker behind the Caddy gateway, which terminates TLS; that needs no change to the broker.

The broker emits structured JSON logs for preset validation, the startup check, load and unload transitions, scheduling decisions (admit, wait, refuse, and why), timeouts, request lifecycle summaries, and reconciliation. Each log line is a typed event: code emits event objects rather than writing messages, and each event's class sets its level. Events carry the ID of the broker request that caused them, and its client, including events from background loads. `MODEL_BROKER_LOG_LEVEL` sets the detail; `DEBUG` includes every router call and every observation behind a scheduling decision. It never logs API keys, Authorization headers, prompts, completions, or streamed output.

## Privileged host facts

The broker runs unprivileged and is never given a Docker socket, root account, or permission to execute host commands. Host facts come from `model-broker-host-facts.py`, a small root-owned systemd service in this repository. It is an observation helper, not another broker: it makes no scheduling decisions, starts no applications, and has no model API.

It exposes one Unix-domain socket, whose directory (not the socket file, which is replaced whenever the helper restarts) is mounted only into the broker container, and accepts a fixed, versioned request set:

| Request | Returned facts |
| --- | --- |
| `inventory` | CUDA-to-kernel-device mapping, PCIe link facts including the bottleneck on each GPU's path to the CPU, CPU, RAM, and NUMA inventory. |
| `utilisation` | CPU use, host RAM use and availability, and GPU utilisation and memory use. |

It reads only procfs, sysfs, and the NVIDIA management library, and returns structured values, never command output. The socket is root-owned and restricted to the broker's dedicated group; the helper also checks each connecting peer's credentials.

The helper is implemented, but the broker does not consume it yet; see Deferred work.

## Testing

Unit tests run the real scheduler, adapters, and application against an in-memory fake router and fake Strata, with short real timings, and cover the scheduling rules, failures, proxying, authentication, and logging. The service-control helper is tested with a stub `docker` that records its arguments and environment, and the broker's client against the real helper server; an opt-in systemd container test checks a clean install. Both were also run once against real Docker Compose with a throwaway project, which is not repeatable in CI.

Acceptance tests prove the broker, over HTTP, against a real llama.cpp router on the GPUs, checking the router's own view of what is loaded and working. The router image is built in this repository from a pinned llama.cpp commit (kept at the commit the deployment runs), so the tests share no build files or configuration with server-containers. The tests take the preset and the models directory from the environment, choose the smallest suitable models from the preset, and run a fresh broker process per test. They are opt-in and need both GPUs idle.

## Deferred work

### Router configuration verification

Beyond the startup `GET /props` check, the broker will eventually verify the router's configuration when it attaches and periodically afterwards, before admitting new work. The profile includes at least: autoload disabled; no automatic idle sleeping, eviction, or other cleanup that can discard a loaded worker; an adequate `models-max`; no router-wide placement settings that override the preset; and the expected model-management API and catalogue. A router that cannot be verified is unhealthy for new scheduling work; the broker does not fall back to router-controlled autoload or eviction. Until then, deployment review enforces the rest of the profile; the acceptance tests prove the broker against a router with this profile, but they start their own router rather than checking the deployment's. A `sleeping` status is unexpected under this profile and marks the router unhealthy until explained.

### Persistent KV-cache sessions

A future orchestrator owns the real session: its messages, tool calls, workflow state, and decisions about which model to use. The broker may provide a persistent KV cache as an optional acceleration. A KV cache is the model's computed attention state for a token prefix; it can avoid reprocessing an unchanged system prompt and conversation history. It is not a portable conversation format and cannot replace the orchestrator's record. It does not keep model weights loaded and does not prevent a model being unloaded.

The cache is valid only for the exact variant and runtime configuration. It must not be restored for another model, quantisation, tokenizer, context or KV-cache setting, or materially changed preset, and a llama.cpp upgrade may invalidate old files. The broker therefore derives a cache key from an opaque orchestrator session ID and a versioned variant-configuration fingerprint, and does not log the raw session ID.

llama.cpp can save and restore a worker slot's prompt cache when the model server has a `--slot-save-path`. The llama container needs a persistent, writable cache volume; the broker needs no filesystem access to it:

```text
POST /slots/{slot}?action=save     {"filename": "<broker-generated name>"}
POST /slots/{slot}?action=restore  {"filename": "<broker-generated name>"}
```

The broker owns slot allocation and generated filenames; no client can select a slot or pass a path. Cache files are prompt-derived data, so the deployment applies protected storage, expiry, and a size quota. For a cache-aware request, identified by a broker-specific session header, the broker:

1. ensures the variant is loaded and reserves an idle slot;
2. restores a compatible snapshot for that session, if one exists;
3. forwards the complete conversation with prompt caching enabled; and
4. after the response completes, saves the slot snapshot and releases the slot.

This needs an end-to-end acceptance test against the pinned router and unified preset: save, restore, unload and reload, router restart, and proof that the next request reuses cached tokens. Without a compatible snapshot the broker sends the full conversation and accepts the normal prompt-processing cost.

### Other deferred work

- A measured CPU, RAM, and VRAM capacity policy, including whether mixed MoE expert-offload and CPU-weight workloads are worthwhile.
- Broker integration of the host-facts helper, first for observation only.
- Additional OpenAI endpoints, each added deliberately as described under Public API.
- Coordination with ComfyUI, Ollama, or other GPU users, each added as a backend like Strata.
- Benchmarking and, only if justified, reintroducing a maintained MoE fork for a named variant.
