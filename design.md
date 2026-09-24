# Model broker design

## Scope and decisions

The broker is a standalone Python service that exposes an authenticated, OpenAI-compatible API. It schedules model *variants*, asks one already-running upstream llama.cpp router to load and unload them, and proxies inference.

The router runs in Docker as the `llama-cpp` Compose service. Docker starts it normally and keeps it running; the broker never calls Docker, Compose, systemd, or a host-side supervisor. There is no service-per-GPU-layout lifecycle in this design.

The deployment in the server-containers repository supplies the unified llama.cpp preset. The broker only needs a readable path to that file; it does not need that repository, Docker Compose files, model directories, or GPU devices. It is deliberately based on upstream llama.cpp. Existing experimental MoE forks remain available for benchmarking, but are not on the normal broker path. In particular, Flash Next initially uses its upstream preset; a fork is not selected merely because it has a potentially better benchmark result.

The initial implementation schedules only llama.cpp. It is the first compute
adapter in the broker. Future adapters may manage ComfyUI, Ollama, or another
significant-compute application, but they extend the same single broker rather
than creating another broker service.

## Compute adapters

An adapter connects the broker's scheduling policy to one application. The
first adapter, `LlamaCppAdapter`, knows how to inspect the unified preset,
query the llama.cpp router, load and unload a variant, and proxy supported
inference requests. The broker owns global resource accounting, request
lifetimes, authentication, and public API policy; an adapter does not make
independent GPU scheduling decisions.

The adapter boundary will be deliberately small. It will use either an
abstract base class or a Python typing protocol; that choice is still to be
made. It covers application integrations only, such as discovering configured
workloads, reporting their state, loading or stopping a workload, and
forwarding a supported request. For example, a future ComfyUI adapter can
expose workflow execution without being forced to pretend it has
llama.cpp-style model loading.

The broker calls an adapter through that boundary and keeps the resulting
resource claims in one central scheduler. New adapters are registered during
application startup from explicit configuration; they are not plugins uploaded
through the public API.

## Standalone configuration

The model-broker repository contains its own container image, application configuration, and `.env.example`. It has no build-time dependency on server-containers. A deployment provides its environment through Docker Compose, systemd, Kubernetes, or a normal `.env` file.

Only these settings are required to start:

```dotenv
# Address reachable from the broker container or host process.
MODEL_BROKER_LLAMA_URL=http://llama-cpp:8080

# Read-only path inside the broker container or host process.
MODEL_BROKER_PRESET_PATH=/config/models-preset.ini
```

The server-containers Compose service will mount the same unified preset into
the broker at the configured path and set `MODEL_BROKER_LLAMA_URL` to the
llama service's Compose name. A non-Docker deployment can instead use an IP or
DNS name and a local mounted copy of the preset.

Other settings have safe development defaults and should be documented in
`.env.example`: listen address and port, API/admin key sources, load/unload
timeouts, idle-unload timeout, bounded request wait, CPU concurrency budget,
and log level. Secrets must be supplied through the deployment environment or
a secrets mount, never committed in `.env`.

## Unified llama.cpp contract

The `llama-cpp` container can see both CUDA devices. The router starts with the unified preset and `--no-models-autoload`. It does not receive router-wide placement arguments such as `--device`, `--split-mode`, `--main-gpu`, `--tensor-split`, or `--n-gpu-layers`: such arguments would override the placement encoded in an individual preset. Router-driven idle sleeping or any other automatic model discard is disabled: the broker, not the router, decides when a loaded model leaves memory.

The preset contains all effective settings for each model section. This is important because a model section is self-contained rather than relying on router-wide defaults. CPU-only variants use `device = none` and `n-gpu-layers = 0`; GPU variants declare their CUDA device(s), GPU layer count, and any tensor split themselves.

The router has a small `models-max` limit (currently three) as a process-count backstop. It is not a resource scheduler: it neither understands which GPU a model occupies nor resolves an out-of-memory conflict. In normal operation the broker reserves this process-count capacity itself and explicitly unloads its chosen worker before it calls load, so the router's LRU-style fallback is never asked to choose a victim.

At deployment time, verify the relation between `CUDA0`/`CUDA1` and physical cards with `llama-server --list-devices` in the image. The broker configuration uses those llama.cpp device identifiers, not a guessed PCI bus order.

## Preset-derived catalogue

`models-preset.ini` is the sole model catalogue. It is mounted read-only into both the router and the broker. The broker does not scan model directories, maintain a JSON catalogue, or discover models by briefly starting alternative services.

On startup, the broker parses and validates every preset section and builds immutable `Variant` dataclasses in memory. It materialises a `[*]` defaults section if a future preset uses one; the present unified file has already materialised its effective values. A variant contains at least:

```python
@dataclass(frozen=True)
class Variant:
    id: str                     # INI section / public model ID
    model_path: str
    settings: Mapping[str, str] # effective llama.cpp settings
    claims: tuple[ResourceClaim, ...]
    resource_class: str         # gpu_resident, moe_offload, or cpu_weights
```

The public model ID is exactly the preset section name, for example `Qwen3.8-Flash-Next-UD-Q3_K_XL--cuda1`. The final suffix is an intentional, stable part of the ID, not a display label:

| Suffix | Required preset placement | Broker resource claim |
| --- | --- | --- |
| `--cuda0` | `device = CUDA0`, normally `split-mode = none` | `{GPU0}` |
| `--cuda1` | `device = CUDA1`, normally `split-mode = none` | `{GPU1}` |
| `--cuda0-cuda1` | `device = CUDA0,CUDA1` | `{GPU0, GPU1}` |
| `--cpu` | `device = none`, `n-gpu-layers = 0` | no GPU |

The suffix table describes the GPU part of a claim. Every variant also has CPU
and RAM-for-weights claims, derived from its preset settings.

Startup validation rejects a malformed or ambiguous preset rather than silently scheduling it. It verifies that every section has a model and a recognised suffix, that its `device` agrees with that suffix, and that CPU variants have zero GPU layers. It derives CPU and RAM-for-weights demand classes from `threads`, `threads-batch`, `n-gpu-layers`, and `n-cpu-moe`.

## Managed resources and hardware interfaces

The broker builds its managed-resource collection from the parsed preset. It
does not require a hand-maintained list of GPUs. The collection always contains
`CPU` and `RAM:weights`; it adds one GPU resource for every CUDA identifier
used by a preset, for example `GPU0` and `GPU1`. A preset with a new
`CUDA2` variant therefore creates a `GPU2` resource when it is validated.

Each variant produces explicit, typed claims:

| Preset evidence | Derived claim |
| --- | --- |
| `device = CUDA0[,CUDA1...]` | The listed GPU resources. |
| `threads` and `threads-batch` | CPU compute demand. |
| `device = none` / CPU-resident layers | `RAM:weights` demand for CPU-resident weights. |
| `n-cpu-moe > 0` | `RAM:weights` demand for expert offload, plus CPU compute demand. |

This makes CPU and RAM part of scheduling from the start, even where the
broker cannot yet state an exact byte or core requirement. The initial policy
is deliberately binary: a CPU-weights model and an MoE model with CPU expert
offload both claim `RAM:weights` exclusively. They therefore do not run
together. Fully GPU-resident variants still record their CPU/RAM demand, but
the initial policy allows them to coexist when their GPU claims do not
overlap.

The claim format must support a future capacity policy without changing the
public model IDs or the llama adapter boundary. It can later add measured RAM weight
bytes, CPU thread budgets, memory bandwidth, and a performance cost for
coexistence. That permits a deliberate choice such as an MoE on GPU1 with
expert weights in RAM while a small CPU model also uses RAM: slower, but not
automatically impossible. Such combinations remain disabled until measurements
define safe limits and acceptable performance.

The design reserves room for optional hardware-interface facts. When enabled
in a later deployment, the broker can map llama.cpp CUDA identifiers to kernel
devices and obtain the facts needed here: a GPU's PCI/PCIe interface, negotiated
link generation, and link width. It may also obtain reliable RAM/NUMA interface
facts. Unknown means unknown; the broker must not infer a RAM-interface
characteristic that the host does not expose.

Hardware-interface observation is not implemented or collected in the initial
broker, and it has no scheduling effect yet. It is ordinary broker
functionality, not a compute adapter and not part of the llama adapter
interface.

## Privileged host facts

The broker runs unprivileged and is never given a Docker socket, root account,
or permission to execute arbitrary host commands. When the deployment needs
host facts, it uses a small root-owned `model-broker-host-facts` systemd
service. This is an observation helper, not another broker: it makes no
scheduling decisions, starts no applications, and has no model API.

The service exposes one Unix-domain socket, mounted only into the broker
container. It accepts a small fixed, versioned request set:

| Request | Returned facts |
| --- | --- |
| `inventory` | CUDA-to-kernel-device mapping and available hardware-interface facts. |
| `utilisation` | CPU use, host RAM use/availability, and GPU utilisation and memory use. |

The helper reads only the operating system's read-only information sources:
procfs/sysfs for CPU, RAM, NUMA, and PCIe facts; and the NVIDIA management
interface for GPU identity, utilisation, and memory use. It returns structured
values, never an arbitrary command's output. The socket is root-owned and
restricted to the broker's dedicated group; the helper verifies the connecting
peer and rejects all other operations. It has no write operations and no
ability to change kernel, GPU, Docker, or llama state.

`MODEL_BROKER_HOST_FACTS_SOCKET` is optional. Without it, the broker remains
fully functional with its preset-derived resource inventory and llama activity
monitoring; it simply reports host-interface and host-utilisation data as
unavailable. When enabled, the broker samples the helper at a bounded interval
and caches the latest timestamped facts. Sampling is for management visibility
and analytics initially, not an automatic scheduling input.

Changing the preset is a controlled configuration deployment. An admin reload parses and validates the new file first. It is permitted only when affected variants have no active leases; it refreshes the router's model list (using the router's documented reload mechanism) and atomically swaps the in-memory registry. A broker restart performs the same parse/validate operation.

## Router API contract

The router is a private backend. The broker reaches it on the Compose network as `http://llama-cpp:8080`; clients do not call it directly. Once clients have migrated, the router should have no published host/LAN port. This makes the broker's scheduling and lease accounting authoritative.

The broker uses the router's model-management endpoints:

```text
GET  /models
POST /models/load       {"model": "<preset-section-id>"}
POST /models/unload     {"model": "<preset-section-id>"}
```

After a load or unload it polls `GET /models` until the desired state is reported, or until a bounded timeout. It then proxies the normal OpenAI-compatible request, initially `POST /v1/chat/completions`, preserving streaming responses. The upstream request's `model` field is the selected variant ID.

`--no-models-autoload` is intentional. A direct request for an unloaded model returns the router's “model is not loaded” error rather than starting a worker behind the broker's back. The broker is the only component that calls load and unload.

### Deferred router configuration verification

Eventually, when `LlamaCppAdapter` attaches to a router and periodically while it remains attached, it will verify that the router has an acceptable configuration before admitting new inference work. This protects the broker's resource and residency decisions from a changed deployment or an incompatible router version.

The verification profile includes at least:

- autoload is disabled for all paths the broker uses;
- automatic idle sleeping, autonomous eviction, or another router cleanup rule cannot discard a broker-resident worker;
- `models-max` is configured as an agreed process-count backstop and the broker can reserve its slots before loading;
- router-wide placement settings do not override the placement encoded in each preset; and
- the router exposes the expected model-management API and preset catalogue.

The exact verification mechanism depends on the pinned llama.cpp version and its safe configuration/status APIs. A router that cannot be verified, or whose observed configuration is unacceptable, is unhealthy for new scheduling work; the broker must not silently fall back to router-controlled autoload or eviction. Until this feature is implemented, deployment review and the pinned-router acceptance test enforce the same profile.

## Optional persistent KV-cache sessions

The future orchestrator owns the real session: its messages, tool calls,
intermediate results, workflow state, and decisions about which model to use.
The broker may provide a persistent KV-cache as an optional acceleration. A KV
cache is the model's computed attention state for a token prefix; it can avoid
reprocessing an unchanged system prompt and conversation history. It is not a
portable conversation format and cannot replace the orchestrator's record.

Model residency is separate from a session. The broker may hold a bounded residency reservation for a model, but it stores no conversation content, tool state, or model-selection policy. An orchestrator may create, renew, and release such a reservation using an opaque identifier; ordinary API clients receive only the broker's default warm-residency treatment. The broker does not infer a conversation lifetime from that identifier and does not log it as a raw session identifier.

A saved KV cache is also separate from residency. It can reduce prompt processing after a model has been unloaded and loaded again, but it does not keep model weights in VRAM and does not prevent replacement.

The cache is valid only for the exact compatible model variant and runtime
configuration. In particular, it must not be restored for another model,
quantisation, tokenizer, context/KV-cache setting, or materially changed
preset. A deployment or llama.cpp upgrade may also invalidate old cache files.
The broker therefore derives a cache key from an opaque orchestrator session ID
and a versioned variant-configuration fingerprint.

llama.cpp supports saving and restoring an individual worker slot's prompt
cache when the model server has a configured `--slot-save-path`. The llama
container needs a persistent, writable cache volume; the broker needs no direct
filesystem access to it. The adapter requests the documented internal APIs:

```text
POST /slots/{slot}?action=save     {"filename": "<broker-generated name>"}
POST /slots/{slot}?action=restore  {"filename": "<broker-generated name>"}
```

The broker owns slot allocation and generated cache filenames. Neither the
orchestrator nor an ordinary API client can select a slot or pass a filesystem
path. Cache files are prompt-derived data, so the deployment applies protected
storage, expiry, and a size quota.

For a cache-aware request, the orchestrator provides a stable opaque session ID
to the broker, for example in a broker-specific request header. The
`LlamaCppAdapter` then:

1. Ensures that the requested variant is loaded and reserves an idle slot.
2. Restores a compatible snapshot for that session when one exists.
3. Forwards the complete canonical conversation with prompt caching enabled.
4. After the response is complete, saves the slot snapshot and releases the
   slot for another request.

This allows a later request to reload the same variant and recover useful
prefix state after it was displaced by another model. It does not allow a
cache to follow a conversation from one model to another.

This feature is deliberately not enabled in the initial deployment. It needs
an end-to-end acceptance test against the pinned router image and unified
preset, including save, restore, model unload/reload, router restart, and
verification that the next request actually reuses cached tokens. The broker
must continue to work correctly when the test fails or no compatible snapshot
exists: it simply sends the full canonical conversation and accepts the normal
prefill cost.

## Public API and server

The broker, rather than llama.cpp, serves the public OpenAI-compatible API.
The Python application is an ASGI application, initially implemented with
FastAPI. Uvicorn is simply the ASGI web server: it listens for HTTP
connections, turns them into ASGI requests, and streams responses back. API
implementation, authentication, scheduling, model loading, and proxying remain
application code; Uvicorn does not perform any of those jobs.

The supported, broker-owned endpoints are registered explicitly:

| Endpoint | Behaviour |
| --- | --- |
| `GET /health` | Returns broker health and whether its router observation is current; intended for service health checks. |
| `GET /metrics` | Exposes broker and collected llama activity in Prometheus format; restricted to the monitoring network. |
| `GET /v1/models` | Returns the validated preset variants in OpenAI's `object: "list"` / `object: "model"` shape. |
| `GET /v1/models/{model}` | Returns one variant or an OpenAI-style not-found error. |
| `POST /v1/chat/completions` | Validates that `model` is a known variant, schedules it, loads it if necessary, and proxies the request and any SSE stream. |
| `POST /admin/reload` | Authenticated administration endpoint that reloads the preset under the safety rules below. |

The broker also obtains the router's OpenAPI document at startup. For every
router operation not explicitly owned above, it registers a corresponding
route from that schema. Initially those generated routes return a consistent
OpenAI-style `501 not_implemented` response; they do not silently proxy a
request that might use an unloaded model without scheduling it. This gives
clients and the broker's generated `/openapi.json` a visible
surface from the beginning.

Initially, the generated OpenAPI entries preserve the path, method, operation ID, summary,
and description only. They do not yet reproduce router parameters, request bodies, response
schemas, security requirements, tags, or other metadata. This is a temporary limitation: the
full router contract must be copied before the broker OpenAPI document is treated as suitable
input for generated clients.

Adding support for another endpoint is a deliberate override: implement and
test a broker-owned handler, register it in place of the generated placeholder,
then proxy it only after extracting and scheduling any model reference it
contains. Endpoints which have no model effect may later become straightforward
pass-through handlers. If the router's schema cannot be fetched at startup,
the explicit endpoints still start; generated routes are omitted and the
failure is logged.

## Scheduling, residency, and request lifecycle

There is one broker service. It is the central authority for scheduling
significant compute on this host. It keeps track of loaded models, router
process-count capacity, and resource claims in its own memory. The design does
not include broker replicas.

llama.cpp can host a model and route a request to it, but it cannot make this
host's placement decision. Its model count is global rather than GPU-aware: it
cannot know that a GPU1 model and a GPU0 model can remain loaded together, or
that a request for another GPU1 model must make only the GPU1 worker a possible
victim. Its request order and LRU behaviour are therefore not a scheduling
policy for this deployment. The broker owns the resident set and makes every
normal load and unload decision explicitly.

The broker distinguishes three independent things:

- An **execution lease** exists only while an inference request or stream is active. Its worker is never interrupted or selected for replacement.
- A **residency reservation** keeps an otherwise idle worker loaded. It is a resource-management record, not a session. An orchestrator may own a protected reservation; a normal request creates an unprotected warm reservation.
- A **resource claim** reserves the worker's GPU, CPU, RAM, and router process-count capacity while it is loaded, including while it is warm but idle.

Warm residency is governed by configurable, explainable rules, not just one
idle timeout. A deployment may consider time since use, measured load cost,
recent demand, reservation priority, queued work, resource pressure, and other
available operational data. The initial rule set may be time-based, but the
broker API and state model must not assume that time is the only input.

For each request, the broker:

1. Authenticates the caller and resolves its requested preset ID to a validated `Variant`.
2. Takes the scheduling lock and reconciles its state with `GET /models` when necessary.
3. Preserves every loaded worker whose claims do not conflict with the requested variant. It never removes an unrelated GPU0 worker merely to load a GPU1 worker.
4. If the variant is not loaded, considers only idle, unprotected workers whose claims conflict with the request or whose router process-count slot is required. It selects an explicit victim according to the residency rules, unloads it, and waits for its unloaded state.
5. If no eligible victim can make sufficient capacity, waits for the bounded request wait. It then returns `503 resource_busy` with `Retry-After`; it does not delegate the choice to router LRU.
6. Reserves the requested variant's resource claims and router process-count slot, calls `/models/load`, and waits for it to be ready.
7. Creates an execution lease, releases the scheduling lock, and proxies the request or streaming response.
8. Releases the execution lease only after the response stream ends, is cancelled, or fails. It then creates or refreshes the applicable warm reservation; later cleanup follows the configured residency rules and releases the resource claim only when the worker is actually unloaded.

An in-progress request is never interrupted to make room for another model.
Protected orchestrator reservations remain loaded until their expiry or
explicit release. An unprotected warm worker may be replaced only when the
residency rules permit it. The broker serialises conflicting replacement
decisions and applies a bounded queue, so alternating requests cannot race in
the router. The exact queueing and warm-residency rules may change with
measurements, but their configured result—not llama.cpp's incidental request
order—must decide whether to wait, replace a worker, or return `503`.

This yields the expected basic policy:

| Active variant | Requested variant | Result |
| --- | --- | --- |
| `--cuda1` | `--cuda0` | May run concurrently. |
| `--cuda0` | `--cuda1` | May run concurrently. |
| warm `--cuda1` and warm `--cuda0` | another `--cuda1` | Preserve the GPU0 worker; wait, reject, or explicitly replace the eligible GPU1 worker under the residency rules. |
| `--cuda1` | `--cuda0-cuda1` | Wait/reject while active; otherwise unload the GPU1 worker, then load dual-GPU. |
| `--cuda0` or `--cuda1` | `--cuda0-cuda1` | Same: dual-GPU requires both cards. |
| `--cuda0-cuda1` | any GPU variant | Wait/reject while dual-GPU lease is active. |

Consequently, Flash Next on GPU1 can run with a less-than-16-GB GPU0 variant, and two less-than-16-GB variants can remain loaded together. An MoE variant with `n-cpu-moe` still has its declared GPU claim, plus its CPU class; the PCIe characteristics are accounted for by selecting the right preset ID, not by moving a loaded model at runtime.

CPU-only variants make no GPU claim. Their initial policy is conservative: variants with `n-cpu-moe > 0` are `moe_cpu` and consume a CPU-heavy scheduling slot. They do not run beside another CPU-heavy inference until measurements establish a safe capacity model. Pure CPU variants use the parsed thread settings and a small configurable CPU concurrency budget. This is a policy decision, not an assertion that llama.cpp itself reserves host CPU resources.

More specifically, the initial `RAM:weights` rule treats CPU-weight models
and MoE expert-offload models as mutually exclusive. This is intentionally
more conservative than the hardware: the two may fit and run at the same
time, but they compete for RAM capacity, memory bandwidth, and CPU work. The
future capacity policy described above may admit a measured combination rather
than treating it as a binary conflict.

## Llama activity monitoring and analytics

`LlamaCppAdapter` monitors the router independently of client requests. For
management, it polls `GET /models` and, where available, consumes
`GET /models/sse`. This records each configured model's status
(unloaded, loading, loaded, sleeping, or failed), observes completion of
broker-requested loads and unloads, and reconciles the broker's resource
claims after a router or broker restart. A sleeping status is unexpected under
the accepted router profile and marks the router unhealthy for new scheduling
work until it is explained or corrected.

The unified llama service enables its metrics endpoint. The adapter scrapes
the router's model-scoped metrics as an input to operations and analytics,
alongside information already known to the broker: queue time, load/unload
duration, request/stream duration, cancellation, error outcome, and token or
timing data returned by a request. The broker exposes its own aggregate health
and metrics endpoint for the deployment's monitoring system; it does not
expose the router management endpoints to ordinary clients.

When the optional host-facts socket is configured, the same monitoring loop
also records timestamped CPU, RAM, and GPU utilisation. It keeps those facts
separate from the router's model status and from the scheduler's resource
claims: utilisation describes what is happening, while a claim describes what
the broker has reserved.

Analytics are observational. They never trigger a scheduling action on their
own in the initial design, and they never include prompts, completions, API
keys, Authorization headers, or raw session identifiers. A later policy can
use retained measurements to set resource capacities and make informed
trade-offs, such as whether a mixed MoE/CPU workload is worthwhile.

## Restart and failure handling

On broker startup, it reads the preset and queries `GET /models`. It rebuilds loaded-worker/resource claims from the router's reported loaded variants before accepting new work. Request streams cannot survive a broker restart, but already-loaded router workers are not needlessly discarded.

If loading fails or never reaches ready state, the broker removes the pending reservation, records the router state and safe error details, and returns a backend failure to the caller. If an unload fails, its resource claim remains reserved until reconciliation proves it is gone. The broker does not request an unsafe router LRU eviction and does not try to migrate a loaded model to a different GPU or to the CPU.

## Networking, authentication, and logging

The broker is the only externally reachable API service. It authenticates clients with broker API keys; administrative operations such as preset reload use separate admin credentials. The broker and router share only their internal Compose network. By default, the deployment gives the broker only the router URL and a read-only preset mount, plus normal service configuration and secrets. The optional host-facts Unix socket is the only additional host integration; it is read-only in effect and explicitly configured.

The broker emits structured JSON logs for preset validation, load/unload transitions, resource decisions, timeouts, request lifecycle summaries, and reconciliation. It must not log API keys, Authorization headers, prompts, completions, or streamed model output.

## Deferred work

- Benchmarking and, only if justified, selectively reintroducing a maintained MoE fork for a named variant.
- A measured CPU/RAM/VRAM capacity model, including controlled mixed MoE
  expert-offload and CPU-weight workloads.
- Optional hardware-interface and host-utilisation observation through the
  host-facts service.
- Coordination with ComfyUI, Ollama, or other external GPU users.
- Additional OpenAI endpoints such as Responses API and non-llama backends.
- Router-configuration verification and a pinned-router acceptance suite. It
  must prove that autoload and autonomous discard are disabled, that a GPU0
  worker remains loaded while a GPU1 worker is replaced, that active work is
  not interrupted, and that alternating conflicting requests follow the broker
  residency policy rather than router LRU.
