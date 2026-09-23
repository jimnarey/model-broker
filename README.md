# Model broker scaffolding

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
