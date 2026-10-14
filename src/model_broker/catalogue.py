"""Parse the read-only llama.cpp preset into validated broker variants."""

from __future__ import annotations

import configparser
import itertools
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

PRESET_VERSION = "1"
CUDA_NUMBER = r"(?:0|[1-9][0-9]*)"
CUDA_DEVICE = re.compile(rf"CUDA{CUDA_NUMBER}")
CUDA_SUFFIX = re.compile(rf"--(cuda{CUDA_NUMBER}(?:-cuda{CUDA_NUMBER})*)$")
CPU_SUFFIX = "--cpu"
SERVICE_KEYS = frozenset({"service", "device", "exclusive"})
SERVICE_ENV_PREFIX = "env."
SERVICE_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]*")
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
RESOURCE_NAME = re.compile(r"[A-Za-z0-9:_-]+")
# llama.cpp accepts either spelling for the draft model's devices; it defaults to ``device``.
DRAFT_DEVICE_KEYS = ("spec-draft-device", "device-draft")
GPU_ONLY_KEYS = frozenset({"n-cpu-moe", "split-mode", "main-gpu", "tensor-split"})


class CatalogueError(ValueError):
    """A preset cannot safely become the broker's model catalogue."""


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    """A resource a loaded variant uses.

    Two exclusive claims on the same resource conflict. A non-exclusive claim records demand
    for a later capacity policy and never blocks another variant.
    """

    resource: str
    exclusive: bool


@dataclass(frozen=True, slots=True)
class Variant:
    """One validated preset section and the resources it requires while loaded."""

    id: str
    model_path: str
    settings: Mapping[str, str] = field(hash=False)
    claims: tuple[ResourceClaim, ...]
    resource_class: str
    # Which server loads and serves it: "llama" (the router), or the name of a service
    # started and stopped by restarting its container, such as "strata".
    backend: str = "llama"
    # The environment variables a restart-managed service's container is started with.
    env: Mapping[str, str] = field(default_factory=lambda: no_env(), hash=False)


def no_env() -> Mapping[str, str]:
    return MappingProxyType({})


def invalid(variant_id: str, problem: str) -> CatalogueError:
    return CatalogueError(f"preset section {variant_id!r} {problem}")


def suffix_devices(variant_id: str) -> tuple[str, ...]:
    """Return the CUDA devices named by a public model ID's suffix; none for ``--cpu``."""
    if variant_id.endswith(CPU_SUFFIX):
        return ()
    match = CUDA_SUFFIX.search(variant_id)
    if match is None:
        raise invalid(variant_id, "must end in --cpu or one or more --cuda<N> suffixes")
    names = tuple(f"CUDA{item[4:]}" for item in match.group(1).split("-"))
    if len(set(names)) != len(names):
        raise invalid(variant_id, "repeats a CUDA device in its suffix")
    return names


def setting_devices(variant_id: str, key: str, value: str | None) -> tuple[str, ...]:
    """Parse a device list setting: ``none`` or comma-separated ``CUDA<N>`` names."""
    if value is None:
        raise invalid(variant_id, f"has no {key} setting")
    if value == "none":
        return ()
    names = tuple(part.strip() for part in value.split(","))
    if any(CUDA_DEVICE.fullmatch(name) is None for name in names):
        raise invalid(
            variant_id, f"has invalid {key} {value!r}; use none or comma-separated CUDA<N> names"
        )
    if len(set(names)) != len(names):
        raise invalid(variant_id, f"repeats a CUDA device in {key}")
    return names


def int_setting(variant_id: str, settings: Mapping[str, str], key: str, default: int) -> int:
    text = settings.get(key)
    if text is None:
        return default
    try:
        return int(text)
    except ValueError as error:
        raise invalid(variant_id, f"has invalid {key} value {text!r}") from error


def check_threads(variant_id: str, settings: Mapping[str, str]) -> None:
    """Thread counts must be positive, or -1 for llama.cpp's automatic default."""
    for key in ("threads", "threads-batch"):
        threads = int_setting(variant_id, settings, key, default=-1)
        if threads == 0 or threads < -1:
            raise invalid(variant_id, f"has {key} = {threads}; use a positive count or -1")


def resource_class(variant_id: str, settings: Mapping[str, str], devices: tuple[str, ...]) -> str:
    """Check that the layer placement matches the claim, and classify the variant.

    A GPU variant must keep every layer on its GPUs (``all``): ``auto`` or a layer count lets
    llama.cpp put weights in host RAM, which a GPU-resident claim does not cover. Expert
    offload to RAM is only allowed explicitly, through ``n-cpu-moe``.
    """
    layers = settings.get("n-gpu-layers")
    if not devices:
        if layers != "0":
            raise invalid(
                variant_id, f"is a CPU variant and must set n-gpu-layers = 0, not {layers!r}"
            )
        gpu_only = sorted(GPU_ONLY_KEYS & settings.keys())
        if gpu_only:
            raise invalid(variant_id, f"is a CPU variant but sets GPU-only {', '.join(gpu_only)}")
        return "cpu_weights"
    if layers != "all":
        raise invalid(
            variant_id, f"is a GPU variant and must set n-gpu-layers = all, not {layers!r}"
        )
    cpu_moe = int_setting(variant_id, settings, "n-cpu-moe", default=0)
    if cpu_moe < 0:
        raise invalid(variant_id, "has negative n-cpu-moe")
    return "moe_offload" if cpu_moe else "gpu_resident"


def resource_claims(kind: str, devices: tuple[str, ...]) -> tuple[ResourceClaim, ...]:
    """Derive the initial, binary claims from the design.

    Every variant records CPU and RAM-for-weights use. Only CPU-weight and MoE expert-offload
    variants hold ``RAM:weights`` exclusively, so no two of them run together. GPU-resident
    variants only record it, so they coexist with anything that uses different GPUs. Each GPU
    is exclusive.
    """
    return (
        ResourceClaim("CPU", exclusive=False),
        ResourceClaim("RAM:weights", exclusive=kind != "gpu_resident"),
        *(ResourceClaim(f"GPU{device[4:]}", exclusive=True) for device in devices),
    )


def variant(variant_id: str, settings: Mapping[str, str]) -> Variant:
    """Validate one effective preset section and return its immutable broker record."""
    model_path = settings.get("model")
    if not model_path:
        raise invalid(variant_id, "has no model setting")
    devices = suffix_devices(variant_id)
    configured = setting_devices(variant_id, "device", settings.get("device"))
    if configured != devices:
        raise invalid(
            variant_id,
            f"suffix requires {devices or ('none',)}, but device is {configured or ('none',)}",
        )
    for key in DRAFT_DEVICE_KEYS:
        if key in settings and setting_devices(variant_id, key, settings[key]) != devices:
            raise invalid(variant_id, f"places its draft model outside its devices with {key}")
    check_threads(variant_id, settings)
    kind = resource_class(variant_id, settings, devices)
    return Variant(
        id=variant_id,
        model_path=model_path,
        settings=MappingProxyType(dict(settings)),
        claims=resource_claims(kind, devices),
        resource_class=kind,
    )


def preset_header(lines: list[str]) -> dict[str, str]:
    """Parse the ``key = value`` lines before the first section, where llama.cpp puts version."""
    header: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith((";", "#")):
            continue
        key, separator, value = text.partition("=")
        if not separator:
            raise CatalogueError(f"invalid preset line {number}: {text!r}")
        if key.strip() in header:
            raise CatalogueError(f"preset line {number} repeats top-level setting {key.strip()!r}")
        header[key.strip()] = value.strip()
    return header


def read_preset(text: str, keep_case: bool = False) -> configparser.ConfigParser:
    """Parse a preset's version header and sections, as llama.cpp's and the services' both have
    them. Option names are lowercased unless ``keep_case``, which environment variables need."""
    lines = text.splitlines()
    first_section = next(
        (number for number, line in enumerate(lines) if line.lstrip().startswith("[")),
        len(lines),
    )
    header = preset_header(lines[:first_section])
    if header != {"version": PRESET_VERSION}:
        raise CatalogueError(
            f"preset must start with version = {PRESET_VERSION} and no other top-level "
            f"settings; found {header}"
        )
    parser = configparser.ConfigParser(interpolation=None)
    if keep_case:
        parser.optionxform = str  # type: ignore[assignment,method-assign]
    try:
        # Blank lines replace the header so that configparser reports true line numbers.
        parser.read_string("\n" * first_section + "\n".join(lines[first_section:]))
    except configparser.Error as error:
        raise CatalogueError(f"invalid preset: {error}") from error
    if parser.defaults():
        raise CatalogueError("[DEFAULT] is not supported; use [*] for shared llama settings")
    return parser


def parse_catalogue(text: str) -> dict[str, Variant]:
    """Parse a llama.cpp preset, materialise ``[*]`` defaults, and validate every variant."""
    parser = read_preset(text)
    defaults = dict(parser.items("*", raw=True)) if parser.has_section("*") else {}
    variants = {
        section: variant(section, defaults | dict(parser.items(section, raw=True)))
        for section in parser.sections()
        if section != "*"
    }
    if not variants:
        raise CatalogueError("preset has no model sections")
    return variants


def service_variant(variant_id: str, settings: Mapping[str, str]) -> Variant:
    """Validate one section of a restart-managed service's preset.

    The section names the service, the GPUs it holds (the suffix, confirmed by ``device``),
    any other resources it holds exclusively, and the environment variables its container is
    started with. Its GPUs and those resources are exclusive; CPU is recorded.
    """
    unknown = sorted(
        key
        for key in settings
        if key not in SERVICE_KEYS and not key.startswith(SERVICE_ENV_PREFIX)
    )
    if unknown:
        raise invalid(variant_id, f"has unknown setting {unknown[0]!r}")
    service = settings.get("service", "")
    if SERVICE_NAME.fullmatch(service) is None:
        raise invalid(variant_id, f"has an invalid or missing service {service!r}")
    devices = suffix_devices(variant_id)
    configured = setting_devices(variant_id, "device", settings.get("device"))
    if configured != devices:
        raise invalid(
            variant_id,
            f"suffix requires {devices or ('none',)}, but device is {configured or ('none',)}",
        )
    env = {
        key.removeprefix(SERVICE_ENV_PREFIX): value
        for key, value in settings.items()
        if key.startswith(SERVICE_ENV_PREFIX)
    }
    bad = sorted(name for name in env if ENV_NAME.fullmatch(name) is None)
    if bad:
        raise invalid(variant_id, f"has an invalid environment variable name {bad[0]!r}")
    extra = settings.get("exclusive", "").replace(",", " ").split()
    if any(RESOURCE_NAME.fullmatch(resource) is None for resource in extra):
        raise invalid(variant_id, f"has an invalid exclusive resource in {extra}")
    exclusive = {f"GPU{device[4:]}" for device in devices} | set(extra)
    claims = tuple(ResourceClaim(resource, exclusive=True) for resource in sorted(exclusive))
    if "CPU" not in exclusive:
        claims = (ResourceClaim("CPU", exclusive=False), *claims)
    return Variant(
        id=variant_id,
        model_path=f"service:{service}",
        settings=MappingProxyType(dict(settings)),
        claims=claims,
        resource_class="service",
        backend=service,
        env=MappingProxyType(env),
    )


def parse_service_catalogue(text: str) -> dict[str, Variant]:
    """Parse a preset of restart-managed services, which has no ``[*]`` defaults: each section
    is complete."""
    parser = read_preset(text, keep_case=True)
    if parser.has_section("*"):
        raise CatalogueError("[*] is not supported in a service preset; each section is complete")
    variants = {
        section: service_variant(section, dict(parser.items(section, raw=True)))
        for section in parser.sections()
    }
    if not variants:
        raise CatalogueError("service preset has no model sections")
    return variants


def load_catalogue(path: Path) -> dict[str, Variant]:
    """Read a mounted preset file and turn it into a validated catalogue."""
    try:
        return parse_catalogue(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CatalogueError(f"could not read preset {path}: {error}") from error


def load_service_catalogue(path: Path) -> dict[str, Variant]:
    """Read a mounted service preset file and turn it into a validated catalogue."""
    try:
        return parse_service_catalogue(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CatalogueError(f"could not read service preset {path}: {error}") from error


def merge_catalogues(*catalogues: Mapping[str, Variant]) -> dict[str, Variant]:
    """Combine catalogues, refusing a model ID that two of them define."""
    merged: dict[str, Variant] = {}
    for catalogue in catalogues:
        for model_id, item in catalogue.items():
            if model_id in merged:
                raise CatalogueError(f"model {model_id!r} is defined in more than one preset")
            merged[model_id] = item
    return merged


def managed_resources(variants: Mapping[str, Variant]) -> frozenset[str]:
    """Return the CPU, RAM, and GPU resource names claimed by a parsed catalogue."""
    return frozenset(claim.resource for item in variants.values() for claim in item.claims)


def max_loaded_models(variants: Mapping[str, Variant]) -> int:
    """Return the largest number of variants that can be loaded together without conflict.

    Two variants conflict when they share an exclusive resource, so this is the largest
    group of variants whose exclusive resources do not overlap. Variants with the same
    exclusive resources are interchangeable here, and there are few distinct sets (each is
    drawn from the GPUs and RAM:weights), so trying every combination of them is cheap.
    """
    resource_sets = list(
        {frozenset(c.resource for c in v.claims if c.exclusive) for v in variants.values()}
    )
    largest = 0
    for size in range(1, len(resource_sets) + 1):
        if not any(
            sum(len(group) for group in combination) == len(frozenset[str]().union(*combination))
            for combination in itertools.combinations(resource_sets, size)
        ):
            break  # any larger group would contain an overlapping group of this size
        largest = size
    return largest


def claims_conflict(first: Variant, second: Variant) -> bool:
    """Return whether both variants claim some resource exclusively."""
    exclusive = {claim.resource for claim in first.claims if claim.exclusive}
    return any(claim.exclusive and claim.resource in exclusive for claim in second.claims)
