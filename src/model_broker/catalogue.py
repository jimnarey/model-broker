"""Parse the read-only llama.cpp preset into validated broker variants."""

from __future__ import annotations

import configparser
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

CUDA_DEVICE = re.compile(r"CUDA[0-9]+$")
CUDA_SUFFIX = re.compile(r"--(cuda[0-9]+(?:-cuda[0-9]+)*)$")
CPU_SUFFIX = "--cpu"
BASE_CLAIMS = ("CPU", "RAM:weights")


class CatalogueError(ValueError):
    """A preset cannot safely become the broker's model catalogue."""


@dataclass(frozen=True, slots=True)
class Variant:
    """One validated preset section and the resources it requires while loaded."""

    id: str
    model_path: str
    settings: Mapping[str, str]
    claims: tuple[str, ...]
    resource_class: str


def device_names(variant_id: str) -> tuple[str, ...]:
    """Return the CUDA devices encoded in a public model ID, or none for a CPU variant."""
    if variant_id.endswith(CPU_SUFFIX):
        return ()
    match = CUDA_SUFFIX.search(variant_id)
    if match is None:
        raise CatalogueError(
            f"preset section {variant_id!r} must end in --cpu or one or more --cuda<N> suffixes"
        )
    names = tuple(f"CUDA{item[4:]}" for item in match.group(1).split("-"))
    if len(set(names)) != len(names):
        raise CatalogueError(f"preset section {variant_id!r} repeats a CUDA device in its suffix")
    return names


def configured_devices(variant_id: str, settings: Mapping[str, str]) -> tuple[str, ...]:
    """Read and validate the exact comma-separated device setting for one variant."""
    value = settings.get("device")
    if value is None:
        raise CatalogueError(f"preset section {variant_id!r} has no device setting")
    if value == "none":
        return ()
    names = tuple(part.strip() for part in value.split(","))
    if not names or any(CUDA_DEVICE.fullmatch(name) is None for name in names):
        raise CatalogueError(
            f"preset section {variant_id!r} has invalid device setting {value!r}; "
            "use none or comma-separated CUDA<N> names"
        )
    if len(set(names)) != len(names):
        raise CatalogueError(f"preset section {variant_id!r} repeats a CUDA device")
    return names


def integer_setting(
    variant_id: str, settings: Mapping[str, str], key: str, default: int | None = None
) -> int:
    """Read one integer preset setting, reporting a useful error for malformed values."""
    text = settings.get(key)
    if text is None:
        if default is not None:
            return default
        raise CatalogueError(f"preset section {variant_id!r} has no {key} setting")
    try:
        return int(text)
    except ValueError as error:
        raise CatalogueError(
            f"preset section {variant_id!r} has invalid {key} value {text!r}"
        ) from error


def validate_integer_settings(variant_id: str, settings: Mapping[str, str]) -> int:
    """Validate numeric scheduling inputs and return the model's CPU expert-offload count."""
    for key in ("threads", "threads-batch"):
        if key in settings and integer_setting(variant_id, settings, key) < 0:
            raise CatalogueError(f"preset section {variant_id!r} has negative {key}")
    cpu_moe = integer_setting(variant_id, settings, "n-cpu-moe", default=0)
    if cpu_moe < 0:
        raise CatalogueError(f"preset section {variant_id!r} has negative n-cpu-moe")
    return cpu_moe


def variant(variant_id: str, settings: Mapping[str, str]) -> Variant:
    """Validate one effective preset section and return its immutable broker record."""
    model_path = settings.get("model")
    if not model_path:
        raise CatalogueError(f"preset section {variant_id!r} has no model setting")
    suffix_devices = device_names(variant_id)
    actual_devices = configured_devices(variant_id, settings)
    if actual_devices != suffix_devices:
        raise CatalogueError(
            f"preset section {variant_id!r} suffix requires {suffix_devices or ('none',)}, "
            f"but device is {actual_devices or ('none',)}"
        )

    gpu_layers = integer_setting(variant_id, settings, "n-gpu-layers")
    if not suffix_devices and gpu_layers != 0:
        raise CatalogueError(f"CPU preset section {variant_id!r} must set n-gpu-layers = 0")
    if suffix_devices and gpu_layers == 0:
        raise CatalogueError(f"GPU preset section {variant_id!r} must set non-zero n-gpu-layers")

    cpu_moe = validate_integer_settings(variant_id, settings)
    resource_class = (
        "moe_offload" if cpu_moe else "cpu_weights" if not suffix_devices else "gpu_resident"
    )
    claims = BASE_CLAIMS + tuple(f"GPU{device[4:]}" for device in suffix_devices)
    return Variant(
        id=variant_id,
        model_path=model_path,
        settings=MappingProxyType(dict(settings)),
        claims=claims,
        resource_class=resource_class,
    )


def parse_catalogue(text: str) -> dict[str, Variant]:
    """Parse a llama.cpp preset, materialise ``[*]`` defaults, and validate every variant."""
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error as error:
        raise CatalogueError(f"invalid preset: {error}") from error
    if parser.defaults():
        raise CatalogueError("[DEFAULT] is not supported; use [*] for shared llama settings")

    defaults = dict(parser.items("*", raw=True)) if parser.has_section("*") else {}
    variants = {
        section: variant(section, defaults | dict(parser.items(section, raw=True)))
        for section in parser.sections()
        if section != "*"
    }
    if not variants:
        raise CatalogueError("preset has no model sections")
    return variants


def load_catalogue(path: Path) -> dict[str, Variant]:
    """Read a mounted preset file and turn it into a validated catalogue."""
    try:
        return parse_catalogue(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CatalogueError(f"could not read preset {path}: {error}") from error


def managed_resources(variants: Mapping[str, Variant]) -> frozenset[str]:
    """Return the CPU, RAM, and GPU resource names claimed by a parsed catalogue."""
    return frozenset(claim for item in variants.values() for claim in item.claims)
