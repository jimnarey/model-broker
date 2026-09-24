from __future__ import annotations

from pathlib import Path

import pytest

from model_broker.catalogue import (
    CatalogueError,
    ResourceClaim,
    Variant,
    claims_conflict,
    load_catalogue,
    managed_resources,
    parse_catalogue,
)

# Real sections from the unified preset; see the comment at the top of the file.
REAL_PRESET = Path(__file__).parent / "fixtures" / "models-preset.ini"
CUDA0 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda0"
CUDA1 = "qwen2.5-coder-7b-instruct-q4_k_m--cuda1"
MOE_CUDA1 = "GLM-4.5-Air-UD-Q4_K_XL--cuda1"
DUAL = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0--cuda0-cuda1"
CPU = "Qwen3.5-9B-Q4_K_M--cpu"


@pytest.fixture(scope="module")
def real() -> dict[str, Variant]:
    return load_catalogue(REAL_PRESET)


def section(name: str, **settings: str) -> str:
    """A minimal valid preset: the version header and one section with the given settings."""
    lines = [f"{key.replace('_', '-')} = {value}" for key, value in settings.items()]
    return "\n".join(["version = 1", f"[{name}]", *lines]) + "\n"


def gpu(name: str = "m--cuda0", **overrides: str) -> str:
    settings = {"model": "/models/m.gguf", "device": "CUDA0", "n_gpu_layers": "all"}
    return section(name, **(settings | overrides))


def cpu(name: str = "m--cpu", **overrides: str) -> str:
    settings = {"model": "/models/m.gguf", "device": "none", "n_gpu_layers": "0"}
    return section(name, **(settings | overrides))


# The real preset


def test_real_preset_parses_every_section(real: dict[str, Variant]) -> None:
    """The fixture, copied from the deployed unified preset, is accepted in full.

    It has the real format: a top-level ``version = 1`` before any section, and
    ``n-gpu-layers = all`` on GPU entries. An earlier version rejected both, and so rejected
    every section of the deployed file.
    """
    assert set(real) == {CUDA0, CUDA1, MOE_CUDA1, DUAL, CPU}


@pytest.mark.parametrize(
    ("variant_id", "resource_class", "gpus"),
    [
        (CUDA0, "gpu_resident", ["GPU0"]),
        (CUDA1, "gpu_resident", ["GPU1"]),
        (MOE_CUDA1, "moe_offload", ["GPU1"]),
        (DUAL, "gpu_resident", ["GPU0", "GPU1"]),
        (CPU, "cpu_weights", []),
    ],
    ids=["cuda0", "cuda1", "moe-offload", "dual-gpu", "cpu"],
)
def test_real_sections_are_classified_and_claim_their_gpus(
    real: dict[str, Variant], variant_id: str, resource_class: str, gpus: list[str]
) -> None:
    """Each section's class and GPU claims follow from its suffix, device and n-cpu-moe.

    ``moe-offload`` has ``n-cpu-moe = 46``, so its experts sit in host RAM. ``dual-gpu`` has
    an MTP draft model that follows ``device``, so it claims nothing extra.
    """
    variant = real[variant_id]
    assert variant.resource_class == resource_class
    claimed = [c.resource for c in variant.claims if c.resource.startswith("GPU")]
    assert claimed == gpus


def test_managed_resources_come_from_the_preset(real: dict[str, Variant]) -> None:
    """The resource set is derived from the preset: CPU, RAM:weights, and each GPU used."""
    assert managed_resources(real) == {"CPU", "RAM:weights", "GPU0", "GPU1"}


# Claims and conflicts


def test_claims_record_cpu_and_ram_and_reserve_gpus(real: dict[str, Variant]) -> None:
    """A GPU-resident variant records CPU and RAM use; only its GPU is exclusive.

    A CPU-weights variant holds RAM:weights exclusively instead.
    """
    assert real[CUDA0].claims == (
        ResourceClaim("CPU", exclusive=False),
        ResourceClaim("RAM:weights", exclusive=False),
        ResourceClaim("GPU0", exclusive=True),
    )
    assert real[CPU].claims == (
        ResourceClaim("CPU", exclusive=False),
        ResourceClaim("RAM:weights", exclusive=True),
    )


@pytest.mark.parametrize(
    ("first", "second", "conflict"),
    [
        (CUDA0, CUDA1, False),
        (MOE_CUDA1, CUDA0, False),
        (CUDA0, CPU, False),
        (CUDA1, MOE_CUDA1, True),
        (CUDA1, DUAL, True),
        (MOE_CUDA1, CPU, True),
    ],
    ids=[
        "gpu0-and-gpu1",
        "moe-gpu1-and-gpu0",
        "gpu-resident-and-cpu",
        "same-gpu",
        "one-gpu-and-dual",
        "moe-offload-and-cpu-weights",
    ],
)
def test_conflicts_follow_the_design_policy(
    real: dict[str, Variant], first: str, second: str, conflict: bool
) -> None:
    """Two variants conflict only when both hold some resource exclusively.

    Expected results are the design's scheduling table: different GPUs coexist (so a GPU0
    worker is kept while a GPU1 model loads); a GPU-resident model coexists with a CPU model,
    because it only records RAM use; a shared GPU conflicts, as does one GPU against dual-GPU;
    and MoE expert offload conflicts with CPU weights because both hold RAM:weights
    exclusively. The check is symmetric, so each pair is tested both ways.
    """
    assert claims_conflict(real[first], real[second]) is conflict
    assert claims_conflict(real[second], real[first]) is conflict


# Preset structure


def test_section_settings_override_star_defaults() -> None:
    """``[*]`` settings apply to every section, and a section's own value wins.

    Setup: ``[*]`` sets threads = 8 and n-gpu-layers = all; the CPU section sets
    n-gpu-layers = 0. Expect the GPU section to inherit both defaults, and the CPU section to
    keep threads = 8 but use its own 0, which is the only value a CPU variant accepts.
    """
    text = (
        "version = 1\n[*]\nthreads = 8\nn-gpu-layers = all\n"
        "[g--cuda0]\nmodel = /m.gguf\ndevice = CUDA0\n"
        "[c--cpu]\nmodel = /m.gguf\ndevice = none\nn-gpu-layers = 0\n"
    )
    variants = parse_catalogue(text)

    assert set(variants) == {"g--cuda0", "c--cpu"}
    assert variants["g--cuda0"].settings["n-gpu-layers"] == "all"
    assert variants["c--cpu"].settings["threads"] == "8"
    assert variants["c--cpu"].settings["n-gpu-layers"] == "0"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("[m--cpu]\nmodel = /m.gguf\n", "must start with version = 1"),
        ("version = 2\n[m--cpu]\n", "must start with version = 1"),
        ("version = 1\nthreads = 8\n[m--cpu]\n", "no other top-level settings"),
        ("version = 1\nnot a setting\n[m--cpu]\n", "invalid preset line 2"),
        ("version = 1\n[DEFAULT]\nmodel = /m.gguf\n", r"\[DEFAULT\] is not supported"),
        ("version = 1\n[*]\nthreads = 8\n", "no model sections"),
        ("version = 1\n[a--cpu]\n[a--cpu]\n", r"\[line +3\]"),
    ],
    ids=[
        "no-version",
        "wrong-version",
        "other-top-level-key",
        "malformed-header",
        "configparser-default",
        "no-models",
        "duplicate-section",
    ],
)
def test_malformed_preset_structure_is_rejected(text: str, error: str) -> None:
    """The file-level format is checked before any section.

    ``[DEFAULT]`` is Python-only inheritance that llama.cpp would not apply, and top-level
    keys other than version would be settings the broker cannot attribute to a variant.
    ``duplicate-section`` checks that error line numbers still match the file although the
    header is parsed separately.
    """
    with pytest.raises(CatalogueError, match=error):
        parse_catalogue(text)


# Section validation


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (section("m--cuda0", device="CUDA0", n_gpu_layers="all"), "no model setting"),
        (gpu("m"), "must end in --cpu or one or more --cuda<N>"),
        (gpu("m--CUDA0"), "must end in --cpu"),
        (gpu("m--cuda01", device="CUDA01"), "must end in --cpu"),
        (gpu("m--cuda1"), "suffix requires"),
        (gpu("m--cuda0-cuda1", device="CUDA1,CUDA0"), "suffix requires"),
        (gpu("m--cuda0-cuda0", device="CUDA0,CUDA0"), "repeats a CUDA device"),
        (gpu(device="GPU0"), "invalid device"),
        (section("m--cuda0", model="/m.gguf", n_gpu_layers="all"), "no device setting"),
        (gpu(n_gpu_layers="auto"), "must set n-gpu-layers = all"),
        (gpu(n_gpu_layers="99"), "must set n-gpu-layers = all"),
        (gpu(n_cpu_moe="-1"), "negative n-cpu-moe"),
        (gpu(n_cpu_moe="some"), "invalid n-cpu-moe"),
        (gpu(spec_draft_device="CUDA1"), "draft model outside its devices"),
        (gpu(device_draft="none"), "draft model outside its devices"),
        (cpu(n_gpu_layers="all"), "must set n-gpu-layers = 0"),
        (cpu(n_cpu_moe="4"), "GPU-only n-cpu-moe"),
        (cpu(split_mode="none", main_gpu="0"), "GPU-only main-gpu, split-mode"),
        (cpu(threads="0"), "threads = 0"),
        (cpu(threads_batch="-2"), "threads-batch = -2"),
    ],
    ids=[
        "no-model",
        "no-suffix",
        "uppercase-suffix",
        "leading-zero",
        "suffix-device-mismatch",
        "device-order-mismatch",
        "repeated-device",
        "unknown-device-name",
        "no-device",
        "gpu-layers-auto",
        "gpu-layers-count",
        "negative-n-cpu-moe",
        "non-numeric-n-cpu-moe",
        "draft-on-other-gpu",
        "draft-on-cpu",
        "cpu-with-gpu-layers",
        "cpu-with-n-cpu-moe",
        "cpu-with-gpu-placement",
        "zero-threads",
        "negative-threads",
    ],
)
def test_unsafe_sections_are_rejected(text: str, error: str) -> None:
    """A section whose placement differs from what its ID claims stops startup.

    The public ID's suffix is the claim the scheduler relies on, so the device list (and any
    draft-model device list) must name exactly those GPUs, in order. GPU variants must use
    ``n-gpu-layers = all``: ``auto`` or a count lets llama.cpp spill weights into RAM that
    the claim does not cover. ``cuda01`` is refused because it would become a GPU01 resource
    separate from GPU1.
    """
    with pytest.raises(CatalogueError, match=error):
        parse_catalogue(text)


def test_threads_accept_llama_automatic_default() -> None:
    """``threads = -1`` is llama.cpp's own default (automatic), so it is valid."""
    assert parse_catalogue(cpu(threads="-1", threads_batch="16"))["m--cpu"]


def test_variants_are_hashable(real: dict[str, Variant]) -> None:
    """Variants can be set members or dict keys, although their settings are a mapping."""
    assert len({real[CUDA0], real[CUDA1]}) == 2


def test_load_catalogue_reports_unreadable_path(tmp_path: Path) -> None:
    """A missing preset file is a CatalogueError that names the path."""
    with pytest.raises(CatalogueError, match="could not read preset"):
        load_catalogue(tmp_path / "missing.ini")
