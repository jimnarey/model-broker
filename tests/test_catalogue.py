from __future__ import annotations

from pathlib import Path

import pytest

from model_broker.catalogue import (
    CatalogueError,
    load_catalogue,
    managed_resources,
    parse_catalogue,
)

PRESET = """\
[*]
threads = 8
threads-batch = 16
n-gpu-layers = 99

[Flash Next--cuda1]
model = /models/flash-next.gguf
device = CUDA1

[gpt-oss20b--cuda0]
model = /models/gpt-oss20b.gguf
device = CUDA0
n-cpu-moe = 4

[CPU helper--cpu]
model = /models/helper.gguf
device = none
n-gpu-layers = 0
"""


def test_parse_catalogue_materialises_defaults_and_derives_claims() -> None:
    """Shared ``[*]`` settings become part of each variant before validation.

    The preset has a GPU1 model, an MoE model on GPU0, and a CPU-only model. The test checks
    that the defaults section is not exposed as a model and that every parsed model records the
    resource names needed by the future scheduler. This matters because the scheduler must keep
    the GPU0 model when loading a non-conflicting GPU1 model.
    """
    variants = parse_catalogue(PRESET)

    assert set(variants) == {"Flash Next--cuda1", "gpt-oss20b--cuda0", "CPU helper--cpu"}
    assert variants["Flash Next--cuda1"].settings["threads"] == "8"
    assert variants["Flash Next--cuda1"].claims == ("CPU", "RAM:weights", "GPU1")
    assert variants["Flash Next--cuda1"].resource_class == "gpu_resident"
    assert variants["gpt-oss20b--cuda0"].resource_class == "moe_offload"
    assert variants["CPU helper--cpu"].claims == ("CPU", "RAM:weights")
    assert variants["CPU helper--cpu"].resource_class == "cpu_weights"
    assert managed_resources(variants) == frozenset({"CPU", "RAM:weights", "GPU0", "GPU1"})


@pytest.mark.parametrize(
    ("section", "device", "expected_claims"),
    [
        ("one--cuda0", "CUDA0", ("CPU", "RAM:weights", "GPU0")),
        ("two--cuda1", "CUDA1", ("CPU", "RAM:weights", "GPU1")),
        ("both--cuda0-cuda1", "CUDA0,CUDA1", ("CPU", "RAM:weights", "GPU0", "GPU1")),
        ("future--cuda2", "CUDA2", ("CPU", "RAM:weights", "GPU2")),
    ],
)
def test_parse_catalogue_matches_cuda_suffix_to_device_setting(
    section: str, device: str, expected_claims: tuple[str, ...]
) -> None:
    """A CUDA suffix and ``device`` setting name the same GPUs in the same order.

    The catalogue supports the two current GPUs and a future CUDA2 device without maintaining a
    separate GPU list. A mismatch would let the public model ID describe one placement while
    llama.cpp loads another, which would make later scheduling unsafe.
    """
    preset = f"""\
[{section}]
model = /models/example.gguf
device = {device}
n-gpu-layers = 99
"""

    assert parse_catalogue(preset)[section].claims == expected_claims


@pytest.mark.parametrize(
    ("preset", "error"),
    [
        (
            "[missing-model--cuda0]\ndevice = CUDA0\nn-gpu-layers = 99\n",
            "no model setting",
        ),
        (
            (
                "[unknown-placement]\n"
                "model = /models/example.gguf\n"
                "device = CUDA0\n"
                "n-gpu-layers = 99\n"
            ),
            "must end",
        ),
        (
            "[wrong-gpu--cuda1]\nmodel = /models/example.gguf\ndevice = CUDA0\nn-gpu-layers = 99\n",
            "suffix requires",
        ),
        (
            (
                "[cpu-with-gpu-layers--cpu]\n"
                "model = /models/example.gguf\n"
                "device = none\n"
                "n-gpu-layers = 1\n"
            ),
            "must set n-gpu-layers = 0",
        ),
        (
            (
                "[gpu-without-layers--cuda0]\n"
                "model = /models/example.gguf\n"
                "device = CUDA0\n"
                "n-gpu-layers = 0\n"
            ),
            "must set non-zero n-gpu-layers",
        ),
        (
            (
                "[bad-number--cuda0]\n"
                "model = /models/example.gguf\n"
                "device = CUDA0\n"
                "n-gpu-layers = many\n"
            ),
            "invalid n-gpu-layers",
        ),
        (
            (
                "[duplicate-device--cuda0-cuda0]\n"
                "model = /models/example.gguf\n"
                "device = CUDA0,CUDA0\n"
                "n-gpu-layers = 99\n"
            ),
            "repeats a CUDA device",
        ),
    ],
)
def test_parse_catalogue_rejects_unsafe_or_ambiguous_model_sections(
    preset: str, error: str
) -> None:
    """Invalid placement details prevent the whole preset from becoming a catalogue.

    These failures cover a missing model, an unrecognised public ID, disagreement between the
    ID and llama.cpp placement, unsuitable GPU-layer settings, malformed numbers, and repeated
    GPUs. Startup must fail in these cases rather than leave a scheduler with a misleading model
    description.
    """
    with pytest.raises(CatalogueError, match=error):
        parse_catalogue(preset)


def test_parse_catalogue_rejects_default_section_and_empty_catalogue() -> None:
    """Only ``[*]`` may be shared, and at least one actual model must be configured.

    The llama preset convention uses ``[*]`` for explicit shared settings. Rejecting Python
    ConfigParser's separate ``[DEFAULT]`` behaviour avoids silent inheritance that the router
    may not share; rejecting an empty file prevents a healthy-looking broker with no models.
    """
    with pytest.raises(CatalogueError, match=r"\[DEFAULT\]"):
        parse_catalogue("[DEFAULT]\nmodel = /models/example.gguf\n")
    with pytest.raises(CatalogueError, match="no model sections"):
        parse_catalogue("[*]\nthreads = 8\n")


def test_load_catalogue_reads_the_mounted_preset_path(tmp_path: Path) -> None:
    """The catalogue is read from its supplied path, not discovered by scanning model files.

    The broker receives a read-only mounted preset from its deployment. Keeping the path
    explicit lets it validate exactly the catalogue that llama.cpp was configured to use and
    avoids treating unrelated files in a model directory as runnable variants.
    """
    path = tmp_path / "models-preset.ini"
    path.write_text(PRESET)

    assert load_catalogue(path)["Flash Next--cuda1"].model_path == "/models/flash-next.gguf"
