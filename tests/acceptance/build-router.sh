#!/usr/bin/env bash
# Build the acceptance tests' llama.cpp router image from the values in llama.env.
# The image is tagged with the commit, so changing LLAMA_COMMIT builds a new one.
set -euo pipefail

directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
set -a
# shellcheck source=llama.env
source "$directory/llama.env"
set +a

tag="model-broker-acceptance-llama:${LLAMA_COMMIT:0:12}"
docker build \
    --build-arg LLAMA_REPOSITORY \
    --build-arg LLAMA_COMMIT \
    --build-arg CUDA_VERSION \
    --build-arg UBUNTU_VERSION \
    --build-arg GCC_VERSION \
    --build-arg CUDA_DOCKER_ARCH \
    --build-arg BUILD_JOBS="${BUILD_JOBS:-$(nproc)}" \
    --tag "$tag" \
    "$directory"
echo "$tag"
