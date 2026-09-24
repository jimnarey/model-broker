# A minimal llama.cpp router for the broker's acceptance tests: llama-server built
# with CUDA from one pinned commit. Build it with build-router.sh, which reads the
# pinned values from llama.env.
ARG CUDA_VERSION
ARG UBUNTU_VERSION

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG LLAMA_REPOSITORY
ARG LLAMA_COMMIT
ARG GCC_VERSION
ARG CUDA_DOCKER_ARCH
ARG BUILD_JOBS=4

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN test -n "${LLAMA_REPOSITORY}" && test -n "${LLAMA_COMMIT}" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates git build-essential cmake gcc-${GCC_VERSION} g++-${GCC_VERSION} libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

# Fetch only the pinned commit, and check that it is the one checked out.
RUN git init /src && \
    git -C /src fetch --depth 1 "${LLAMA_REPOSITORY}" "${LLAMA_COMMIT}" && \
    git -C /src checkout --detach FETCH_HEAD && \
    test "$(git -C /src rev-parse HEAD)" = "${LLAMA_COMMIT}" && \
    cmake -S /src -B /src/build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build /src/build --config Release --target llama-server --parallel "${BUILD_JOBS}" && \
    mkdir -p /output && \
    find /src/build -name '*.so*' -exec cp -P {} /output/ \; && \
    cp /src/build/bin/llama-server /output/

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG LLAMA_REPOSITORY
ARG LLAMA_COMMIT

LABEL org.opencontainers.image.source="${LLAMA_REPOSITORY}" \
      org.opencontainers.image.revision="${LLAMA_COMMIT}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /output/ /app/

ENV LD_LIBRARY_PATH=/app
WORKDIR /app
ENTRYPOINT ["/app/llama-server"]
