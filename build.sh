#!/bin/bash
set -euo pipefail

# Clean host build dir
rm -rf build

DOCKER_IMAGE="${DOCKER_IMAGE:-conanio/gcc11-ubuntu16.04:2.20.1}"

docker run --rm \
  -v "$PWD":/work -w /work \
  --user "$(id -u)":"$(id -g)" \
  "$DOCKER_IMAGE" \
  bash -lc '
    # Start from a pristine Conan home each run
    rm -rf ~/.conan2

    # Fresh profile
    conan profile detect --force

    # Create a custom profile with the defines injected
    cat > ~/.conan2/profiles/custom <<EOF
include(default)
[buildenv]
CFLAGS=-DUNITY_INCLUDE_DOUBLE -DUNITY_INCLUDE_FLOAT
EOF

    # Install deps using the custom profile
    conan install . \
      -of build/conan \
      --profile=custom \
      --build=unity/* \
      --build=missing \
      -g CMakeDeps -g CMakeToolchain \
      --deployer=full_deploy \
      --deployer-folder build/conan/full_deploy
  '

# Host configure using the generated toolchain
cmake -S . -B build \
  -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan/conan_toolchain.cmake \
  -DCMAKE_C_FLAGS="-DUNITY_INCLUDE_DOUBLE -DUNITY_INCLUDE_FLOAT"

# Host build
cmake --build build -j