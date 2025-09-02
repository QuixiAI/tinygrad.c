#!/bin/bash
set -e

rm -rf build

# Configure if needed
[ ! -d "build" ] && cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON

# Build
cmake --build build -j
