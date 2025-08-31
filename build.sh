#!/bin/bash
set -e

echo "Building tinygrad.c..."

# Configure if build directory doesn't exist or if CMakeLists.txt is newer
if [ ! -d "build" ] || [ "CMakeLists.txt" -nt "build/CMakeCache.txt" ]; then
    echo "Configuring CMake..."
    cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
fi

# Build
echo "Compiling..."
cmake --build build -j

echo "Build completed successfully!"
echo ""
echo "Available executables:"
echo "  Examples:    ./build/resnet18_cpu"
echo "  Tests:       ./build/test_tensor, ./build/test_ops, ./build/test_resnet18, ./build/test_dtype"
echo "  Run tests:   ./test.sh"