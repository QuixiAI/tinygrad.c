#!/bin/bash
set -e

echo "Running tinygrad.c test suite..."

# Ensure project is built
if [ ! -f "build/test_tensor" ]; then
    echo "Build directory not found. Building first..."
    ./build.sh
fi

echo ""
echo "=== Running Individual Tests ==="

# Run each test individually for better output
echo "Running test_dtype..."
./build/test_dtype

echo ""
echo "Running test_tensor..."
./build/test_tensor

echo ""
echo "Running test_ops..."
./build/test_ops

echo ""
echo "Running test_resnet18..."
./build/test_resnet18

echo ""
echo "=== Running CTest Suite ==="
cd build && ctest --output-on-failure
cd ..

echo ""
echo "All tests completed successfully!"