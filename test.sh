#!/bin/bash
set -e

# Build if needed
[ ! -f "build/test_tensor" ] && ./build.sh

# Run all Unity tests directly for proper reporting
for test in build/test_*; do
    [ -x "$test" ] && "$test"
done