# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

tinygrad.c uses CMake with the following commands:

**Quick build and test (recommended):**
```bash
./build.sh    # Configure and build project
./test.sh     # Run all tests with detailed output
```

**Manual build (if needed):**
```bash
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j
```

**Manual test commands:**
```bash
# Run individual tests directly
./build/test_dtype      # DType system tests (newly added)
./build/test_tensor
./build/test_ops  
./build/test_resnet18

# Or run all tests with CTest
cd build && ctest --output-on-failure
```

**Run examples:**
```bash
./build/resnet18_cpu
```

**Generate manifest for language bindings:**
```bash
cmake --build build --target tinygradc_manifest
```

## Architecture

tinygrad.c is a C port of [tinygrad](https://github.com/tinygrad/tinygrad), designed as a minimal deep learning engine with:

- **Stable C ABI** - main API in `include/tg.h`
- **Modular backends** - CPU implementation active, GPU backends stubbed in `backends/`
- **File-for-file mirroring** - C source in `src/` mirrors original Python structure

### Core Components

- **Tensor API** (`src/tensor/tensor.c`) - Core tensor operations and autograd
- **Runtime** (`src/runtime/ops_cpu/`) - CPU backend with hand-written kernels
- **Engine** (`src/engine/`) - Graph execution and automatic differentiation
- **Shape System** (`src/shape/`) - View operations and shape tracking
- **Code Generation** (`src/codegen/`) - Kernel generation (mostly stubbed)
- **Neural Networks** (`src/nn/`) - Layers and optimizers (SGD implemented)

### Porting Status

The project follows a phased porting approach documented in `PORTING.md`:

- **Phase 1-2**: Core foundation (dtype, helpers, shape system) - partially complete
- **Phase 5**: Core tensor operations - basic implementation done
- **Phase 9**: CPU runtime - functional for basic ops
- **Phase 11**: Neural network layers - SGD optimizer implemented

Many modules are stubbed (return `TG_ERR_UNIMPL`) to allow compilation while incrementally porting functionality.

## Key Files

- `include/tg.h` - Public C API
- `src/tensor/tensor.h` - Internal tensor structure
- `examples/resnet18_cpu.c` - Working neural network training example
- `CMakeLists.txt` - Build configuration with modular object libraries
- `PORTING.md` - Detailed porting roadmap and dependency order

## Development Workflow

1. The codebase uses C11 standard with position-independent code enabled
2. Object libraries allow modular compilation of different components
3. Tests validate functionality against expected outputs
4. Manual manifest generation creates language binding specifications
5. Original Python reference code available in `reference/` directory

The current focus is on CPU-only functionality with plans for modular GPU backend plugins.

## Testing

The project uses the Unity test framework for C unit testing. Tests follow Test-Driven Development (TDD) principles.

### Running Tests

```bash
./test.sh          # Run all tests with dot-reporter style output
./test.sh -p       # Run tests in parallel
./build/test_NAME  # Run individual test suite
```

Test output shows:
- `.` for passing tests
- `F` for failing tests  
- `X` for crashed test suites
- `i` for ignored tests

### Writing Tests

Tests are located in `tests/` with Unity test framework patterns:

```c
#include "test_common.h"  // Includes Unity and common test utilities

// Define test functions using TEST() macro
TEST(test_example) {
    TEST_ASSERT_EQUAL(expected, actual);
    TEST_ASSERT_TRUE(condition);
    TEST_ASSERT_FALSE(!condition);
    TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual);
}

// Use TEST_MAIN() to auto-register and run all tests
TEST_MAIN()
```

The `TEST()` macro automatically registers test functions, eliminating manual test listing. Common Unity assertions:
- `TEST_ASSERT_EQUAL(expected, actual)` - Compare values
- `TEST_ASSERT_TRUE/FALSE(condition)` - Boolean checks
- `TEST_ASSERT_NULL/NOT_NULL(pointer)` - Pointer validation
- `TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual)` - Float comparison with tolerance
- `TEST_FAIL_MESSAGE(message)` - Explicit failure with message

### Test Organization

- `tests/test_*.c` - Main test files
- `tests/uop/test_*.c` - UOp subsystem tests (75 test cases)
- `tests/test_common.h` - Shared test configuration and utilities

### TDD Approach

Following TDD principles:
1. Write failing tests first
2. Implement minimal code to pass tests
3. Refactor while keeping tests green
4. **Never suppress test failures** - let tests fail naturally to drive implementation

Stub implementations should compile but fail at runtime (return `TG_ERR_UNIMPL` or similar).