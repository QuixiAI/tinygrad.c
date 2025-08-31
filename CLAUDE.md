# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

tinygrad.c uses CMake with the following commands:

**Configure and build:**
```bash
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j
```

**Run tests:**
```bash
# Run individual tests directly
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