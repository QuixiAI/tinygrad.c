# tinygrad.c

**tinygrad.c** is a **C port and re-architecture** of [tinygrad](https://github.com/tinygrad/tinygrad) —  
keeping its minimalist deep learning philosophy, but rebuilding the core in pure C with:

- **Stable C ABI** — usable from any language (C++, Rust, Zig, Go, …)
- **Modular GPU backends** — pluggable backends for CUDA, HIP, Metal, Vulkan, WebGPU, etc.
- **Modular client generation** — automated language bindings via a machine-readable manifest
- **File-for-file mirroring** of the original tinygrad structure for incremental porting

The long-term goal is to make tinygrad.c a small, hackable, and portable low-level deep learning engine that can  
train and run modern AI models in pure C or through bindings in other languages.

---

## Credits

This project is directly inspired by and based on  
**[tinygrad](https://github.com/tinygrad/tinygrad)**, created by [George Hotz (geohot)](https://github.com/geohot) and contributors.

All credit for the original architecture, ideas, and code belongs to the tinygrad community.  
tinygrad.c is an independent re-implementation and is not affiliated with the original authors.

---

## Goals

1. **Faithful port** — maintain close file/module parity with tinygrad for easier diffing and learning.
2. **Language interoperability** — pure C ABI to support generated bindings for many languages.
3. **Hardware modularity** — GPU/accelerator vendors can ship standalone backends without touching the core.
4. **Hackability** — keep the codebase small, clear, and easy to experiment with.

---

## Directory Layout

```

include/                # Public headers (tg.h, tg\_backend.h)
src/                    # Core implementation, mirrors tinygrad modules
dtype/                # Data types and related utilities
shape/                # Shape and view logic
tensor/               # Tensor struct and core tensor API
engine/               # Graph execution, autograd
runtime/              # Device runtime interfaces
ops\_cpu/            # CPU backend ops and kernels
graph/              # Device graph execution plumbing
nn/                   # Optimizers and layers
codegen/              # Kernel code generation
renderer/             # Backend-specific kernel renderers
frontend/             # Model importers (ONNX, Torch, etc.)
uop/                  # Low-level operation and symbolic math
helpers/              # Miscellaneous helper functions
device/               # Device selection and management
schedule/             # Scheduling and kernelization
backends/               # Pluggable GPU/accelerator backend implementations
generators/manifest/    # Manifest generator for client bindings
examples/               # Example programs
tests/                  # Test executables
reference/              # Original tinygrad Python source (for reference only)

````

---

## Build Instructions

**Requirements:**
- [CMake](https://cmake.org/) ≥ 3.16
- C compiler (GCC, Clang, or AppleClang)
- [Python 3](https://www.python.org/) (for manifest generation)

**Configure and build:**
```bash
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j
````

---

## Running Tests

Tests are built as executables and also registered with `ctest`.

**Run directly:**

```bash
./build/test_tensor
./build/test_ops
./build/test_resnet18
```

**Or run all with CTest:**

```bash
cd build
ctest --output-on-failure
```

---

## Generating the Client Manifest

The manifest describes the C API in a machine-readable format for generating language bindings.

To regenerate:

```bash
cmake --build build --target tinygradc_manifest
```

The output will be written to:

```
generators/manifest/tg_manifest.json
```

---

## License

This project follows the license terms of the original [tinygrad](https://github.com/tinygrad/tinygrad) (MIT license).
See [LICENSE](LICENSE) for details.

---

**Original tinygrad links:**

* Repo: [https://github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)
* Author: [George Hotz (geohot)](https://github.com/geohot)
