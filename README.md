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
tests/                  # Test executables (using Unity test framework)
reference/              # Original tinygrad Python source (for reference only)

````

---

## Build Instructions

### Setup (install Conan in your Python env)

This project relies on Conan 2 for C/C++ dependencies.
We recommend you manage your Python environment explicitly (venv, conda, uv, etc.).

Example with Python venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip conan
# Optional: create/update a default profile (Makefile does this if missing)
conan profile detect --force
```

If you skip the venv, ensure a recent `conan` is available on your `PATH`.

**Requirements:**
- [CMake](https://cmake.org/) ≥ 3.16
- C compiler (GCC, Clang, or AppleClang)
- [Python 3](https://www.python.org/) (for manifest generation)

**Test Framework:**
- Tests use the [Unity](https://github.com/ThrowTheSwitch/Unity) test framework (included in third_party/unity/)

**Quick build:**
```bash
make          # Configure (runs Conan if inputs changed) and build
make test     # Run all tests with dot-reporter
```

Note: The build will only re-run Conan when `conanfile.txt`, `profiles/linux-gcc11`,
or the lockfile change. Otherwise it uses the existing toolchain for fast, offline builds.

**Manual build:**
```bash
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j
````

---

## Running Tests

Tests use the Unity framework and are built as executables registered with `ctest`.

**Run all tests:**
```bash
make test
```

Advanced
- `CLEAN=1 make`: Removes `build/` before configuring (fresh build dir).
- `PERSIST_CONAN=1 make`: Persist Conan cache across runs by mounting `$HOME/.conan2` into the Docker container.
- `CLEAN_CONAN=1 PERSIST_CONAN=1 make`: Clear the mounted Conan cache before installing deps.
- `make rebuild`: Alias for `CLEAN=1 make`.

Notes
- Conan runs via the local binary (`.venv/bin/conan` if present, else system `conan`).
- A checked-in Conan profile lives at `profiles/linux-gcc11`; a lockfile is created under `build/conan/locks/` and used on installs for reproducible dependency resolution.

**Run individual tests:**
```bash
./build/test_tensor
./build/test_ops
./build/test_resnet18
./build/test_dtype
./build/test_uop
```

**Or run all with CTest:**

```bash
cd build
ctest --output-on-failure
```

---

## Contributing

- Run `make test` locally and keep tests green.
- For a fresh build directory use `CLEAN=1 make` (alias: `make rebuild`).
- For faster iterations, persist the Conan cache with `PERSIST_CONAN=1 make`.
- To reset dependencies, combine `CLEAN_CONAN=1 PERSIST_CONAN=1 make`.

These advanced toggles are optional; default `make` and `make test` remain one-command workflows.

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

## Runtime Tuning

- LRU cache cap: control the global size of the allocator’s LRU cache via the `LRU_CACHE_CAP` environment variable.
  - Accepts bytes with optional K/M/G suffix (base 1024): examples: `LRU_CACHE_CAP=1048576`, `LRU_CACHE_CAP=512K`, `LRU_CACHE_CAP=2M`, `LRU_CACHE_CAP=1G`.
  - Can also be set at runtime via the C API: `tg_allocator_set_cache_cap(cap_bytes)`.
  - Inspect current values using:
    - `tg_allocator_get_cache_cap()` and `tg_allocator_get_cached_bytes()`.

Example
```bash
LRU_CACHE_CAP=64M make test
```

Process exit automatically finalizes opened devices (registered via `atexit`) to mirror tinygrad behavior.

---

## Helpers Runtime

- Disk cache (helpers diskcache):
  - `CACHEDB`: path to sqlite database (default: `build/cache/cache.db`).
  - `CACHELEVEL`: 0 disables cache; 1 enables (default).
  - Requires sqlite support (`TG_HAVE_SQLITE3`) or falls back to in-memory cache.

- Profiling:
  - `PROFILE`: non-empty enables profiling capture; disabled automatically on CI.
  - `PROFILE_OUT`: output path for event dump (default: `build/profile.json`).
  - C API: `tg_profile_set_enabled/get_enabled`, `tg_cpu_profile_begin/end`.

- Fetch + downloads:
  - Default downloads dir: `/raid/downloads` on tinybox, else `build/downloads[/subdir]`.
  - `DISABLE_HTTP_CACHE`: non-empty disables using a cached file (forces re-download).
  - Requires `TG_HAVE_CURL` (HTTP/S) and `TG_HAVE_ZLIB` (gunzip) for full functionality; local file passthrough always supported.

Example: fetch and gunzip a local .gz
```c
#include "helpers/helpers.h"
char path[512];
int rc = tg_fetch("./file.txt.gz", NULL, NULL, /*gunzip*/1, /*allow_caching*/1, path, sizeof(path));
if (rc == 0) {
  // path now points to "./file.txt.gz.gunzip"
}
```

Example: download with caching (requires libcurl + zlib)
```c
char path[512];
int rc = tg_fetch("https://example.com/model.gz", "model.gz", "weights", /*gunzip*/1, /*allow_caching*/1, path, sizeof(path));
// Downloaded to build/downloads/weights/model.gz.gunzip (or /raid/downloads on tinybox)
```


## License

This project follows the license terms of the original [tinygrad](https://github.com/tinygrad/tinygrad) (MIT license).
See [LICENSE](LICENSE) for details.

---

**Original tinygrad links:**

* Repo: [https://github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)
* Author: [George Hotz (geohot)](https://github.com/geohot)
