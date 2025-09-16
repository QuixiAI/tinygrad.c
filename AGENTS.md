# AGENTS.md

A simple, open format for guiding coding agents, used by over 20k open-source projects.

Think of AGENTS.md as a README for agents: a dedicated, predictable place to provide the context and instructions to help AI coding agents work on your project.  Keep this file up to date—future agents rely on the quick cues here instead of re-discovering the workflow.

## Philosophy

Welcome, agent. tinygrad.c is the C port of tinygrad—the same minimal, beautiful tensor library, rendered with a stable C ABI. Treat every edit as a chance to make the code tighter and more comprehensible.

- Every line must earn its place. Prefer clarity over clever tricks; a well-written 10 lines in C can replace a maze of macros.
- Keep functionality changes focused. Do not mix logic edits with unrelated formatting or whitespace churn.
- Tests guard our parity with Python tinygrad. Any behavioral change should land with tests (new or expanded) and must run cleanly through `make test`.
- Match the house style: C and header files use two-space indentation, trailing whitespace is frowned upon, and 150 characters per line is the soft ceiling unless an existing file dictates otherwise.

### TL;DR for new agents
- `make -j$(nproc)` builds everything; `make test` runs all Unity suites.
- Many modules are still **stubbed** (`return TG_ERR_UNIMPL`). Stubs live in the same relative path as the Python source; real implementations typically live in subdirectories (e.g. `src/tensor/tensor.c`). Remove or replace a stub once the real port lands.
- Always run a full cycle (`make clean && make && make test`) to catch regressions immediately; investigate and fix every warning, error, crash, or failing test—no shortcuts.
- If a `deeply_port_*.md` file exists for the subsystem you are touching, review it for the latest TODO list and context before making changes.

## Setup commands

### Build and test commands
- Build project: `make`
- Run all tests: `make test`
- Run specific tests: `./build/test_dtype`, `./build/test_tensor`, `./build/test_ops`, `./build/test_resnet18`
- Run examples: `./build/resnet18_cpu`
- Rebuild just one test binary: `cmake --build build --target test_NAME`

Advanced:
- `make -j$(nproc)` — parallel build (faster on multi-core machines).
- `CLEAN=1 make` — fresh build dir (removes `build/`).
- `PERSIST_CONAN=1 make` — persist Conan cache across runs by mounting `$HOME/.conan2` in Docker.
- `CLEAN_CONAN=1 PERSIST_CONAN=1 make` — clear the mounted Conan cache before install.
- `make rebuild` — alias for `CLEAN=1 make`.

### Manual build (if needed)
```bash
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j
cd build && ctest --output-on-failure
```

### Testing instructions
The project uses the Unity test framework with a Test-Driven Development (TDD) approach:
- Tests are distributed across multiple suites in `tests/` and `tests/uop/`
- Dot-reporter style output: `.` for pass, `F` for fail, `X` for crash

Run tests with:
```bash
make test                 # Dot-reporter style with summary
make test REPORTER=tap    # TAP v13 output (with plan at end)
./build/test_NAME  # Run individual test suite
```

To change the default reporter without passing it each time, use one of:
- Environment: `export TEST_REPORTER=tap` (or `TINYGRADC_TEST_REPORTER=tap`)
- Local config: create `.env` with `TEST_REPORTER=tap` (gitignored)

Precedence: command-line `REPORTER=` > env (`TEST_REPORTER`/`TINYGRADC_TEST_REPORTER`) > .env/.env.local > default (dot).
Note: environment `REPORTER` is ignored to avoid collisions with other tools.

## Project overview

tinygrad.c is a C port of tinygrad, designed as a minimal deep learning engine with modular backends. The UOp system (`src/uop/`) is a core component providing symbolic computation and graph optimization.

### Architecture
- **Stable C ABI** - main API in `include/tg.h`
- **Modular backends** - CPU implementation active, GPU backends stubbed
- **File-for-file mirroring** - C source in `src/` mirrors original Python structure
- **UOp system** - Symbolic computation engine for graph optimization

### Porting status
The project follows a phased approach documented in `PORTING.md`:
- **Phase 1-2**: Core foundation (dtype, helpers, shape system) - partially complete
- **Phase 5**: Core tensor operations - basic implementation done
- **Phase 9**: CPU runtime - functional for basic ops
- **Phase 11**: Neural network layers - SGD optimizer implemented

Many modules are stubbed (return `TG_ERR_UNIMPL`) to allow compilation while incrementally porting functionality.

### Stub map
- Source files directly under `src/` with the header `// Auto-generated stub...` are placeholders. The real implementations usually live in subdirectories (e.g. `src/tensor/tensor.c`, `src/dtype/dtype.c`, `src/runtime/ops_cpu/ops.c`).
- Matching stub tests live under `tests/**/test_stub_*.c`. They exist only to keep the build green; feel free to delete them when a real implementation + tests are added.
- Before adding new code, double-check for an existing non-stub version to avoid duplicates.

## Development workflow

### Code style
- C11 standard with position-independent code enabled
- Modular compilation using object libraries
- Tests validate functionality against expected outputs
- Reference Python code available in `reference/` directory

### Current focus
The project follows Test-Driven Development with Unity framework:
- Tests written first to drive implementation
- Stub implementations compile but fail at runtime (`TG_ERR_UNIMPL`)
- Current test status visible via dot-reporter output
- Progressive approach: fix test failures to drive feature completion

## Key files

### Core components
- `include/uop/uop.h` - UOp system public API
- `src/uop/ops.c` - Core UOp operations and implementation
- `src/uop/mathtraits.c` - Mathematical operations and traits
- `src/uop/symbolic.c` - Symbolic computation and simplification
- `tests/test_uop.c` - Comprehensive UOp test suite

### Build system
- `Makefile` - Main entrypoint (build, test, clean)
- `CMakeLists.txt` - Build configuration
- Conan profile at `profiles/linux-gcc11` and lockfiles under `profiles/locks/`
- Code generation for language bindings via `tinygradc_manifest`

### Reference materials
- `reference/tinygrad/uop/` - Original Python implementation
- `PORTING.md` - Detailed porting roadmap
- `CLAUDE.md` - Additional developer guidance

## Testing guidelines

### How to write tests
Tests use Unity framework with automatic registration:

```c
#include "test_common.h"  // Includes Unity and utilities

// Define test functions with TEST() macro
TEST(test_feature_name) {
    // Arrange
    int expected = 42;
    int actual = my_function();
    
    // Assert
    TEST_ASSERT_EQUAL(expected, actual);
}

// Auto-register and run all tests
TEST_MAIN()
```

Common Unity assertions:
- `TEST_ASSERT_EQUAL(expected, actual)` - Value comparison
- `TEST_ASSERT_TRUE/FALSE(condition)` - Boolean checks  
- `TEST_ASSERT_NULL/NOT_NULL(pointer)` - Pointer validation
- `TEST_ASSERT_FLOAT_WITHIN(delta, expected, actual)` - Float comparison
- `TEST_FAIL_MESSAGE(message)` - Explicit failure

### TDD workflow
1. **Write failing tests first** - Tests drive implementation
2. **Implement minimal code** - Just enough to pass tests
3. **Refactor** - Improve code while keeping tests green
4. **Never suppress failures** - Let tests fail naturally

### Test organization
- `tests/test_*.c` - Module-level suites (tensor, dtype, device, etc.)
- `tests/uop/test_*.c` - UOp subsystem suites covering ops, optional features, symbolic simplification, transcendental math, pattern matching, and spec handling.

### Running tests
After making changes:
1. Run `make test` to see dot-reporter output
2. Failed tests show detailed error messages
3. Crashed tests marked with `X` indicate segfaults
4. Check specific suite with `./build/test_NAME`

## Special considerations

### Memory management
The UOp system uses reference counting - be careful with circular references and ensure proper cleanup when testing.

### Symbolic computation
Many tests expect symbolic variables to have proper bounds propagation (`vmin`/`vmax`). The system should infer ranges from operations.

### Type system operations
Pay attention to type promotion rules and ensure operations work across different dtypes.

### Porting faithfulness
When implementing features, compare against the reference Python implementation in `reference/tinygrad/uop/` to maintain behavior compatibility.

## Contribution instructions

### Before committing
- Run `make test` and ensure all tests pass or test count increases
- Verify no new memory leaks with valgrind if available
- Check that the UOp system maintains reference counting integrity

### Pull requests
- Update test coverage when implementing new features
- Maintain compatibility with the existing API in `include/uop/uop.h`
- Document any breaking changes or new functionality
