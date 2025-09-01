# AGENTS.md

A simple, open format for guiding coding agents, used by over 20k open-source projects.

Think of AGENTS.md as a README for agents: a dedicated, predictable place to provide the context and instructions to help AI coding agents work on your project.

## Setup commands

### Build and test commands
- Build project: `./build.sh`
- Run all tests: `./test.sh`
- Run specific tests: `./build/test_dtype`, `./build/test_tensor`, `./build/test_ops`, `./build/test_resnet18`
- Run examples: `./build/resnet18_cpu`

### Manual build (if needed)
```bash
cmake -S . -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j
cd build && ctest --output-on-failure
```

### Testing instructions
The project uses a single comprehensive test suite that covers multiple components:
- UOp system tests in `tests/test_uop.c` (496 tests)
- DType system tests in `tests/test_dtype.c`
- Tensor operations in `tests/test_tensor.c`
- End-to-end examples in `tests/test_resnet18.c`

You must run `./test.sh` to see the overall test coverage and identify specific failures.

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

## Development workflow

### Code style
- C11 standard with position-independent code enabled
- Modular compilation using object libraries
- Tests validate functionality against expected outputs
- Reference Python code available in `reference/` directory

### Current focus
The UOp system is currently undergoing implementation and optimization:
- 91.5% test coverage achieved (454/496 tests passing)
- Remaining failures focus on reference counting, symbolic variables, and constant folding
- Progressive approach: fix basic issues first, then optimize performance

## Key files

### Core components
- `include/uop/uop.h` - UOp system public API
- `src/uop/ops.c` - Core UOp operations and implementation
- `src/uop/mathtraits.c` - Mathematical operations and traits
- `src/uop/symbolic.c` - Symbolic computation and simplification
- `tests/test_uop.c` - Comprehensive UOp test suite

### Build system
- `build.sh` - Main build script (recommended)
- `CMakeLists.txt` - Build configuration
- Code generation for language bindings via `tinygradc_manifest`

### Reference materials
- `reference/tinygrad/uop/` - Original Python implementation
- `PORTING.md` - Detailed porting roadmap
- `CLAUDE.md` - Additional developer guidance

## Testing guidelines

### How to test
After making changes to the UOp system:
1. Run `./test.sh` to see overall status
2. Focus on `tests/test_uop.c` for UOp-specific failures
3. Check for memory leaks and reference counting issues
4. Verify symbolic variable inference (`uop_sym_infer`)
5. Test constant folding and simplification rules

### Test priorities
1. Reference counting and memory management
2. Core operations (ADD, SUB, MUL, DIV, comparisons)
3. Symbolic variable creation and resolution
4. Constant folding and expression simplification
5. Advanced features (WMMA, vectorization, etc.)

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
- Run `./test.sh` and ensure all tests pass or test count increases
- Verify no new memory leaks with valgrind if available
- Check that the UOp system maintains reference counting integrity

### Pull requests
- Update test coverage when implementing new features
- Maintain compatibility with the existing API in `include/uop/uop.h`
- Document any breaking changes or new functionality