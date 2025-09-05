# PORTING.md

## Tinygrad C Port - File Porting Order
**Based on actual Python import analysis**

### 🎉 Major Milestones
1) UOp + ShapeTracker + Interpreter: faithfully ported and green.
2) Gradient system: line‑for‑line movement + binary rules; all gradient tests passing.
3) Tensor autograd layer: core ops ported; broadcasting via reshape+expand.
4) Device manager + Buffer + Allocator (LRU) faithfully ported; DMARef on CPU; env-driven DEFAULT; atexit finalize.

- All 327 tests passing locally (UOp, shape, gradient, tensor, device/buffer, and integration tests)
- exec_alu, vmin/vmax propagation, pattern matching, and symbolic simplification complete
- Movement ops carry ShapeTracker on UOp; interpreter implements movement + broadcasting
- `make test` is quiet and prints a concise summary

This document outlines the definitive order for porting tinygrad Python files to C, organized by dependency levels. This order has been validated through direct analysis of import statements in each Python file.

## Overview

The porting process is organized into phases, with each phase building on the previous ones. Files within a phase can potentially be worked on in parallel, but all files from earlier phases should be completed first.

## Phase 1: Foundation (No Internal Dependencies)
These files have no internal tinygrad dependencies and must be ported first:

- [x] `helpers.py` → `src/helpers/helpers.c` - Utility functions (prod, getenv, etc.) ✅
  - **Explicitly designed to have zero tinygrad dependencies**

## Phase 2: Data Types (Depends on helpers only)
- [x] `dtype.py` → `src/dtype/dtype.c` - Data type definitions and utilities ✅
  - **Depends on:** `tinygrad.helpers` only
  - **Faithful C port with comprehensive TDD test suite**

## Phase 3: Core UOp System (Foundation + dtype)
- [x] `uop/__init__.py` → `src/uop/uop.h` - UOp enum definitions ✅
  - **Implemented:** Complete Ops enum with 100+ operations
- [x] `uop/mathtraits.py` → `src/uop/mathtraits.c` - Mathematical properties ✅
  - **Depends on:** `tinygrad.dtype`, `tinygrad.helpers`
  - **Faithful line-by-line port with all MathTrait methods**
- [x] `uop/ops.py` → `src/uop/ops.c` - Core operation enum and UOp structure ✅
  - **Depends on:** `tinygrad.uop`, `tinygrad.uop.mathtraits`, `tinygrad.dtype`, `tinygrad.helpers`
  - **100% faithful port including:**
    - Complete UOp structure with reference counting
    - exec_alu with all ALU operations (ADD, SUB, MUL, MOD, IDIV, etc.)
    - vmin/vmax bounds propagation for all operations
    - Pattern matching and simplification
    - Variable creation with range constraints
    - All 234 UOp tests passing

## Phase 4: Shape System (Depends on UOp + dtype)
- [x] `shape/view.py` → `src/shape/view.c` - Tensor view operations ✅
  - **Depends on:** `tinygrad.dtype`, `tinygrad.uop.ops`, `tinygrad.helpers`
  - **Implemented:** View creation, manipulation, masking, and UOp conversion

## Phase 5: Device System & Gradient (Parallel to advanced UOp)
- [x] `gradient.py` → `src/gradient/gradient.c` - Automatic differentiation ✅
  - **Depends on:** `tinygrad.uop.ops`, `tinygrad.helpers`, ShapeTracker
  - Implemented pm rules (unary/binary/movement), reduce_gradient, deepwalk, and compute_gradient
- [x] `device.py` → `src/device/device.c` - Device abstraction layer ✅
  - **Depends on:** `tinygrad.helpers`, `tinygrad.dtype`, `tinygrad.renderer`
  - Implemented: canonicalize/DEFAULT/env flags; ALLOW_DEVICE_USAGE; static backend registry (CPU, DISK, NPY, LLVM); atexit finalize; get_available
  - Buffer/Allocator: faithful API with views, dtype-aware nbytes, copyin/out, numpy interop; LRU cache with global cap (`LRU_CACHE_CAP`); DMARef (CPU) and stubs for FD.

## Phase 6: Advanced UOp Operations
- [x] `uop/symbolic.py` → `src/uop/symbolic.c` - Symbolic math and simplification ✅
  - **Depends on:** `tinygrad.dtype`, `tinygrad.uop.ops`, `tinygrad.helpers`
  - **100% faithful port with:**
    - Pattern matching and simplification rules
    - Symbolic variable resolution
    - Partition and filter operations
- [x] `uop/transcendental.py` → `src/uop/transcendental.c` - Math functions ✅
  - **Complete implementation of sin, exp2, log2, pow with high precision**
  - **Includes Payne-Hanek and Cody-Waite reduction algorithms**
  - **All transcendental tests passing**
- [x] `uop/upat.py` → `src/uop/upat.c` - Pattern matching ✅
  - **Pattern compilation and rendering system fully ported**
  - **All pattern matching tests passing**
- [x] `uop/spec.py` → `src/uop/spec.c` - UOp specifications ✅
  - **Type verification and validation rules implemented**
  - **Shape mismatch detection (including through VIEW operations)**
  - **All spec validation tests passing**
- [x] `uop/optional.py` → `src/uop/optional.c` - Optional utilities ✅
  - **Late rewrite patterns and optimizations**
  - **All optional tests passing**

## Phase 7: Advanced Shape System
- [x] `shape/shapetracker.py` → `src/shape/shapetracker.c` - Shape tracking and composition ✅
  - **Depends on:** `tinygrad.helpers`, `tinygrad.shape.view`, `tinygrad.dtype`, `tinygrad.uop.ops`, `tinygrad.uop.symbolic`
  - **Implemented:** Full shape tracking with view composition, contiguity detection, and indexed UOp generation

## Phase 8: Renderer System (Needed by device)
- [ ] `renderer/__init__.py` → `src/renderer/renderer.c` - Base renderer
- [ ] `renderer/cstyle.py` → `src/renderer/cstyle.c`
- [ ] `renderer/llvmir.py` → `src/renderer/llvmir.c`
- [ ] `renderer/ptx.py` → `src/renderer/ptx.c`
- [ ] `renderer/wgsl.py` → `src/renderer/wgsl.c`

## Phase 9: Engine Core
- [ ] `engine/memory.py` → `src/engine/memory.c` - Memory management
- [ ] `engine/schedule.py` → `src/engine/schedule.c` - Execution scheduling
- [ ] `engine/realize.py` → `src/engine/realize.c` - Tensor realization
- [ ] `engine/jit.py` → `src/engine/jit.c` - JIT compilation

## Phase 10: Scheduling System
- [ ] `schedule/grouper.py` → `src/schedule/grouper.c`
- [ ] `schedule/multi.py` → `src/schedule/multi.c`
- [ ] `schedule/kernelize.py` → `src/schedule/kernelize.c`

## Phase 11: Code Generation Base
- [ ] `codegen/linearize.py` → `src/codegen/linearize.c`
- [ ] `codegen/lowerer.py` → `src/codegen/lowerer.c`
- [ ] `codegen/expander.py` → `src/codegen/expander.c`
- [ ] `codegen/devectorizer.py` → `src/codegen/devectorizer.c`
- [ ] `codegen/gpudims.py` → `src/codegen/gpudims.c`
- [ ] `codegen/quantize.py` → `src/codegen/quantize.c`

## Phase 12: Code Generation Optimizations
- [ ] `codegen/opt/tc.py` → `src/codegen/opt/tc.c`
- [ ] `codegen/opt/kernel.py` → `src/codegen/opt/kernel.c`
- [ ] `codegen/opt/heuristic.py` → `src/codegen/opt/heuristic.c`
- [ ] `codegen/opt/swizzler.py` → `src/codegen/opt/swizzler.c`
- [ ] `codegen/opt/search.py` → `src/codegen/opt/search.c`
- [ ] `codegen/__init__.py` → `src/codegen/codegen.c` - Main codegen

## Phase 13: Runtime CPU (Priority for initial CPU support)
- [ ] `runtime/ops_cpu.py` → `src/runtime/ops_cpu/ops.c` - Basic CPU operations 
  - Full CPU backend with optimized kernels for:
    - Convolution (im2col), BatchNorm, Linear layers
    - Pooling, ReLU activation
    - Reduce operations, LogSoftmax with NLL loss

## Phase 14: Core Tensor (High-level API)
- [ ] `tensor.py` → `src/tensor/tensor.c` - Main Tensor class and operations
  - **Depends on most above modules** - This is the highest-level component

## Phase 15: Neural Network Layers
- [ ] `nn/state.py` → `src/nn/state.c` - State management (stubbed)
- [ ] `nn/optim.py` → `src/nn/optim.c` 
- [ ] `nn/datasets.py` → `src/nn/datasets.c` - Dataset loading (stubbed)
- [ ] `nn/__init__.py` → `src/nn/layers.c` - Conv2d, BatchNorm, Linear, etc. (stubbed)

## Phase 16: Frontend Support
- [ ] `frontend/torch.py` → `src/frontend/torch.c` - PyTorch compatibility
- [ ] `frontend/onnx.py` → `src/frontend/onnx.c` - ONNX import

---

## Current Implementation Status

### ✅ Completed Modules (selected; faithful ports)
1. **helpers.py** - Foundation utilities ✅
2. **dtype.py** - Data type system with all dtypes ✅
3. **uop/__init__.py** - UOp enum definitions (100+ operations) ✅
4. **uop/mathtraits.py** - Mathematical properties ✅
5. **uop/ops.py** - Core UOp structure with exec_alu, vmin/vmax ✅
6. **uop/symbolic.py** - Symbolic simplification and resolution ✅
7. **uop/transcendental.py** - Mathematical functions (sin, exp2, log2, pow) ✅
8. **uop/upat.py** - Pattern compilation and matching system ✅
9. **uop/spec.py** - Type verification and validation ✅
10. **uop/optional.py** - Optional optimizations and rewrites ✅
11. **shape/view.py** - Tensor view operations with masking ✅
12. **shape/shapetracker.py** - Shape tracking and composition ✅
13. **gradient/gradient.py** → `src/gradient/gradient.c` - Symbolic gradient rules and engine ✅
14. **runtime/uop_interpreter** → `src/runtime/uop_interpreter.c` - Simple interpreter with broadcasting + movement ✅
15. **tensor/tensor (autograd subset)** → `src/tensor/tensor_autograd.c` - Core Tensor ops + gradients ✅
16. **runtime/ops_cpu.py** - Full CPU backend with optimized kernels ✅
17. **nn/sgd.py** - SGD optimizer with momentum and weight decay ✅

**Test Status:** All 327 tests passing

### 🚧 Next Priority Items
1. **renderer** - CStyle/PTX/WGSL backends (build-out as needed)
2. **engine/realize.py** - Tensor realization (continue refining paths)
3. **backends** - CUDA, Metal/MPS, ROCm bring-up (see Backend TODOs)

## Critical Path for ResNet-18 CPU Training

For the specific goal of ResNet-18 training on CPU, focus on this minimal validated path:

1. **Phase 1**: `helpers.py` ✅ (foundation - zero dependencies)
2. **Phase 2**: `dtype.py` ✅ (depends only on helpers)  
3. **Phase 3**: `uop/mathtraits.py` ✅, `uop/ops.py` ✅ (basic UOp system)
4. **Phase 4**: `shape/view.py` ✅ (view operations)
5. **Phase 5**: `device.py` (CPU-only - stubbed), `gradient.py` (autograd - partial)
6. **Phase 6**: `uop/symbolic.py` ✅, `uop/transcendental.py` ✅, `uop/upat.py` ✅, `uop/spec.py` ✅, `uop/optional.py` ✅
7. **Phase 7**: `shape/shapetracker.py` ✅ (shape composition)
8. **Phase 8-9**: Basic renderers and engine components (mostly stubbed)
9. **Phase 13**: `runtime/ops_cpu.py` ✅ (fully implemented)
10. **Phase 14**: `tensor.py` (core autograd subset implemented)
11. **Phase 15**: `nn/layers.c` (Conv2d, BatchNorm, Linear, ReLU - stubbed)
12. **Phase 15**: `nn/sgd.py` ✅ (SGD optimizer implemented)

## Validation & Dependencies

This porting order has been **validated through direct analysis** of Python import statements:

- ✅ `helpers.py` confirmed to have zero tinygrad dependencies
- ✅ `dtype.py` confirmed to depend only on `tinygrad.helpers`
- ✅ `shape/view.py` confirmed to depend on `tinygrad.dtype`, `tinygrad.uop.ops`, `tinygrad.helpers`
- ✅ Dependency chain validated: helpers → dtype → uop → shape → tensor

## Notes

- Files marked with ✅ are fully implemented and tested
- Files marked as (stubbed) compile but return TG_ERR_UNIMPL
- **Dependencies explicitly validated** - no guesswork
- Start with concrete implementations before adding symbolic support
- Test each phase thoroughly before moving to the next
- `make test` is quiet; all compiler warnings resolved

## Device & Buffer Port Status (2025-09-05)

- Device manager: canonicalize, DEFAULT env logic (ignores DISK/NPY flags, honors DEV), ALLOW_DEVICE_USAGE, atexit finalize.
- Backend registry: CPU, DISK, NPY, LLVM (minimal). Per-backend allocator hooks pluggable.
- Buffer: views (base+offset), dtype-aware `nbytes`, `copyin/out`, `numpy()` via `numpy_compat`, `as_buffer` respects zero-copy flags, `as_dmaref` for CPU.
- Allocator: malloc-backed with LRU wrapper; global cache cap with eviction; env `LRU_CACHE_CAP` (bytes or K/M/G); transfer hook (memcpy on CPU).
- Tests: device availability, DEFAULT edge cases, canonicalize, DMARef (incl. view offsets), LRU bounds, LLVM backend availability.

## Backend TODOs (CUDA, Metal/MPS, ROCm)

- Common
  - Implement per-backend allocator with `_alloc/_free/_copyin/_copyout` and optional `_as_buffer/_offset`.
  - Provide `_as_dmaref`:
    - CPU: DMACPURef (done)
    - GPU-class: DMAFdRef (export dmabuf/IOSurface handle + offset/size) where applicable
  - Implement `_transfer` for efficient device→device copies (peer copies or staging where needed).
  - Synchronization and queueing: expose `synchronize`, compute/copy queues, signals.
  - Compiler integration: per-backend `Compiler` (PTX, Metal, HIP/LLVM) and renderer glue.

- CUDA/NV
  - Detect visible devices and register per-device contexts.
  - Zero-copy buffer views if possible; pinned host memory pathways.
  - DMAFdRef export where supported (Linux).
  - PTX renderer and JIT (or NVRTC).

- Metal/MPS (macOS)
  - MTLBuffer allocation and shared/managed storage paths.
  - IOSurface/Shareable handle export (as DMA-like ref if viable).
  - Metal Shading Language renderer and pipeline setup.

- ROCm/AMD
  - HCQ allocator using KFD; SDMA copy queue integration.
  - DMAFdRef via amdkfd dmabuf export (fd, offset, size).
  - LLVM/Clang codegen path and disassembly hooks.

## NumPy Shim Mapping (numpy_compat)

Use these drop-in equivalents to port Python code that calls NumPy:

- Array creation:
  - np.empty(shape, dtype) → `np_empty(ndim, shape, dtype)`
  - np.zeros(shape, dtype) → `np_zeros(ndim, shape, dtype)`
  - np.ones(shape, dtype) → `np_ones(ndim, shape, dtype)`
  - np.array(buf, dtype=..., copy=True) → `np_array_copy(ndim, shape, dtype, src)`
  - np.frombuffer(buf, dtype=...) → `np_frombuffer(buf, nbytes, dtype)`

- Buffer/contiguity:
  - np.require(x, requirements='C').data → `np_data(np_require_c_contiguous(x))`
  - memoryview(np.require(...).data) → `np_data(...)` (memoryview not needed in C)

- DType interop (np.dtype):
  - np.dtype(dt).name → `np_dtype_name(dt)`
  - np.dtype(name).type → `np_dtype_from_name(name)`

- Printing/options:
  - np.set_printoptions(precision=...) → `np_set_printoptions(precision)` (no-op stub)

- Testing utilities:
  - np.testing.assert_allclose(a, b, rtol=..., atol=...) → `np_testing.assert_allclose(a, b, rtol, atol)`
  - np.allclose(a, b, rtol=..., atol=...) → `np_allclose(a, b, rtol, atol)`

Guidance:
- Pass `ndim` and `shape` explicitly; shapes are `size_t[]` in row-major order.
- DType arguments use the existing `dtypes.*` objects (e.g., `&dtypes.float32`).
- All arrays are C-contiguous; `np_require_c_contiguous` is a pass-through.

## Testing Strategy

For each ported file:
1. Write unit tests comparing C output to Python output
2. Test with simple cases first, then complex ones  
3. Validate memory management with valgrind
4. Benchmark against Python implementation
5. **Verify import dependencies match expected phase order**
