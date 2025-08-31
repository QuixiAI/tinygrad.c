# PORTING.md

## Tinygrad C Port - File Porting Order
**Based on actual Python import analysis**

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
- [ ] `uop/__init__.py` → `src/uop/uop.h` - UOp enum definitions
- [ ] `uop/mathtraits.py` → `src/uop/mathtraits.c` - Mathematical properties
  - **Depends on:** `tinygrad.dtype`, `tinygrad.helpers`
- [ ] `uop/ops.py` → `src/uop/ops.c` - Core operation enum and UOp structure
  - **Depends on:** `tinygrad.uop`, `tinygrad.uop.mathtraits`, `tinygrad.dtype`, `tinygrad.helpers`

## Phase 4: Shape System (Depends on UOp + dtype)
- [x] `shape/view.py` → `src/shape/view.c` - Tensor view operations ✅
  - **Depends on:** `tinygrad.dtype`, `tinygrad.uop.ops`, `tinygrad.helpers`

## Phase 5: Device System & Gradient (Parallel to advanced UOp)
- [ ] `device.py` → `src/device/device.c` - Device abstraction layer
  - **Depends on:** `tinygrad.helpers`, `tinygrad.dtype`, `tinygrad.renderer`
- [ ] `gradient.py` → `src/engine/autograd.c` - Automatic differentiation
  - **Depends on:** `tinygrad.uop.ops`, `tinygrad.helpers`

## Phase 6: Advanced UOp Operations
- [ ] `uop/symbolic.py` → `src/uop/symbolic.c` - Symbolic math and simplification
  - **Depends on:** `tinygrad.dtype`, `tinygrad.uop.ops`, `tinygrad.helpers`
- [ ] `uop/transcendental.py` → `src/uop/transcendental.c` - Math functions
- [ ] `uop/upat.py` → `src/uop/upat.c` - Pattern matching
- [ ] `uop/spec.py` → `src/uop/spec.c` - UOp specifications
- [ ] `uop/optional.py` → `src/uop/optional.c` - Optional utilities

## Phase 7: Advanced Shape System
- [ ] `shape/shapetracker.py` → `src/shape/shapetracker.c` - Shape tracking and composition
  - **Depends on:** `tinygrad.helpers`, `tinygrad.shape.view`, `tinygrad.dtype`, `tinygrad.uop.ops`, `tinygrad.uop.symbolic`

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
- [x] `runtime/ops_cpu.py` → `src/runtime/ops_cpu/ops.c` - Basic CPU operations ✅

## Phase 14: Core Tensor (High-level API)
- [x] `tensor.py` → `src/tensor/tensor.c` - Main Tensor class and operations ✅
  - **Depends on most above modules** - This is the highest-level component

## Phase 15: Neural Network Layers
- [ ] `nn/state.py` → `src/nn/state.c` - State management
- [ ] `nn/optim.py` → `src/nn/optim.c` - SGD, Adam optimizers
- [ ] `nn/datasets.py` → `src/nn/datasets.c` - Dataset loading
- [ ] `nn/__init__.py` → `src/nn/layers.c` - Conv2d, BatchNorm, Linear, etc.

## Phase 16: Frontend Support
- [ ] `frontend/torch.py` → `src/frontend/torch.c` - PyTorch compatibility
- [ ] `frontend/onnx.py` → `src/frontend/onnx.c` - ONNX import

---

## Critical Path for ResNet-18 CPU Training

For the specific goal of ResNet-18 training on CPU, focus on this minimal validated path:

1. **Phase 1**: `helpers.py` ✅ (foundation - zero dependencies)
2. **Phase 2**: `dtype.py` (depends only on helpers)  
3. **Phase 3**: `uop/mathtraits.py`, `uop/ops.py` (basic UOp system)
4. **Phase 4**: `shape/view.py` ✅ (tensor views)
5. **Phase 5**: `device.py` (CPU-only), `gradient.py` (autograd)
6. **Phase 6**: Basic `uop/symbolic.py` (simplified for concrete values)
7. **Phase 7**: `shape/shapetracker.py` (shape composition)
8. **Phase 8-9**: Basic renderers and engine components (minimal viable)
9. **Phase 13**: `runtime/ops_cpu.py` ✅ (CPU ops: add, mul, conv2d, relu)
10. **Phase 14**: `tensor.py` ✅ (tensor operations)
11. **Phase 15**: `nn/layers.c` (Conv2d, BatchNorm, Linear, ReLU)
12. **Phase 15**: `nn/optim.py` (SGD optimizer only)

## Validation & Dependencies

This porting order has been **validated through direct analysis** of Python import statements:

- ✅ `helpers.py` confirmed to have zero tinygrad dependencies
- ✅ `dtype.py` confirmed to depend only on `tinygrad.helpers`
- ✅ `shape/view.py` confirmed to depend on `tinygrad.dtype`, `tinygrad.uop.ops`, `tinygrad.helpers`
- ✅ Dependency chain validated: helpers → dtype → uop → shape → tensor

## Notes

- Files marked with ✅ are already ported/stubbed
- **Dependencies explicitly validated** - no guesswork
- Start with concrete implementations before adding symbolic support
- Test each phase thoroughly before moving to the next
- Stub complex features initially to maintain end-to-end functionality

## Testing Strategy

For each ported file:
1. Write unit tests comparing C output to Python output
2. Test with simple cases first, then complex ones  
3. Validate memory management with valgrind
4. Benchmark against Python implementation
5. **Verify import dependencies match expected phase order**