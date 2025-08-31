# PORTING.md

## Tinygrad C Port - File Porting Order

This document outlines the recommended order for porting tinygrad Python files to C, organized by dependency levels.

## Overview

The porting process is organized into phases, with each phase building on the previous ones. Files within a phase can potentially be worked on in parallel, but all files from earlier phases should be completed first.

## Phase 1: Core Foundation
These files have no dependencies and form the base of the system.

- [ ] `helpers.py` → `src/helpers/helpers.c` - Utility functions (prod, getenv, etc.)
- [x] `dtype.py` → `src/dtype/dtype.c` - Data type definitions
- [ ] `uop/ops.py` → `src/uop/ops.c` - Core operation enum and UOp structure

## Phase 2: Shape System
Depends on: Phase 1

- [x] `shape/view.py` → `src/shape/view.c` - Tensor view operations
- [ ] `shape/shapetracker.py` → `src/shape/shapetracker.c` - Shape tracking and composition

## Phase 3: Symbolic System (UOps)
Depends on: Phases 1-2

- [ ] `uop/symbolic.py` → `src/uop/symbolic.c` - Symbolic simplification
- [ ] `uop/mathtraits.py` → `src/uop/mathtraits.c` - Mathematical properties
- [ ] `uop/upat.py` → `src/uop/upat.c` - Pattern matching
- [ ] `uop/spec.py` → `src/uop/spec.c` - UOp specifications
- [ ] `uop/optional.py` → `src/uop/optional.c` - Optional utilities
- [ ] `uop/transcendental.py` → `src/uop/transcendental.c` - Transcendental functions

## Phase 4: Device and Memory
Depends on: Phases 1-3

- [ ] `device.py` → `src/device/device.c` - Device abstraction layer
- [ ] `engine/memory.py` → `src/engine/memory.c` - Memory management

## Phase 5: Core Tensor
Depends on: Phases 1-4

- [ ] `tensor.py` → `src/tensor/tensor.c` - Main Tensor class
- [ ] `gradient.py` → `src/engine/autograd.c` - Automatic differentiation
- [ ] `engine/realize.py` → `src/engine/realize.c` - Tensor realization
- [ ] `engine/schedule.py` → `src/engine/schedule.c` - Execution scheduling

## Phase 6: Code Generation Base
Depends on: Phases 1-5

- [ ] `codegen/linearize.py` → `src/codegen/linearize.c`
- [ ] `codegen/lowerer.py` → `src/codegen/lowerer.c`
- [ ] `codegen/expander.py` → `src/codegen/expander.c`
- [ ] `codegen/devectorizer.py` → `src/codegen/devectorizer.c`
- [ ] `codegen/gpudims.py` → `src/codegen/gpudims.c`
- [ ] `codegen/quantize.py` → `src/codegen/quantize.c`

## Phase 7: Optimizations
Depends on: Phase 6

- [ ] `codegen/opt/kernel.py` → `src/codegen/opt/kernel.c`
- [ ] `codegen/opt/heuristic.py` → `src/codegen/opt/heuristic.c`
- [ ] `codegen/opt/tc.py` → `src/codegen/opt/tc.c`
- [ ] `codegen/opt/swizzler.py` → `src/codegen/opt/swizzler.c`
- [ ] `codegen/opt/search.py` → `src/codegen/opt/search.c`

## Phase 8: Scheduling
Depends on: Phases 1-6

- [ ] `schedule/kernelize.py` → `src/schedule/kernelize.c`
- [ ] `schedule/grouper.py` → `src/schedule/grouper.c`
- [ ] `schedule/multi.py` → `src/schedule/multi.c`

## Phase 9: Runtime Backends
Depends on: Phases 1-5

### Priority (for CPU execution):
- [ ] `runtime/ops_cpu.py` → `src/runtime/ops_cpu/ops.c`

### Optional backends:
- [ ] `runtime/ops_cuda.py` → `src/runtime/ops_cuda/`
- [ ] `runtime/ops_metal.py` → `src/runtime/ops_metal/`
- [ ] `runtime/ops_webgpu.py` → `src/runtime/ops_webgpu/`

## Phase 10: Renderers
Depends on: Phases 1-6

- [ ] `renderer/cstyle.py` → `src/renderer/cstyle.c`
- [ ] `renderer/llvmir.py` → `src/renderer/llvmir.c`
- [ ] `renderer/ptx.py` → `src/renderer/ptx.c`
- [ ] `renderer/wgsl.py` → `src/renderer/wgsl.c`

## Phase 11: Neural Network
Depends on: Phases 1-5, 9

- [ ] `nn/__init__.py` → `src/nn/layers.c` - Conv2d, BatchNorm, Linear, etc.
- [ ] `nn/optim.py` → `src/nn/optim.c` - SGD, Adam optimizers
- [ ] `nn/state.py` → `src/nn/state.c` - State management
- [ ] `nn/datasets.py` → `src/nn/datasets.c` - Dataset loading

## Phase 12: Optional/Advanced
Depends on: Various phases

- [ ] `engine/jit.py` → `src/engine/jit.c` - JIT compilation
- [ ] `frontend/torch.py` → `src/frontend/torch.c` - PyTorch compatibility
- [ ] `frontend/onnx.py` → `src/frontend/onnx.c` - ONNX import

---

## Minimal Path for ResNet-18 CPU Training

For the specific goal of ResNet-18 training on CPU, focus on this critical path:

1. **Phase 1**: `helpers.py`, `dtype.py` ✓, `uop/ops.py`
2. **Phase 2**: `shape/view.py` ✓, `shape/shapetracker.py`
3. **Simplified Phase 3**: Basic UOp support (concrete values only)
4. **Minimal Phase 4**: `device.py` (CPU only)
5. **Minimal Phase 5**: `tensor.py` (core ops: add, mul, conv2d, etc.)
6. **Phase 5**: `gradient.py` (backward pass)
7. **Phase 9**: `runtime/ops_cpu.py`
8. **Phase 11**: `nn/__init__.py` (Conv2d, BatchNorm, Linear, ReLU)
9. **Phase 11**: `nn/optim.py` (SGD only)

## Notes

- Files marked with ✓ are already ported
- Start with concrete implementations before adding symbolic support
- Test each phase thoroughly before moving to the next
- Consider stubbing complex features initially to get end-to-end functionality

## Testing Strategy

For each ported file:
1. Write unit tests comparing C output to Python output
2. Test with simple cases first, then complex ones
3. Validate memory management with valgrind
4. Benchmark against Python implementation