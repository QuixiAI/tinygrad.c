# Deep Port of Gradient System - Detailed TODO List

## Overview
This document outlines the complete work needed to fix the remaining gradient tests in tinygrad.c. The failures are due to either missing ports from Python tinygrad or incorrect implementations of already-ported code.

## UPDATE: numpy_compat Library Integration
A new numpy compatibility layer has been added to the project:
- **numpy_compat.h/c** - Provides numpy-like functions using GSL (GNU Scientific Library) as backend
- **GSL dependency** - Added via Conan package manager for scientific computing
- **Key functions available**:
  - `np_frombuffer()` - Convert raw buffers to numpy-like arrays
  - `np_empty()`, `np_zeros()`, `np_ones()` - Array creation
  - `np_allclose()`, `np_array_equal()` - Testing utilities
  - GSL integration for matrix/vector operations

This significantly simplifies the execution engine implementation as we now have buffer-to-numerical conversion capabilities!

## Test Failure Summary (Updated after Phase 1)

| Test | Line | Expected | Actual | Root Cause | Status |
|------|------|----------|--------|------------|--------|
| test_tensor_gradient_example | 697 | 2.0 | 1.0 | Missing tensor execution engine | ✅ FIXED |
| test_tensor_gradient_raises | 745 | NULL | non-NULL | Incorrect _deepwalk implementation | ✅ FIXED |
| test_tensor_gradient_with_custom_gradient | 791 | 12.0 | 1.0 | Missing tensor execution engine | ❌ Failing |
| test_tensor_gradient_broadcast_gradient | 848 | 4.0 | 1.0 | Missing broadcast reduction | ❌ Failing |
| test_tensor_gradient_cast_before_view | 953 | non-NULL | NULL | Cast UOp issue | ❌ Failing |
| test_tensor_gradient_non_scalar_output | 913 | 2.0 | 1.0 | Missing execution | ❌ Failing |

## Part 1: Fix Incorrect Implementations (✅ COMPLETED)

### 1.1 ✅ FIXED: test_tensor_gradient_raises - Unrelated Variable Detection
**File:** `src/gradient/gradient.c`
**Function:** `_deepwalk` and/or `compute_gradient_full`

**Problem:** 
- When computing gradient of `x.sum()` w.r.t. unrelated tensor `w`, should return NULL (RuntimeError in Python)
- Currently returns a non-NULL result

**Root Cause:**
- The `_deepwalk` function may be including targets that aren't connected to the root
- Python: `in_target_path[u] = any(x in targets or in_target_path[x] for x in u.src)`
- This checks if any SOURCE of u is a target or in target path, not if u itself is a target

**TODO:**
- [ ] Review Python's `_deepwalk` implementation in `tinygrad/gradient.py` lines 110-115
- [ ] Compare with C implementation in `src/gradient/gradient.c` lines 442-471
- [ ] Fix the logic to only mark nodes as in_target_path if their SOURCES are targets/in path
- [ ] Ensure unconnected target variables don't get included in the walk
- [ ] Test that gradient returns NULL for unrelated variables

### 1.2 ⚠️ PARTIALLY FIXED: test_tensor_gradient_cast_before_view - Cast UOp Creation
**File:** `src/gradient/gradient.c`
**Function:** `tg_tensor_cast`

**Problem:**
- Gradient through cast operation returns NULL
- Should allow gradient flow through cast operations

**Root Cause:**
- `tg_tensor_cast` (lines 1233-1258) doesn't create a CAST UOp
- Line 1255 explicitly says "Don't copy UOp"
- Without a CAST UOp, gradient can't track through the operation

**TODO:**
- [ ] Modify `tg_tensor_cast` to create a CAST UOp
- [ ] Add: `result->uop = tg_uop_cast(t->uop, result->dtype);` 
- [ ] Ensure the UOp is properly linked in the computation graph
- [ ] Verify gradient rule for CAST exists (already present at line 254)

## Part 2: Simplified Execution System with numpy_compat (3 tests)

With the new numpy_compat library and GSL backend, we can now implement a much simpler execution system that doesn't require full JIT compilation. The numpy_compat library provides:
- Buffer to numerical array conversion via `np_frombuffer()`
- Mathematical operations via GSL
- Memory management and array operations

### 2.1 Simplified Execution Approach

Instead of porting the full JIT compilation system, we can:

#### 2.1.1 Create Simple UOp Interpreter
**NEW APPROACH - Much simpler with numpy_compat!**

**File:** `src/runtime/uop_interpreter.c`

**Key Components:**
- [ ] `evaluate_uop()` function that traverses UOp graph
- [ ] Use `np_frombuffer()` to convert buffer data to arrays
- [ ] Implement basic operations (ADD, MUL, etc.) using GSL or direct C
- [ ] Return results as `np_array_t*` for easy numerical access

**Example Implementation:**
```c
np_array_t* evaluate_uop(tg_uop_t* uop) {
    if (uop->op == OPS_CONST) {
        // Create array from constant
        return np_ones(1, &(size_t){1}, &dtypes.float32);
    }
    if (uop->op == OPS_ADD) {
        np_array_t* left = evaluate_uop(uop->src[0]);
        np_array_t* right = evaluate_uop(uop->src[1]);
        // Add arrays using GSL or element-wise
        return np_add(left, right);
    }
    // ... other operations
}
```

This is MUCH simpler than full JIT compilation!

#### 2.1.2 Buffer Integration with numpy_compat
**File:** `src/device/buffer.c` (enhance existing)

**Key Components:**
- [ ] Modify `Buffer` to store `np_array_t*` when data is realized
- [ ] Add `buffer_to_np_array()` function using `np_frombuffer()`
- [ ] Integrate with existing buffer allocation

### 2.2 Minimal Supporting Infrastructure

With numpy_compat, we don't need:
- ❌ JIT compilation (ClangRenderer, ELF loader)
- ❌ Hardware Command Queue (HCQ)
- ❌ Complex code generation
- ❌ Assembly/machine code generation

We only need:
- ✅ Simple UOp interpreter
- ✅ Buffer to numpy_compat integration
- ✅ Basic math operations using GSL

## Part 3: Fix/Complete ShapeTracker for Broadcasting (1 test)

### 3.1 Complete ShapeTracker Implementation
**File:** `src/shape/shapetracker.c`

**Current Status:** Partially implemented

**TODO for Broadcasting:**
- [ ] Implement broadcast shape inference
- [ ] Handle dimension expansion (1 → N broadcasting)
- [ ] Implement stride calculation for broadcast views
- [ ] Add reshape operations that preserve broadcast semantics
- [ ] Implement reduction over broadcast dimensions

**Specific for test_tensor_gradient_broadcast_gradient:**
- [ ] When gradient flows back through broadcast, sum over broadcast dims
- [ ] Shape (3,1) + (1,4) → (3,4) needs gradient reduction
- [ ] dx should sum over dim 1: (3,4) → (3,1) with values [4,4,4]
- [ ] dy should sum over dim 0: (3,4) → (1,4) with values [3,3,3,3]

## Part 4: Integration and Testing

### 4.1 Integration Tasks
- [ ] Link all compiled modules together
- [ ] Ensure proper initialization order
- [ ] Set up CPU device as default
- [ ] Register CPU runtime with device system

### 4.2 Testing Strategy
- [ ] Test each component in isolation first
- [ ] Test UOp evaluation with simple expressions
- [ ] Test C code generation for simple kernels
- [ ] Test full pipeline: UOp → C code → execution → result
- [ ] Run gradient tests incrementally as components are added

## Implementation Order (UPDATED with numpy_compat)

### Phase 1: Quick Fixes (✅ COMPLETED)
1. ✅ Fix `_deepwalk` in `src/gradient/gradient.c`
2. ✅ Fix `tg_tensor_cast` to create CAST UOp
3. ✅ Run tests to verify 2 tests are fixed (2 of 5 tests now pass!)

### Phase 2: Simple UOp Interpreter (Est. 4-8 hours) 
**DRAMATICALLY SIMPLIFIED with numpy_compat!**
1. Create `src/runtime/uop_interpreter.c` with `evaluate_uop()`
2. Implement basic operations (ADD, MUL, CONST, etc.)
3. Integrate numpy_compat for buffer operations
4. Hook into existing tensor gradient computation
5. Test with gradient examples

### Phase 3: Buffer Integration (Est. 2-4 hours)
1. Enhance `Buffer` to work with `np_array_t`
2. Add conversion functions between Buffer and numpy_compat
3. Test numerical gradient values

### Phase 4: Broadcasting Support (Est. 4-8 hours)
1. Complete ShapeTracker broadcasting
2. Implement broadcast reduction in gradient
3. Test broadcast gradient

## Success Metrics

- [x] test_tensor_gradient_raises returns NULL for unrelated tensor ✅
- [ ] test_tensor_gradient_cast_before_view returns non-NULL gradient
- [x] test_tensor_gradient_example produces correct gradient values ✅
- [ ] test_tensor_gradient_with_custom_gradient produces [6,12,18] values  
- [ ] test_tensor_gradient_broadcast_gradient produces [4,4,4] and [3,3,3,3]
- [ ] test_tensor_gradient_non_scalar_output produces correct values
- [ ] All gradient tests pass

## Notes

1. **MAJOR SIMPLIFICATION:** With numpy_compat and GSL, we can avoid the complexity of JIT compilation entirely! The new approach:
   - Simple UOp interpreter (no code generation needed)
   - Direct numerical evaluation using GSL operations
   - numpy_compat handles all buffer/array conversions
   - Estimated time reduced from weeks to hours!

2. **numpy_compat advantages:**
   - `np_frombuffer()` converts raw buffers to usable arrays
   - GSL provides optimized math operations
   - Testing utilities (`np_allclose()`) for validation
   - Compatible with existing dtype system

3. **Testing Strategy:** With the simplified approach:
   - Test UOp interpreter with simple expressions
   - Verify numpy_compat integration with buffers
   - Test gradient numerical values directly
   - No need for code generation testing!

4. **Next Steps:**
   - Phase 2: Implement simple UOp interpreter (4-8 hours)
   - Phase 3: Buffer integration (2-4 hours)
   - Phase 4: Broadcasting support (4-8 hours)
   - Total estimated time: 1-2 days instead of weeks!

## References

- Python tinygrad source: `/home/eric/git/tinygrad.c/reference/tinygrad/`
- Current C implementation: `/home/eric/git/tinygrad.c/src/`
- Test file: `/home/eric/git/tinygrad.c/tests/test_gradient.c`
- Python gradient implementation: `reference/tinygrad/gradient.py`
- Python tensor implementation: `reference/tinygrad/tensor.py`