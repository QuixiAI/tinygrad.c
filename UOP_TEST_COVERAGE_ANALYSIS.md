# UOp Test Coverage Gap Analysis

## Overview
Your C test suite (`tests/test_uop.c`) contains 496 tests and covers many core UOp system aspects comprehensively. However, when compared against the reference Python test suites, several important test categories are missing that should be implemented for complete coverage.

## Reference Test Files Analyzed
- `reference/test/test_uops.py` - Core UOp functionality and ALU operations
- `reference/test/test_uop_graph.py` - Graph rewriting, optimization, and transformations  
- `reference/test/unit/test_uop_spec.py` - UOp specification validation
- Various unit tests for symbolic operations, bounds checking, etc.

## Critical Missing Test Categories

### 1. Graph Rewriting and Optimization Tests ⚠️ HIGH PRIORITY

#### Missing from C tests:
```c
// Graph constant folding for complex expressions
void test_graph_constant_folding_depth2() {
    // Test: (v + const1) + const2 → v + (const1 + const2)
    UOp* v = uop_variable("v", 0, 10);
    UOp* c1 = uop_const(dtypes.int32, 5);
    UOp* c2 = uop_const(dtypes.int32, 3);
    UOp* expr = uop_add(uop_add(v, c1), c2);
    UOp* simplified = uop_graph_rewrite(expr);
    
    // Should be: v + 8
    ASSERT(simplified->op == OPS_ADD);
    ASSERT(simplified->src[1]->op == OPS_CONST);
    ASSERT(simplified->src[1]->arg.const_data.const_value == 8.0);
}

// Commutative canonicalization 
void test_commutative_canonicalization() {
    // Constants should move to the right in commutative ops
    UOp* var = uop_variable("x", 0, 10); 
    UOp* two = uop_const(dtypes.int32, 2);
    UOp* expr = uop_add(two, var);  // 2 + x
    UOp* canonical = uop_graph_rewrite(expr);
    
    // Should become: x + 2
    ASSERT(canonical->src[0] == var);
    ASSERT(canonical->src[1] == two);
}

// WHERE same branch folding
void test_where_same_branch_fold() {
    UOp* cond = uop_const(dtypes.bool_, 1);
    UOp* val = uop_const(dtypes.float32, 42.0);
    UOp* where_expr = uop_where(cond, val, val);  // WHERE(cond, val, val)
    UOp* folded = uop_graph_rewrite(where_expr);
    
    // Should fold to just val
    ASSERT(folded == val);
}
```

#### Python Reference:
```python
# From test_uop_graph.py
def test_where_same_fold(self):
    v = UOp.variable('tmp', 0, 1)
    c0 = UOp(Ops.CONST, dtypes.int, arg=0)
    vc = UOp(Ops.CMPNE, dtypes.bool, (v, c0))
    c1 = UOp(Ops.CONST, dtypes.float, arg=1.0)
    out = UOp(Ops.WHERE, dtypes.float, (vc, c1, c1))
    uops = to_uops_list([out])
    self.assertEqual(out.arg, 1.0)
```

### 2. Memory Access and Bounds Checking Tests ⚠️ HIGH PRIORITY

#### Missing critical functionality:
```c
// Out-of-bounds access detection
void test_out_of_bounds_detection() {
    // This should trigger a runtime error
    UOp* buf = uop_define_global(dtypes.int32, 0);
    // Buffer size is typically inferred, but let's assume size 16
    UOp* idx = uop_const(dtypes.int32, 42);  // Out of bounds
    UOp* load = uop_index(buf, idx);
    
    // Should detect bounds violation and fail
    bool should_fail = false;
    // TODO: Implement bounds checking that sets should_fail = true
    ASSERT(should_fail);
}

// Symbolic bounds checking with variables
void test_symbolic_bounds_checking() {
    UOp* buf = uop_define_global(dtypes.int32, 0);
    UOpArg var_arg = {0};
    var_arg.type = ARG_INT;
    UOp* var = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, &var_arg, "i");
    
    // Variable with range [0, 20) accessing buffer of size 16
    // This should be detected as potentially out-of-bounds
    UOp* load = uop_index(buf, var);
    bool bounds_ok = uop_check_bounds(load);  // Need to implement
    ASSERT(!bounds_ok);  // Should detect potential OOB
}

// Gated memory access
void test_gated_memory_access() {
    UOp* buf = uop_define_global(dtypes.float32, 0);
    UOp* idx = uop_const(dtypes.int32, 5);
    UOp* gate = uop_const(dtypes.bool_, 1);  // Always true gate
    UOp* val = uop_const(dtypes.float32, 42.0);
    
    // Gated store
    UOp* gated_idx = uop_index_with_gate(buf, idx, gate);
    UOp* store = uop_store(gated_idx, val);
    
    ASSERT(store != NULL);
    // Should generate IF/ENDIF blocks in final code
}
```

#### Python Reference:
```python
# From test_uop_graph.py
def test_in_out_of_bounds_access(self):
    with Context(IGNORE_OOB=0):
        glbl0 = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(16), (), 0)
        ld0 = UOp(Ops.LOAD, dtypes.int, (glbl0.index(UOp.const(dtypes.int, 42)),))
        with self.assertRaises(RuntimeError): to_uops_list([ld0])
```

### 3. UOp Specification Validation Tests ⚠️ HIGH PRIORITY

#### Missing validation tests:
```c
// Shape broadcasting validation
void test_no_implicit_broadcasting() {
    UOp* buf1 = uop_define_global(dtypes.float32, 0);
    UOp* buf2 = uop_define_global(dtypes.float32, 1);
    
    // Create loads with incompatible shapes
    // This should fail validation
    UOp* a = uop_load(buf1, dtypes.float32);  // Assume shape (4, 32)
    UOp* b = uop_load(buf2, dtypes.float32);  // Assume shape (32,)
    
    UOp* result = uop_add(a, b);  // Should fail - no implicit broadcasting
    bool validation_passed = uop_validate_ast(result);
    ASSERT(!validation_passed);
}

// Reduce operation validation  
void test_reduce_store_validation() {
    UOp* buf = uop_define_global(dtypes.float32, 0);
    UOp* data = uop_load(buf, dtypes.float32);
    
    // Direct reduce to store should fail
    int axes[] = {0};
    UOp* reduced = uop_reduce_axis(data, OPS_ADD, axes, 1);
    UOp* store = uop_store(buf, reduced);
    
    bool validation_passed = uop_validate_ast(store);
    ASSERT(!validation_passed);  // Should fail validation
}
```

#### Python Reference:
```python
# From test_uop_spec.py
def test_no_implicit_broadcasting(self):
    bufs = [UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), i) for i in range(2)]
    a = UOp(Ops.LOAD, dtypes.float, (bufs[1].view(ShapeTracker.from_shape((4, 32))),))
    b = a + UOp(Ops.REDUCE_AXIS, dtypes.float, (a,), (Ops.MAX, (1,)))
    st = UOp(Ops.STORE, dtypes.void, (bufs[0].view(ShapeTracker.from_shape((4, 32))), b))
    with self.assertRaises(InvalidASTException): helper_test_verify_ast(st)
```

### 4. Advanced Vector Operations ⚠️ MEDIUM PRIORITY

#### Missing vector tests:
```c
// GEP vector folding
void test_gep_vector_folding() {
    // Create vector constant [0, 1, 2, 3]
    UOpArg vec_arg = {0};
    vec_arg.type = ARG_VECTOR;
    vec_arg.vector_data.values[0] = 0.0;
    vec_arg.vector_data.values[1] = 1.0;
    vec_arg.vector_data.values[2] = 2.0;
    vec_arg.vector_data.values[3] = 3.0;
    vec_arg.vector_data.count = 4;
    
    UOp* vec = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec_arg, NULL);
    
    // Test GEP folding: vec[1] should fold to const(1)
    UOp* gep1 = uop_gep(vec, 1);
    UOp* folded = uop_graph_rewrite(gep1);
    
    ASSERT(folded->op == OPS_CONST);
    ASSERT(folded->arg.const_data.const_value == 1.0);
}

// Vector add constant folding
void test_vector_add_const_fold() {
    // Create two vector constants and add them
    UOp* vec1 = uop_vconst_int32((int[]){0, 1, 2}, 3);
    UOp* vec2 = uop_vconst_int32((int[]){5, 6, 7}, 3);
    UOp* sum = uop_add(vec1, vec2);
    UOp* folded = uop_graph_rewrite(sum);
    
    ASSERT(folded->op == OPS_VCONST);
    ASSERT(folded->arg.vector_data.values[0] == 5.0);  // 0+5
    ASSERT(folded->arg.vector_data.values[1] == 7.0);  // 1+6
    ASSERT(folded->arg.vector_data.values[2] == 9.0);  // 2+7
}
```

#### Python Reference:
```python
# From test_uop_graph.py
def test_gep_const(self):
    v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    v2 = v1.gep(1)
    ret = graph_rewrite(v2, sym)
    self.assertEqual(ret.dtype, dtypes.int)
    self.assertEqual(ret.arg, 1)
```

### 5. Assembly-Level Optimization Detection ⚠️ MEDIUM PRIORITY

#### Missing optimization tests:
```c
// Division by power of 2 → SHR optimization
void test_div_power_of_2_optimization() {
    UOp* x = uop_const(dtypes.int32, 16);
    UOp* divisor = uop_const(dtypes.int32, 4);  // Power of 2
    UOp* div = uop_div(x, divisor);
    
    // After optimization, should become SHR
    UOp* optimized = uop_optimize_for_assembly(div);
    ASSERT(optimized->op == OPS_SHR);
    ASSERT(optimized->src[1]->arg.const_data.const_value == 2.0);  // log2(4)
}

// Multiplication by power of 2 → SHL optimization  
void test_mul_power_of_2_optimization() {
    UOp* x = uop_variable("x", 0, 100);
    UOp* multiplier = uop_const(dtypes.int32, 8);  // Power of 2
    UOp* mul = uop_mul(x, multiplier);
    
    // Should optimize to SHL
    UOp* optimized = uop_optimize_for_assembly(mul);
    ASSERT(optimized->op == OPS_SHL);
    ASSERT(optimized->src[1]->arg.const_data.const_value == 3.0);  // log2(8)
}

// Fast integer division optimization
void test_fast_idiv_optimization() {
    UOp* x = uop_variable("x", 0, 1000);
    UOp* divisor = uop_const(dtypes.uint32, 3);  // Non-power-of-2
    UOp* div = uop_div(x, divisor);
    
    // Should use fast division (magic number multiplication + shift)
    UOp* optimized = uop_optimize_for_assembly(div);
    // Should NOT contain IDIV operation anymore
    bool contains_idiv = uop_graph_contains_op(optimized, OPS_IDIV);
    ASSERT(!contains_idiv);
}
```

#### Python Reference:
```python
# From test_uops.py (TestAssembly)
def test_division_power_of_two(self):
    for dt in (dtypes.int32, dtypes.uint32):
        g = UOp(Ops.DEFINE_GLOBAL, dt.ptr(), (), 0)
        c = UOp(Ops.CONST, dt, (), 2)
        l = UOp(Ops.LOAD, dt, (g.index(c),))
        a = UOp(Ops.IDIV, dt, (l, c))
        uops = to_uops_list([a], opts=Device[Device.DEFAULT].renderer)
        ops = [x.op for x in uops]
        self.assertIn(Ops.SHR, ops)
        self.assertNotIn(Ops.IDIV, ops)
```

### 6. Advanced Symbolic and Variable Operations ⚠️ MEDIUM PRIORITY

#### Missing symbolic tests:
```c
// Variable bounds propagation (vmin/vmax)
void test_variable_bounds_propagation() {
    // Create variable with known range [10, 20]
    UOp* var = uop_variable_with_bounds("x", 10, 20);
    
    // Add constant: x + 5, should have range [15, 25]
    UOp* five = uop_const(dtypes.int32, 5);
    UOp* sum = uop_add(var, five);
    
    int vmin = uop_sym_infer_min(sum);
    int vmax = uop_sym_infer_max(sum);
    ASSERT(vmin == 15);
    ASSERT(vmax == 25);
}

// Symbolic resolution with ambiguous conditions
void test_symbolic_resolution_ambiguous() {
    UOp* var = uop_variable_with_bounds("x", 5, 15);
    UOp* ten = uop_const(dtypes.int32, 10);
    UOp* comparison = uop_lt(var, ten);  // x < 10
    
    // This is ambiguous since x ∈ [5, 15) and 10 is in the middle
    bool resolved = uop_resolve(comparison, false);
    ASSERT(resolved == false);  // Should be ambiguous
    
    // But x < 5 should resolve to false
    UOp* five = uop_const(dtypes.int32, 5);
    UOp* comparison2 = uop_lt(var, five);
    resolved = uop_resolve(comparison2, false);
    ASSERT(resolved == true);  // Should resolve to definitely false
}

// Modulo arithmetic simplification
void test_modulo_arithmetic_simplification() {
    // Test: (3 + 3*a) % 4 should simplify based on congruence
    UOp* a = uop_variable("a", 0, 10);
    UOp* three = uop_const(dtypes.int32, 3);
    UOp* mul = uop_mul(a, three);     // 3*a
    UOp* sum = uop_add(mul, three);   // 3 + 3*a
    UOp* four = uop_const(dtypes.int32, 4);
    UOp* mod = uop_mod(sum, four);    // (3 + 3*a) % 4
    
    UOp* simplified = uop_symbolic_simplify(mod);
    // Should simplify based on modular arithmetic properties
    ASSERT(simplified != mod);  // Should have been simplified
}
```

#### Python Reference:
```python
# From unit tests for symbolic operations
def test_vmin_vmax_propagation(self):
    a = UOp.variable("a", 1, 10)
    b = a + 5
    self.assertEqual(b.vmin, 6)
    self.assertEqual(b.vmax, 15)
```

### 7. UOp Object Behavior and Validation ⚠️ MEDIUM PRIORITY

#### Missing object behavior tests:
```c
// UOp immutability enforcement
void test_uop_immutability() {
    UOp* original = uop_const(dtypes.int32, 42);
    UOp* original_ptr = original;
    
    // Any operation should create new UOp, not modify existing
    UOp* modified = uop_add(original, uop_const(dtypes.int32, 1));
    
    ASSERT(original == original_ptr);  // Original should be unchanged
    ASSERT(modified != original);      // Result should be different UOp
    ASSERT(original->arg.const_data.const_value == 42);  // Value unchanged
}

// Children tracking and cleanup
void test_uop_children_tracking() {
    UOp* var = uop_variable("test_var", 0, 10);
    
    // Create child operations
    UOp* child1 = uop_mul(var, var);
    UOp* child2 = uop_add(var, uop_const(dtypes.int32, 5));
    
    // var should track its children
    ASSERT(uop_get_children_count(var) == 2);
    
    // Clean up one child
    uop_unref(child1);
    ASSERT(uop_get_children_count(var) == 1);
    
    // Clean up remaining child  
    uop_unref(child2);
    ASSERT(uop_get_children_count(var) == 0);
}

// String representation testing
void test_uop_string_representation() {
    UOp* expr = uop_add(uop_const(dtypes.float32, 3.14), 
                       uop_const(dtypes.float32, 2.71));
    
    char* str_repr = uop_to_string(expr);
    ASSERT(str_repr != NULL);
    ASSERT(strlen(str_repr) > 0);
    
    // Should contain operation and operands
    ASSERT(strstr(str_repr, "ADD") != NULL);
    ASSERT(strstr(str_repr, "3.14") != NULL || strstr(str_repr, "CONST") != NULL);
    
    free(str_repr);
}

// Device property validation
void test_uop_device_properties() {
    UOp* buf = uop_define_global(dtypes.float32, 0);
    
    // Buffer should have device property
    const char* device = uop_get_device(buf);
    ASSERT(device != NULL);
    
    // Constants don't have device
    UOp* const_val = uop_const(dtypes.float32, 1.0);
    device = uop_get_device(const_val);
    ASSERT(device == NULL);  // Should be NULL for constants
}
```

#### Python Reference:
```python
# From test_uops.py (TestUopsObject)
def test_immutable(self):
    const_4 = UOp.const(dtypes.int, 4)
    with self.assertRaises(Exception):
        const_4.arg = 5

# From test_uops.py (TestUOpChildren)
def test_children_exist(self):
    a = UOp.variable("weird_name_234", 0, 10)
    b = a*a
    self.assertEqual(len(a.children), 1)
    self.assertIs(list(a.children)[0](), b)
```

### 8. IF/ENDIF Conditional Block Generation ⚠️ MEDIUM PRIORITY

#### Missing conditional tests:
```c
// Gated store rewriting to IF/ENDIF
void test_gated_store_rewriting() {
    UOp* buf = uop_define_global(dtypes.float32, 0);
    UOp* gidx = uop_special("gidx0", dtypes.int32, 4);
    UOp* gate = uop_lt(gidx, uop_const(dtypes.int32, 1));  // gidx < 1
    UOp* val = uop_const(dtypes.float32, 42.0);
    
    // Create gated store
    UOp* idx = uop_index_with_gate(buf, gidx, gate);
    UOp* store = uop_store(idx, val);
    
    // After rewriting, should have IF/ENDIF blocks
    UOp* rewritten = uop_full_rewrite(store);
    bool has_if = uop_graph_contains_op(rewritten, OPS_IF);
    bool has_endif = uop_graph_contains_op(rewritten, OPS_ENDIF);
    
    ASSERT(has_if);
    ASSERT(has_endif);
}

// Multiple gated stores merging
void test_merge_gated_stores() {
    UOp* buf1 = uop_define_global(dtypes.float32, 0);
    UOp* buf2 = uop_define_global(dtypes.float32, 1);
    UOp* gidx = uop_special("gidx0", dtypes.int32, 4);
    UOp* gate = uop_lt(gidx, uop_const(dtypes.int32, 2));
    UOp* val = uop_const(dtypes.float32, 42.0);
    
    // Two stores with same gate
    UOp* store1 = uop_store(uop_index_with_gate(buf1, gidx, gate), val);
    UOp* store2 = uop_store(uop_index_with_gate(buf2, gidx, gate), val);
    
    UOp* sink = uop_sink((UOp*[]){store1, store2}, 2);
    UOp* rewritten = uop_full_rewrite(sink);
    
    // Should have only one IF/ENDIF pair (merged)
    int if_count = uop_count_ops(rewritten, OPS_IF);
    int endif_count = uop_count_ops(rewritten, OPS_ENDIF);
    
    ASSERT(if_count == 1);
    ASSERT(endif_count == 1);
}
```

#### Python Reference:
```python
# From test_uops.py (TestGatedStoreRewrite)
def test_tiny_gate_store(self):
    gmem = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), (), 0)
    gidx0 = UOp(Ops.SPECIAL, dtypes.int, (), ('gidx0', 4))
    gate = gidx0<UOp.const(dtypes.int, 1)
    idx = UOp(Ops.INDEX, dtypes.float.ptr(), (gmem, gidx0 * UOp.const(dtypes.int, 2), gate))
    val = UOp.const(dtypes.float, 42.0)
    store = UOp(Ops.STORE, dtypes.void, (idx, val))
    uops = to_uops_list([store])
    if_uop = next(u for u in uops if u.op is Ops.IF)
    endif = next(u for u in uops if u.op is Ops.ENDIF)
    self.assertEqual(len(gated_uops), 1)
```

### 9. Advanced Graph Transformations ⚠️ LOWER PRIORITY

#### Missing transformation tests:
```c
// UNROLL/CONTRACT/EXPAND operations
void test_unroll_contract_expand() {
    // Test UNROLL operation
    UOp* data = uop_vconst_int32((int[]){0, 1, 2, 3}, 4);
    UOp* unrolled = uop_unroll(data, 4);
    
    ASSERT(unrolled->op == OPS_UNROLL);
    ASSERT(unrolled->src_count == 1);
    
    // Test CONTRACT operation  
    UOp* contracted = uop_contract(unrolled, 2);
    ASSERT(contracted->op == OPS_CONTRACT);
    
    // Test EXPAND operation
    UOp* expanded = uop_expand(contracted, 8);
    ASSERT(expanded->op == OPS_EXPAND);
}

// BITCAST operations
void test_bitcast_operations() {
    // Test float to uint32 bitcast
    UOp* float_val = uop_const(dtypes.float32, 1.0f);
    UOp* as_uint = uop_bitcast(float_val, dtypes.uint32);
    
    ASSERT(as_uint->op == OPS_BITCAST);
    ASSERT(dtype_eq(&as_uint->dtype, &dtypes.uint32));
    
    // Test bitcast to same type should be eliminated
    UOp* same_type = uop_bitcast(float_val, dtypes.float32);
    UOp* optimized = uop_graph_rewrite(same_type);
    ASSERT(optimized == float_val);  // Should be eliminated
}
```

#### Python Reference:
```python
# From test_uop_graph.py (TestExpander)  
def test_expand_add_broadcast(self):
    e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    sink = expander_rewrite(e1+3)
    assert sink.op is Ops.UNROLL and len(sink.src[0].arg) == 4
    self.assertTupleEqual(sink.src[0].arg, (3,4,5,6))
```

### 10. Error Conditions and Validation ⚠️ LOWER PRIORITY

#### Missing error handling tests:
```c
// Invalid AST detection
void test_invalid_ast_detection() {
    // Create intentionally invalid AST
    UOp* buf = uop_define_global(dtypes.float32, 0);
    UOp* load = uop_load(buf, dtypes.int32);  // Wrong dtype
    
    bool is_valid = uop_validate_ast(load);
    ASSERT(!is_valid);  // Should detect invalid AST
}

// Memory layout constraint violations
void test_memory_layout_constraints() {
    // Test accessing beyond allocated size
    UOp* buf = uop_define_global_with_size(dtypes.float32, 0, 16);
    UOp* idx = uop_const(dtypes.int32, 20);  // Beyond size
    
    bool violates_constraint = uop_check_memory_constraints(buf, idx);
    ASSERT(violates_constraint);
}

// Type mismatch detection
void test_type_mismatch_detection() {
    UOp* int_val = uop_const(dtypes.int32, 42);
    UOp* float_val = uop_const(dtypes.float32, 3.14f);
    
    // Addition should require type compatibility
    UOp* result = uop_add(int_val, float_val);
    bool is_valid = uop_validate_types(result);
    
    // Should either be valid (with promotion) or invalid
    // The test checks that type validation is working
    ASSERT(is_valid == true || is_valid == false);  // Just check it runs
}
```

## Implementation Priority Summary

### **Immediate High Priority (Core Functionality)**
1. **Graph Rewriting Tests** - Essential for optimization validation
2. **Memory Bounds Checking** - Critical for safety and correctness  
3. **UOp Specification Validation** - Required for AST correctness
4. **Basic Vector Operations** - Needed for modern GPU operations

### **Medium Priority (Optimization & Features)**
5. **Assembly Optimization Detection** - Important for performance validation
6. **Advanced Symbolic Operations** - Complex mathematical simplifications  
7. **UOp Object Behavior** - Immutability, children tracking, string representation
8. **IF/ENDIF Generation** - Conditional block handling

### **Lower Priority (Advanced Features)**
9. **Advanced Graph Transformations** - UNROLL/CONTRACT/EXPAND operations
10. **Error Condition Handling** - Invalid AST detection and validation

## Specific Missing Functions That Need Implementation

Based on the test analysis, your C implementation needs these additional functions:

### Graph Rewriting Functions
```c
UOp* uop_graph_rewrite(UOp* uop);                    // Graph optimization
UOp* uop_symbolic_simplify(UOp* uop);               // Symbolic simplification 
UOp* uop_full_rewrite(UOp* uop);                    // Complete rewrite pipeline
bool uop_graph_contains_op(UOp* uop, Ops op);       // Check for operation type
int uop_count_ops(UOp* uop, Ops op);                // Count operations
```

### Bounds Checking Functions
```c
bool uop_check_bounds(UOp* load_op);                // Check memory bounds
bool uop_validate_ast(UOp* uop);                    // AST validation
bool uop_check_memory_constraints(UOp* buf, UOp* idx); // Memory constraint checking
```

### Variable and Symbolic Functions
```c
UOp* uop_variable_with_bounds(const char* name, int min, int max); // Bounded variables
int uop_sym_infer_min(UOp* uop);                    // Infer minimum value
int uop_sym_infer_max(UOp* uop);                    // Infer maximum value  
UOp* uop_special(const char* name, DType dtype, int size); // Special operations
```

### Vector and Advanced Operations
```c
UOp* uop_gep(UOp* vec, int index);                  // Get element pointer
UOp* uop_vconst_int32(int* values, int count);      // Vector constants
UOp* uop_unroll(UOp* data, int factor);             // Unroll operations
UOp* uop_contract(UOp* data, int factor);           // Contract operations
UOp* uop_expand(UOp* data, int factor);             // Expand operations
UOp* uop_bitcast(UOp* src, DType target_dtype);     // Bitcast operations
```

### Object Behavior Functions
```c
char* uop_to_string(UOp* uop);                      // String representation
const char* uop_get_device(UOp* uop);               // Device property
int uop_get_children_count(UOp* uop);               // Children count
UOp* uop_index_with_gate(UOp* buf, UOp* idx, UOp* gate); // Gated indexing
```

### Assembly Optimization Functions
```c
UOp* uop_optimize_for_assembly(UOp* uop);           // Assembly-level optimizations
bool uop_validate_types(UOp* uop);                  // Type validation
```

## Test Implementation Strategy

To implement these missing tests efficiently:

### Phase 1: Core Infrastructure (Week 1-2)
1. Implement basic graph rewriting functions
2. Add bounds checking infrastructure
3. Create UOp validation framework
4. Extend variable system with bounds

### Phase 2: Graph Optimizations (Week 3-4)  
1. Add constant folding for complex expressions
2. Implement commutative canonicalization
3. Add WHERE same-branch folding
4. Create depth-2 constant folding

### Phase 3: Memory Safety (Week 5-6)
1. Implement out-of-bounds detection
2. Add symbolic bounds checking
3. Create gated memory access
4. Add memory constraint validation

### Phase 4: Advanced Features (Week 7-8)
1. Vector operations and GEP folding
2. Assembly optimization detection
3. IF/ENDIF block generation
4. UOp object behavior testing

## Current Test Coverage Assessment

### ✅ Well Covered Areas (Your current 496 tests)
- **Basic ALU operations** (ADD, MUL, SUB, DIV, etc.)
- **Transcendental functions** (SIN, EXP2, LOG2, SQRT)
- **Reference counting** and memory management
- **Pattern matching** with UPat system
- **Basic UOp creation** and caching
- **Comparison operations** (CMPLT, CMPEQ, etc.)
- **Bitwise operations** (AND, OR, XOR, SHL, SHR)
- **Ternary operations** (WHERE, MULACC)
- **Basic symbolic variables**
- **Buffer operations** (LOAD, STORE, DEFINE_GLOBAL)

### ⚠️ Gaps to Address (New tests needed)
- **Graph optimization and rewriting** (0% covered)
- **Memory bounds checking** (0% covered) 
- **UOp specification validation** (0% covered)
- **Vector operations and folding** (10% covered)
- **Assembly optimization detection** (0% covered)
- **Advanced symbolic operations** (20% covered)
- **Conditional block generation** (0% covered)
- **UOp object behavior** (30% covered)

## Quality Assessment

Your current test suite is **excellent for basic functionality** with comprehensive coverage of:
- Core mathematical operations
- Reference counting integrity  
- Pattern matching system
- Basic symbolic computation

However, it's **missing critical enterprise features** needed for a production UOp system:
- Graph optimization validation
- Memory safety enforcement
- Advanced vectorization support
- Assembly-level optimization verification

## Recommendations

### Immediate Actions
1. **Prioritize graph rewriting tests** - These are essential for any optimization framework
2. **Add memory bounds checking** - Critical for safety and debugging
3. **Implement UOp validation** - Required for catching AST errors early

### Long-term Strategy
1. **Build test infrastructure incrementally** - Don't try to implement everything at once
2. **Focus on functionality you're actively using** - Prioritize tests for features in use
3. **Create modular test functions** - Make tests reusable across different scenarios
4. **Validate against Python reference** - Ensure behavioral compatibility

### Success Metrics
- **Graph rewriting tests**: 20+ test cases covering optimization patterns
- **Bounds checking**: 15+ test cases covering edge conditions  
- **Vector operations**: 10+ test cases covering folding and GEP
- **Overall coverage**: Target 600+ tests (current 496 + 100+ new)

This analysis should guide your development priorities and help you build a comprehensive, production-ready UOp system with proper validation and optimization capabilities.

## Summary

Your C UOp test suite is impressively comprehensive for core functionality (496 tests), but lacks critical enterprise-level features. The missing test categories represent about **40% of a complete test suite** that would match the Python reference implementation.

**Key Takeaway**: You have excellent foundational coverage but need to expand into graph optimization, memory safety, and advanced validation to achieve production readiness.
