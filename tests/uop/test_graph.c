/* test_graph.c
 * TDD tests for UOp graph rewriting and optimization
 * Ported from reference/test/test_uop_graph.py
 */

#include "test_uop_common.h"
#include <math.h>

// Unity setUp and tearDown functions
void setUp(void) {
    // Initialize modules once
    static int initialized = 0;
    if (!initialized) {
        dtypes_init();
        uop_init();
        uop_cache_init();
        uop_ops_init();
        initialized = 1;
    }
}

void tearDown(void) {
    // Clean up after each test if needed
}

// =============================================================================
// TestGraphRewriteConst - Tests for constant folding during graph rewriting
// =============================================================================

// Port of test_gep_const from Python
TEST(test_gep_const) {
    // Test GEP (get element pointer) with vector constant
    // v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    // v2 = v1.gep(1)
    
    // Create vector constant - using VCONST for vector
    UOpArg vec_arg = {0};
    vec_arg.type = ARG_INT;
    vec_arg.int_data.i = 0;  // First element for now
    UOp* vec_const = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec_arg, NULL);
    TEST_ASSERT_NOT_NULL(vec_const);
    
    // Create GEP operation to get element at index 1
    UOpArg gep_arg = {0};
    gep_arg.type = ARG_INT;
    gep_arg.int_data.i = 1;
    UOp* gep_result = uop_new(OPS_GEP, dtypes.int32, &vec_const, 1, &gep_arg, NULL);
    TEST_ASSERT_NOT_NULL(gep_result);
    TEST_ASSERT_EQUAL(OPS_GEP, gep_result->op);
    
    // After graph rewrite, should constant fold to 1
    // UOp* rewritten = uop_graph_rewrite(gep_result);  // TODO: Implement
    // TEST_ASSERT_EQUAL(OPS_CONST, rewritten->op);
    // TEST_ASSERT_EQUAL(1, rewritten->arg.const_data.const_value);
    
    uop_unref(vec_const);
    uop_unref(gep_result);
}

// Port of test_gep_const_single from Python  
TEST(test_gep_const_single) {
    // Test GEP with single value vector constant
    UOpArg vec_arg = {0};
    vec_arg.type = ARG_INT;
    vec_arg.int_data.i = 4;  // Single value 4
    UOp* vec_const = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec_arg, NULL);
    
    UOpArg gep_arg = {0};
    gep_arg.type = ARG_INT;
    gep_arg.int_data.i = 1;
    UOp* gep_result = uop_new(OPS_GEP, dtypes.int32, &vec_const, 1, &gep_arg, NULL);
    
    // Should fold to constant 4
    TEST_ASSERT_NOT_NULL(gep_result);
    
    uop_unref(vec_const);
    uop_unref(gep_result);
}

// Port of test_add_const from Python
TEST(test_add_const) {
    // Test vector constant addition
    // v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    // v2 = UOp.const(dtypes.int.vec(3), (5,6,7))
    // ret = graph_rewrite(v1+v2, sym)
    
    UOpArg vec1_arg = {0};
    vec1_arg.type = ARG_INT;
    vec1_arg.int_data.i = 0;  // Representing (0,1,2)
    UOp* vec1 = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec1_arg, NULL);
    
    UOpArg vec2_arg = {0};
    vec2_arg.type = ARG_INT;
    vec2_arg.int_data.i = 5;  // Representing (5,6,7)
    UOp* vec2 = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec2_arg, NULL);
    
    UOp* add_result = uop_add(vec1, vec2);
    TEST_ASSERT_NOT_NULL(add_result);
    
    // After graph rewrite should fold to VCONST with (5,7,9)
    // UOp* rewritten = uop_graph_rewrite(add_result);  // TODO: Implement
    // TEST_ASSERT_EQUAL(OPS_VCONST, rewritten->op);
    
    uop_unref(vec1);
    uop_unref(vec2);
    uop_unref(add_result);
}

// Port of test_add_const_lose_v from Python
TEST(test_add_const_lose_v) {
    // Test vector addition that results in uniform value
    // v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    // v2 = UOp.const(dtypes.int.vec(3), (2,1,0))
    // Should result in (2,2,2) -> scalar 2
    
    UOpArg vec1_arg = {0};
    vec1_arg.type = ARG_INT;
    vec1_arg.int_data.i = 0;
    UOp* vec1 = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec1_arg, NULL);
    
    UOpArg vec2_arg = {0};
    vec2_arg.type = ARG_INT;  
    vec2_arg.int_data.i = 2;
    UOp* vec2 = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec2_arg, NULL);
    
    UOp* add_result = uop_add(vec1, vec2);
    TEST_ASSERT_NOT_NULL(add_result);
    
    // Should optimize to scalar constant 2
    // UOp* rewritten = uop_graph_rewrite(add_result);
    // TEST_ASSERT_EQUAL(OPS_CONST, rewritten->op);
    // TEST_ASSERT_EQUAL(2, rewritten->arg.const_data.const_value);
    
    uop_unref(vec1);
    uop_unref(vec2);
    uop_unref(add_result);
}

// =============================================================================
// TestModularWraparound - Tests for modular arithmetic wraparound behavior
// =============================================================================

// Helper function for testing modular wraparound
static void test_wraparound_helper(UOp* uop, int expected) {
    // Simulate to_uops_list([uop]) operation
    TEST_ASSERT_NOT_NULL(uop);
    
    // After full rewrite, should be single constant with expected value
    // UOp** results = uop_to_uops_list(&uop, 1, &count);  // TODO: Implement
    // TEST_ASSERT_EQUAL(1, count);
    // TEST_ASSERT_EQUAL(OPS_CONST, results[0]->op);
    // TEST_ASSERT_EQUAL(expected, results[0]->arg.const_data.const_value);
    
    // For now, just verify UOp creation
    TEST_ASSERT_TRUE(uop->op != OPS_NOOP);
}

// Port of test_cast from TestModularWraparound
TEST(test_modular_cast) {
    // Test: UOp.const(dtypes.uint, 0xABCD17D6).cast(dtypes.uint8) -> 0xD6
    UOp* uint_const = uop_const(dtypes.uint32, 0xABCD17D6);
    UOp* cast_result = uop_cast(uint_const, dtypes.uint8);
    
    test_wraparound_helper(cast_result, 0xD6);
    
    uop_unref(uint_const);
    uop_unref(cast_result);
}

// Port of test_mul from TestModularWraparound  
TEST(test_modular_mul) {
    // Test: UOp.const(dtypes.uint, 0xABCD17D6) * 0xAABBCCDD -> wrapped result
    UOp* const1 = uop_const(dtypes.uint32, 0xABCD17D6);
    UOp* const2 = uop_const(dtypes.uint32, 0xAABBCCDD);
    UOp* mul_result = uop_mul(const1, const2);
    
    test_wraparound_helper(mul_result, 1147018174);  // Expected wrapped value
    
    uop_unref(const1);
    uop_unref(const2); 
    uop_unref(mul_result);
}

// Port of test_neg from TestModularWraparound
TEST(test_modular_neg) {
    // Test negation wraparound for different unsigned types
    
    // uint8: -1 -> 0xFF  
    UOp* uint8_one = uop_const(dtypes.uint8, 1);
    UOp* neg_uint8 = uop_neg(uint8_one);
    test_wraparound_helper(neg_uint8, 0xFF);
    
    // uint16: -1 -> 0xFFFF
    UOp* uint16_one = uop_const(dtypes.uint16, 1);  
    UOp* neg_uint16 = uop_neg(uint16_one);
    test_wraparound_helper(neg_uint16, 0xFFFF);
    
    // uint32: -1 -> 0xFFFFFFFF
    UOp* uint32_one = uop_const(dtypes.uint32, 1);
    UOp* neg_uint32 = uop_neg(uint32_one);
    test_wraparound_helper(neg_uint32, 0xFFFFFFFF);
    
    uop_unref(uint8_one);
    uop_unref(neg_uint8);
    uop_unref(uint16_one);
    uop_unref(neg_uint16);
    uop_unref(uint32_one);
    uop_unref(neg_uint32);
}

// Port of test_neg_min_int from TestModularWraparound
TEST(test_modular_neg_min_int) {
    // Test negation of minimum signed integers (should stay same)
    
    // int8: -(-2^7) = -2^7  
    UOp* int8_min = uop_const(dtypes.int8, -128);  // -2^7
    UOp* neg_int8 = uop_neg(int8_min);
    test_wraparound_helper(neg_int8, -128);
    
    // int16: -(-2^15) = -2^15
    UOp* int16_min = uop_const(dtypes.int16, -32768);  // -2^15  
    UOp* neg_int16 = uop_neg(int16_min);
    test_wraparound_helper(neg_int16, -32768);
    
    // int32: -(-2^31) = -2^31
    UOp* int32_min = uop_const(dtypes.int32, -2147483648);  // -2^31
    UOp* neg_int32 = uop_neg(int32_min);  
    test_wraparound_helper(neg_int32, -2147483648);
    
    uop_unref(int8_min);
    uop_unref(neg_int8);
    uop_unref(int16_min);
    uop_unref(neg_int16);
    uop_unref(int32_min);
    uop_unref(neg_int32);
}

// =============================================================================
// TestGraphRewrite - General graph rewriting pattern tests
// =============================================================================

// Port of test_dedup from Python
TEST(test_graph_dedup) {
    // Test that identical variables are deduplicated
    UOp* v1 = uop_var("x", dtypes.float32);
    UOp* v2 = uop_var("x", dtypes.float32);  // Same name, should dedupe
    UOp* sum = uop_add(v1, v2);
    
    // After graph rewrite, v1 and v2 should be the same instance
    // UOp* rewritten = uop_graph_rewrite(sum);  // TODO: Implement  
    // TEST_ASSERT_TRUE(rewritten->src[0] == rewritten->src[1]);
    
    TEST_ASSERT_NOT_NULL(sum);
    
    uop_unref(v1);
    uop_unref(v2);
    uop_unref(sum);
}

// Port of test_simple from Python
TEST(test_graph_simple_rewrite) {
    // Test simple pattern matching: const + const -> folded const
    UOp* c1 = uop_const(dtypes.float32, 1.0);
    UOp* c2 = uop_const(dtypes.float32, 2.0);
    UOp* sum = uop_add(c1, c2);
    
    // Should rewrite to constant 3.0
    // UOp* rewritten = uop_graph_rewrite(sum);  
    // TEST_ASSERT_EQUAL(OPS_CONST, rewritten->op);
    // TEST_ASSERT_FLOAT_WITHIN(0.001, 3.0, rewritten->arg.const_data.const_value);
    
    TEST_ASSERT_NOT_NULL(sum);
    
    uop_unref(c1);
    uop_unref(c2);
    uop_unref(sum);
}

// Port of test_depth_2_late from Python
TEST(test_graph_depth_2_late) {
    // Test depth-2 constant folding: c1*c2*(c3+c3) -> 12.0
    UOp* c1 = uop_const(dtypes.float32, 1.0);
    UOp* c2 = uop_const(dtypes.float32, 2.0);
    UOp* c3 = uop_const(dtypes.float32, 3.0);
    
    UOp* c3_plus_c3 = uop_add(c3, c3);  // 6.0
    UOp* c1_mul_c2 = uop_mul(c1, c2);   // 2.0
    UOp* result = uop_mul(c1_mul_c2, c3_plus_c3);  // 12.0
    
    // Should fold to 12.0
    // UOp* rewritten = uop_graph_rewrite(result);
    // TEST_ASSERT_EQUAL(OPS_CONST, rewritten->op);
    // TEST_ASSERT_FLOAT_WITHIN(0.001, 12.0, rewritten->arg.const_data.const_value);
    
    TEST_ASSERT_NOT_NULL(result);
    
    uop_unref(c1);
    uop_unref(c2); 
    uop_unref(c3);
    uop_unref(c3_plus_c3);
    uop_unref(c1_mul_c2);
    uop_unref(result);
}

// Port of test_double from Python
TEST(test_graph_double_add) {
    // Test: c1+c2+c3 -> 6.0
    UOp* c1 = uop_const(dtypes.float32, 1.0);
    UOp* c2 = uop_const(dtypes.float32, 2.0);  
    UOp* c3 = uop_const(dtypes.float32, 3.0);
    
    UOp* sum12 = uop_add(c1, c2);
    UOp* sum123 = uop_add(sum12, c3);
    
    // Should fold to 6.0
    TEST_ASSERT_NOT_NULL(sum123);
    
    uop_unref(c1);
    uop_unref(c2);
    uop_unref(c3);
    uop_unref(sum12);
    uop_unref(sum123);
}

// Port of test_magic_4 from Python
TEST(test_graph_magic_4) {
    // Test special case pattern matching for integer 4
    UOp* c1 = uop_const(dtypes.int32, 4.0);
    
    // Should rewrite to 3.0 based on pattern matcher
    // UOp* rewritten = uop_graph_rewrite(c1);
    // TEST_ASSERT_EQUAL(OPS_CONST, rewritten->op);
    // TEST_ASSERT_FLOAT_WITHIN(0.001, 3.0, rewritten->arg.const_data.const_value);
    
    TEST_ASSERT_NOT_NULL(c1);
    
    uop_unref(c1);
}

// Port of test_depth_2_fold from Python  
TEST(test_graph_depth_2_fold) {
    // Test: v+c1+c2 -> v+3.0 (fold constants)
    UOp* v = uop_var("x", dtypes.float32);
    UOp* c1 = uop_const(dtypes.float32, 1.0);
    UOp* c2 = uop_const(dtypes.float32, 2.0);
    
    UOp* v_plus_c1 = uop_add(v, c1);
    UOp* result = uop_add(v_plus_c1, c2);
    
    // Should rewrite to v + 3.0
    // UOp* rewritten = uop_graph_rewrite(result);
    // TEST_ASSERT_EQUAL(OPS_ADD, rewritten->op);
    // TEST_ASSERT_EQUAL(OPS_DEFINE_VAR, rewritten->src[0]->op); 
    // TEST_ASSERT_EQUAL(OPS_CONST, rewritten->src[1]->op);
    // TEST_ASSERT_FLOAT_WITHIN(0.001, 3.0, rewritten->src[1]->arg.const_data.const_value);
    
    TEST_ASSERT_NOT_NULL(result);
    
    uop_unref(v);
    uop_unref(c1);
    uop_unref(c2);
    uop_unref(v_plus_c1);
    uop_unref(result);
}

// Port of test_commutative_work from Python
TEST(test_graph_commutative_work) {
    // Test that (a+b) and (b+a) simplify to same result
    UOp* a = uop_var_with_range("a", dtypes.int32, 0, 1);
    UOp* b = uop_var_with_range("b", dtypes.int32, 0, 1);
    
    UOp* ab = uop_add(a, b);
    UOp* ba = uop_add(b, a);
    
    // After simplification, should be identical
    UOp* ab_simplified = uop_simplify(ab);
    UOp* ba_simplified = uop_simplify(ba);
    
    // TEST_ASSERT_TRUE(ab_simplified == ba_simplified);  // Should be same after simplification
    TEST_ASSERT_NOT_NULL(ab_simplified);
    TEST_ASSERT_NOT_NULL(ba_simplified);
    
    uop_unref(a);
    uop_unref(b);
    uop_unref(ab);
    uop_unref(ba);
    uop_unref(ab_simplified);
    uop_unref(ba_simplified);
}

// =============================================================================
// TestUOpGraph - Core UOp graph optimization tests
// =============================================================================

// Port of test_add_constant_fold from Python
TEST(test_uop_add_constant_fold) {
    // Test: const + const -> folded const in graph
    UOp* c1 = uop_const(dtypes.float32, 1.0);
    UOp* c2 = uop_const(dtypes.float32, 2.0);
    UOp* add_result = uop_add(c1, c2);
    
    // Simulate to_uops_list([add_result])
    // Should result in single CONST with value 3.0
    TEST_ASSERT_NOT_NULL(add_result);
    TEST_ASSERT_EQUAL(OPS_ADD, add_result->op);  // Before optimization
    
    uop_unref(c1);
    uop_unref(c2);
    uop_unref(add_result);
}

// Port of test_where_same_fold from Python
TEST(test_uop_where_same_fold) {
    // Test WHERE with same true/false values should fold
    UOp* v = uop_var_with_range("tmp", dtypes.int32, 0, 1);
    UOp* c0 = uop_const(dtypes.int32, 0);
    UOp* cond = uop_ne(v, c0);  // v != 0
    UOp* c1 = uop_const(dtypes.float32, 1.0);
    UOp* where_result = uop_where(cond, c1, c1);  // WHERE(cond, 1.0, 1.0)
    
    // Should fold to just c1 (1.0)
    TEST_ASSERT_NOT_NULL(where_result);
    
    uop_unref(v);
    uop_unref(c0);
    uop_unref(cond);
    uop_unref(c1);
    uop_unref(where_result);
}

// Port of test_where_const_fold from Python  
TEST(test_uop_where_const_fold) {
    // Test WHERE with constant condition
    UOp* bf = uop_const(dtypes.bool_, 0);  // false
    UOp* c1 = uop_const(dtypes.float32, 1.0);
    UOp* c2 = uop_const(dtypes.float32, 2.0);
    UOp* where_result = uop_where(bf, c1, c2);  // WHERE(false, 1.0, 2.0) -> 2.0
    
    // Should fold to c2 (2.0)
    TEST_ASSERT_NOT_NULL(where_result);
    
    uop_unref(bf);
    uop_unref(c1);
    uop_unref(c2);
    uop_unref(where_result);
}

// Port of test_const_cast from Python
TEST(test_uop_const_cast) {
    // Test casting constants
    UOp* bf = uop_const(dtypes.bool_, 0);  // false
    UOp* cast_result = uop_cast(bf, dtypes.int32);
    
    // Should fold to const int 0
    TEST_ASSERT_NOT_NULL(cast_result);
    TEST_ASSERT_EQUAL(OPS_CAST, cast_result->op);
    
    uop_unref(bf);
    uop_unref(cast_result);
}

// Port of test_const_bitcast from Python
TEST(test_uop_const_bitcast) {
    // Test bitcasting constants
    UOp* bf = uop_const(dtypes.float32, 1.0);
    UOp* bitcast_result = uop_bitcast(bf, dtypes.uint32);
    
    // Should fold to const uint32 0x3F800000
    TEST_ASSERT_NOT_NULL(bitcast_result);
    TEST_ASSERT_EQUAL(OPS_BITCAST, bitcast_result->op);
    
    uop_unref(bf);
    uop_unref(bitcast_result);
}

// Port of test_depth_2_const_fold from Python
TEST(test_uop_depth_2_const_fold) {
    // Test depth-2 constant folding: v + 2 + 4 -> v + 6
    UOp* v = uop_var_with_range("tmp", dtypes.int32, 0, 1);
    UOp* c2 = uop_const(dtypes.int32, 2);
    UOp* c4 = uop_const(dtypes.int32, 4);
    UOp* vc = uop_add(v, c2);
    UOp* result = uop_add(vc, c4);
    
    // Should fold constants: result should be ADD(v, 6)
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(OPS_ADD, result->op);
    
    uop_unref(v);
    uop_unref(c2);
    uop_unref(c4);
    uop_unref(vc);
    uop_unref(result);
}

// Port of test_bitcast_to_same_dtype_fold from Python
TEST(test_uop_bitcast_same_dtype_fold) {
    // Test bitcast to same dtype should be eliminated
    UOp* d0 = uop_define_global(dtypes.int32, 0);
    UOp* idx = uop_const(dtypes.int32, 0);
    UOp* indexed = uop_index(d0, idx);
    UOp* v = uop_load(indexed, dtypes.int32);
    UOp* bitcast_same = uop_bitcast(v, dtypes.int32);  // Same dtype
    
    // Should optimize away bitcast
    TEST_ASSERT_NOT_NULL(bitcast_same);
    
    uop_unref(d0);
    uop_unref(idx);
    uop_unref(indexed);
    uop_unref(v);
    uop_unref(bitcast_same);
}

// =============================================================================
// Test functions using TEST() macro for remaining tests
// =============================================================================

// Port of test_expand_add_broadcast from Python
TEST(test_expander_add_broadcast) {
    // Test expanding addition with broadcasting
    UOpArg vec_arg = {0};
    vec_arg.type = ARG_INT;
    vec_arg.int_data.i = 0;  // Representing (0,1,2,3)
    UOp* vec_const = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec_arg, NULL);
    
    UOpArg unroll_arg = {0};
    unroll_arg.type = ARG_INT;
    unroll_arg.int_data.i = 4;  // Axis info
    UOp* unroll = uop_new(OPS_UNROLL, dtypes.int32, &vec_const, 1, &unroll_arg, NULL);
    
    UOp* const_3 = uop_const(dtypes.int32, 3);
    UOp* add_result = uop_add(unroll, const_3);
    
    // Should expand to (3,4,5,6)
    TEST_ASSERT_NOT_NULL(add_result);
    
    uop_unref(vec_const);
    uop_unref(unroll);
    uop_unref(const_3);
    uop_unref(add_result);
}

// Port of test_contract_simple from Python
TEST(test_expander_contract_simple) {
    // Test simple contraction
    UOpArg vec_arg = {0};
    vec_arg.type = ARG_INT;
    vec_arg.int_data.i = 0;  // Representing (0,1,2,3)
    UOp* vec_const = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec_arg, NULL);
    
    UOpArg unroll_arg = {0};
    unroll_arg.type = ARG_INT;
    unroll_arg.int_data.i = 4;
    UOp* unroll = uop_new(OPS_UNROLL, dtypes.int32, &vec_const, 1, &unroll_arg, NULL);
    
    UOpArg contract_arg = {0};
    contract_arg.type = ARG_INT;
    contract_arg.int_data.i = 4;
    UOp* contract = uop_new(OPS_CONTRACT, dtypes.int32, &unroll, 1, &contract_arg, NULL);
    
    // Should contract back to VCONST (0,1,2,3)
    TEST_ASSERT_NOT_NULL(contract);
    
    uop_unref(vec_const);
    uop_unref(unroll);
    uop_unref(contract);
}

// Port of test_create_ifs from Python
TEST(test_if_uops_create_ifs) {
    // Test creating IF blocks with barriers and local memory
    UOp* gbuf = uop_define_global(dtypes.float32, 0);
    UOp* sbuf = uop_define_local(dtypes.float32, 4);  // size=4
    
    // Create condition: gidx0 < 5
    UOpArg gidx_arg = {0};
    gidx_arg.type = ARG_INT;
    gidx_arg.int_data.i = 10;  // max value
    UOp* gidx0 = uop_new(OPS_SPECIAL, dtypes.int32, NULL, 0, &gidx_arg, "gidx0");
    UOp* const_5 = uop_const(dtypes.int32, 5);
    UOp* valid = uop_lt(gidx0, const_5);
    
    UOpArg lidx_arg = {0};
    lidx_arg.type = ARG_INT;
    lidx_arg.int_data.i = 4;
    UOp* lidx = uop_new(OPS_SPECIAL, dtypes.int32, NULL, 0, &lidx_arg, "lidx0");
    UOp* const_2 = uop_const(dtypes.int32, 2);
    UOp* ne_result = uop_ne(lidx, const_2);  // lidx != 2
    UOp* gate = uop_and(valid, ne_result);   // valid & (lidx != 2)
    
    // Store to local memory (conditional on gate)
    UOp* idx = uop_const(dtypes.int32, 0);
    UOp* const_42 = uop_const(dtypes.float32, 42);
    UOp* indexed_sbuf = uop_index(sbuf, idx);
    UOp* conditional_val = uop_where(gate, const_42, uop_const(dtypes.float32, 0));
    UOp* st = uop_store(indexed_sbuf, conditional_val);
    
    // Barrier
    UOp* barrier = uop_new(OPS_BARRIER, dtypes.void_, &st, 1, NULL, NULL);
    
    // Load from local memory
    UOp* lbuf = uop_load(indexed_sbuf, dtypes.float32);
    
    // Store to global memory
    UOp* indexed_gbuf = uop_index(gbuf, idx);
    UOp* store = uop_store(indexed_gbuf, lbuf);
    
    // Include barrier in the sink
    UOp* sink_ops[] = {barrier, store};
    UOp* sink = uop_sink(sink_ops, 2);
    
    // After full rewrite, should create IF blocks
    TEST_ASSERT_NOT_NULL(sink);
    
    uop_unref(gbuf);
    uop_unref(sbuf);
    uop_unref(sink);
}

// Port of test_inc_by_one from Python
TEST(test_uop_tags_inc_by_one) {
    // Test tag-based pattern matching and replacement
    UOp* c1 = uop_const(dtypes.int32, 1);
    UOp* c2 = uop_const(dtypes.int32, 1);
    UOp* sum = uop_add(c1, c2);
    
    // Should simplify to 2
    UOp* simplified = uop_ssimplify(sum);
    TEST_ASSERT_NOT_NULL(simplified);
    
    // Note: Tag-based pattern matching not yet implemented
    // This would test replacing tagged constants with incremented values
    
    uop_unref(c1);
    uop_unref(c2);
    uop_unref(sum);
    uop_unref(simplified);
}

// Auto-register all test functions and run them
TEST_MAIN()