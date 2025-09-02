#include "test_uop_common.h"

void setUp(void) {
    // Initialize test fixtures if needed
}

void tearDown(void) {
    // Clean up after each test if needed
}

TEST(test_pattern_matching) {
    
    // Create a simple expression: a + b
    UOp* a = uop_const(dtypes.float32, 10.0);
    UOp* b = uop_const(dtypes.float32, 20.0);
    UOp* sum = uop_add(a, b);
    
    // Create pattern: ADD(x, y)
    UPat* x = upat_var(0);
    UPat* y = upat_var(1);
    UPat* patterns[] = {x, y};
    UPat* add_pattern = upat_op(OPS_ADD, patterns, 2);
    
    // Test matching
    ASSERT(upat_match(add_pattern, sum) == true);
    
    // Test non-matching
    UOp* prod = uop_mul(a, b);
    ASSERT(upat_match(add_pattern, prod) == false);
    
    // Create const pattern
    UPat* const_pattern = upat_const(10.0);
    ASSERT(upat_match(const_pattern, a) == true);
    ASSERT(upat_match(const_pattern, b) == false);
    
    // Create any pattern
    UPat* any_pattern = upat_any();
    ASSERT(upat_match(any_pattern, sum) == true);
    ASSERT(upat_match(any_pattern, a) == true);
    ASSERT(upat_match(any_pattern, b) == true);
    
    // Clean up
    upat_free(add_pattern);
    upat_free(const_pattern);
    upat_free(any_pattern);
    uop_unref(a);
    uop_unref(b);
    uop_unref(sum);
    uop_unref(prod);
}

TEST(test_upat_helpers) {
    
    // Test pattern location tracking
    UPat* pat = upat_op(OPS_ADD, NULL, 0);
    ASSERT(pat != NULL);  // Will fail until implemented
    
    // Test pattern variables
    UPat* var = upat_var(1);
    ASSERT(var != NULL);  // Will fail until implemented
}

TEST(test_graph_rewrite_patterns) {
    
    // Test pattern matching and rewriting like test_uop_graph.py
    UOp* x = uop_const(dtypes.float32, 2.0);
    UOp* y = uop_const(dtypes.float32, 3.0);
    UOp* add = uop_add(x, y);
    
    // This should fold to const(5.0)
    UOp* simplified = uop_simplify(add);
    ASSERT(simplified->op == OPS_CONST);  // Will fail until simplification works
}

TEST(test_graph_rewrite_const) {
    
    // Test GEP constant folding
    UOpArg vec_arg = {0};
    vec_arg.type = ARG_INT;
    // Note: This test expects a vec3i interface that's not in our structure
    UOp* vec = uop_new(OPS_VCONST, dtypes.int32, NULL, 0, &vec_arg, NULL);
    UOpArg idx_arg = {0};
    idx_arg.type = ARG_INT;
    idx_arg.int_data.i = 1;
    UOp* gep = uop_new(OPS_GEP, dtypes.int32, &vec, 1, &idx_arg, NULL);
    
    // Should fold to const(2)
    UOp* folded = uop_simplify(gep);
    ASSERT(folded->op == OPS_CONST);  // Will fail until implemented
}

// Auto-register all test functions and run them
TEST_MAIN()