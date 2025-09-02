#include "test_uop_common.h"

void setUp(void) {
    // Initialize test fixtures if needed
}

void tearDown(void) {
    // Clean up after each test if needed
}

// Test MathTrait operations
TEST(test_math_traits) {
    
    
    // Create test UOps
    UOp* a = uop_const(dtypes.float32, 10.0);
    UOp* b = uop_const(dtypes.float32, 20.0);
    
    // Test arithmetic operations through MathTrait
    UOp* sum = a->math_ops->add(a, b, false);
    ASSERT(sum != NULL);
    ASSERT(sum->op == OPS_ADD);
    
    UOp* prod = a->math_ops->mul(a, b, false);
    ASSERT(prod != NULL);
    ASSERT(prod->op == OPS_MUL);
    
    UOp* diff = b->math_ops->sub(b, a, false);
    ASSERT(diff != NULL);
    ASSERT(diff->op == OPS_SUB);
    
    // Dump test values for debugging
    
    // Try the most minimal call possible
    UOp* neg_a = a->math_ops->neg(a);
    
    // If we get here, the corruption is deeper in neg_impl
    ASSERT(neg_a != NULL);
    ASSERT(neg_a->op == OPS_NEG);
    
    // Test comparison operations
    UOp* lt = a->math_ops->lt(a, b);
    ASSERT(lt != NULL);
    ASSERT(lt->op == OPS_CMPLT);
    
    UOp* eq = a->math_ops->eq(a, b);
    ASSERT(eq != NULL);
    ASSERT(eq->op == OPS_CMPEQ);
    
    // Test math functions
    UOp* sqrt_a = a->math_ops->sqrt(a);
    ASSERT(sqrt_a != NULL);
    ASSERT(sqrt_a->op == OPS_SQRT);
    
    UOp* recip_a = a->math_ops->reciprocal(a);
    ASSERT(recip_a != NULL);
    ASSERT(recip_a->op == OPS_RECIP);
    
    // Clean up
    uop_unref(a);
    uop_unref(b);
    uop_unref(sum);
    uop_unref(prod);
    uop_unref(diff);
    uop_unref(neg_a);
    uop_unref(lt);
    uop_unref(eq);
    uop_unref(sqrt_a);
    uop_unref(recip_a);
}

// Test identity elements
TEST(test_identity_elements) {
    
    // Test ADD identity (0)
    ASSERT(identity_element(OPS_ADD, &dtypes.int32) == 0.0);
    ASSERT(identity_element(OPS_ADD, &dtypes.float32) == 0.0);
    
    // Test MUL identity (1)
    ASSERT(identity_element(OPS_MUL, &dtypes.int32) == 1.0);
    ASSERT(identity_element(OPS_MUL, &dtypes.float32) == 1.0);
    
    // Test MAX identity (min value of dtype)
    double max_identity_int32 = identity_element(OPS_MAX, &dtypes.int32);
    ASSERT(max_identity_int32 == dtypes_min(&dtypes.int32));
    
    double max_identity_float = identity_element(OPS_MAX, &dtypes.float32);
    ASSERT(isinf(max_identity_float) && max_identity_float < 0);
}

// Test special math operations
TEST(test_special_math_ops) {
    
    // Test EXP2
    UOp* val = uop_const(dtypes.float32, 3.0);
    UOp* exp2_val = uop_exp2(val);
    ASSERT(exp2_val != NULL);
    ASSERT(exp2_val->op == OPS_EXP2);
    
    // Test LOG2
    UOp* log2_val = uop_log2(val);
    ASSERT(log2_val != NULL);
    ASSERT(log2_val->op == OPS_LOG2);
    
    // Test SIN
    UOp* sin_val = uop_sin(val);
    ASSERT(sin_val != NULL);
    ASSERT(sin_val->op == OPS_SIN);
    
    // Test SQRT
    UOp* sqrt_val = uop_sqrt(val);
    ASSERT(sqrt_val != NULL);
    ASSERT(sqrt_val->op == OPS_SQRT);
    
    // Test RECIP
    UOp* recip_val = uop_recip(val);
    ASSERT(recip_val != NULL);
    ASSERT(recip_val->op == OPS_RECIP);
    
    // Clean up
    uop_unref(val);
    uop_unref(exp2_val);
    uop_unref(log2_val);
    uop_unref(sin_val);
    uop_unref(sqrt_val);
    uop_unref(recip_val);
}

TEST(test_commutative_canonicalization) {
    
    // Constants should go to the right in commutative ops
    UOp* two = uop_const(dtypes.int32, 2);
    UOp* x = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, NULL, "x");
    
    // 2 + x should become x + 2
    UOp* sum = uop_add(two, x);
    UOp* simplified = uop_ssimplify(sum);
    ASSERT(simplified != NULL);
    // In canonical form, variable should be on left, constant on right
    
    // Test with MUL (also commutative)
    UOp* prod = uop_mul(two, x);
    simplified = uop_ssimplify(prod);
    ASSERT(simplified != NULL);
}

TEST(test_constant_folding) {
    
    // Test ADD constant folding
    UOp* c1 = uop_const(dtypes.float32, 1.0);
    UOp* c2 = uop_const(dtypes.float32, 2.0);
    UOp* sum = uop_add(c1, c2);
    UOp* folded = uop_ssimplify(sum);
    ASSERT(folded->op == OPS_CONST);
    ASSERT_NEAR(folded->arg.const_data.const_value, 3.0, 0.001);
    
    // Test WHERE with same branches
    UOp* cond = uop_const(dtypes.bool_, 1);
    UOp* val = uop_const(dtypes.float32, 5.0);
    UOp* where = uop_where(cond, val, val);
    folded = uop_ssimplify(where);
    ASSERT(folded->op == OPS_CONST);
    ASSERT_NEAR(folded->arg.const_data.const_value, 5.0, 0.001);
    
    // Test WHERE with const condition
    UOp* false_cond = uop_const(dtypes.bool_, 0);
    UOp* v1 = uop_const(dtypes.float32, 1.0);
    UOp* v2 = uop_const(dtypes.float32, 2.0);
    where = uop_where(false_cond, v1, v2);
    folded = uop_ssimplify(where);
    ASSERT(folded->op == OPS_CONST);
    ASSERT_NEAR(folded->arg.const_data.const_value, 2.0, 0.001);
}

TEST(test_simplification) {
    
    // Test x + 0 = x
    UOp* x = uop_const(dtypes.int32, 5);
    UOp* zero = uop_const(dtypes.int32, 0);
    UOp* sum = uop_add(x, zero);
    UOp* simplified = uop_ssimplify(sum);
    ASSERT(uop_equals(simplified, x));  // Should simplify to just x
    
    // Test x * 1 = x
    UOp* one = uop_const(dtypes.int32, 1);
    UOp* prod = uop_mul(x, one);
    simplified = uop_ssimplify(prod);
    ASSERT(uop_equals(simplified, x));  // Should simplify to just x
    
    // Test x - x = 0
    UOp* diff = uop_sub(x, x);
    simplified = uop_ssimplify(diff);
    ASSERT(simplified->op == OPS_CONST);
    ASSERT(simplified->arg.int_data.i == 0);
}

// Auto-register all test functions and run them
TEST_MAIN()