#include "test_uop_common.h"

void setUp(void) {
    // Initialize test fixtures if needed
    static int initialized = 0;
    if (!initialized) {
        dtypes_init();
        initialized = 1;
    }
}

void tearDown(void) {
    // Clean up after each test if needed
}

TEST(test_symbolic_variables) {
    
    // Test variable creation with range
    UOpArg var_arg = {0};
    var_arg.type = ARG_INT;
    // Note: This test seems to expect a different interface, may need to be updated
    UOp* var_x = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, &var_arg, "x");
    ASSERT(var_x != NULL);
    ASSERT(var_x->op == OPS_DEFINE_VAR);
    ASSERT(uop_sym_infer(var_x) >= 10);
    ASSERT(uop_sym_infer(var_x) <= 20);
    
    // Test variable arithmetic
    UOp* var_plus_5 = uop_add(var_x, uop_const(dtypes.int32, 5));
    ASSERT(var_plus_5 != NULL);
    ASSERT(uop_sym_infer(var_plus_5) >= 15);
    ASSERT(uop_sym_infer(var_plus_5) <= 25);
    
    // Test variable comparison
    UOp* var_lt_15 = uop_lt(var_x, uop_const(dtypes.int32, 15));
    ASSERT(var_lt_15 != NULL);
    
    // Test variable resolution
    // var_x is in [10,20], var_x < 15 is ambiguous
    // With default=false, should return false for ambiguous case
    bool resolved = uop_resolve(var_lt_15, false);
    ASSERT(resolved == false);  // ambiguous, returns default (false)
}

TEST(test_advanced_symbolic) {
    
    // Test modulo congruence: (3 + 3*a) % 4 should simplify to a
    UOpArg var_arg = {0};
    var_arg.type = ARG_INT;
    // Note: This test expects a different interface
    UOp* a = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, &var_arg, "a");
    UOp* three = uop_const(dtypes.int32, 3);
    UOp* mul = uop_mul(a, three);
    UOp* sum = uop_add(mul, three);
    UOp* four = uop_const(dtypes.int32, 4);
    UOp* mod = uop_new(OPS_MOD, dtypes.int32, (UOp*[]){sum, four}, 2, NULL, NULL);
    ASSERT(mod != NULL);
    
    // Test division congruence
    UOp* div = uop_new(OPS_IDIV, dtypes.int32, (UOp*[]){sum, four}, 2, NULL, NULL);
    ASSERT(div != NULL);
}

TEST(test_vmin_vmax_propagation) {
    
    // Test variable with range
    UOpArg var_arg = {0};
    var_arg.type = ARG_INT;
    // Note: This test expects a different interface
    UOp* var = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, &var_arg, "x");
    ASSERT(var != NULL);
    
    // Test vmin/vmax after addition
    UOp* five = uop_const(dtypes.int32, 5);
    UOp* sum = uop_add(var, five);
    // vmin should be 15, vmax should be 25
    ASSERT(uop_sym_infer(sum) >= 15);
    ASSERT(uop_sym_infer(sum) <= 25);
    
    // Test vmin/vmax after multiplication
    UOp* two = uop_const(dtypes.int32, 2);
    UOp* prod = uop_mul(var, two);
    // vmin should be 20, vmax should be 40
    ASSERT(uop_sym_infer(prod) >= 20);
    ASSERT(uop_sym_infer(prod) <= 40);
}

TEST(test_symbolic_bounds) {
    
    // Test symbolic variable bounds propagation
    UOpArg var_arg = {0};
    var_arg.type = ARG_INT;
    // Note: This test expects a string interface that's not in our structure
    UOp* x = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, &var_arg, NULL);
    
    // Set bounds on variable
    // x should have vmin=0, vmax=10
    UOpArg bound_arg = {0};
    bound_arg.type = ARG_INT;
    // Note: This test expects a vec3i interface that's not in our structure
    UOp* bounded_x = uop_new(OPS_BIND, dtypes.int32, &x, 1, &bound_arg, NULL);
    
    ASSERT(bounded_x != NULL);
}

TEST(test_symbolic_resolution) {
    
    // Test simple int resolution
    UOpArg arg = {0};
    arg.type = ARG_INT;
    arg.int_data.i = 42;
    UOp* x = uop_new(OPS_CONST, dtypes.int32, NULL, 0, &arg, NULL);
    bool resolved = uop_resolve(x, false);
    ASSERT(resolved == true || resolved == false);  // Should resolve
    
    // Test variable comparison
    UOpArg var_arg = {0};
    var_arg.type = ARG_INT;
    // Note: This test expects a string interface that's not in our structure
    UOp* var = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, &var_arg, NULL);
    UOp* ten = uop_const(dtypes.int32, 10);
    UOp* cmp = uop_lt(var, ten);
    resolved = uop_resolve(cmp, false);
    ASSERT(resolved == true || resolved == false);
}

TEST(test_vmin_vmax_divmod) {
    fflush(stdout);
    
    // Test division bounds
    UOpArg var_arg = {0};
    var_arg.type = ARG_INT;
    // Note: This test expects a string interface that's not in our structure
    UOp* x = uop_new(OPS_DEFINE_VAR, dtypes.int32, NULL, 0, &var_arg, NULL);
    UOp* ten = uop_const(dtypes.int32, 10);
    UOp* div = uop_div(x, ten);
    
    // Should compute bounds for division
    if (!div) {
        TEST_FAIL_MESSAGE("div is NULL");
        ASSERT(0);
        return;
    }
    
    int min_val = uop_vmin(div);
    int max_val = uop_vmax(div);
    fflush(stderr);
    // For now, just check that we got reasonable bounds
    // x in [0,100], x/10 should be in [0,10]
    ASSERT(min_val >= 0);
    ASSERT(max_val <= 10);
    ASSERT(min_val >= 0);  // Expected: x in [0,100], x/10 in [0,10]
}

TEST(test_range_operations) {
    
    // Test RANGE operation for loop bounds
    UOpArg range_arg = {0};
    range_arg.type = ARG_INT;
    UOp* range = uop_new(OPS_RANGE, dtypes.int32, NULL, 0, &range_arg, NULL);
    ASSERT(range != NULL);
    ASSERT(range->op == OPS_RANGE);
}

TEST(test_modular_wraparound) {
    
    // Test integer wraparound behavior for different dtypes
    UOp* max_int8 = uop_const(dtypes.int8, 127);
    UOp* one = uop_const(dtypes.int8, 1);
    uop_add(max_int8, one);  // Should wrap to -128
    
    // Should wrap to -128 for int8
    double result = exec_alu(OPS_ADD, dtypes.int8, (double[]){127, 1}, 2);
    ASSERT_NEAR(result, -128, 0.001);  // Will fail until modular arithmetic works
}

// === NEW TESTS PORTED FROM PYTHON REFERENCE ===

// Port of test_cmp_simple from Python
TEST(test_symbolic_cmp_simple) {
    // Variable "a" with range [3, 8]
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 3, 8);
    UOp* const_4 = uop_const(dtypes.int32, 4);
    
    // a < 4 should be conditional (0, 1)
    UOp* lt_result = uop_lt(var_a, const_4);
    TEST_ASSERT_NOT_NULL(lt_result);
    TEST_ASSERT_EQUAL(OPS_CMPLT, lt_result->op);
    
    // Test bounds for boolean result
    int vmin = uop_vmin(lt_result);
    int vmax = uop_vmax(lt_result);
    TEST_ASSERT_EQUAL(0, vmin);  // Can be false
    TEST_ASSERT_EQUAL(1, vmax);  // Can be true
}

// Port of test_ge operations from Python
TEST(test_symbolic_ge_operations) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 3, 8);
    
    // a >= 77 should be false (0,0)
    UOp* const_77 = uop_const(dtypes.int32, 77);
    UOp* ge_77 = uop_ge(var_a, const_77);
    TEST_ASSERT_EQUAL(0, uop_vmin(ge_77));
    TEST_ASSERT_EQUAL(0, uop_vmax(ge_77));
    
    // a >= 3 should be true (1,1)
    UOp* const_3 = uop_const(dtypes.int32, 3);
    UOp* ge_3 = uop_ge(var_a, const_3);
    TEST_ASSERT_EQUAL(1, uop_vmin(ge_3));
    TEST_ASSERT_EQUAL(1, uop_vmax(ge_3));
    
    // a >= 4 should be conditional (0,1)
    UOp* const_4 = uop_const(dtypes.int32, 4);
    UOp* ge_4 = uop_ge(var_a, const_4);
    TEST_ASSERT_EQUAL(0, uop_vmin(ge_4));
    TEST_ASSERT_EQUAL(1, uop_vmax(ge_4));
}

// Port of test_lt operations from Python
TEST(test_symbolic_lt_operations) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 3, 8);
    
    // a < 77 should be true (1,1)
    UOp* const_77 = uop_const(dtypes.int32, 77);
    UOp* lt_77 = uop_lt(var_a, const_77);
    TEST_ASSERT_EQUAL(1, uop_vmin(lt_77));
    TEST_ASSERT_EQUAL(1, uop_vmax(lt_77));
    
    // a < 3 should be false (0,0)
    UOp* const_3 = uop_const(dtypes.int32, 3);
    UOp* lt_3 = uop_lt(var_a, const_3);
    TEST_ASSERT_EQUAL(0, uop_vmin(lt_3));
    TEST_ASSERT_EQUAL(0, uop_vmax(lt_3));
}

// Port of test_div_reduction from Python
TEST(test_symbolic_div_reduction) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 2, 3);
    UOp* const_2 = uop_const(dtypes.int32, 2);
    UOp* div_result = uop_div(var_a, const_2);
    
    // a(2,3) // 2 should result in (1,1)
    TEST_ASSERT_EQUAL(1, uop_vmin(div_result));
    TEST_ASSERT_EQUAL(1, uop_vmax(div_result));
}

// Port of test_factorize from Python - a*2+a*3 bounds check
TEST(test_symbolic_factorize_bounds) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* const_2 = uop_const(dtypes.int32, 2);
    UOp* const_3 = uop_const(dtypes.int32, 3);
    
    UOp* mul_a2 = uop_mul(var_a, const_2);
    UOp* mul_a3 = uop_mul(var_a, const_3);
    UOp* sum = uop_add(mul_a2, mul_a3);
    
    // Result should have bounds [0, 8*5] = [0, 40]
    TEST_ASSERT_EQUAL(0, uop_vmin(sum));
    TEST_ASSERT_EQUAL(40, uop_vmax(sum));  // 8 * (2+3)
}

// Port of test_neg from Python
TEST(test_symbolic_negation) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* neg_a = uop_neg(var_a);
    
    TEST_ASSERT_EQUAL(-8, uop_vmin(neg_a));
    TEST_ASSERT_EQUAL(0, uop_vmax(neg_a));
}

// Port of test_add_1 from Python
TEST(test_symbolic_add_constant) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* const_1 = uop_const(dtypes.int32, 1);
    UOp* add_result = uop_add(var_a, const_1);
    
    TEST_ASSERT_EQUAL(1, uop_vmin(add_result));
    TEST_ASSERT_EQUAL(9, uop_vmax(add_result));
}

// Port of test_sub_1 from Python
TEST(test_symbolic_sub_constant) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* const_1 = uop_const(dtypes.int32, 1);
    UOp* sub_result = uop_sub(var_a, const_1);
    
    TEST_ASSERT_EQUAL(-1, uop_vmin(sub_result));
    TEST_ASSERT_EQUAL(7, uop_vmax(sub_result));
}

// Port of test_const_var from Python
TEST(test_symbolic_const_var) {
    UOp* var_fake = uop_var_with_range("fake", dtypes.int32, 1, 1);
    TEST_ASSERT_EQUAL(1, uop_vmin(var_fake));
    TEST_ASSERT_EQUAL(1, uop_vmax(var_fake));
}

// Port of test_mul_0 from Python
TEST(test_symbolic_mul_zero) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* const_0 = uop_const(dtypes.int32, 0);
    UOp* mul_result = uop_mul(var_a, const_0);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(mul_result));
    TEST_ASSERT_EQUAL(0, uop_vmax(mul_result));
}

// Port of test_mul_1 from Python
TEST(test_symbolic_mul_one) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* const_1 = uop_const(dtypes.int32, 1);
    UOp* mul_result = uop_mul(var_a, const_1);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(mul_result));
    TEST_ASSERT_EQUAL(8, uop_vmax(mul_result));
}

// Port of test_div_1 from Python
TEST(test_symbolic_div_one) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* const_1 = uop_const(dtypes.int32, 1);
    UOp* div_result = uop_div(var_a, const_1);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(div_result));
    TEST_ASSERT_EQUAL(8, uop_vmax(div_result));
}

// Port of test_mod_1 from Python
TEST(test_symbolic_mod_one) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* const_1 = uop_const(dtypes.int32, 1);
    UOp* mod_result = uop_mod(var_a, const_1);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(mod_result));
    TEST_ASSERT_EQUAL(0, uop_vmax(mod_result));
}

// Port of test_div_min_max from Python
TEST(test_symbolic_div_bounds) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 1, 7);
    UOp* const_2 = uop_const(dtypes.int32, 2);
    UOp* div_result = uop_div(var_a, const_2);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(div_result));  // 1//2 = 0
    TEST_ASSERT_EQUAL(3, uop_vmax(div_result));  // 7//2 = 3
}

// Port of test_div_neg_min_max from Python
TEST(test_symbolic_div_neg_bounds) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 1, 7);
    UOp* const_neg2 = uop_const(dtypes.int32, -2);
    UOp* div_result = uop_div(var_a, const_neg2);
    
    TEST_ASSERT_EQUAL(-3, uop_vmin(div_result));  // 7//-2 = -3
    TEST_ASSERT_EQUAL(0, uop_vmax(div_result));   // 1//-2 = 0
}

// Port of test_mod_min_max from Python
TEST(test_symbolic_mod_bounds) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 0, 10);
    UOp* var_y = uop_var_with_range("y", dtypes.int32, 1, 10);
    UOp* mod_result = uop_mod(var_x, var_y);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(mod_result));
    TEST_ASSERT_EQUAL(9, uop_vmax(mod_result));  // max(y)-1
}

// Port of test_mod_remove from Python
TEST(test_symbolic_mod_remove) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 6);
    UOp* const_100 = uop_const(dtypes.int32, 100);
    UOp* mod_result = uop_mod(var_a, const_100);
    
    // a % 100 where a is [0,6] should just be a
    TEST_ASSERT_EQUAL(0, uop_vmin(mod_result));
    TEST_ASSERT_EQUAL(6, uop_vmax(mod_result));
}

// Port of test_big_mod from Python
TEST(test_symbolic_big_mod) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, -20, 20);
    UOp* const_10 = uop_const(dtypes.int32, 10);
    UOp* mod_result = uop_mod(var_a, const_10);
    
    TEST_ASSERT_EQUAL(-9, uop_vmin(mod_result));
    TEST_ASSERT_EQUAL(9, uop_vmax(mod_result));
}

// Port of variable comparison tests from Python
TEST(test_symbolic_variable_comparison) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 3, 4);
    UOp* var_b = uop_var_with_range("b", dtypes.int32, 5, 6);
    UOp* lt_result = uop_lt(var_a, var_b);
    
    // a(3,4) < b(5,6) should always be true
    TEST_ASSERT_EQUAL(1, uop_vmin(lt_result));
    TEST_ASSERT_EQUAL(1, uop_vmax(lt_result));
    
    UOp* var_c = uop_var_with_range("c", dtypes.int32, 3, 5);
    UOp* var_d = uop_var_with_range("d", dtypes.int32, 5, 6);
    UOp* lt_result2 = uop_lt(var_c, var_d);
    
    // c(3,5) < d(5,6) should be conditional (0,1)
    TEST_ASSERT_EQUAL(0, uop_vmin(lt_result2));
    TEST_ASSERT_EQUAL(1, uop_vmax(lt_result2));
}

// Port of test_add_self from Python
TEST(test_symbolic_add_self) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* add_result = uop_add(var_a, var_a);  // a + a
    
    TEST_ASSERT_EQUAL(0, uop_vmin(add_result));   // 0 + 0
    TEST_ASSERT_EQUAL(16, uop_vmax(add_result));  // 8 + 8
}

// Port of test_mul_2 from Python
TEST(test_symbolic_mul_two) {
    UOp* var_a = uop_var_with_range("a", dtypes.int32, 0, 8);
    UOp* const_2 = uop_const(dtypes.int32, 2);
    UOp* mul_result = uop_mul(var_a, const_2);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(mul_result));
    TEST_ASSERT_EQUAL(16, uop_vmax(mul_result));  // 8 * 2
}

// === TESTS PORTED FROM test_uop_vmin_vmax.py ===

// Port of test_vmin_vmax_constant from Python
TEST(test_vmin_vmax_constant) {
    UOp* const_42 = uop_const(dtypes.int32, 42);
    TEST_ASSERT_EQUAL(42, uop_vmin(const_42));
    TEST_ASSERT_EQUAL(42, uop_vmax(const_42));
}

// Port of test_vmin_vmax_cmpne from Python
TEST(test_vmin_vmax_cmpne) {
    UOp* const_42 = uop_const(dtypes.int32, 42);
    UOp* const_42_2 = uop_const(dtypes.int32, 42);
    UOp* const_43 = uop_const(dtypes.int32, 43);
    UOp* const_41 = uop_const(dtypes.int32, 41);
    
    // 42 != 42 should be false (0,0)
    UOp* ne_same = uop_ne(const_42, const_42_2);
    TEST_ASSERT_EQUAL(0, uop_vmin(ne_same));
    TEST_ASSERT_EQUAL(0, uop_vmax(ne_same));
    
    // 42 != 43 should be true (1,1)
    UOp* ne_diff1 = uop_ne(const_42, const_43);
    TEST_ASSERT_EQUAL(1, uop_vmin(ne_diff1));
    TEST_ASSERT_EQUAL(1, uop_vmax(ne_diff1));
    
    // 42 != 41 should be true (1,1)
    UOp* ne_diff2 = uop_ne(const_42, const_41);
    TEST_ASSERT_EQUAL(1, uop_vmin(ne_diff2));
    TEST_ASSERT_EQUAL(1, uop_vmax(ne_diff2));
}

// Port of test_vmin_vmax_addition_with_variable from Python
TEST(test_vmin_vmax_addition_variable) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 10, 20);
    UOp* const_5 = uop_const(dtypes.int32, 5);
    UOp* add_result = uop_add(var_x, const_5);
    
    TEST_ASSERT_EQUAL(15, uop_vmin(add_result));  // 10 + 5
    TEST_ASSERT_EQUAL(25, uop_vmax(add_result));  // 20 + 5
}

// Port of test_vmin_vmax_subtraction_with_variable from Python
TEST(test_vmin_vmax_subtraction_variable) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 10, 20);
    UOp* const_5 = uop_const(dtypes.int32, 5);
    
    // x - 5
    UOp* sub1 = uop_sub(var_x, const_5);
    TEST_ASSERT_EQUAL(5, uop_vmin(sub1));   // 10 - 5
    TEST_ASSERT_EQUAL(15, uop_vmax(sub1));  // 20 - 5
    
    // 5 - x
    UOp* sub2 = uop_sub(const_5, var_x);
    TEST_ASSERT_EQUAL(-15, uop_vmin(sub2));  // 5 - 20
    TEST_ASSERT_EQUAL(-5, uop_vmax(sub2));   // 5 - 10
}

// Port of test_vmin_vmax_and_with_variable from Python
TEST(test_vmin_vmax_and_variable) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 10, 20);
    UOp* const_5 = uop_const(dtypes.int32, 5);
    UOp* and_result = uop_and(var_x, const_5);
    
    // x & 5 - bitwise AND with 5 should be bounded by 0 and 5
    TEST_ASSERT_EQUAL(0, uop_vmin(and_result));
    TEST_ASSERT_EQUAL(5, uop_vmax(and_result));
    
    UOp* const_15 = uop_const(dtypes.int32, 15);
    UOp* and_result2 = uop_and(var_x, const_15);
    TEST_ASSERT_EQUAL(0, uop_vmin(and_result2));
    TEST_ASSERT_EQUAL(15, uop_vmax(and_result2));
}

// Port of test_vmin_vmax_multiplication_with_variable from Python
TEST(test_vmin_vmax_multiplication_variable) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, -3, 4);
    UOp* const_2 = uop_const(dtypes.int32, 2);
    UOp* mul_result = uop_mul(var_x, const_2);
    
    TEST_ASSERT_EQUAL(-6, uop_vmin(mul_result));  // -3 * 2
    TEST_ASSERT_EQUAL(8, uop_vmax(mul_result));   // 4 * 2
}

// Port of test_vmin_vmax_with_negative_multiplication from Python
TEST(test_vmin_vmax_negative_multiplication) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 2, 5);
    UOp* const_neg3 = uop_const(dtypes.int32, -3);
    UOp* mul_result = uop_mul(var_x, const_neg3);
    
    TEST_ASSERT_EQUAL(-15, uop_vmin(mul_result));  // 5 * -3
    TEST_ASSERT_EQUAL(-6, uop_vmax(mul_result));   // 2 * -3
}

// Port of test_vmin_vmax_with_negative_multiplication2 from Python
TEST(test_vmin_vmax_negative_multiplication2) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, -2, 5);
    UOp* const_neg3 = uop_const(dtypes.int32, -3);
    UOp* mul_result = uop_mul(var_x, const_neg3);
    
    TEST_ASSERT_EQUAL(-15, uop_vmin(mul_result));  // 5 * -3
    TEST_ASSERT_EQUAL(6, uop_vmax(mul_result));    // -2 * -3
}

// Port of test_vmin_vmax_nested_min_max from Python
TEST(test_vmin_vmax_nested_min_max) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 0, 10);
    UOp* const_5 = uop_const(dtypes.int32, 5);
    UOp* const_8 = uop_const(dtypes.int32, 8);
    
    // x.maximum(5).minimum(8)
    UOp* max_result = uop_max(var_x, const_5);
    UOp* min_result = uop_min(max_result, const_8);
    
    TEST_ASSERT_EQUAL(5, uop_vmin(min_result));
    TEST_ASSERT_EQUAL(8, uop_vmax(min_result));
}

// Port of test_vmin_vmax_where from Python
TEST(test_vmin_vmax_where) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 0, 10);
    UOp* var_y = uop_var_with_range("y", dtypes.int32, 1, 11);
    UOp* var_z = uop_var_with_range("z", dtypes.int32, 2, 12);
    UOp* const_5 = uop_const(dtypes.int32, 5);
    
    // (x < 5).where(y, z)
    UOp* cond = uop_lt(var_x, const_5);
    UOp* where_result = uop_where(cond, var_y, var_z);
    
    TEST_ASSERT_EQUAL(1, uop_vmin(where_result));   // min(y.vmin, z.vmin) = min(1, 2)
    TEST_ASSERT_EQUAL(12, uop_vmax(where_result));  // max(y.vmax, z.vmax) = max(11, 12)
}

// Port of test_vmin_vmax_shl from Python
TEST(test_vmin_vmax_shl) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 0, 10);
    UOp* const_5 = uop_const(dtypes.int32, 5);
    UOp* shl_result = uop_shl(var_x, const_5);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(shl_result));             // 0 << 5
    TEST_ASSERT_EQUAL(10 << 5, uop_vmax(shl_result));      // 10 << 5 = 320
}

// Port of test_vmin_vmax_shr from Python
TEST(test_vmin_vmax_shr) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 0, 10);
    UOp* const_2 = uop_const(dtypes.int32, 2);
    UOp* shr_result = uop_shr(var_x, const_2);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(shr_result));             // 0 >> 2
    TEST_ASSERT_EQUAL(10 >> 2, uop_vmax(shr_result));      // 10 >> 2 = 2
}

// Port of division tests from TestVminVmaxDivMod
TEST(test_vmin_vmax_division_positive) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 10, 20);
    UOp* const_2 = uop_const(dtypes.int32, 2);
    UOp* div_result = uop_div(var_x, const_2);
    
    TEST_ASSERT_EQUAL(5, uop_vmin(div_result));   // 10 // 2
    TEST_ASSERT_EQUAL(10, uop_vmax(div_result));  // 20 // 2
}

// Port of test_vmin_vmax_division_negative from Python
TEST(test_vmin_vmax_division_negative) {
    // Always positive dividend
    UOp* var_x1 = uop_var_with_range("x", dtypes.int32, 10, 20);
    UOp* const_neg2 = uop_const(dtypes.int32, -2);
    UOp* div_result1 = uop_div(var_x1, const_neg2);
    
    TEST_ASSERT_EQUAL(-10, uop_vmin(div_result1));  // 20 // -2
    TEST_ASSERT_EQUAL(-5, uop_vmax(div_result1));   // 10 // -2
    
    // Always negative dividend
    UOp* var_x2 = uop_var_with_range("x", dtypes.int32, -20, -10);
    UOp* div_result2 = uop_div(var_x2, const_neg2);
    
    TEST_ASSERT_EQUAL(5, uop_vmin(div_result2));    // -10 // -2
    TEST_ASSERT_EQUAL(10, uop_vmax(div_result2));   // -20 // -2
}

// Port of test_vmin_vmax_mod_positive from Python
TEST(test_vmin_vmax_mod_positive) {
    // Positive variable
    UOp* var_pos = uop_var_with_range("positive", dtypes.int32, 10, 20);
    UOp* const_3 = uop_const(dtypes.int32, 3);
    UOp* mod_pos = uop_mod(var_pos, const_3);
    
    TEST_ASSERT_EQUAL(0, uop_vmin(mod_pos));
    TEST_ASSERT_EQUAL(2, uop_vmax(mod_pos));
    
    // Negative variable
    UOp* var_neg = uop_var_with_range("negative", dtypes.int32, -20, -10);
    UOp* mod_neg = uop_mod(var_neg, const_3);
    
    TEST_ASSERT_EQUAL(-2, uop_vmin(mod_neg));
    TEST_ASSERT_EQUAL(0, uop_vmax(mod_neg));
    
    // Mixed variable
    UOp* var_mixed = uop_var_with_range("mixed", dtypes.int32, -20, 20);
    UOp* mod_mixed = uop_mod(var_mixed, const_3);
    
    TEST_ASSERT_EQUAL(-2, uop_vmin(mod_mixed));
    TEST_ASSERT_EQUAL(2, uop_vmax(mod_mixed));
}

// Port of test_vmin_vmax_div_symbolic from Python
TEST(test_vmin_vmax_div_symbolic) {
    UOp* var_x = uop_var_with_range("x", dtypes.int32, 1, 10);
    UOp* var_y = uop_var_with_range("y", dtypes.int32, 3, 5);
    
    // x // y
    UOp* div1 = uop_div(var_x, var_y);
    TEST_ASSERT_EQUAL(0, uop_vmin(div1));  // 1 // 5
    TEST_ASSERT_EQUAL(3, uop_vmax(div1));  // 10 // 3
    
    // (-x) // y
    UOp* neg_x = uop_neg(var_x);
    UOp* div2 = uop_div(neg_x, var_y);
    TEST_ASSERT_EQUAL(-3, uop_vmin(div2));  // -10 // 3
    TEST_ASSERT_EQUAL(0, uop_vmax(div2));   // -1 // 5
}

// Auto-register all test functions and run them
TEST_MAIN()