#include "test_uop_common.h"

void setUp(void) {
    // Initialize test fixtures if needed
    static int initialized = 0;
    if (!initialized) {
        dtypes_init();
        uop_ops_init();
        initialized = 1;
    }
}

void tearDown(void) {
    // Clean up after each test if needed
}

// Test Ops enum and basic operations
TEST(test_ops_enum) {
    
    // Test that ops have unique values
    ASSERT(OPS_NOOP != OPS_SINK);
    ASSERT(OPS_ADD != OPS_MUL);
    ASSERT(OPS_LOAD != OPS_STORE);
    
    // Test ops_to_string
    ASSERT(strcmp(ops_to_string(OPS_ADD), "ADD") == 0);
    ASSERT(strcmp(ops_to_string(OPS_MUL), "MUL") == 0);
    ASSERT(strcmp(ops_to_string(OPS_LOAD), "LOAD") == 0);
    
    // Test ops_is_valid
    ASSERT(ops_is_valid(OPS_ADD) == true);
    ASSERT(ops_is_valid(OPS_MUL) == true);
    ASSERT(ops_is_valid(OPS_MAX_VALUE) == false);
    ASSERT(ops_is_valid(0) == false);
    
    // Test ops_get_arity
    ASSERT(ops_get_arity(OPS_NEG) == 1);  // Unary
    ASSERT(ops_get_arity(OPS_ADD) == 2);  // Binary
    ASSERT(ops_get_arity(OPS_WHERE) == 3); // Ternary
    ASSERT(ops_get_arity(OPS_CONST) == 0); // No sources
}

// Test GroupOp classifications
TEST(test_group_ops) {
    
    // Intentional failure to verify Unity is working
    // ASSERT(1 == 2);  // Uncomment to test
    
    // Test Unary ops
    ASSERT(group_op.is_unary[OPS_NEG] == true);
    ASSERT(group_op.is_unary[OPS_EXP2] == true);
    ASSERT(group_op.is_unary[OPS_LOG2] == true);
    ASSERT(group_op.is_unary[OPS_SIN] == true);
    ASSERT(group_op.is_unary[OPS_SQRT] == true);
    ASSERT(group_op.is_unary[OPS_RECIP] == true);
    ASSERT(group_op.is_unary[OPS_ADD] == false);
    
    // Test Binary ops
    ASSERT(group_op.is_binary[OPS_ADD] == true);
    ASSERT(group_op.is_binary[OPS_MUL] == true);
    ASSERT(group_op.is_binary[OPS_MAX] == true);
    ASSERT(group_op.is_binary[OPS_CMPLT] == true);
    ASSERT(group_op.is_binary[OPS_NEG] == false);
    
    // Test Ternary ops
    ASSERT(group_op.is_ternary[OPS_WHERE] == true);
    ASSERT(group_op.is_ternary[OPS_MULACC] == true);
    ASSERT(group_op.is_ternary[OPS_ADD] == false);
    
    // Test ALU ops (union of Unary, Binary, Ternary)
    ASSERT(group_op.is_alu[OPS_ADD] == true);
    ASSERT(group_op.is_alu[OPS_NEG] == true);
    ASSERT(group_op.is_alu[OPS_WHERE] == true);
    ASSERT(group_op.is_alu[OPS_LOAD] == false);
    
    // Test Commutative ops
    ASSERT(group_op.is_commutative[OPS_ADD] == true);
    ASSERT(group_op.is_commutative[OPS_MUL] == true);
    ASSERT(group_op.is_commutative[OPS_MAX] == true);
    ASSERT(group_op.is_commutative[OPS_SUB] == false);
    ASSERT(group_op.is_commutative[OPS_IDIV] == false);
    
    // Test Associative ops
    ASSERT(group_op.is_associative[OPS_ADD] == true);
    ASSERT(group_op.is_associative[OPS_MUL] == true);
    ASSERT(group_op.is_associative[OPS_MAX] == true);
    ASSERT(group_op.is_associative[OPS_SUB] == false);
    
    // Test Movement ops
    ASSERT(group_op.is_movement[OPS_RESHAPE] == true);
    ASSERT(group_op.is_movement[OPS_PERMUTE] == true);
    ASSERT(group_op.is_movement[OPS_EXPAND] == true);
    ASSERT(group_op.is_movement[OPS_PAD] == true);
    ASSERT(group_op.is_movement[OPS_ADD] == false);
    
    // Test Define ops
    ASSERT(group_op.is_define[OPS_DEFINE_GLOBAL] == true);
    ASSERT(group_op.is_define[OPS_DEFINE_LOCAL] == true);
    ASSERT(group_op.is_define[OPS_DEFINE_REG] == true);
    ASSERT(group_op.is_define[OPS_ADD] == false);
}

// Test exec_alu function
TEST(test_exec_alu) {
    
    // Test ADD
    double args_add[] = {10.0, 20.0};
    double result = exec_alu(OPS_ADD, dtypes.float32, args_add, 2);
    ASSERT_FLOAT_EQ(result, 30.0, 1e-6);
    
    // Test MUL
    double args_mul[] = {3.0, 7.0};
    result = exec_alu(OPS_MUL, dtypes.float32, args_mul, 2);
    ASSERT_FLOAT_EQ(result, 21.0, 1e-6);
    
    // Test NEG
    double args_neg[] = {-5.0};
    result = exec_alu(OPS_NEG, dtypes.float32, args_neg, 1);
    ASSERT_FLOAT_EQ(result, 5.0, 1e-6);
    
    // Test SQRT
    double args_sqrt[] = {16.0};
    result = exec_alu(OPS_SQRT, dtypes.float32, args_sqrt, 1);
    ASSERT_FLOAT_EQ(result, 4.0, 1e-6);
    
    // Test MAX
    double args_max[] = {10.0, 20.0};
    result = exec_alu(OPS_MAX, dtypes.float32, args_max, 2);
    ASSERT_FLOAT_EQ(result, 20.0, 1e-6);
    
    // Test CMPLT
    double args_lt[] = {10.0, 20.0};
    result = exec_alu(OPS_CMPLT, dtypes.bool_, args_lt, 2);
    ASSERT(result == 1.0);  // true
    
    double args_lt2[] = {20.0, 10.0};
    result = exec_alu(OPS_CMPLT, dtypes.bool_, args_lt2, 2);
    ASSERT(result == 0.0);  // false
    
    // Test WHERE
    double args_where[] = {1.0, 42.0, 99.0};  // condition, true_val, false_val
    result = exec_alu(OPS_WHERE, dtypes.float32, args_where, 3);
    ASSERT_FLOAT_EQ(result, 42.0, 1e-6);
    
    double args_where2[] = {0.0, 42.0, 99.0};
    result = exec_alu(OPS_WHERE, dtypes.float32, args_where2, 3);
    ASSERT_FLOAT_EQ(result, 99.0, 1e-6);
}

// Test bitwise operations
TEST(test_bitwise_operations) {
    
    // Create integer constants
    UOp* a = uop_const(dtypes.int32, 0b1010);  // 10
    UOp* b = uop_const(dtypes.int32, 0b1100);  // 12
    
    // Test AND
    UOp* and_result = a->math_ops->bitwise_and(a, b, false);
    ASSERT(and_result != NULL);
    ASSERT(and_result->op == OPS_AND);
    
    // Test OR
    UOp* or_result = a->math_ops->bitwise_or(a, b, false);
    ASSERT(or_result != NULL);
    ASSERT(or_result->op == OPS_OR);
    
    // Test XOR
    UOp* xor_result = a->math_ops->bitwise_xor(a, b, false);
    ASSERT(xor_result != NULL);
    ASSERT(xor_result->op == OPS_XOR);
    
    // Test SHL (shift left)
    UOp* shift_amount = uop_const(dtypes.int32, 2);
    UOp* shl_result = a->math_ops->lshift(a, shift_amount, false);
    ASSERT(shl_result != NULL);
    ASSERT(shl_result->op == OPS_SHL);
    
    // Test SHR (shift right)
    UOp* shr_result = a->math_ops->rshift(a, shift_amount, false);
    ASSERT(shr_result != NULL);
    ASSERT(shr_result->op == OPS_SHR);
    
    // Clean up
    uop_unref(a);
    uop_unref(b);
    uop_unref(and_result);
    uop_unref(or_result);
    uop_unref(xor_result);
    uop_unref(shift_amount);
    uop_unref(shl_result);
    uop_unref(shr_result);
}

// Test comparison operations in detail
TEST(test_comparison_operations) {
    
    UOp* a = uop_const(dtypes.float32, 10.0);
    UOp* b = uop_const(dtypes.float32, 20.0);
    
    // Test CMPLT (less than)
    UOp* lt = uop_lt(a, b);
    ASSERT(lt != NULL);
    ASSERT(lt->op == OPS_CMPLT);
    ASSERT(dtype_eq(&lt->dtype, &dtypes.bool_));
    
    // Test CMPEQ (equal)
    UOp* eq = uop_eq(a, b);
    ASSERT(eq != NULL);
    ASSERT(eq->op == OPS_CMPEQ);
    ASSERT(dtype_eq(&eq->dtype, &dtypes.bool_));
    
    // Test CMPNE (not equal)
    UOp* ne = uop_ne(a, b);
    ASSERT(ne != NULL);
    ASSERT(ne->op == OPS_CMPNE);
    ASSERT(dtype_eq(&ne->dtype, &dtypes.bool_));
    
    // Test compound comparisons (>=, <=, >)
    // a <= b is !(b < a)
    UOp* b_lt_a = uop_lt(b, a);
    UOp* le = b_lt_a->math_ops->logical_not(b_lt_a);
    ASSERT(le != NULL);
    
    // a > b is b < a
    UOp* gt = uop_lt(b, a);
    ASSERT(gt != NULL);
    
    // a >= b is !(a < b)
    UOp* ge = lt->math_ops->logical_not(lt);
    ASSERT(ge != NULL);
    
    // Clean up
    uop_unref(a);
    uop_unref(b);
    uop_unref(lt);
    uop_unref(eq);
    uop_unref(ne);
    uop_unref(b_lt_a);
    uop_unref(le);
    uop_unref(gt);
    uop_unref(ge);
}

// Test MIN/MAX operations
TEST(test_min_max_operations) {
    
    UOp* a = uop_const(dtypes.float32, 10.0);
    UOp* b = uop_const(dtypes.float32, 20.0);
    
    // Test MAX
    UOp* max_val = uop_max(a, b);
    ASSERT(max_val != NULL);
    ASSERT(max_val->op == OPS_MAX);
    
    // Test MIN (using maximum with negation or WHERE)
    UOp* min_val = uop_min(a, b);
    ASSERT(min_val != NULL);
    // MIN might be implemented as WHERE(a < b, a, b)
    
    // Clean up
    uop_unref(a);
    uop_unref(b);
    uop_unref(max_val);
    uop_unref(min_val);
}

// Test ternary operations
TEST(test_ternary_operations) {
    
    // Test WHERE
    UOp* cond = uop_const(dtypes.bool_, 1.0);  // true
    UOp* true_val = uop_const(dtypes.float32, 42.0);
    UOp* false_val = uop_const(dtypes.float32, 99.0);
    
    UOp* where_result = uop_where(cond, true_val, false_val);
    ASSERT(where_result != NULL);
    ASSERT(where_result->op == OPS_WHERE);
    ASSERT(where_result->src_count == 3);
    ASSERT(where_result->src[0] == cond);
    ASSERT(where_result->src[1] == true_val);
    ASSERT(where_result->src[2] == false_val);
    
    // Test MULACC (multiply-accumulate: a * b + c)
    UOp* a = uop_const(dtypes.float32, 2.0);
    UOp* b = uop_const(dtypes.float32, 3.0);
    UOp* c = uop_const(dtypes.float32, 10.0);
    
    UOp* mulacc_result = uop_mulacc(a, b, c);
    ASSERT(mulacc_result != NULL);
    ASSERT(mulacc_result->op == OPS_MULACC);
    ASSERT(mulacc_result->src_count == 3);
    
    // Clean up
    uop_unref(cond);
    uop_unref(true_val);
    uop_unref(false_val);
    uop_unref(where_result);
    uop_unref(a);
    uop_unref(b);
    uop_unref(c);
    uop_unref(mulacc_result);
}

TEST(test_overflow_behavior) {
    
    // Test uint8 overflow
    double args_overflow[] = {250.0, 250.0};
    double result = exec_alu(OPS_ADD, dtypes.uint8, args_overflow, 2);
    ASSERT_NEAR(result, 244.0, 0.01);  // 500 & 0xFF = 244
    
    // Test int8 overflow
    double args_int8[] = {127.0, 10.0};
    result = exec_alu(OPS_ADD, dtypes.int8, args_int8, 2);
    ASSERT_NEAR(result, -119.0, 0.01);  // Wraps around
}

TEST(test_float_edge_cases) {
    
    // Test NaN
    double nan_val = 0.0/0.0;
    double args_nan[] = {nan_val};
    double result = exec_alu(OPS_SQRT, dtypes.float32, args_nan, 1);
    ASSERT(isnan(result));
    
    // Test infinity
    double inf_val = 1.0/0.0;
    double args_inf[] = {inf_val};
    result = exec_alu(OPS_RECIP, dtypes.float32, args_inf, 1);
    ASSERT_NEAR(result, 0.0, 0.0001);
    
    // Test division by zero  
    double args_div[] = {0.0};
    result = exec_alu(OPS_RECIP, dtypes.float32, args_div, 1);
    ASSERT(isinf(result));
}

TEST(test_integer_division) {
    
    // Test IDIV with truncation
    double args1[] = {7.0, 3.0};
    double result = exec_alu(OPS_IDIV, dtypes.int32, args1, 2);
    ASSERT_NEAR(result, 2.0, 0.01);
    
    // Test IDIV with negative
    double args2[] = {7.0, -3.0};
    result = exec_alu(OPS_IDIV, dtypes.int32, args2, 2);
    ASSERT_NEAR(result, -2.0, 0.01);
    
    // Test MOD with negative
    double args3[] = {-7.0, 3.0};
    result = exec_alu(OPS_MOD, dtypes.int32, args3, 2);
    ASSERT_NEAR(result, -1.0, 0.01);
}

TEST(test_shift_operations) {
    
    // Test SHL (shift left)
    double args_shl[] = {5.0, 2.0};
    double result = exec_alu(OPS_SHL, dtypes.int32, args_shl, 2);
    ASSERT_NEAR(result, 20.0, 0.01);  // 5 << 2 = 20
    
    // Test SHR (shift right)
    double args_shr[] = {20.0, 2.0};
    result = exec_alu(OPS_SHR, dtypes.int32, args_shr, 2);
    ASSERT_NEAR(result, 5.0, 0.01);  // 20 >> 2 = 5
}

TEST(test_boolean_alu) {
    
    // Test XOR on booleans
    double args_xor[] = {1.0, 0.0};
    double result = exec_alu(OPS_XOR, dtypes.bool_, args_xor, 2);
    ASSERT_NEAR(result, 1.0, 0.01);
    
    // Test AND on booleans
    double args_and[] = {1.0, 1.0};
    result = exec_alu(OPS_AND, dtypes.bool_, args_and, 2);
    ASSERT_NEAR(result, 1.0, 0.01);
    
    // Test OR on booleans
    double args_or[] = {0.0, 1.0};
    result = exec_alu(OPS_OR, dtypes.bool_, args_or, 2);
    ASSERT_NEAR(result, 1.0, 0.01);
}

TEST(test_comprehensive_alu) {
    
    // Test MULACC operation
    double args_mulacc[] = {2.0, 3.0, 4.0};
    double result = exec_alu(OPS_MULACC, dtypes.float32, args_mulacc, 3);
    ASSERT_NEAR(result, 10.0, 0.01);  // 2*3+4 = 10
    
    // Test CMPNE
    double args_ne[] = {5.0, 5.0};
    result = exec_alu(OPS_CMPNE, dtypes.float32, args_ne, 2);
    ASSERT_NEAR(result, 0.0, 0.01);  // 5 != 5 is false
    
    // Test CMPEQ  
    double args_eq[] = {5.0, 5.0};
    result = exec_alu(OPS_CMPEQ, dtypes.float32, args_eq, 2);
    ASSERT_NEAR(result, 1.0, 0.01);  // 5 == 5 is true
}

TEST(test_exec_alu_overflow) {
    
    // Test overflow handling like TestExecALU.test_overflow
    double max_int32 = 2147483647.0;
    double result = exec_alu(OPS_ADD, dtypes.int32, (double[]){max_int32, 1}, 2);
    
    // Should wrap to negative
    ASSERT_NEAR(result, -2147483648.0, 0.001);  // Will fail until overflow handled
}

TEST(test_division_by_zero) {
    
    // Test division by zero handling
    double result = exec_alu(OPS_FDIV, dtypes.float32, (double[]){1.0, 0.0}, 2);
    ASSERT(isinf(result));  // Should be infinity
    
    // Test RECIP of zero
    result = exec_alu(OPS_RECIP, dtypes.float32, (double[]){0.0}, 1);
    ASSERT(isinf(result));  // Should be infinity
}

TEST(test_nan_inf_handling) {
    
    // Test NaN propagation
    double nan = NAN;
    double result = exec_alu(OPS_ADD, dtypes.float32, (double[]){nan, 1.0}, 2);
    ASSERT(isnan(result));  // NaN should propagate
    
    // Test infinity arithmetic
    double inf = INFINITY;
    result = exec_alu(OPS_ADD, dtypes.float32, (double[]){inf, 1.0}, 2);
    ASSERT(isinf(result));  // Should remain infinity
}

TEST(test_boolean_logic_comprehensive) {
    
    // Test all boolean operations like TestBoolUOps
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            // AND
            double result = exec_alu(OPS_AND, dtypes.bool_, (double[]){a, b}, 2);
            ASSERT_NEAR(result, a && b, 0.001);
            
            // OR
            result = exec_alu(OPS_OR, dtypes.bool_, (double[]){a, b}, 2);
            ASSERT_NEAR(result, a || b, 0.001);
            
            // XOR
            result = exec_alu(OPS_XOR, dtypes.bool_, (double[]){a, b}, 2);
            ASSERT_NEAR(result, a != b, 0.001);
        }
    }
}

TEST(test_shift_edge_cases) {
    
    // Test shift with negative values (should be prevented)
    double result = exec_alu(OPS_SHL, dtypes.int32, (double[]){1, -1}, 2);
    // Negative shift should be handled properly
    ASSERT(!isnan(result));
    
    // Test shift by large amounts
    result = exec_alu(OPS_SHR, dtypes.int32, (double[]){256, 32}, 2);
    ASSERT_NEAR(result, 0, 0.001);  // Should shift to zero
}

TEST(test_int32_operations) {
    
    // Test int32 ADD
    double result = exec_alu(OPS_ADD, dtypes.int32, (double[]){100, 200}, 2);
    ASSERT_NEAR(result, 300, 0.001);
    
    // Test int32 MUL
    result = exec_alu(OPS_MUL, dtypes.int32, (double[]){5, 6}, 2);
    ASSERT_NEAR(result, 30, 0.001);
    
    // Test int32 AND
    result = exec_alu(OPS_AND, dtypes.int32, (double[]){0xFF, 0x0F}, 2);
    ASSERT_NEAR(result, 0x0F, 0.001);
    
    // Test int32 OR
    result = exec_alu(OPS_OR, dtypes.int32, (double[]){0xF0, 0x0F}, 2);
    ASSERT_NEAR(result, 0xFF, 0.001);
    
    // Test int32 comparisons
    result = exec_alu(OPS_CMPLT, dtypes.int32, (double[]){5, 10}, 2);
    ASSERT_NEAR(result, 1.0, 0.001);
}

TEST(test_float16_operations) {
    
    // Test float16 WHERE
    UOp* cond = uop_const(dtypes.bool_, 1);
    UOp* true_val = uop_const(dtypes.float16, 1.0);
    UOp* false_val = uop_const(dtypes.float16, 2.0);
    UOp* result = uop_where(cond, true_val, false_val);
    ASSERT(result != NULL);
}

TEST(test_modulo_operations) {
    
    // Test basic MOD
    double args[] = {10.0, 3.0};
    double result = exec_alu(OPS_MOD, dtypes.int32, args, 2);
    ASSERT_NEAR(result, 1.0, 0.01);
    
    // Test MOD with negative dividend
    double args2[] = {-10.0, 3.0};
    result = exec_alu(OPS_MOD, dtypes.int32, args2, 2);
    ASSERT_NEAR(result, -1.0, 0.01);
    
    // Test MOD with negative divisor
    double args3[] = {10.0, -3.0};
    result = exec_alu(OPS_MOD, dtypes.int32, args3, 2);
    ASSERT_NEAR(result, 1.0, 0.01);
}

// Auto-register all test functions and run them
TEST_MAIN()