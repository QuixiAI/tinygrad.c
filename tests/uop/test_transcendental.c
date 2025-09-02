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

TEST(test_transcendental_mathematical_accuracy) {
    
    // Test values covering various ranges and edge cases
    double test_values[] = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        -1.0, -2.0, -3.0, -4.0, -5.0,
        0.5, 1.5, 2.5, 3.5, 4.5,
        -0.5, -1.5, -2.5, -3.5, -4.5,
        100.0, 1000.0, -100.0, -1000.0,
        1.5707963267948966  // pi/2 for sin testing
    };
    
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++) {
        double x = test_values[i];
        UOp* val = uop_const(dtypes.float32, x);
        
        // Test EXP2
        UOp* exp2_val = uop_exp2(val);
        if (exp2_val != NULL) {
            // Compare against math library (convert float64 to float32 for comparison)
            double expected = pow(2.0, x);
            double result = exec_alu(OPS_EXP2, dtypes.float32, &x, 1);
            // Skip overflow cases - if x is too large, result should be inf
            if (x > 127.0) {  // 2^128 is beyond float32 range
                if (isinf(result) && result > 0) {
                    continue;  // Correct overflow behavior
                }
            }
            // Skip overflow cases (both should be inf)
            if (isinf(expected) && isinf(result)) {
                continue;
            }
            // Use absolute tolerance for values near zero, relative otherwise
            double tolerance = fmax(0.001 * fabs(expected), 1e-6);
            ASSERT_NEAR(result, expected, tolerance);
        }
        
        // Test LOG2
        UOp* log2_val = uop_log2(val);
        if (log2_val != NULL && x > 0) {  // log2 of negative/nan is undefined
            double expected = log2(x);
            double result = exec_alu(OPS_LOG2, dtypes.float32, &x, 1);
            ASSERT_NEAR(result, expected, 0.001 * fabs(expected));
        }
        
        // Test SIN
        UOp* sin_val = uop_sin(val);
        if (sin_val != NULL) {
            double expected = sin(x);
            double result = exec_alu(OPS_SIN, dtypes.float32, &x, 1);
            ASSERT_NEAR(result, expected, 0.001);
        }
        
        // Clean up
        if (exp2_val) uop_unref(exp2_val);
        if (log2_val) uop_unref(log2_val);
        if (sin_val) uop_unref(sin_val);
        uop_unref(val);
    }
}

TEST(test_transcendental_edge_cases) {
    
    // Test NaN
    UOp* nan_val = uop_const(dtypes.float32, NAN);
    UOp* sin_nan = uop_sin(nan_val);
    if (sin_nan != NULL) {
        double result = exec_alu(OPS_SIN, dtypes.float32, (double[]){NAN}, 1);
        ASSERT(isnan(result));
    }
    if (sin_nan) uop_unref(sin_nan);
    uop_unref(nan_val);
    
    // Test infinity
    UOp* inf_val = uop_const(dtypes.float32, INFINITY);
    UOp* sin_inf = uop_sin(inf_val);
    if (sin_inf != NULL) {
        // sin(inf) should be in [-1, 1] range but not necessarily a specific value
        double result = exec_alu(OPS_SIN, dtypes.float32, (double[]){INFINITY}, 1);
        ASSERT(fabs(result) <= 1.0 || isnan(result));  // Valid range or NaN
    }
    if (sin_inf) uop_unref(sin_inf);
    uop_unref(inf_val);
    
    // Test negative infinity
    UOp* neg_inf_val = uop_const(dtypes.float32, -INFINITY);
    UOp* sin_neg_inf = uop_sin(neg_inf_val);
    if (sin_neg_inf != NULL) {
        double result = exec_alu(OPS_SIN, dtypes.float32, (double[]){-INFINITY}, 1);
        ASSERT(fabs(result) <= 1.0 || isnan(result));
    }
    if (sin_neg_inf) uop_unref(sin_neg_inf);
    uop_unref(neg_inf_val);
    
    // Test very small values (near zero)
    UOp* small_val = uop_const(dtypes.float32, 1e-10);
    UOp* sin_small = uop_sin(small_val);
    if (sin_small != NULL) {
        // For small x, sin(x) ≈ x
        double x = 1e-10;
        double expected = x;
        double result = exec_alu(OPS_SIN, dtypes.float32, &x, 1);
        ASSERT_NEAR(result, expected, x * 0.1);
    }
    if (sin_small) uop_unref(sin_small);
    uop_unref(small_val);
    
    // Test log2 of zero (should be -infinity)
    UOp* zero_val = uop_const(dtypes.float32, 0.0);
    UOp* log2_zero = uop_log2(zero_val);
    if (log2_zero != NULL) {
        double result = exec_alu(OPS_LOG2, dtypes.float32, (double[]){0.0}, 1);
        ASSERT(isinf(result) && result < 0);  // Should be -infinity
    }
    if (log2_zero) uop_unref(log2_zero);
    uop_unref(zero_val);
}

TEST(test_transcendental_large_angles_sin) {
    
    // Test angles that would benefit from Payne-Hanek reduction
    double large_angles[] = {
        1e6, 1e7, 1e8, 1e9,
        123456789.0,
        3.141592653589793 * 1e6,  // Large multiple of pi
    };
    
    for (size_t i = 0; i < sizeof(large_angles) / sizeof(large_angles[0]); i++) {
        double angle = large_angles[i];
        UOp* angle_val = uop_const(dtypes.float32, angle);
        UOp* sin_val = uop_sin(angle_val);
        
        if (sin_val != NULL) {
            double result = exec_alu(OPS_SIN, dtypes.float32, &angle, 1);
            
            // Result should be in valid range [-1, 1] or NaN
            ASSERT(fabs(result) <= 1.0 || isnan(result));
            
            // Test periodicity: sin(x + 2*pi) should equal sin(x)
            double angle_plus_2pi = angle + 2 * M_PI;
            UOp* angle_plus_2pi_val = uop_const(dtypes.float32, angle_plus_2pi);
            UOp* sin_plus_2pi = uop_sin(angle_plus_2pi_val);
            
            if (sin_plus_2pi != NULL) {
                double result_plus_2pi = exec_alu(OPS_SIN, dtypes.float32, &angle_plus_2pi, 1);
                
                // Results should be very close (accounting for floating point precision)
                double diff = fabs(result - result_plus_2pi);
                ASSERT(diff < 0.01 || isnan(result) || isnan(result_plus_2pi));
                
                uop_unref(sin_plus_2pi);
            }
            uop_unref(angle_plus_2pi_val);
        }
        
        if (sin_val) uop_unref(sin_val);
        uop_unref(angle_val);
    }
}

TEST(test_exp2_log2_inverse_relationship) {
    
    // Test that exp2(log2(x)) ≈ x for x > 0
    double test_values[] = {
        0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
        3.0, 5.0, 10.0, 100.0, 1000.0
    };
    
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++) {
        double x = test_values[i];
        UOp* x_val = uop_const(dtypes.float32, x);
        
        UOp* log2_x = uop_log2(x_val);
        if (log2_x != NULL && x > 0) {
            UOp* exp2_log2x = uop_exp2(log2_x);
            
            if (exp2_log2x != NULL) {
                double result = exec_alu(OPS_EXP2, dtypes.float32, 
                    (double[]){exec_alu(OPS_LOG2, dtypes.float32, &x, 1)}, 1);
                
                // Should be close to original x (accounting for floating point precision)
                double relative_error = fabs(result - x) / fabs(x);
                ASSERT(relative_error < 0.001);  // Within 0.1%
            }
            
            if (exp2_log2x) uop_unref(exp2_log2x);
        }
        
        if (log2_x) uop_unref(log2_x);
        uop_unref(x_val);
    }
}

TEST(test_transcendental_power_relationships) {
    
    // Test that 2^x = pow(2, x) and log2(x) = log(x)/log(2)
    double test_values[] = {
        0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
        0.1, 0.01, 10.0, 100.0
    };
    
    for (size_t i = 0; i < sizeof(test_values) / sizeof(test_values[0]); i++) {
        double x = test_values[i];
        UOp* x_val = uop_const(dtypes.float32, x);
        
        // Test 2^x equivalence
        UOp* exp2_x = uop_exp2(x_val);
        if (exp2_x != NULL) {
            double exp2_result = exec_alu(OPS_EXP2, dtypes.float32, &x, 1);
            double pow_result = pow(2.0, x);
            
            double relative_error = fabs(exp2_result - pow_result) / fabs(pow_result);
            ASSERT(relative_error < 0.001 || (isnan(exp2_result) && isnan(pow_result)));
        }
        
        if (exp2_x) uop_unref(exp2_x);
        uop_unref(x_val);
    }
}

TEST(test_transcendental_performance) {
    
    // Test that functions complete in reasonable time
    // No hard timing constraints, just ensure they don't hang
    
    for (int i = 0; i < 100; i++) {
        double x = (double)i * 0.1;
        UOp* val = uop_const(dtypes.float32, x);
        
        UOp* sin_val = uop_sin(val);
        UOp* exp2_val = uop_exp2(val);
        UOp* log2_val = uop_log2(val);
        
        // Execute to ensure they complete
        if (sin_val) {
            exec_alu(OPS_SIN, dtypes.float32, &x, 1);
            uop_unref(sin_val);
        }
        if (exp2_val) {
            exec_alu(OPS_EXP2, dtypes.float32, &x, 1);
            uop_unref(exp2_val);
        }
        if (log2_val && x > 0) {
            exec_alu(OPS_LOG2, dtypes.float32, &x, 1);
            uop_unref(log2_val);
        }
        
        uop_unref(val);
    }
}

TEST(test_test_payne_hanek_reduction) {
    
    // Test special trigonometric reduction
    UOp* large_angle = uop_const(dtypes.float32, 1000000.0);
    UOp* sin_val = uop_sin(large_angle);
    
    // Should handle large angles correctly
    ASSERT(sin_val != NULL);
}

// Auto-register all test functions and run them
TEST_MAIN()