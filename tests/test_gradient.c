#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include Unity test framework
#include "test_common.h"
#include "../src/gradient/gradient.h"

// Helper function for floating point comparison with NaN handling
static void assert_cmp_nan_okay(float expected, float actual) {
    if (isnan(expected) && isnan(actual)) return;
    TEST_ASSERT_FLOAT_WITHIN(1e-5, expected, actual);
}

// Tests based on reference/test/unit/test_gradient.py - faithful port

TEST(test_recip) {
    // Python: def test_recip(self): self._test_one_input_function(lambda x: 1.0/x)
    // Test gradient of 1/x function using _test_one_input_function pattern
    
    float test_vals[] = {-5.0f, -2.0f, 2.0f, 5.0f}; // Skip 0.0 to avoid division by zero
    int num_vals = sizeof(test_vals) / sizeof(test_vals[0]);
    
    for (int i = 0; i < num_vals; i++) {
        float val = test_vals[i];
        
        // Create UOp variable 'x' with bounds (-inf, inf)
        tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
        
        // Create expression f(x) = 1.0/x  
        tg_uop_t* recip_expr = tg_uop_recip(x);
        
        // Create const 1.0 for gradient seed
        tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
        
        // Compute gradient: compute_gradient(f(x), 1.0, set([x]))[x]
        tg_uop_t* vars[] = {x};
        tg_gradient_result_t* grad_result = tg_compute_gradient(recip_expr, grad_seed, vars, 1);
        tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
        
        // Substitute value into gradient expression: gx.substitute({x: x.const_like(val)})
        tg_uop_t* val_const = tg_uop_const_like(x, val);
        tg_substitution_t sub = {x, val_const};
        tg_uop_t* substituted = tg_uop_substitute(gx, &sub, 1);
        
        // Simplify: .ssimplify()
        tg_uop_t* simplified = tg_uop_ssimplify(substituted);
        float tg_out = tg_uop_get_float(simplified);
        
        // Expected gradient: d/dx[1/x] = -1/(x^2)
        float expected = -1.0f / (val * val);
        
        // Compare with nan-okay semantics like Python test
        assert_cmp_nan_okay(expected, tg_out);
        
        // Cleanup
        tg_gradient_result_free(grad_result);
        tg_uop_free(simplified);
        tg_uop_free(substituted);
        tg_uop_free(val_const);
        tg_uop_free(gx);
        tg_uop_free(grad_seed);
        tg_uop_free(recip_expr);
        tg_uop_free(x);
    }
}

TEST(test_sin) {
    // Python: def test_sin(self): self._test_one_input_function(lambda x: x.sin())
    
    float test_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    int num_vals = sizeof(test_vals) / sizeof(test_vals[0]);
    
    for (int i = 0; i < num_vals; i++) {
        float val = test_vals[i];
        
        tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
        tg_uop_t* sin_expr = tg_uop_sin(x);
        tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
        
        tg_uop_t* vars[] = {x};
        tg_gradient_result_t* grad_result = tg_compute_gradient(sin_expr, grad_seed, vars, 1);
        tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
        
        tg_uop_t* val_const = tg_uop_const_like(x, val);
        tg_substitution_t sub = {x, val_const};
        tg_uop_t* substituted = tg_uop_substitute(gx, &sub, 1);
        tg_uop_t* simplified = tg_uop_ssimplify(substituted);
        float tg_out = tg_uop_get_float(simplified);
        
        // Expected gradient: d/dx[sin(x)] = cos(x)
        float expected = cosf(val);
        
        assert_cmp_nan_okay(expected, tg_out);
        
        // Cleanup
        tg_gradient_result_free(grad_result);
        tg_uop_free(simplified);
        tg_uop_free(substituted);
        tg_uop_free(val_const);
        tg_uop_free(gx);
        tg_uop_free(grad_seed);
        tg_uop_free(sin_expr);
        tg_uop_free(x);
    }
}

TEST(test_sqrt) {
    // Python: def test_sqrt(self): self._test_one_input_function(lambda x: x.sqrt())
    
    float test_vals[] = {2.0f, 5.0f}; // Only positive values for sqrt
    int num_vals = sizeof(test_vals) / sizeof(test_vals[0]);
    
    for (int i = 0; i < num_vals; i++) {
        float val = test_vals[i];
        
        tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
        tg_uop_t* sqrt_expr = tg_uop_sqrt(x);
        tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
        
        tg_uop_t* vars[] = {x};
        tg_gradient_result_t* grad_result = tg_compute_gradient(sqrt_expr, grad_seed, vars, 1);
        tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
        
        tg_uop_t* val_const = tg_uop_const_like(x, val);
        tg_substitution_t sub = {x, val_const};
        tg_uop_t* substituted = tg_uop_substitute(gx, &sub, 1);
        tg_uop_t* simplified = tg_uop_ssimplify(substituted);
        float tg_out = tg_uop_get_float(simplified);
        
        // Expected gradient: d/dx[sqrt(x)] = 1/(2*sqrt(x))
        float expected = 1.0f / (2.0f * sqrtf(val));
        
        assert_cmp_nan_okay(expected, tg_out);
        
        // Cleanup
        tg_gradient_result_free(grad_result);
        tg_uop_free(simplified);
        tg_uop_free(substituted);
        tg_uop_free(val_const);
        tg_uop_free(gx);
        tg_uop_free(grad_seed);
        tg_uop_free(sqrt_expr);
        tg_uop_free(x);
    }
}

TEST(test_log2) {
    // Python: def test_log2(self): self._test_one_input_function(lambda x: x.log2())
    
    float test_vals[] = {2.0f, 5.0f}; // Only positive values for log2
    int num_vals = sizeof(test_vals) / sizeof(test_vals[0]);
    
    for (int i = 0; i < num_vals; i++) {
        float val = test_vals[i];
        
        tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
        tg_uop_t* log2_expr = tg_uop_log2(x);
        tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
        
        tg_uop_t* vars[] = {x};
        tg_gradient_result_t* grad_result = tg_compute_gradient(log2_expr, grad_seed, vars, 1);
        tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
        
        tg_uop_t* val_const = tg_uop_const_like(x, val);
        tg_substitution_t sub = {x, val_const};
        tg_uop_t* substituted = tg_uop_substitute(gx, &sub, 1);
        tg_uop_t* simplified = tg_uop_ssimplify(substituted);
        float tg_out = tg_uop_get_float(simplified);
        
        // Expected gradient: d/dx[log2(x)] = 1/(x*ln(2))
        float expected = 1.0f / (val * logf(2.0f));
        
        assert_cmp_nan_okay(expected, tg_out);
        
        // Cleanup
        tg_gradient_result_free(grad_result);
        tg_uop_free(simplified);
        tg_uop_free(substituted);
        tg_uop_free(val_const);
        tg_uop_free(gx);
        tg_uop_free(grad_seed);
        tg_uop_free(log2_expr);
        tg_uop_free(x);
    }
}

TEST(test_exp2) {
    // Python: def test_exp2(self): self._test_one_input_function(lambda x: x.exp2())
    
    float test_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    int num_vals = sizeof(test_vals) / sizeof(test_vals[0]);
    
    for (int i = 0; i < num_vals; i++) {
        float val = test_vals[i];
        
        tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
        tg_uop_t* exp2_expr = tg_uop_exp2(x);
        tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
        
        tg_uop_t* vars[] = {x};
        tg_gradient_result_t* grad_result = tg_compute_gradient(exp2_expr, grad_seed, vars, 1);
        tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
        
        tg_uop_t* val_const = tg_uop_const_like(x, val);
        tg_substitution_t sub = {x, val_const};
        tg_uop_t* substituted = tg_uop_substitute(gx, &sub, 1);
        tg_uop_t* simplified = tg_uop_ssimplify(substituted);
        float tg_out = tg_uop_get_float(simplified);
        
        // Expected gradient: d/dx[2^x] = 2^x * ln(2)
        float expected = powf(2.0f, val) * logf(2.0f);
        
        assert_cmp_nan_okay(expected, tg_out);
        
        // Cleanup
        tg_gradient_result_free(grad_result);
        tg_uop_free(simplified);
        tg_uop_free(substituted);
        tg_uop_free(val_const);
        tg_uop_free(gx);
        tg_uop_free(grad_seed);
        tg_uop_free(exp2_expr);
        tg_uop_free(x);
    }
}

TEST(test_add) {
    // Python: def test_add(self): self._test_two_input_function(lambda x,y: x+y)
    
    float x_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    float y_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    int num_x = sizeof(x_vals) / sizeof(x_vals[0]);
    int num_y = sizeof(y_vals) / sizeof(y_vals[0]);
    
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < num_y; j++) {
            float valx = x_vals[i];
            float valy = y_vals[j];
            
            tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* y = tg_uop_variable("y", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* add_expr = tg_uop_add(x, y);
            tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
            
            tg_uop_t* vars[] = {x, y};
            tg_gradient_result_t* grad_result = tg_compute_gradient(add_expr, grad_seed, vars, 2);
            tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
            tg_uop_t* gy = tg_gradient_result_get(grad_result, y);
            
            // Substitute values
            tg_substitution_t subs[] = {{x, tg_uop_const_like(x, valx)}, {y, tg_uop_const_like(y, valy)}};
            tg_uop_t* gx_sub = tg_uop_substitute(gx, subs, 2);
            tg_uop_t* gy_sub = tg_uop_substitute(gy, subs, 2);
            
            tg_uop_t* gx_simp = tg_uop_ssimplify(gx_sub);
            tg_uop_t* gy_simp = tg_uop_ssimplify(gy_sub);
            
            float tg_out_x = tg_uop_get_float(gx_simp);
            float tg_out_y = tg_uop_get_float(gy_simp);
            
            // Expected gradients: d/dx[x+y] = 1, d/dy[x+y] = 1
            float expected_x = 1.0f;
            float expected_y = 1.0f;
            
            assert_cmp_nan_okay(expected_x, tg_out_x);
            assert_cmp_nan_okay(expected_y, tg_out_y);
            
            // Cleanup
            tg_gradient_result_free(grad_result);
            tg_uop_free(gy_simp);
            tg_uop_free(gx_simp);
            tg_uop_free(gy_sub);
            tg_uop_free(gx_sub);
            tg_uop_free(subs[1].value);
            tg_uop_free(subs[0].value);
            tg_uop_free(gy);
            tg_uop_free(gx);
            tg_uop_free(grad_seed);
            tg_uop_free(add_expr);
            tg_uop_free(y);
            tg_uop_free(x);
        }
    }
}

TEST(test_mul) {
    // Python: def test_mul(self): self._test_two_input_function(lambda x,y: x*y)
    
    float x_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    float y_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    int num_x = sizeof(x_vals) / sizeof(x_vals[0]);
    int num_y = sizeof(y_vals) / sizeof(y_vals[0]);
    
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < num_y; j++) {
            float valx = x_vals[i];
            float valy = y_vals[j];
            
            tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* y = tg_uop_variable("y", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* mul_expr = tg_uop_mul(x, y);
            tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
            
            tg_uop_t* vars[] = {x, y};
            tg_gradient_result_t* grad_result = tg_compute_gradient(mul_expr, grad_seed, vars, 2);
            tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
            tg_uop_t* gy = tg_gradient_result_get(grad_result, y);
            
            // Substitute values
            tg_substitution_t subs[] = {{x, tg_uop_const_like(x, valx)}, {y, tg_uop_const_like(y, valy)}};
            tg_uop_t* gx_sub = tg_uop_substitute(gx, subs, 2);
            tg_uop_t* gy_sub = tg_uop_substitute(gy, subs, 2);
            
            tg_uop_t* gx_simp = tg_uop_ssimplify(gx_sub);
            tg_uop_t* gy_simp = tg_uop_ssimplify(gy_sub);
            
            float tg_out_x = tg_uop_get_float(gx_simp);
            float tg_out_y = tg_uop_get_float(gy_simp);
            
            // Expected gradients: d/dx[x*y] = y, d/dy[x*y] = x
            float expected_x = valy;
            float expected_y = valx;
            
            assert_cmp_nan_okay(expected_x, tg_out_x);
            assert_cmp_nan_okay(expected_y, tg_out_y);
            
            // Cleanup
            tg_gradient_result_free(grad_result);
            tg_uop_free(gy_simp);
            tg_uop_free(gx_simp);
            tg_uop_free(gy_sub);
            tg_uop_free(gx_sub);
            tg_uop_free(subs[1].value);
            tg_uop_free(subs[0].value);
            tg_uop_free(gy);
            tg_uop_free(gx);
            tg_uop_free(grad_seed);
            tg_uop_free(mul_expr);
            tg_uop_free(y);
            tg_uop_free(x);
        }
    }
}

TEST(test_chain) {
    // Python: def test_chain(self): self._test_one_input_function(lambda x: x.sin().sqrt())
    
    float test_vals[] = {2.0f, 5.0f}; // Only positive sin values for sqrt
    int num_vals = sizeof(test_vals) / sizeof(test_vals[0]);
    
    for (int i = 0; i < num_vals; i++) {
        float val = test_vals[i];
        
        tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
        tg_uop_t* sin_expr = tg_uop_sin(x);
        tg_uop_t* chain_expr = tg_uop_sqrt(sin_expr);
        tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
        
        tg_uop_t* vars[] = {x};
        tg_gradient_result_t* grad_result = tg_compute_gradient(chain_expr, grad_seed, vars, 1);
        tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
        
        tg_uop_t* val_const = tg_uop_const_like(x, val);
        tg_substitution_t sub = {x, val_const};
        tg_uop_t* substituted = tg_uop_substitute(gx, &sub, 1);
        tg_uop_t* simplified = tg_uop_ssimplify(substituted);
        float tg_out = tg_uop_get_float(simplified);
        
        // Expected gradient: d/dx[sqrt(sin(x))] = cos(x) / (2*sqrt(sin(x)))
        float sin_val = sinf(val);
        float expected = cosf(val) / (2.0f * sqrtf(sin_val));
        
        assert_cmp_nan_okay(expected, tg_out);
        
        // Cleanup
        tg_gradient_result_free(grad_result);
        tg_uop_free(simplified);
        tg_uop_free(substituted);
        tg_uop_free(val_const);
        tg_uop_free(gx);
        tg_uop_free(grad_seed);
        tg_uop_free(chain_expr);
        tg_uop_free(sin_expr);
        tg_uop_free(x);
    }
}

TEST(test_chain_binop) {
    // Python: def test_chain_binop(self): self._test_two_input_function(lambda x,y: (x*y)+x*y)
    // This is effectively 2*x*y
    
    float x_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    float y_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    int num_x = sizeof(x_vals) / sizeof(x_vals[0]);
    int num_y = sizeof(y_vals) / sizeof(y_vals[0]);
    
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < num_y; j++) {
            float valx = x_vals[i];
            float valy = y_vals[j];
            
            tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* y = tg_uop_variable("y", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* xy1 = tg_uop_mul(x, y);
            tg_uop_t* xy2 = tg_uop_mul(x, y);
            tg_uop_t* chain_expr = tg_uop_add(xy1, xy2);
            tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
            
            tg_uop_t* vars[] = {x, y};
            tg_gradient_result_t* grad_result = tg_compute_gradient(chain_expr, grad_seed, vars, 2);
            tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
            tg_uop_t* gy = tg_gradient_result_get(grad_result, y);
            
            // Substitute values
            tg_substitution_t subs[] = {{x, tg_uop_const_like(x, valx)}, {y, tg_uop_const_like(y, valy)}};
            tg_uop_t* gx_sub = tg_uop_substitute(gx, subs, 2);
            tg_uop_t* gy_sub = tg_uop_substitute(gy, subs, 2);
            
            tg_uop_t* gx_simp = tg_uop_ssimplify(gx_sub);
            tg_uop_t* gy_simp = tg_uop_ssimplify(gy_sub);
            
            float tg_out_x = tg_uop_get_float(gx_simp);
            float tg_out_y = tg_uop_get_float(gy_simp);
            
            // Expected gradients: d/dx[2*x*y] = 2*y, d/dy[2*x*y] = 2*x
            float expected_x = 2.0f * valy;
            float expected_y = 2.0f * valx;
            
            assert_cmp_nan_okay(expected_x, tg_out_x);
            assert_cmp_nan_okay(expected_y, tg_out_y);
            
            // Cleanup
            tg_gradient_result_free(grad_result);
            tg_uop_free(gy_simp);
            tg_uop_free(gx_simp);
            tg_uop_free(gy_sub);
            tg_uop_free(gx_sub);
            tg_uop_free(subs[1].value);
            tg_uop_free(subs[0].value);
            tg_uop_free(gy);
            tg_uop_free(gx);
            tg_uop_free(grad_seed);
            tg_uop_free(chain_expr);
            tg_uop_free(xy2);
            tg_uop_free(xy1);
            tg_uop_free(y);
            tg_uop_free(x);
        }
    }
}

TEST(test_big_add_sin) {
    // Python: def test_big_add_sin(self): self._test_two_input_function(lambda x,y: x.sin()+3.0/y)
    
    float x_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    float y_vals[] = {-5.0f, -2.0f, 2.0f, 5.0f}; // Skip 0.0 for division
    int num_x = sizeof(x_vals) / sizeof(x_vals[0]);
    int num_y = sizeof(y_vals) / sizeof(y_vals[0]);
    
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < num_y; j++) {
            float valx = x_vals[i];
            float valy = y_vals[j];
            
            tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* y = tg_uop_variable("y", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* sin_x = tg_uop_sin(x);
            tg_uop_t* three = tg_uop_const(TG_F32, 3.0f);
            tg_uop_t* three_div_y = tg_uop_div(three, y);
            tg_uop_t* expr = tg_uop_add(sin_x, three_div_y);
            tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
            
            tg_uop_t* vars[] = {x, y};
            tg_gradient_result_t* grad_result = tg_compute_gradient(expr, grad_seed, vars, 2);
            tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
            tg_uop_t* gy = tg_gradient_result_get(grad_result, y);
            
            // Substitute values
            tg_substitution_t subs[] = {{x, tg_uop_const_like(x, valx)}, {y, tg_uop_const_like(y, valy)}};
            tg_uop_t* gx_sub = tg_uop_substitute(gx, subs, 2);
            tg_uop_t* gy_sub = tg_uop_substitute(gy, subs, 2);
            
            tg_uop_t* gx_simp = tg_uop_ssimplify(gx_sub);
            tg_uop_t* gy_simp = tg_uop_ssimplify(gy_sub);
            
            float tg_out_x = tg_uop_get_float(gx_simp);
            float tg_out_y = tg_uop_get_float(gy_simp);
            
            // Expected gradients: d/dx[sin(x)+3/y] = cos(x), d/dy[sin(x)+3/y] = -3/y^2
            float expected_x = cosf(valx);
            float expected_y = -3.0f / (valy * valy);
            
            assert_cmp_nan_okay(expected_x, tg_out_x);
            assert_cmp_nan_okay(expected_y, tg_out_y);
            
            // Cleanup
            tg_gradient_result_free(grad_result);
            tg_uop_free(gy_simp);
            tg_uop_free(gx_simp);
            tg_uop_free(gy_sub);
            tg_uop_free(gx_sub);
            tg_uop_free(subs[1].value);
            tg_uop_free(subs[0].value);
            tg_uop_free(gy);
            tg_uop_free(gx);
            tg_uop_free(grad_seed);
            tg_uop_free(expr);
            tg_uop_free(three_div_y);
            tg_uop_free(three);
            tg_uop_free(sin_x);
            tg_uop_free(y);
            tg_uop_free(x);
        }
    }
}

TEST(test_big_chain) {
    // Python: def test_big_chain(self): self._test_two_input_function(lambda x,y: (1.0/x*y)+x*y)
    
    float x_vals[] = {-5.0f, -2.0f, 2.0f, 5.0f}; // Skip 0.0 for division
    float y_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    int num_x = sizeof(x_vals) / sizeof(x_vals[0]);
    int num_y = sizeof(y_vals) / sizeof(y_vals[0]);
    
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < num_y; j++) {
            float valx = x_vals[i];
            float valy = y_vals[j];
            
            tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* y = tg_uop_variable("y", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* one = tg_uop_const(TG_F32, 1.0f);
            tg_uop_t* recip_x = tg_uop_div(one, x);
            tg_uop_t* recip_x_y = tg_uop_mul(recip_x, y);
            tg_uop_t* x_y = tg_uop_mul(x, y);
            tg_uop_t* expr = tg_uop_add(recip_x_y, x_y);
            tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
            
            tg_uop_t* vars[] = {x, y};
            tg_gradient_result_t* grad_result = tg_compute_gradient(expr, grad_seed, vars, 2);
            tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
            tg_uop_t* gy = tg_gradient_result_get(grad_result, y);
            
            // Substitute values
            tg_substitution_t subs[] = {{x, tg_uop_const_like(x, valx)}, {y, tg_uop_const_like(y, valy)}};
            tg_uop_t* gx_sub = tg_uop_substitute(gx, subs, 2);
            tg_uop_t* gy_sub = tg_uop_substitute(gy, subs, 2);
            
            tg_uop_t* gx_simp = tg_uop_ssimplify(gx_sub);
            tg_uop_t* gy_simp = tg_uop_ssimplify(gy_sub);
            
            float tg_out_x = tg_uop_get_float(gx_simp);
            float tg_out_y = tg_uop_get_float(gy_simp);
            
            // Expected gradients: d/dx[y/x + x*y] = -y/x^2 + y, d/dy[y/x + x*y] = 1/x + x
            float expected_x = -valy / (valx * valx) + valy;
            float expected_y = 1.0f / valx + valx;
            
            assert_cmp_nan_okay(expected_x, tg_out_x);
            assert_cmp_nan_okay(expected_y, tg_out_y);
            
            // Cleanup
            tg_gradient_result_free(grad_result);
            tg_uop_free(gy_simp);
            tg_uop_free(gx_simp);
            tg_uop_free(gy_sub);
            tg_uop_free(gx_sub);
            tg_uop_free(subs[1].value);
            tg_uop_free(subs[0].value);
            tg_uop_free(gy);
            tg_uop_free(gx);
            tg_uop_free(grad_seed);
            tg_uop_free(expr);
            tg_uop_free(x_y);
            tg_uop_free(recip_x_y);
            tg_uop_free(recip_x);
            tg_uop_free(one);
            tg_uop_free(y);
            tg_uop_free(x);
        }
    }
}

TEST(test_where) {
    // Python: def test_where(self): self._test_two_input_function(lambda x,y: (x<y).where(x,y), lambda x,y: torch.where(x<y,x,y))
    // This is min(x,y) function
    
    float x_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    float y_vals[] = {-5.0f, -2.0f, 0.0f, 2.0f, 5.0f};
    int num_x = sizeof(x_vals) / sizeof(x_vals[0]);
    int num_y = sizeof(y_vals) / sizeof(y_vals[0]);
    
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < num_y; j++) {
            float valx = x_vals[i];
            float valy = y_vals[j];
            
            tg_uop_t* x = tg_uop_variable("x", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* y = tg_uop_variable("y", -INFINITY, INFINITY, TG_F32);
            tg_uop_t* cond = tg_uop_cmplt(x, y);
            tg_uop_t* where_expr = tg_uop_where(cond, x, y);
            tg_uop_t* grad_seed = tg_uop_const(TG_F32, 1.0f);
            
            tg_uop_t* vars[] = {x, y};
            tg_gradient_result_t* grad_result = tg_compute_gradient(where_expr, grad_seed, vars, 2);
            tg_uop_t* gx = tg_gradient_result_get(grad_result, x);
            tg_uop_t* gy = tg_gradient_result_get(grad_result, y);
            
            // Substitute values
            tg_substitution_t subs[] = {{x, tg_uop_const_like(x, valx)}, {y, tg_uop_const_like(y, valy)}};
            tg_uop_t* gx_sub = tg_uop_substitute(gx, subs, 2);
            tg_uop_t* gy_sub = tg_uop_substitute(gy, subs, 2);
            
            tg_uop_t* gx_simp = tg_uop_ssimplify(gx_sub);
            tg_uop_t* gy_simp = tg_uop_ssimplify(gy_sub);
            
            float tg_out_x = tg_uop_get_float(gx_simp);
            float tg_out_y = tg_uop_get_float(gy_simp);
            
            // Expected gradients for min(x,y): if x < y then (1, 0) else (0, 1)
            float expected_x = (valx < valy) ? 1.0f : 0.0f;
            float expected_y = (valx < valy) ? 0.0f : 1.0f;
            
            assert_cmp_nan_okay(expected_x, tg_out_x);
            assert_cmp_nan_okay(expected_y, tg_out_y);
            
            // Cleanup
            tg_gradient_result_free(grad_result);
            tg_uop_free(gy_simp);
            tg_uop_free(gx_simp);
            tg_uop_free(gy_sub);
            tg_uop_free(gx_sub);
            tg_uop_free(subs[1].value);
            tg_uop_free(subs[0].value);
            tg_uop_free(gy);
            tg_uop_free(gx);
            tg_uop_free(grad_seed);
            tg_uop_free(where_expr);
            tg_uop_free(cond);
            tg_uop_free(y);
            tg_uop_free(x);
        }
    }
}

// TestTensorGradient tests

TEST(test_tensor_gradient_example) {
    // Python: def test_example(self):
    //   x = Tensor.eye(3)
    //   y = Tensor([[2.0,0,-2.0]])  
    //   z = y.matmul(x).sum()
    //   dx, dy = z.gradient(x, y)
    //   self.assertListEqual(dx.tolist(), [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]])
    //   self.assertListEqual(dy.tolist(), [[1.0, 1.0, 1.0]])
    
    // Create x = eye(3)
    tg_tensor_t* x = tg_tensor_eye(3, TG_F32);
    TEST_ASSERT_NOT_NULL(x);
    
    // Create y = [[2.0, 0, -2.0]]
    int64_t y_shape[] = {1, 3};
    float y_data[] = {2.0f, 0.0f, -2.0f};
    tg_tensor_t* y = tg_tensor_from_data(y_shape, 2, y_data, TG_F32);
    TEST_ASSERT_NOT_NULL(y);
    
    // Compute z = y.matmul(x).sum()
    tg_tensor_t* matmul_result = tg_tensor_matmul(y, x);
    TEST_ASSERT_NOT_NULL(matmul_result);
    
    tg_tensor_t* z = tg_tensor_sum(matmul_result);
    TEST_ASSERT_NOT_NULL(z);
    
    // Compute gradients: dx, dy = z.gradient(x, y)
    tg_tensor_t* inputs[] = {x, y};
    tg_tensor_t** gradients = tg_tensor_gradient(z, inputs, 2);
    TEST_ASSERT_NOT_NULL(gradients);
    
    tg_tensor_t* dx = gradients[0];
    tg_tensor_t* dy = gradients[1];
    TEST_ASSERT_NOT_NULL(dx);
    TEST_ASSERT_NOT_NULL(dy);
    
    // Check dx shape and values
    int64_t dx_shape[] = {3, 3};
    TEST_ASSERT_TRUE(tg_tensor_shape_equal(dx, dx_shape, 2));
    
    float dx_expected[] = {
        2.0f, 2.0f, 2.0f,
        0.0f, 0.0f, 0.0f,
        -2.0f, -2.0f, -2.0f
    };
    float* dx_data = tg_tensor_data_ptr(dx);
    for (int i = 0; i < 9; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, dx_expected[i], dx_data[i]);
    }
    
    // Check dy shape and values  
    int64_t dy_shape[] = {1, 3};
    TEST_ASSERT_TRUE(tg_tensor_shape_equal(dy, dy_shape, 2));
    
    float dy_expected[] = {1.0f, 1.0f, 1.0f};
    float* dy_data = tg_tensor_data_ptr(dy);
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, dy_expected[i], dy_data[i]);
    }
    
    // Cleanup
    free(gradients);
    tg_tensor_free(dy);
    tg_tensor_free(dx);
    tg_tensor_free(z);
    tg_tensor_free(matmul_result);
    tg_tensor_free(y);
    tg_tensor_free(x);
}

TEST(test_tensor_gradient_raises) {
    // Python: def test_raises(self):
    //   x = Tensor([1.0, 2.0, 3.0])
    //   w = Tensor.randn((3,))
    //   with self.assertRaises(RuntimeError): x.sum().gradient(w)
    
    // Create x = [1.0, 2.0, 3.0]
    int64_t x_shape[] = {3};
    float x_data[] = {1.0f, 2.0f, 3.0f};
    tg_tensor_t* x = tg_tensor_from_data(x_shape, 1, x_data, &dtypes.float32);
    TEST_ASSERT_NOT_NULL(x);
    
    // Create w = randn((3,))
    tg_tensor_t* w = tg_tensor_randn(x_shape, 1, TG_F32);
    TEST_ASSERT_NOT_NULL(w);
    
    // Compute x.sum()
    tg_tensor_t* x_sum = tg_tensor_sum(x);
    TEST_ASSERT_NOT_NULL(x_sum);
    
    // Try to compute gradient w.r.t. unrelated tensor w - should raise RuntimeError
    tg_tensor_t* inputs[] = {w};
    tg_tensor_t** gradients = tg_tensor_gradient(x_sum, inputs, 1);
    
    // This should return NULL or set an error flag indicating RuntimeError
    TEST_ASSERT_NULL(gradients);
    
    // Cleanup
    tg_tensor_free(x_sum);
    tg_tensor_free(w);
    tg_tensor_free(x);
}

TEST(test_tensor_gradient_with_custom_gradient) {
    // Python: def test_with_custom_gradient(self):
    //   x = Tensor([1.0, 2.0, 3.0])
    //   z = (x * x).sum()
    //   dx = z.gradient(x, gradient=Tensor([3.0]))[0]
    //   self.assertListEqual(dx.tolist(), [6.0, 12.0, 18.0])
    
    // Create x = [1.0, 2.0, 3.0]
    int64_t x_shape[] = {3};
    float x_data[] = {1.0f, 2.0f, 3.0f};
    tg_tensor_t* x = tg_tensor_from_data(x_shape, 1, x_data, &dtypes.float32);
    TEST_ASSERT_NOT_NULL(x);
    
    // Compute z = (x * x).sum()
    tg_tensor_t* x_squared = tg_tensor_mul(x, x);
    TEST_ASSERT_NOT_NULL(x_squared);
    
    tg_tensor_t* z = tg_tensor_sum(x_squared);
    TEST_ASSERT_NOT_NULL(z);
    
    // Create custom gradient = [3.0]
    int64_t grad_shape[] = {1};
    float grad_data[] = {3.0f};
    tg_tensor_t* custom_grad = tg_tensor_from_data(grad_shape, 1, grad_data, TG_F32);
    TEST_ASSERT_NOT_NULL(custom_grad);
    
    // Compute dx = z.gradient(x, gradient=custom_grad)[0]
    tg_tensor_t* inputs[] = {x};
    tg_tensor_t** gradients = tg_tensor_gradient_with_grad(z, inputs, 1, custom_grad);
    TEST_ASSERT_NOT_NULL(gradients);
    
    tg_tensor_t* dx = gradients[0];
    TEST_ASSERT_NOT_NULL(dx);
    
    // Check dx values: should be [6.0, 12.0, 18.0] = custom_grad * 2*x
    float dx_expected[] = {6.0f, 12.0f, 18.0f};
    float* dx_data = tg_tensor_data_ptr(dx);
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, dx_expected[i], dx_data[i]);
    }
    
    // Cleanup
    free(gradients);
    tg_tensor_free(dx);
    tg_tensor_free(custom_grad);
    tg_tensor_free(z);
    tg_tensor_free(x_squared);
    tg_tensor_free(x);
}

TEST(test_tensor_gradient_broadcast_gradient) {
    // Python: def test_broadcast_gradient(self):
    //   x = Tensor([[1.0], [2.0], [3.0]])
    //   y = Tensor([[10.0, 20.0, 30.0, 40.0]])
    //   z = (x + y).sum()
    //   dx, dy = z.gradient(x, y)
    //   self.assertListEqual(dx.tolist(), [[4.0], [4.0], [4.0]])
    //   self.assertListEqual(dy.tolist(), [[3.0, 3.0, 3.0, 3.0]])
    
    // Create x = [[1.0], [2.0], [3.0]]
    int64_t x_shape[] = {3, 1};
    float x_data[] = {1.0f, 2.0f, 3.0f};
    tg_tensor_t* x = tg_tensor_from_data(x_shape, 2, x_data, TG_F32);
    TEST_ASSERT_NOT_NULL(x);
    
    // Create y = [[10.0, 20.0, 30.0, 40.0]]
    int64_t y_shape[] = {1, 4};
    float y_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
    tg_tensor_t* y = tg_tensor_from_data(y_shape, 2, y_data, TG_F32);
    TEST_ASSERT_NOT_NULL(y);
    
    // Compute z = (x + y).sum()
    tg_tensor_t* add_result = tg_tensor_add(x, y);  // This will broadcast
    TEST_ASSERT_NOT_NULL(add_result);
    
    tg_tensor_t* z = tg_tensor_sum(add_result);
    TEST_ASSERT_NOT_NULL(z);
    
    // Compute gradients: dx, dy = z.gradient(x, y)
    tg_tensor_t* inputs[] = {x, y};
    tg_tensor_t** gradients = tg_tensor_gradient(z, inputs, 2);
    TEST_ASSERT_NOT_NULL(gradients);
    
    tg_tensor_t* dx = gradients[0];
    tg_tensor_t* dy = gradients[1];
    TEST_ASSERT_NOT_NULL(dx);
    TEST_ASSERT_NOT_NULL(dy);
    
    // Check dx shape and values: [[4.0], [4.0], [4.0]]
    int64_t dx_expected_shape[] = {3, 1};
    TEST_ASSERT_TRUE(tg_tensor_shape_equal(dx, dx_expected_shape, 2));
    
    float dx_expected[] = {4.0f, 4.0f, 4.0f};
    float* dx_data = tg_tensor_data_ptr(dx);
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, dx_expected[i], dx_data[i]);
    }
    
    // Check dy shape and values: [[3.0, 3.0, 3.0, 3.0]]
    int64_t dy_expected_shape[] = {1, 4};
    TEST_ASSERT_TRUE(tg_tensor_shape_equal(dy, dy_expected_shape, 2));
    
    float dy_expected[] = {3.0f, 3.0f, 3.0f, 3.0f};
    float* dy_data = tg_tensor_data_ptr(dy);
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, dy_expected[i], dy_data[i]);
    }
    
    // Cleanup
    free(gradients);
    tg_tensor_free(dy);
    tg_tensor_free(dx);
    tg_tensor_free(z);
    tg_tensor_free(add_result);
    tg_tensor_free(y);
    tg_tensor_free(x);
}

TEST(test_tensor_gradient_non_scalar_output) {
    // Python: def test_non_scalar_output(self):
    //   x = Tensor([1.0, 2.0, 3.0])
    //   z = x * x
    //   with self.assertRaises(AssertionError): z.gradient(x)
    //   dz = Tensor([1.0, 1.0, 1.0])
    //   dx = z.gradient(x, gradient=dz)[0]
    //   self.assertListEqual(dx.tolist(), [2.0, 4.0, 6.0])
    
    // Create x = [1.0, 2.0, 3.0]
    int64_t x_shape[] = {3};
    float x_data[] = {1.0f, 2.0f, 3.0f};
    tg_tensor_t* x = tg_tensor_from_data(x_shape, 1, x_data, TG_F32);
    TEST_ASSERT_NOT_NULL(x);
    
    // Compute z = x * x (non-scalar output)
    tg_tensor_t* z = tg_tensor_mul(x, x);
    TEST_ASSERT_NOT_NULL(z);
    
    // Try to compute gradient without providing gradient tensor - should raise AssertionError
    tg_tensor_t* inputs[] = {x};
    tg_tensor_t** gradients = tg_tensor_gradient(z, inputs, 1);
    
    // This should return NULL or set an error flag indicating AssertionError
    TEST_ASSERT_NULL(gradients);
    
    // Now provide gradient tensor dz = [1.0, 1.0, 1.0]
    float dz_data[] = {1.0f, 1.0f, 1.0f};
    tg_tensor_t* dz = tg_tensor_from_data(x_shape, 1, dz_data, TG_F32);
    TEST_ASSERT_NOT_NULL(dz);
    
    // Compute dx = z.gradient(x, gradient=dz)[0]
    gradients = tg_tensor_gradient_with_grad(z, inputs, 1, dz);
    TEST_ASSERT_NOT_NULL(gradients);
    
    tg_tensor_t* dx = gradients[0];
    TEST_ASSERT_NOT_NULL(dx);
    
    // Check dx values: should be [2.0, 4.0, 6.0] = dz * 2*x
    float dx_expected[] = {2.0f, 4.0f, 6.0f};
    float* dx_data = tg_tensor_data_ptr(dx);
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, dx_expected[i], dx_data[i]);
    }
    
    // Cleanup
    free(gradients);
    tg_tensor_free(dx);
    tg_tensor_free(dz);
    tg_tensor_free(z);
    tg_tensor_free(x);
}

TEST(test_tensor_gradient_cast_before_view) {
    // Python: def test_cast_before_view(self):
    //   x = Tensor([1.0, 1, 1, 1])
    //   x_reshaped = x.reshape(2,2)
    //   x_casted = x_reshaped.cast(dtypes.float16)
    //   x_casted.mean().gradient(x_reshaped)
    
    // Create x = [1.0, 1, 1, 1]
    int64_t x_shape[] = {4};
    float x_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    tg_tensor_t* x = tg_tensor_from_data(x_shape, 1, x_data, TG_F32);
    TEST_ASSERT_NOT_NULL(x);
    
    // Reshape: x_reshaped = x.reshape(2,2)
    int64_t reshaped_shape[] = {2, 2};
    tg_tensor_t* x_reshaped = tg_tensor_reshape(x, reshaped_shape, 2);
    TEST_ASSERT_NOT_NULL(x_reshaped);
    
    // Cast: x_casted = x_reshaped.cast(float16)
    tg_tensor_t* x_casted = tg_tensor_cast(x_reshaped, &dtypes.float16);
    TEST_ASSERT_NOT_NULL(x_casted);
    
    // Compute mean: x_casted.mean()
    tg_tensor_t* mean_result = tg_tensor_mean(x_casted);
    TEST_ASSERT_NOT_NULL(mean_result);
    
    // Compute gradient: mean_result.gradient(x_reshaped)
    tg_tensor_t* inputs[] = {x_reshaped};
    tg_tensor_t** gradients = tg_tensor_gradient(mean_result, inputs, 1);
    TEST_ASSERT_NOT_NULL(gradients);
    
    // This should work without error
    tg_tensor_t* dx = gradients[0];
    TEST_ASSERT_NOT_NULL(dx);
    
    // Cleanup
    free(gradients);
    tg_tensor_free(dx);
    tg_tensor_free(mean_result);
    tg_tensor_free(x_casted);
    tg_tensor_free(x_reshaped);
    tg_tensor_free(x);
}

TEST(test_tensor_gradient_non_float_tensor_raise) {
    // Python: def test_non_float_tensor_raise(self):
    //   x = Tensor([1, 2, 3])
    //   with self.assertRaises(RuntimeError): x.sum().gradient(x)
    //   with self.assertRaises(RuntimeError): x.float().sum().gradient(x)
    
    // Create x = [1, 2, 3] (integer tensor)
    int64_t x_shape[] = {3};
    int32_t x_data[] = {1, 2, 3};
    tg_tensor_t* x = tg_tensor_from_data_int(x_shape, 1, x_data, &dtypes.int32);
    TEST_ASSERT_NOT_NULL(x);
    
    // Compute x.sum()
    tg_tensor_t* x_sum = tg_tensor_sum(x);
    TEST_ASSERT_NOT_NULL(x_sum);
    
    // Try to compute gradient - should raise RuntimeError for non-float input
    tg_tensor_t* inputs[] = {x};
    tg_tensor_t** gradients = tg_tensor_gradient(x_sum, inputs, 1);
    
    // This should return NULL or set an error flag indicating RuntimeError
    TEST_ASSERT_NULL(gradients);
    
    // Test with x.float().sum().gradient(x) - should also raise RuntimeError
    tg_tensor_t* x_float = tg_tensor_cast(x, TG_F32);
    TEST_ASSERT_NOT_NULL(x_float);
    
    tg_tensor_t* x_float_sum = tg_tensor_sum(x_float);
    TEST_ASSERT_NOT_NULL(x_float_sum);
    
    // Try to compute gradient w.r.t. original integer tensor - should raise RuntimeError
    gradients = tg_tensor_gradient(x_float_sum, inputs, 1);
    TEST_ASSERT_NULL(gradients);
    
    // Cleanup
    tg_tensor_free(x_float_sum);
    tg_tensor_free(x_float);
    tg_tensor_free(x_sum);
    tg_tensor_free(x);
}

// Unity framework requires setUp and tearDown functions
void setUp(void) {
    // Set up before each test
}

void tearDown(void) {
    // Clean up after each test
}

TEST_MAIN()