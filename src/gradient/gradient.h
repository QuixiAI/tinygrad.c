#ifndef GRADIENT_H
#define GRADIENT_H

#include "tg.h"
#include "dtype/dtype.h"
#include "uop/uop.h"
#include "shape/shapetracker.h"
#include <stdint.h>
#include <stdbool.h>

// Unify tg_uop_t with core UOp structure
typedef UOp tg_uop_t;
typedef struct tg_gradient_result tg_gradient_result_t;
// Note: tg_tensor_t is defined in tg.h as struct tg_tensor*
typedef struct tg_substitution {
    tg_uop_t* variable;
    tg_uop_t* value;
} tg_substitution_t;

// UOp creation functions (tg_dtype for tests' convenience)
tg_uop_t* tg_uop_variable(const char* name, float vmin, float vmax, tg_dtype dtype);
tg_uop_t* tg_uop_const(tg_dtype dtype, float value);
tg_uop_t* tg_uop_const_like(tg_uop_t* template_uop, float value);

// UOp unary operations
tg_uop_t* tg_uop_recip(tg_uop_t* x);
tg_uop_t* tg_uop_sin(tg_uop_t* x);
tg_uop_t* tg_uop_sqrt(tg_uop_t* x);
tg_uop_t* tg_uop_log2(tg_uop_t* x);
tg_uop_t* tg_uop_exp2(tg_uop_t* x);
tg_uop_t* tg_uop_neg(tg_uop_t* x);
tg_uop_t* tg_uop_cast(tg_uop_t* x, const DType* dtype);
tg_uop_t* tg_uop_contiguous(tg_uop_t* x);
tg_uop_t* tg_uop_flip(tg_uop_t* x, int axis);
tg_uop_t* tg_uop_reshape(tg_uop_t* x, const int32_t* new_shape, int new_ndim);
tg_uop_t* tg_uop_permute(tg_uop_t* x, const int32_t* axes, int num_axes);
tg_uop_t* tg_uop_expand(tg_uop_t* x, const int32_t* target_shape, int target_ndim);
tg_uop_t* tg_uop_pad(tg_uop_t* x, const int32_t* pad_before, const int32_t* pad_after, int ndim);
tg_uop_t* tg_uop_shrink(tg_uop_t* x, const int32_t* start, const int32_t* end, int ndim);

// UOp binary operations
tg_uop_t* tg_uop_add(tg_uop_t* x, tg_uop_t* y);
tg_uop_t* tg_uop_mul(tg_uop_t* x, tg_uop_t* y);
tg_uop_t* tg_uop_div(tg_uop_t* x, tg_uop_t* y);
tg_uop_t* tg_uop_sub(tg_uop_t* x, tg_uop_t* y);
tg_uop_t* tg_uop_pow(tg_uop_t* x, tg_uop_t* y);
tg_uop_t* tg_uop_cmplt(tg_uop_t* x, tg_uop_t* y);
tg_uop_t* tg_uop_cmpgt(tg_uop_t* x, tg_uop_t* y);
tg_uop_t* tg_uop_cmpne(tg_uop_t* x, tg_uop_t* y);

// UOp conditional operations
tg_uop_t* tg_uop_where(tg_uop_t* condition, tg_uop_t* x, tg_uop_t* y);

// UOp manipulation
tg_uop_t* tg_uop_substitute(tg_uop_t* uop, tg_substitution_t* substitutions, int count);
tg_uop_t* tg_uop_ssimplify(tg_uop_t* uop);
float tg_uop_get_float(tg_uop_t* uop);
void tg_uop_free(tg_uop_t* uop);
tg_uop_t** tg_uop_toposort(tg_uop_t* root, int* out_count);
tg_uop_t* tg_uop_reduce_axis(tg_uop_t* src, int reduce_op, int* axes, int axes_count);

// Gradient computation
tg_gradient_result_t* tg_compute_gradient(tg_uop_t* expression, tg_uop_t* grad_seed, tg_uop_t** variables, int var_count);
tg_uop_t* tg_gradient_result_get(tg_gradient_result_t* result, tg_uop_t* variable);
void tg_gradient_result_free(tg_gradient_result_t* result);

// Tensor operations
tg_tensor_t* tg_tensor_eye(int size, const DType* dtype);
tg_tensor_t* tg_tensor_from_data(int64_t* shape, int ndim, float* data, const DType* dtype);
tg_tensor_t* tg_tensor_from_data_int(int64_t* shape, int ndim, int32_t* data, const DType* dtype);
tg_tensor_t* tg_tensor_randn(int64_t* shape, int ndim, const DType* dtype);

tg_tensor_t* tg_tensor_matmul(tg_tensor_t* a, tg_tensor_t* b);
tg_tensor_t* tg_tensor_sum(tg_tensor_t* tensor);
tg_tensor_t* tg_tensor_add(tg_tensor_t* a, tg_tensor_t* b);
tg_tensor_t* tg_tensor_mul(tg_tensor_t* a, tg_tensor_t* b);
tg_tensor_t* tg_tensor_reshape(tg_tensor_t* tensor, int64_t* shape, int ndim);
tg_tensor_t* tg_tensor_cast(tg_tensor_t* tensor, const DType* dtype);
tg_tensor_t* tg_tensor_mean(tg_tensor_t* tensor);

bool tg_tensor_shape_equal(tg_tensor_t* tensor, int64_t* shape, int ndim);
float* tg_tensor_data_ptr(tg_tensor_t* tensor);
void tg_tensor_free(tg_tensor_t* tensor);

// Tensor gradient functions
tg_tensor_t** tg_tensor_gradient(tg_tensor_t* output, tg_tensor_t** inputs, int input_count);
tg_tensor_t** tg_tensor_gradient_with_grad(tg_tensor_t* output, tg_tensor_t** inputs, int input_count, tg_tensor_t* grad_output);

#endif // GRADIENT_H
