#pragma once
#include "dtype/dtype.h"
#include <gsl/gsl_block.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

typedef struct {
    void* data;           // Raw data pointer
    size_t* shape;        // Shape array
    size_t* strides;      // Strides in bytes
    size_t ndim;          // Number of dimensions
    size_t size;          // Total elements
    DType dtype;          // Your existing dtype
    gsl_block* block;     // Optional GSL backing
} np_array_t;

// Core functions you need
np_array_t* np_frombuffer(void* buffer, size_t size, const DType* dtype);
np_array_t* np_empty(size_t ndim, const size_t* shape, const DType* dtype);
np_array_t* np_zeros(size_t ndim, const size_t* shape, const DType* dtype);
np_array_t* np_ones(size_t ndim, const size_t* shape, const DType* dtype);
// Create a new array and copy data from src (np.array(...))
np_array_t* np_array_copy(size_t ndim, const size_t* shape, const DType* dtype, const void* src);
void np_free(np_array_t* arr);

// Buffer access
void* np_data(np_array_t* arr);
np_array_t* np_require_c_contiguous(np_array_t* arr);
// Printing options (stub for compatibility)
void np_set_printoptions(int precision);
// Dtype helpers akin to np.dtype(...)
const char* np_dtype_name(const DType* dt);
const DType* np_dtype_from_name(const char* name);

// Testing
bool np_allclose(np_array_t* a, np_array_t* b, double rtol, double atol);
bool np_array_equal(np_array_t* a, np_array_t* b);
int np_assert_allclose(np_array_t* a, np_array_t* b, double rtol, double atol);

// Conversion to/from GSL
np_array_t* np_from_gsl_matrix(gsl_matrix* m, const DType* dtype);
np_array_t* np_from_gsl_vector(gsl_vector* v, const DType* dtype);
gsl_matrix* np_to_gsl_matrix(np_array_t* arr);
gsl_vector* np_to_gsl_vector(np_array_t* arr);

// Introspection
static inline int np_is_array(const void* p){ return p!=NULL; }
// A tiny namespace to approximate `np.testing.assert_allclose`
typedef struct {
    int (*assert_allclose)(np_array_t* a, np_array_t* b, double rtol, double atol);
} np_testing_t;
extern const np_testing_t np_testing;
