#pragma once
#include "dtype/dtype.h"
#include <gsl/gsl_block.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

typedef struct {
    void* data;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    size_t size;
    DType dtype;
    gsl_block* block;
} np_array_t;

np_array_t* np_frombuffer(void* buffer, size_t size, const DType* dtype);
np_array_t* np_empty(size_t ndim, const size_t* shape, const DType* dtype);
np_array_t* np_zeros(size_t ndim, const size_t* shape, const DType* dtype);
np_array_t* np_ones(size_t ndim, const size_t* shape, const DType* dtype);
np_array_t* np_array_copy(size_t ndim, const size_t* shape, const DType* dtype, const void* src);
void np_free(np_array_t* arr);

void* np_data(np_array_t* arr);
np_array_t* np_require_c_contiguous(np_array_t* arr);
void np_set_printoptions(int precision);
const char* np_dtype_name(const DType* dt);
const DType* np_dtype_from_name(const char* name);

bool np_allclose(np_array_t* a, np_array_t* b, double rtol, double atol);
bool np_array_equal(np_array_t* a, np_array_t* b);
int np_assert_allclose(np_array_t* a, np_array_t* b, double rtol, double atol);

np_array_t* np_from_gsl_matrix(gsl_matrix* m, const DType* dtype);
np_array_t* np_from_gsl_vector(gsl_vector* v, const DType* dtype);
gsl_matrix* np_to_gsl_matrix(np_array_t* arr);
gsl_vector* np_to_gsl_vector(np_array_t* arr);

static inline int np_is_array(const void* p){ return p!=NULL; }
typedef struct { int (*assert_allclose)(np_array_t* a, np_array_t* b, double rtol, double atol); } np_testing_t;
extern const np_testing_t np_testing;

