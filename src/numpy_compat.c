#include "numpy_compat.h"
#include <string.h>
#include <assert.h>

np_array_t* np_frombuffer(void* buffer, size_t size, const DType* dtype) {
    np_array_t* arr = calloc(1, sizeof(np_array_t));
    arr->data = buffer;
    arr->dtype = *dtype;
    arr->size = size / dtype->itemsize;
    arr->ndim = 1;
    arr->shape = calloc(1, sizeof(size_t));
    arr->shape[0] = arr->size;
    arr->strides = calloc(1, sizeof(size_t));
    arr->strides[0] = dtype->itemsize;
    return arr;
}

np_array_t* np_empty(size_t ndim, const size_t* shape, const DType* dtype) {
    np_array_t* arr = calloc(1, sizeof(np_array_t));
    arr->dtype = *dtype;
    arr->ndim = ndim;
    arr->shape = calloc(ndim, sizeof(size_t));
    arr->strides = calloc(ndim, sizeof(size_t));
    
    // Calculate size
    arr->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        arr->shape[i] = shape[i];
        arr->size *= shape[i];
    }
    
    // C-contiguous strides
    arr->strides[ndim-1] = dtype->itemsize;
    for (int i = ndim-2; i >= 0; i--) {
        arr->strides[i] = arr->strides[i+1] * arr->shape[i+1];
    }
    
    // Use GSL for memory if it's float64
    if (dtype_eq(dtype, &dtypes.float64) && ndim <= 2) {
        if (ndim == 1) {
            gsl_vector* v = gsl_vector_alloc(arr->size);
            arr->data = v->data;
            arr->block = v->block;
        } else {
            gsl_matrix* m = gsl_matrix_alloc(shape[0], shape[1]);
            arr->data = m->data;
            arr->block = m->block;
        }
    } else {
        arr->data = malloc(arr->size * dtype->itemsize);
    }
    
    return arr;
}

np_array_t* np_zeros(size_t ndim, const size_t* shape, const DType* dtype) {
    np_array_t* arr = np_empty(ndim, shape, dtype);
    memset(arr->data, 0, arr->size * dtype->itemsize);
    return arr;
}

np_array_t* np_ones(size_t ndim, const size_t* shape, const DType* dtype) {
    np_array_t* arr = np_empty(ndim, shape, dtype);
    
    // Fill with ones based on dtype
    if (dtype_eq(dtype, &dtypes.float64)) {
        double* data = (double*)arr->data;
        for (size_t i = 0; i < arr->size; i++) {
            data[i] = 1.0;
        }
    } else if (dtype_eq(dtype, &dtypes.float32)) {
        float* data = (float*)arr->data;
        for (size_t i = 0; i < arr->size; i++) {
            data[i] = 1.0f;
        }
    } else if (dtype_eq(dtype, &dtypes.int32)) {
        int32_t* data = (int32_t*)arr->data;
        for (size_t i = 0; i < arr->size; i++) {
            data[i] = 1;
        }
    } else {
        // Generic byte fill
        memset(arr->data, 1, arr->size * dtype->itemsize);
    }
    
    return arr;
}

bool np_allclose(np_array_t* a, np_array_t* b, double rtol, double atol) {
    if (!dtype_eq(&a->dtype, &b->dtype) || a->size != b->size) {
        return false;
    }
    
    // For float types, check with tolerance
    if (dtypes_is_float(&a->dtype)) {
        if (dtype_eq(&a->dtype, &dtypes.float64)) {
            double* data_a = (double*)a->data;
            double* data_b = (double*)b->data;
            for (size_t i = 0; i < a->size; i++) {
                double diff = fabs(data_a[i] - data_b[i]);
                if (diff > atol + rtol * fabs(data_b[i])) {
                    return false;
                }
            }
        } else if (dtype_eq(&a->dtype, &dtypes.float32)) {
            float* data_a = (float*)a->data;
            float* data_b = (float*)b->data;
            for (size_t i = 0; i < a->size; i++) {
                float diff = fabsf(data_a[i] - data_b[i]);
                if (diff > atol + rtol * fabsf(data_b[i])) {
                    return false;
                }
            }
        }
    }
    return true;
}