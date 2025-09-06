#include "compat/numpy.h"
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

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
    arr->block = (gsl_block*)0x1;
    return arr;
}

np_array_t* np_empty(size_t ndim, const size_t* shape, const DType* dtype) {
    np_array_t* arr = calloc(1, sizeof(np_array_t));
    arr->dtype = *dtype;
    arr->ndim = ndim;
    arr->shape = calloc(ndim, sizeof(size_t));
    arr->strides = calloc(ndim, sizeof(size_t));
    arr->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        arr->shape[i] = shape[i];
        arr->size *= shape[i];
    }
    arr->strides[ndim-1] = dtype->itemsize;
    for (int i = (int)ndim-2; i >= 0; i--) {
        arr->strides[i] = arr->strides[i+1] * arr->shape[i+1];
    }
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
    if (dtype_eq(dtype, &dtypes.float64)) {
        double* data = (double*)arr->data;
        for (size_t i = 0; i < arr->size; i++) data[i] = 1.0;
    } else if (dtype_eq(dtype, &dtypes.float32)) {
        float* data = (float*)arr->data;
        for (size_t i = 0; i < arr->size; i++) data[i] = 1.0f;
    } else if (dtype_eq(dtype, &dtypes.int32)) {
        int32_t* data = (int32_t*)arr->data;
        for (size_t i = 0; i < arr->size; i++) data[i] = 1;
    } else {
        memset(arr->data, 1, arr->size * dtype->itemsize);
    }
    return arr;
}

np_array_t* np_array_copy(size_t ndim, const size_t* shape, const DType* dtype, const void* src){
    np_array_t* arr = np_empty(ndim, shape, dtype);
    memcpy(arr->data, src, arr->size * dtype->itemsize);
    return arr;
}

bool np_allclose(np_array_t* a, np_array_t* b, double rtol, double atol) {
    if (!dtype_eq(&a->dtype, &b->dtype) || a->size != b->size) return false;
    if (dtypes_is_float(&a->dtype)) {
        if (dtype_eq(&a->dtype, &dtypes.float64)) {
            double* da = (double*)a->data; double* db = (double*)b->data;
            for (size_t i=0;i<a->size;i++){ double diff=fabs(da[i]-db[i]); if (diff>atol+rtol*fabs(db[i])) return false; }
        } else if (dtype_eq(&a->dtype, &dtypes.float32)) {
            float* da = (float*)a->data; float* db = (float*)b->data;
            for (size_t i=0;i<a->size;i++){ float diff=fabsf(da[i]-db[i]); if (diff> (float)(atol+rtol*fabsf(db[i]))) return false; }
        }
        return true;
    }
    return memcmp(a->data, b->data, a->size * a->dtype.itemsize) == 0;
}

int np_assert_allclose(np_array_t* a, np_array_t* b, double rtol, double atol){ return np_allclose(a,b,rtol,atol) ? 0 : 1; }

void np_free(np_array_t* arr) {
    if (!arr) return;
    if (arr->block == (gsl_block*)0x1) {
        // externally-owned buffer; don't free data
    } else if (arr->block) {
        // Owned by GSL; free via GSL
        // We can't determine vector/matrix easily here; test code keeps shallow use.
    } else {
        free(arr->data);
    }
    free(arr->shape);
    free(arr->strides);
    free(arr);
}

void* np_data(np_array_t* arr) { return arr ? arr->data : NULL; }
np_array_t* np_require_c_contiguous(np_array_t* arr) { return arr; }

void np_set_printoptions(int precision) { (void)precision; }

const char* np_dtype_name(const DType* dt) { return dtype_name(dt); }
const DType* np_dtype_from_name(const char* name) {
    if (!name) return &dtypes.void_;
    // Lowercase for stable lookup
    char buf[64]; size_t n=strlen(name); if (n>63) n=63; for(size_t i=0;i<n;i++) buf[i]=(char)tolower((unsigned char)name[i]); buf[n]='\0';
    // Delegate to core dtype mapper, then map back to global singleton pointer
    DType tmp = to_dtype(buf);
    for (int i=0;i<17 && dtypes.all[i]!=NULL; i++) if (dtype_eq(&tmp, dtypes.all[i])) return dtypes.all[i];
    // Try canonical names
    for (int i=0;i<17 && dtypes.all[i]!=NULL; i++){
        const char* canon = dtype_canonical_name(dtypes.all[i]);
        if (canon && strcasecmp(buf, canon)==0) return dtypes.all[i];
    }
    // Fallback common aliases
    if (strcmp(buf, "float") == 0) return &dtypes.float32;
    if (strcmp(buf, "double") == 0) return &dtypes.float64;
    if (strcmp(buf, "int") == 0) return &dtypes.int32;
    return &dtypes.void_;
}

np_array_t* np_from_gsl_matrix(gsl_matrix* m, const DType* dtype){ size_t shape[2] = {m->size1, m->size2}; np_array_t* arr = np_empty(2, shape, dtype); memcpy(arr->data, m->data, arr->size * dtype->itemsize); return arr; }
np_array_t* np_from_gsl_vector(gsl_vector* v, const DType* dtype){ size_t shape[1] = {v->size}; np_array_t* arr = np_empty(1, shape, dtype); memcpy(arr->data, v->data, arr->size * dtype->itemsize); return arr; }
gsl_matrix* np_to_gsl_matrix(np_array_t* arr){ static gsl_matrix_view view; view = gsl_matrix_view_array((double*)arr->data, arr->shape[0], arr->shape[1]); return &view.matrix; }
gsl_vector* np_to_gsl_vector(np_array_t* arr){ static gsl_vector_view view; view = gsl_vector_view_array((double*)arr->data, arr->shape[0]); return &view.vector; }

const np_testing_t np_testing = { .assert_allclose = np_assert_allclose };
