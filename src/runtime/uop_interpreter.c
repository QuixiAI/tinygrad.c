#include "uop/uop.h"  // For OPS_ enums and UOp structure
#include "tensor/tensor.h"  // For struct tg_tensor
#include "runtime/uop_interpreter.h"
#include <math.h>
#include <string.h>
#include <assert.h>

// Cache for already evaluated UOps to avoid recomputation
typedef struct {
    tg_uop_t* uop;
    np_array_t* result;
} uop_cache_entry_t;

static uop_cache_entry_t* cache = NULL;
static size_t cache_size = 0;
static size_t cache_capacity = 0;

static np_array_t* get_cached(tg_uop_t* uop) {
    for (size_t i = 0; i < cache_size; i++) {
        if (cache[i].uop == uop) {
            return cache[i].result;
        }
    }
    return NULL;
}

static void cache_result(tg_uop_t* uop, np_array_t* result) {
    if (cache_size >= cache_capacity) {
        cache_capacity = cache_capacity ? cache_capacity * 2 : 16;
        cache = realloc(cache, cache_capacity * sizeof(uop_cache_entry_t));
    }
    cache[cache_size].uop = uop;
    cache[cache_size].result = result;
    cache_size++;
}

// Forward declarations
void uop_interpreter_clear_cache(void);

// Helper function to perform element-wise binary operations
static void compute_row_major_strides(const size_t* shape, size_t ndim, size_t* strides_out) {
    size_t acc = 1;
    for (size_t i = ndim; i-- > 0;) { strides_out[i] = acc; acc *= shape[i]; }
}

static np_array_t* binary_op(np_array_t* left, np_array_t* right, const char* op) {
    assert(left && right);
    // Determine output shape via numpy-style broadcasting
    size_t ndl = left->ndim, ndr = right->ndim;
    size_t nd = ndl > ndr ? ndl : ndr;
    size_t* out_shape = calloc(nd, sizeof(size_t));
    for (size_t i = 0; i < nd; i++) {
        size_t dl = (i < nd - ndl) ? 1 : left->shape[i - (nd - ndl)];
        size_t dr = (i < nd - ndr) ? 1 : right->shape[i - (nd - ndr)];
        if (dl != dr && dl != 1 && dr != 1) {
            // Incompatible shapes: fallback 1D with max size
            size_t out_size = left->size > right->size ? left->size : right->size;
            np_array_t* fallback = np_zeros(1, &out_size, &dtypes.float32);
            free(out_shape);
            // Fill using flat broadcasting
            float* out = (float*)fallback->data, *ld=(float*)left->data, *rd=(float*)right->data;
            for (size_t i2=0;i2<out_size;i2++) {
                float l = left->size == 1 ? ld[0] : ld[i2 % left->size];
                float r = right->size == 1 ? rd[0] : rd[i2 % right->size];
                if (strcmp(op, "ADD") == 0) out[i2] = l + r;
                else if (strcmp(op, "MUL") == 0) out[i2] = l * r;
                else if (strcmp(op, "SUB") == 0) out[i2] = l - r;
                else if (strcmp(op, "DIV") == 0) out[i2] = r==0? INFINITY : l / r;
                else if (strcmp(op, "POW") == 0) out[i2] = powf(l, r);
                else if (strcmp(op, "MAX") == 0) out[i2] = l > r ? l : r;
                else if (strcmp(op, "CMPLT") == 0) out[i2] = l < r ? 1.0f : 0.0f;
                else if (strcmp(op, "CMPEQ") == 0) out[i2] = l == r ? 1.0f : 0.0f;
                else if (strcmp(op, "CMPNE") == 0) out[i2] = l != r ? 1.0f : 0.0f;
            }
            return fallback;
        }
        out_shape[i] = dl == 1 ? dr : dl;
    }
    np_array_t* result = np_zeros(nd, out_shape, &dtypes.float32);
    size_t* lshape = calloc(nd, sizeof(size_t));
    size_t* rshape = calloc(nd, sizeof(size_t));
    for (size_t i=0;i<nd;i++) {
        lshape[i] = (i < nd - ndl) ? 1 : left->shape[i - (nd - ndl)];
        rshape[i] = (i < nd - ndr) ? 1 : right->shape[i - (nd - ndr)];
    }
    size_t* lstr = calloc(nd, sizeof(size_t));
    size_t* rstr = calloc(nd, sizeof(size_t));
    compute_row_major_strides(lshape, nd, lstr);
    compute_row_major_strides(rshape, nd, rstr);
    size_t* ostr = calloc(nd, sizeof(size_t));
    compute_row_major_strides(out_shape, nd, ostr);
    float* outd = (float*)result->data; float* ld=(float*)left->data; float* rd=(float*)right->data;
    size_t total = result->size;
    for (size_t idx=0; idx<total; idx++) {
        // unravel idx into indices
        size_t rem = idx; size_t li=0, ri=0;
        for (size_t d=0; d<nd; d++) {
            size_t coord = rem / ostr[d]; rem %= ostr[d];
            li += (lshape[d]==1 ? 0 : coord) * lstr[d];
            ri += (rshape[d]==1 ? 0 : coord) * rstr[d];
        }
        float l = ld[li], r = rd[ri];
        if (strcmp(op, "ADD") == 0) outd[idx] = l + r;
        else if (strcmp(op, "MUL") == 0) outd[idx] = l * r;
        else if (strcmp(op, "SUB") == 0) outd[idx] = l - r;
        else if (strcmp(op, "DIV") == 0) outd[idx] = r==0? INFINITY : l / r;
        else if (strcmp(op, "POW") == 0) outd[idx] = powf(l, r);
        else if (strcmp(op, "MAX") == 0) outd[idx] = l > r ? l : r;
        else if (strcmp(op, "CMPLT") == 0) outd[idx] = l < r ? 1.0f : 0.0f;
        else if (strcmp(op, "CMPEQ") == 0) outd[idx] = l == r ? 1.0f : 0.0f;
        else if (strcmp(op, "CMPNE") == 0) outd[idx] = l != r ? 1.0f : 0.0f;
    }
    free(out_shape); free(lshape); free(rshape); free(lstr); free(rstr); free(ostr);
    return result;
}

// Helper function for unary operations
static np_array_t* unary_op(np_array_t* input, const char* op) {
    assert(input);
    
    np_array_t* result = np_zeros(1, &input->size, &dtypes.float32);
    float* res_data = (float*)result->data;
    float* in_data = (float*)input->data;
    
    for (size_t i = 0; i < input->size; i++) {
        if (strcmp(op, "NEG") == 0) {
            res_data[i] = -in_data[i];
        } else if (strcmp(op, "RECIP") == 0) {
            res_data[i] = 1.0f / in_data[i];
        } else if (strcmp(op, "SIN") == 0) {
            res_data[i] = sinf(in_data[i]);
        } else if (strcmp(op, "COS") == 0) {
            res_data[i] = cosf(in_data[i]);
        } else if (strcmp(op, "EXP") == 0) {
            res_data[i] = expf(in_data[i]);
        } else if (strcmp(op, "EXP2") == 0) {
            res_data[i] = exp2f(in_data[i]);
        } else if (strcmp(op, "LOG") == 0) {
            res_data[i] = logf(in_data[i]);
        } else if (strcmp(op, "LOG2") == 0) {
            res_data[i] = log2f(in_data[i]);
        } else if (strcmp(op, "SQRT") == 0) {
            res_data[i] = sqrtf(in_data[i]);
        }
    }
    
    return result;
}

// Global context for current evaluation
static eval_context_t* current_context = NULL;

np_array_t* uop_interpreter_evaluate_with_context(tg_uop_t* uop, eval_context_t* ctx) {
    // Save previous context and set new one
    eval_context_t* prev_context = current_context;
    current_context = ctx;
    
    // Clear cache when context changes
    uop_interpreter_clear_cache();
    
    // Evaluate with the new context
    np_array_t* result = uop_interpreter_evaluate(uop);
    
    // Restore previous context
    current_context = prev_context;
    
    return result;
}

np_array_t* uop_interpreter_evaluate(tg_uop_t* uop) {
    if (!uop) return NULL;
    
    // Check cache first
    np_array_t* cached = get_cached(uop);
    if (cached) return cached;
    
    // Debug: Print what operation we're evaluating
    if (getenv("DEBUG_INTERPRETER")) {
        fprintf(stderr, "Evaluating UOp op=%d\n", uop->op);
    }
    
    np_array_t* result = NULL;
    
    switch (uop->op) {
        case OPS_CONST: {
            // Create a scalar constant
            result = np_ones(1, &(size_t){1}, &dtypes.float32);
            float* data = (float*)result->data;
            // The const value is stored in arg.const_data.const_value
            data[0] = (float)uop->arg.const_data.const_value;
            break;
        }
        
        case OPS_ADD: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "ADD");
            break;
        }
        
        case OPS_MUL: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "MUL");
            if (getenv("DEBUG_INTERPRETER")) {
                fprintf(stderr, "MUL: left[0]=%f, right[0]=%f, result[0]=%f\n", 
                       left ? ((float*)left->data)[0] : 0.0f,
                       right ? ((float*)right->data)[0] : 0.0f,
                       result ? ((float*)result->data)[0] : 0.0f);
            }
            break;
        }
        
        case OPS_SUB: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "SUB");
            break;
        }
        case OPS_FDIV: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "DIV");
            break;
        }
        
        case OPS_NEG: {
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            result = unary_op(input, "NEG");
            break;
        }
        
        case OPS_RECIP: {
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            result = unary_op(input, "RECIP");
            break;
        }
        
        case OPS_POW: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "POW");
            break;
        }
        
        case OPS_MAX: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "MAX");
            break;
        }
        case OPS_CMPLT: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "CMPLT");
            break;
        }
        case OPS_CMPEQ: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "CMPEQ");
            break;
        }
        case OPS_CMPNE: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "CMPNE");
            break;
        }
        
        case OPS_SIN: {
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            result = unary_op(input, "SIN");
            break;
        }
        
        case OPS_EXP2: {
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            result = unary_op(input, "EXP2");
            break;
        }
        
        case OPS_LOG2: {
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            result = unary_op(input, "LOG2");
            break;
        }
        
        case OPS_SQRT: {
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            result = unary_op(input, "SQRT");
            break;
        }
        
        case OPS_WHERE: {
            np_array_t* cond = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* true_val = uop_interpreter_evaluate(uop->src[1]);
            np_array_t* false_val = uop_interpreter_evaluate(uop->src[2]);
            
            size_t size = cond->size;
            result = np_zeros(1, &size, &dtypes.float32);
            float* res = (float*)result->data;
            float* c = (float*)cond->data;
            float* t = (float*)true_val->data;
            float* f = (float*)false_val->data;
            
            for (size_t i = 0; i < size; i++) {
                res[i] = c[i] != 0.0f ? t[i] : f[i];
            }
            break;
        }
        
        case OPS_CAST: {
            // Pass-through in float32 interpreter
            result = uop_interpreter_evaluate(uop->src[0]);
            break;
        }
        
        case OPS_REDUCE_AXIS: {
            // Sum over specified axes
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            int axc = uop->arg.reduce_data.axes_count;
            if (axc <= 0) {
                // Reduce all dims
                result = np_ones(1, &(size_t){1}, &dtypes.float32);
                float s=0.0f; float* in=(float*)input->data; for(size_t i=0;i<input->size;i++) s+=in[i];
                ((float*)result->data)[0]=s; break;
            }
            // Build output shape by removing axes (assume unique and sorted for now)
            bool* red = calloc(input->ndim, sizeof(bool));
            for (int i=0;i<axc;i++) { int a=uop->arg.reduce_data.axes[i]; if (a<0) a += (int)input->ndim; if (a>=0 && a<(int)input->ndim) red[a]=true; }
            size_t out_nd=0; for (size_t i=0;i<input->ndim;i++) if (!red[i]) out_nd++;
            if (out_nd==0) { result = np_ones(1, &(size_t){1}, &dtypes.float32); float s=0.0f; float* in=(float*)input->data; for(size_t i=0;i<input->size;i++) s+=in[i]; ((float*)result->data)[0]=s; free(red); break; }
            size_t* out_shape = calloc(out_nd, sizeof(size_t));
            for (size_t i=0, j=0;i<input->ndim;i++) if (!red[i]) out_shape[j++]=input->shape[i];
            result = np_zeros(out_nd, out_shape, &dtypes.float32);
            size_t* istr = calloc(input->ndim, sizeof(size_t)); compute_row_major_strides(input->shape, input->ndim, istr);
            size_t* ostr = calloc(out_nd, sizeof(size_t)); compute_row_major_strides(out_shape, out_nd, ostr);
            float* out=(float*)result->data; float* in=(float*)input->data;
            // Iterate over out indices and sum over reduced dims
            size_t total_out = result->size;
            for (size_t oid=0; oid<total_out; oid++) {
                // unravel out indices
                size_t rem=oid; size_t coords_out[32]={0}; for (size_t d=0; d<out_nd; d++){ coords_out[d]= rem / ostr[d]; rem%=ostr[d]; }
                // build full coords for input
                size_t coords_in[32]={0}; for (size_t d=0, j=0; d<input->ndim; d++){ if (!red[d]) coords_in[d]=coords_out[j++]; else coords_in[d]=0; }
                float sum=0.0f; // iterate reduced dims naive
                // brute force over reduced dims
                size_t loops=1; for(size_t d=0; d<input->ndim; d++) if (red[d]) loops*=input->shape[d];
                for (size_t r=0; r<loops; r++) {
                    // map r into reduced dims coords
                    size_t tmp=r; for (size_t d=input->ndim; d-- >0;) if (red[d]) { size_t dim = input->shape[d]; coords_in[d] = tmp % dim; tmp/=dim; }
                    size_t li=0; for(size_t d=0; d<input->ndim; d++) li += coords_in[d]*istr[d];
                    sum += in[li];
                }
                out[oid]=sum;
            }
            if (getenv("DEBUG_GRAD") && result && result->size>0) {
                fprintf(stderr, "REDUCE_AXIS ndim=%zu axes=%d -> out_nd=%zu first=%f\n", input->ndim, axc, result->ndim, ((float*)result->data)[0]);
            }
            free(red); free(out_shape); free(istr); free(ostr);
            break;
        }
        
        case OPS_DEFINE_VAR: {
            // Variable placeholder - check if we have bound data in the context
            if (getenv("DEBUG_INTERPRETER")) {
                fprintf(stderr, "DEFINE_VAR: context=%p, binding_count=%d\n", 
                       (void*)current_context, current_context ? current_context->binding_count : 0);
            }
            if (current_context) {
                for (int i = 0; i < current_context->binding_count; i++) {
                    if (current_context->bindings[i].var_uop == uop) {
                        // Found the binding - return a copy of the data
                        np_array_t* bound_data = current_context->bindings[i].data;
                        result = np_zeros(bound_data->ndim, bound_data->shape, &bound_data->dtype);
                        memcpy(result->data, bound_data->data, bound_data->size * bound_data->dtype.itemsize);
                        if (getenv("DEBUG_INTERPRETER")) {
                            fprintf(stderr, "  Found binding %d: data[0]=%f\n", i, ((float*)result->data)[0]);
                        }
                        break;
                    }
                }
            }
            
            // Fallback if no binding found
            if (!result) {
                result = np_ones(1, &(size_t){1}, &dtypes.float32);
                // If we have variable bounds, use them to generate test data
                if (uop->vmin_vmax_valid && uop->vmax > uop->vmin) {
                    float* data = (float*)result->data;
                    // Use a value within the bounds for testing
                    data[0] = (float)((uop->vmin + uop->vmax) / 2.0);
                }
            }
            break;
        }
        
        case OPS_CONTIGUOUS: {
            if (uop->src_count>0) result = uop_interpreter_evaluate(uop->src[0]);
            break;
        }
        case OPS_RESHAPE: {
            // Use attached ShapeTracker if present
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            ShapeTracker* st = uop->st;
            if (!st) { result = input; break; }
            int32_t nd = shapetracker_ndim(st); const int32_t* shp = shapetracker_shape(st);
            if (nd <= 0 || !shp) { result = input; break; }
            size_t* out_shape = calloc((size_t)nd, sizeof(size_t)); for (int i=0;i<nd;i++) out_shape[i]=(size_t)shp[i];
            result = np_zeros((size_t)nd, out_shape, &dtypes.float32);
            size_t copy = input->size < result->size ? input->size : result->size;
            memcpy(result->data, input->data, copy*sizeof(float));
            free(out_shape);
            break;
        }
        case OPS_PERMUTE: {
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            int axc = uop->arg.reduce_data.axes_count;
            if (axc <= 0 || (size_t)axc != input->ndim) { result = input; break; }
            size_t* out_shape = calloc(input->ndim, sizeof(size_t));
            for (size_t i=0;i<input->ndim;i++) out_shape[i] = input->shape[uop->arg.reduce_data.axes[i]];
            result = np_zeros(input->ndim, out_shape, &dtypes.float32);
            float* in=(float*)input->data; float* out=(float*)result->data;
            size_t* istr=calloc(input->ndim,sizeof(size_t)); compute_row_major_strides(input->shape,input->ndim,istr);
            size_t* ostr=calloc(result->ndim,sizeof(size_t)); compute_row_major_strides(result->shape,result->ndim,ostr);
            // For each output index, map to input using axes: icoords[axes[i]] = ocoords[i]
            for (size_t oi=0; oi<result->size; oi++){
                size_t rem=oi; size_t ocoords[32]={0}; for(size_t d=0; d<result->ndim; d++){ ocoords[d]= rem/ostr[d]; rem%=ostr[d]; }
                size_t icoords[32]={0}; for (size_t d=0; d<result->ndim; d++){ icoords[uop->arg.reduce_data.axes[d]] = ocoords[d]; }
                size_t ii=0; for(size_t d=0; d<input->ndim; d++) ii += icoords[d]*istr[d];
                out[oi]=in[ii];
            }
            free(out_shape); free(istr); free(ostr);
            break;
        }
        case OPS_EXPAND: {
            // Broadcast input to target shape from attached ShapeTracker
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            ShapeTracker* st = uop->st;
            if (!st) { result = input; break; }
            int32_t nd = shapetracker_ndim(st); const int32_t* shp = shapetracker_shape(st);
            if (nd <= 0 || !shp) { result = input; break; }
            size_t* out_shape = calloc((size_t)nd, sizeof(size_t)); for (int i=0;i<nd;i++) out_shape[i]=(size_t)shp[i];
            np_array_t* target = np_zeros((size_t)nd, out_shape, &dtypes.float32);
            // Use binary_op to broadcast-add zero; reuse broadcasting logic
            result = binary_op(input, target, "ADD");
            np_free(target); free(out_shape);
            break;
        }
        case OPS_PAD:
        {
            // Materialize padding with zeros
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            int ndim = input->ndim;
            int n = uop->arg.pad_data.ndim;
            if (n <= 0 || n != (int)ndim) { result = input; break; }
            size_t* out_shape = calloc(ndim, sizeof(size_t));
            for (int i=0;i<ndim;i++) out_shape[i] = (size_t)( (int)input->shape[i] + uop->arg.pad_data.before[i] + uop->arg.pad_data.after[i] );
            result = np_zeros(ndim, out_shape, &dtypes.float32);
            free(out_shape);
            // copy data into padded region at offsets = before[]
            size_t* istr = calloc(ndim, sizeof(size_t)); compute_row_major_strides(input->shape, ndim, istr);
            size_t* ostr = calloc(ndim, sizeof(size_t)); compute_row_major_strides(result->shape, result->ndim, ostr);
            float* in=(float*)input->data; float* out=(float*)result->data;
            // Iterate over input indices and place into output
            size_t total = input->size; size_t* coords = calloc(ndim, sizeof(size_t));
            for (size_t ii=0; ii<total; ii++) {
                size_t rem=ii; for (int d=0; d<ndim; d++) { coords[d] = rem / istr[d]; rem %= istr[d]; }
                size_t oi=0; for (int d=0; d<ndim; d++) oi += (coords[d] + (size_t)uop->arg.pad_data.before[d]) * ostr[d];
                out[oi] = in[ii];
            }
            free(istr); free(ostr); free(coords);
            break;
        }
        case OPS_SHRINK: {
            // Slice input per start/end
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            int ndim = input->ndim;
            int n = uop->arg.shrink_data.ndim;
            if (n <= 0 || n != (int)ndim) { result = input; break; }
            size_t* out_shape = calloc(ndim, sizeof(size_t));
            for (int i=0;i<ndim;i++) {
                int32_t st = uop->arg.shrink_data.start[i];
                int32_t ed = uop->arg.shrink_data.end[i];
                if (ed < 0) ed = (int32_t)input->shape[i];
                out_shape[i] = (size_t)(ed - st);
            }
            result = np_zeros(ndim, out_shape, &dtypes.float32);
            size_t* istr = calloc(ndim, sizeof(size_t)); compute_row_major_strides(input->shape, ndim, istr);
            size_t* ostr = calloc(ndim, sizeof(size_t)); compute_row_major_strides(result->shape, result->ndim, ostr);
            float* in=(float*)input->data; float* out=(float*)result->data;
            // Iterate over out indices, map to input with +start offsets
            size_t total = result->size; size_t* coords = calloc(ndim, sizeof(size_t));
            for (size_t oi=0; oi<total; oi++) {
                size_t rem=oi; for (int d=0; d<ndim; d++) { coords[d] = rem / ostr[d]; rem %= ostr[d]; }
                size_t ii=0; for (int d=0; d<ndim; d++) ii += (coords[d] + (size_t)uop->arg.shrink_data.start[d]) * istr[d];
                out[oi] = in[ii];
            }
            free(out_shape); free(istr); free(ostr); free(coords);
            break;
        }
        case OPS_FLIP: {
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            int axis = uop->arg.int_data.i;
            if (axis < 0) axis += (int)input->ndim;
            if (axis < 0 || (size_t)axis >= input->ndim) { result = input; break; }
            result = np_zeros(input->ndim, input->shape, &dtypes.float32);
            // naive flip
            size_t* istr=calloc(input->ndim,sizeof(size_t)); compute_row_major_strides(input->shape,input->ndim,istr);
            size_t* ostr=calloc(input->ndim,sizeof(size_t)); compute_row_major_strides(input->shape,input->ndim,ostr);
            float* in=(float*)input->data; float* out=(float*)result->data;
            for (size_t oi=0; oi<result->size; oi++){
                size_t rem=oi; size_t coords[32]={0}; for(size_t d=0; d<input->ndim; d++){ coords[d]= rem/ostr[d]; rem%=ostr[d]; }
                coords[axis] = input->shape[axis]-1 - coords[axis];
                size_t ii=0; for(size_t d=0; d<input->ndim; d++) ii += coords[d]*istr[d];
                out[oi]=in[ii];
            }
            free(istr); free(ostr);
            break;
        }
            
        default:
            fprintf(stderr, "WARNING: Unhandled UOp operation: %d (%s)\n", uop->op, ops_to_string(uop->op));
            // Return a default value
            result = np_zeros(1, &(size_t){1}, &dtypes.float32);
            break;
    }
    
    // Cache the result
    if (result) {
        cache_result(uop, result);
    }
    
    return result;
}

np_array_t* tensor_data_to_np_array(tg_tensor_t* tensor) {
    struct tg_tensor* t = (struct tg_tensor*)tensor;
    if (!t || !t->data) return NULL;
    
    // Create np_array from tensor data
    size_t size = t->numel;
    np_array_t* arr = np_frombuffer(t->data, size * sizeof(float), &dtypes.float32);
    
    // Set proper shape if needed
    if (t->rank > 1) {
        free(arr->shape);
        arr->shape = calloc(t->rank, sizeof(size_t));
        arr->ndim = t->rank;
        for (int i = 0; i < t->rank; i++) {
            arr->shape[i] = t->shape[i];
        }
    }
    
    return arr;
}

float np_array_get_scalar(np_array_t* arr) {
    if (!arr || !arr->data || arr->size == 0) return 0.0f;
    return ((float*)arr->data)[0];
}

// Clear cache when needed
void uop_interpreter_clear_cache(void) {
    if (cache) {
        // Note: We don't free the np_arrays as they may be used elsewhere
        free(cache);
        cache = NULL;
        cache_size = 0;
        cache_capacity = 0;
    }
}
