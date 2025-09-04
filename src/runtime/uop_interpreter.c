#define TG_UOP_INTERNAL  // Enable access to internal UOp structure
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

// Helper function to perform element-wise binary operations
static np_array_t* binary_op(np_array_t* left, np_array_t* right, const char* op) {
    assert(left && right);
    assert(left->size == right->size || left->size == 1 || right->size == 1);
    
    size_t out_size = left->size > right->size ? left->size : right->size;
    np_array_t* result = np_zeros(1, &out_size, &dtypes.float32);
    float* res_data = (float*)result->data;
    float* left_data = (float*)left->data;
    float* right_data = (float*)right->data;
    
    for (size_t i = 0; i < out_size; i++) {
        float l = left->size == 1 ? left_data[0] : left_data[i];
        float r = right->size == 1 ? right_data[0] : right_data[i];
        
        if (strcmp(op, "ADD") == 0) {
            res_data[i] = l + r;
        } else if (strcmp(op, "MUL") == 0) {
            res_data[i] = l * r;
        } else if (strcmp(op, "SUB") == 0) {
            res_data[i] = l - r;
        } else if (strcmp(op, "DIV") == 0) {
            res_data[i] = l / r;
        } else if (strcmp(op, "POW") == 0) {
            res_data[i] = powf(l, r);
        } else if (strcmp(op, "MAX") == 0) {
            res_data[i] = l > r ? l : r;
        } else if (strcmp(op, "CMPLT") == 0) {
            res_data[i] = l < r ? 1.0f : 0.0f;
        } else if (strcmp(op, "CMPEQ") == 0) {
            res_data[i] = l == r ? 1.0f : 0.0f;
        } else if (strcmp(op, "CMPNE") == 0) {
            res_data[i] = l != r ? 1.0f : 0.0f;
        }
    }
    
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
            // The const value is stored in arg.const_value
            data[0] = uop->arg.const_value;
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
            break;
        }
        
        case OPS_SUB: {
            np_array_t* left = uop_interpreter_evaluate(uop->src[0]);
            np_array_t* right = uop_interpreter_evaluate(uop->src[1]);
            result = binary_op(left, right, "SUB");
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
            // For now, just pass through the value
            result = uop_interpreter_evaluate(uop->src[0]);
            break;
        }
        
        case OPS_REDUCE_AXIS: {
            // Simple sum reduction for now
            np_array_t* input = uop_interpreter_evaluate(uop->src[0]);
            result = np_ones(1, &(size_t){1}, &dtypes.float32);
            float* res = (float*)result->data;
            float* in = (float*)input->data;
            
            res[0] = 0.0f;
            for (size_t i = 0; i < input->size; i++) {
                res[0] += in[i];
            }
            break;
        }
        
        case OPS_DEFINE_VAR: {
            // Variable placeholder - check if there's associated tensor data
            // The arg.var.name might help identify which tensor this represents
            // For gradient computation, input tensors usually have their data
            // For now, create array with values based on variable properties
            result = np_ones(1, &(size_t){1}, &dtypes.float32);
            
            // If we have variable bounds, use them to generate test data
            if (uop->arg.var.vmax > uop->arg.var.vmin) {
                float* data = (float*)result->data;
                // Use a value within the bounds for testing
                data[0] = (uop->arg.var.vmin + uop->arg.var.vmax) / 2.0f;
            }
            break;
        }
        
        case OPS_CONTIGUOUS:
        case OPS_RESHAPE:
        case OPS_PERMUTE:
        case OPS_EXPAND:
        case OPS_PAD:
        case OPS_SHRINK:
        case OPS_FLIP:
            // Shape operations - for now just pass through
            if (uop->src_count > 0) {
                result = uop_interpreter_evaluate(uop->src[0]);
            }
            break;
            
        default:
            fprintf(stderr, "WARNING: Unhandled UOp operation: %d\n", uop->op);
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