#pragma once

#include "gradient/gradient.h"
#include "compat/numpy.h"

// Variable binding for the interpreter
typedef struct {
    tg_uop_t* var_uop;      // The DEFINE_VAR UOp
    np_array_t* data;       // The actual data for this variable
} var_binding_t;

typedef struct {
    var_binding_t* bindings;
    int binding_count;
} eval_context_t;

// Evaluate a UOp graph into a numpy-compatible array
np_array_t* uop_interpreter_evaluate(tg_uop_t* uop);

// Evaluate with variable bindings
np_array_t* uop_interpreter_evaluate_with_context(tg_uop_t* uop, eval_context_t* ctx);

// Helper to convert tensor data to np_array
np_array_t* tensor_data_to_np_array(tg_tensor_t* tensor);

// Helper to extract scalar value from np_array
float np_array_get_scalar(np_array_t* arr);
