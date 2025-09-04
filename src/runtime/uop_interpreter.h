#pragma once

#include "gradient/gradient.h"
#include "numpy_compat.h"

// Evaluate a UOp graph into a numpy-compatible array
np_array_t* uop_interpreter_evaluate(tg_uop_t* uop);

// Helper to convert tensor data to np_array
np_array_t* tensor_data_to_np_array(tg_tensor_t* tensor);

// Helper to extract scalar value from np_array
float np_array_get_scalar(np_array_t* arr);